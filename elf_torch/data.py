from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .sampling import build_self_attn_cond_masks, get_pad_token_id


def _as_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def pad_and_truncate(
    ids_list: list[list[int]],
    target_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = []
    lengths = []
    for ids in ids_list:
        length = min(len(ids), target_len)
        ids = ids[:target_len]
        ids = ids + [pad_token_id] * (target_len - len(ids))
        padded.append(ids)
        lengths.append(length)
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def collate_elf_batch(
    batch_list: list[dict[str, Any]],
    *,
    max_seq_length: int,
    pad_token_id: int,
    max_input_seq_length: int | None = None,
) -> dict[str, Any]:
    if not batch_list:
        raise ValueError("Cannot collate an empty batch.")

    has_condition = "condition_input_ids" in batch_list[0]
    seq_list = []
    cond_lens = []

    for item in batch_list:
        if has_condition:
            max_cond_len = max_input_seq_length or max_seq_length
            cond = _as_list(item["condition_input_ids"])[:max_cond_len]
            target = _as_list(item["input_ids"])
            seq_list.append(cond + target)
            cond_lens.append(min(len(cond), max_seq_length))
        else:
            seq_list.append(_as_list(item["input_ids"]))
            cond_lens.append(0)

    input_ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
    cond_lens_tensor = torch.tensor(cond_lens, dtype=torch.long)
    pos = torch.arange(max_seq_length)[None, :]
    is_cond = pos < cond_lens_tensor[:, None]
    is_valid = pos < total_lens[:, None]
    encoder_attention_mask, attention_mask, cond_seq_mask = build_self_attn_cond_masks(
        is_cond, is_valid
    )

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "attention_mask": attention_mask,
        "cond_seq_mask": cond_seq_mask,
    }
    for key in ("index", "input", "target"):
        if key in batch_list[0]:
            result[key] = [item.get(key) for item in batch_list]
    return result


def get_dataloader(
    dataset,
    tokenizer,
    config,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    def collate_fn(batch_list):
        return collate_elf_batch(
            batch_list,
            max_seq_length=config.max_length,
            pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def add_label_drop_mask(batch: dict[str, Any], label_drop_prob: float) -> dict[str, Any]:
    input_ids = batch["input_ids"]
    if label_drop_prob > 0:
        batch["label_drop_mask"] = (
            torch.rand(input_ids.shape[0], device=input_ids.device) < label_drop_prob
        )
    else:
        batch["label_drop_mask"] = torch.zeros(
            input_ids.shape[0], device=input_ids.device, dtype=torch.bool
        )
    return batch


def load_jsonl_dataset(path: str | Path, tokenizer, input_key: str = "input", output_key: str = "output"):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            source = data[input_key]
            target = data[output_key]
            examples.append(
                {
                    "index": index,
                    "input": source,
                    "target": target,
                    "condition_input_ids": tokenizer(source, add_special_tokens=False)["input_ids"],
                    "input_ids": tokenizer(target, add_special_tokens=False)["input_ids"],
                }
            )
    return examples


def _looks_like_save_to_disk_arrow(ds) -> bool:
    return (
        len(ds) == 1
        and any(column.startswith("_") for column in ds.column_names)
        and not any(not column.startswith("_") for column in ds.column_names)
    )


def _single_split(ds, path: str):
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected {path!r} to have one split, got {splits}.")
        return ds[splits[0]]
    return ds


def load_dataset_split(path: str, *, dataset_cache_dir: str | None = None):
    from datasets import load_dataset as hf_load_dataset
    from datasets import load_from_disk

    path_obj = Path(path)
    if path_obj.suffix == ".jsonl":
        raise ValueError("JSONL datasets need a tokenizer; call load_jsonl_dataset directly.")

    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    ds = _single_split(ds, path)
    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=path,
            repo_type="dataset",
            cache_dir=dataset_cache_dir,
        )
        ds = _single_split(load_from_disk(local_dir), path)

    torch_columns = [
        column
        for column in ds.column_names
        if column in {"input_ids", "condition_input_ids", "index"}
    ]
    ds.set_format(type="torch", columns=torch_columns, output_all_columns=True)
    return ds


def load_train_eval_datasets(config, tokenizer, *, dataset_cache_dir: str | None = None):
    if not config.data_path:
        raise ValueError("config.data_path is required for training.")

    data_path = Path(config.data_path)
    if data_path.suffix == ".jsonl":
        train_dataset = load_jsonl_dataset(data_path, tokenizer)
    else:
        train_dataset = load_dataset_split(config.data_path, dataset_cache_dir=dataset_cache_dir)

    eval_dataset = None
    if config.eval_data_path:
        eval_path = Path(config.eval_data_path)
        if eval_path.suffix == ".jsonl":
            eval_dataset = load_jsonl_dataset(eval_path, tokenizer)
        else:
            eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir=dataset_cache_dir)
    return train_dataset, eval_dataset
