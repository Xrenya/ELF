from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def _split_hf_path(value: str) -> tuple[str, str] | None:
    if "://" in value or value.startswith(("/", ".", "~")):
        return None
    if Path(value).exists():
        return None
    parts = value.split("/")
    if len(parts) < 2:
        return None
    return "/".join(parts[:2]), "/".join(parts[2:])


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem if path.suffix else path.name)
    return int(match.group(1)) if match else -1


def _candidate_checkpoints(path: Path, *, prefer_training: bool = False) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    preferred = []
    for name in ("elf_model.pt", "final.pt", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file():
            preferred.append(candidate)
    numbered = sorted(
        [p for p in path.rglob("checkpoint_*") if p.is_file()],
        key=_checkpoint_step,
    )
    pt_files = sorted(path.rglob("*.pt"), key=_checkpoint_step)
    bin_files = sorted(path.rglob("*.bin"), key=_checkpoint_step)
    if prefer_training:
        return list(reversed(numbered)) + preferred + list(reversed(pt_files)) + list(reversed(bin_files))
    return preferred + list(reversed(numbered)) + list(reversed(pt_files)) + list(reversed(bin_files))


def resolve_checkpoint_path(value: str | Path, *, prefer_training: bool = False) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        candidates = _candidate_checkpoints(path, prefer_training=prefer_training)
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No checkpoint file found in {path}")

    hf_path = _split_hf_path(str(value))
    if hf_path is None:
        raise FileNotFoundError(f"Checkpoint path does not exist: {value}")

    repo_id, sub_path = hf_path
    from huggingface_hub import snapshot_download

    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[sub_path, f"{sub_path}/**"] if sub_path else None,
        )
    )
    root = local_dir / sub_path if sub_path else local_dir
    candidates = _candidate_checkpoints(root, prefer_training=prefer_training)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in downloaded HF repo {value}")
    return candidates[0]


def normalize_elf_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        key = key.replace(".mlp_0.", ".mlp.0.")
        key = key.replace(".mlp_2.", ".mlp.2.")
        normalized[key] = value
    return normalized


def select_model_state(payload: Any, *, model_key: str = "ema_params1") -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if model_key in payload and isinstance(payload[model_key], dict):
            return normalize_elf_state_dict(payload[model_key])
        for key in ("state_dict", "params", "raw_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return normalize_elf_state_dict(payload[key])
    if isinstance(payload, dict):
        return normalize_elf_state_dict(payload)
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def load_model_state(
    path_or_repo: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    model_key: str = "ema_params1",
) -> tuple[dict[str, torch.Tensor], dict[str, Any], Path]:
    checkpoint_path = resolve_checkpoint_path(path_or_repo)
    payload = torch.load(checkpoint_path, map_location=map_location)
    state = select_model_state(payload, model_key=model_key)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        metadata = {
            **metadata,
            "config": payload.get("config", metadata.get("config", {})),
            "source": payload.get("source", metadata.get("source", "")),
            "step": int(payload.get("step", metadata.get("step", 0))),
            "epoch": float(payload.get("epoch", metadata.get("epoch", 0.0))),
            "model_key": model_key if model_key in payload else metadata.get("model_key", ""),
        }
    return state, metadata, checkpoint_path
