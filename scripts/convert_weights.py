#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
PORT_ROOT = THIS_FILE.parents[1]


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "pytorch_port").is_dir():
            return candidate
    return THIS_FILE.parents[2]


REPO_ROOT_DEFAULT = find_repo_root(THIS_FILE)
if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))

from elf_torch.config import load_config_from_yaml
from elf_torch.model import build_elf_from_config
from elf_torch.t5_encoder import T5Encoder, T5EncoderConfig
from elf_torch.weight_conversion import (
    convert_elf_params,
    convert_t5_encoder_params,
    load_strict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ELF Flax/JAX checkpoints and T5 encoder pickle files to PyTorch."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT_DEFAULT), help="Path to the ELF repo root.")
    parser.add_argument(
        "--config",
        default="ELF-B-xsum/ELF-B-xsum.yml",
        help="ELF YAML config for the checkpoint being converted.",
    )
    parser.add_argument(
        "--checkpoint",
        default="ELF-B-xsum",
        help="ELF checkpoint path or HF repo id, e.g. embedded-language-flows/ELF-B-xsum.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default="t5_small_encoder_jax.pkl",
        help="T5 encoder pickle path or HF file path.",
    )
    parser.add_argument(
        "--output-dir",
        default="pytorch_port/checkpoints/ELF-B-xsum",
        help="Where to write elf_model.pt, t5_encoder.pt, and metadata.json.",
    )
    parser.add_argument(
        "--model-key",
        choices=("ema_params1", "params"),
        default="ema_params1",
        help="Which ELF parameter tree to export from the training state.",
    )
    parser.add_argument(
        "--jax-platform",
        default=None,
        help="Optional JAX_PLATFORMS value, e.g. cpu, to avoid a broken CUDA plugin during conversion.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override tokenizer vocab size if AutoTokenizer cannot be loaded.",
    )
    parser.add_argument("--skip-elf", action="store_true", help="Only convert the T5 encoder.")
    parser.add_argument("--skip-encoder", action="store_true", help="Only convert the ELF checkpoint.")
    return parser.parse_args()


def add_original_src_to_path(repo_root: Path):
    if repo_root.name == "pytorch_port":
        repo_root = repo_root.parent
    src = repo_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"Could not find original src directory: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def resolve_path(repo_root: Path, value: str) -> str:
    if repo_root.name == "pytorch_port":
        repo_root = repo_root.parent
    path = Path(value)
    if path.is_absolute() or "://" in value:
        return value
    local = repo_root / path
    if local.exists():
        return str(local)
    return value


def resolve_output_dir(repo_root: Path, value: str) -> Path:
    if repo_root.name == "pytorch_port":
        repo_root = repo_root.parent
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def looks_like_lfs_pointer(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 2048:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/")


def resolve_encoder_checkpoint(repo_root: Path, value: str) -> str:
    resolved = resolve_path(repo_root, value)
    if looks_like_lfs_pointer(resolved):
        fallback = "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
        print(
            f"Encoder checkpoint {resolved} is a Git LFS pointer, not the real pickle; "
            f"using {fallback}"
        )
        return fallback
    return resolved


def normalize_checkpoint_path(repo_root: Path, value: str) -> str:
    resolved = resolve_path(repo_root, value)
    path = Path(resolved)
    if path.is_dir() and path.name.startswith("checkpoint_"):
        parent = str(path.parent)
        print(f"Using checkpoint parent directory for Flax restore: {parent}")
        return parent
    return resolved


def load_original_config(repo_root: Path, config_path: str):
    from configs.config import load_config_from_yaml as load_jax_config

    if repo_root.name == "pytorch_port":
        repo_root = repo_root.parent
    old_cwd = os.getcwd()
    try:
        os.chdir(repo_root / "src")
        return load_jax_config(config_path)
    finally:
        os.chdir(old_cwd)


def tokenizer_vocab_size(config, override: int | None) -> int:
    if override is not None:
        return override
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    return tokenizer.vocab_size


def load_jax_elf_params(repo_root: Path, config_path: str, checkpoint_path: str, model_key: str, vocab_size: int):
    add_original_src_to_path(repo_root)

    import jax
    import jax.numpy as jnp
    import optax

    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint
    from utils.train_utils import TrainState

    config = load_original_config(repo_root, config_path)
    encoder_config, _, _ = get_encoder(config.encoder_model_name, jnp.float32)

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    input_dim = 2 * encoder_config.d_model if config.self_cond_prob > 0 else encoder_config.d_model
    dummy_x = jnp.ones((1, config.max_length, input_dim))
    dummy_t = jnp.ones((1,))
    dummy_self_cond_cfg_scale = (
        jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
    )

    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    init_args = dict(
        x=dummy_x,
        t=dummy_t,
        deterministic=True,
        self_cond_cfg_scale=dummy_self_cond_cfg_scale,
    )
    elf_params = model.init(init_rng, **init_args)
    state = TrainState.create(
        apply_fn=model.apply,
        params=elf_params["params"],
        tx=optax.adamw(learning_rate=1e-4),
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(elf_params["params"]),
    )
    try:
        state, step = load_checkpoint(checkpoint_path, state)
    except Exception as exc:
        raise RuntimeError(
            f"Could not restore ELF checkpoint from {checkpoint_path!r}. "
            "If this is a Hugging Face clone, it is probably missing Git LFS checkpoint blobs. "
            "Run `git lfs pull` inside that model folder, or pass "
            "`--checkpoint embedded-language-flows/ELF-B-xsum` so the converter downloads a full snapshot."
        ) from exc
    selected = state.ema_params1 if model_key == "ema_params1" else state.params
    final_kernel = jax.device_get(selected["final_layer"]["linear"]["kernel"])
    final_kernel_norm = float(jnp.linalg.norm(final_kernel))
    if final_kernel_norm < 1e-8:
        raise RuntimeError(
            "Loaded ELF weights look untrained: final_layer/linear/kernel has near-zero norm. "
            "This usually means the checkpoint path is incomplete or Flax restored the init target. "
            "Use the HF repo id `embedded-language-flows/ELF-B-xsum` or run `git lfs pull` for the checkpoint."
        )
    selected = jax.device_get(selected)
    metadata = {
        "step": int(step),
        "epoch": int(jax.device_get(state.epoch)),
        "model_key": model_key,
        "vocab_size": vocab_size,
        "text_encoder_dim": int(encoder_config.d_model),
        "final_layer_kernel_norm": final_kernel_norm,
    }
    return selected, metadata


def load_jax_encoder_params(repo_root: Path, encoder_checkpoint: str):
    add_original_src_to_path(repo_root)
    from utils.checkpoint_utils import load_encoder_checkpoint

    return load_encoder_checkpoint(encoder_checkpoint)


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    if args.jax_platform:
        os.environ["JAX_PLATFORMS"] = args.jax_platform

    repo_root = Path(args.repo_root).resolve()
    if repo_root.name == "pytorch_port":
        repo_root = repo_root.parent
    config_path = resolve_path(repo_root, args.config)
    checkpoint_path = normalize_checkpoint_path(repo_root, args.checkpoint)
    encoder_checkpoint = resolve_encoder_checkpoint(repo_root, args.encoder_checkpoint)
    output_dir = resolve_output_dir(repo_root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_config = load_config_from_yaml(config_path)
    t5_config = T5EncoderConfig.from_pretrained(torch_config.encoder_model_name)

    manifest = {
        "source_config": config_path,
        "source_checkpoint": checkpoint_path,
        "source_encoder_checkpoint": encoder_checkpoint,
        "model": torch_config.model,
        "encoder_model_name": torch_config.encoder_model_name,
    }

    if not args.skip_encoder:
        print(f"Loading T5 encoder checkpoint: {encoder_checkpoint}")
        encoder_params = load_jax_encoder_params(repo_root, encoder_checkpoint)
        encoder_state = convert_t5_encoder_params(encoder_params, t5_config)
        encoder = T5Encoder(t5_config)
        load_strict(encoder, encoder_state, "T5 encoder")
        encoder_path = output_dir / "t5_encoder.pt"
        torch.save(
            {
                "state_dict": encoder_state,
                "config": asdict(t5_config),
                "source": encoder_checkpoint,
            },
            encoder_path,
        )
        manifest["t5_encoder"] = str(encoder_path)
        print(f"Wrote {encoder_path}")

    if not args.skip_elf:
        add_original_src_to_path(repo_root)
        original_config = load_original_config(repo_root, config_path)
        vocab_size = tokenizer_vocab_size(original_config, args.vocab_size)
        print(f"Loading ELF checkpoint: {checkpoint_path}")
        elf_params, elf_metadata = load_jax_elf_params(
            repo_root,
            config_path,
            checkpoint_path,
            args.model_key,
            vocab_size,
        )
        elf_state = convert_elf_params(elf_params, torch_config)
        elf = build_elf_from_config(torch_config, t5_config.d_model, vocab_size)
        load_strict(elf, elf_state, "ELF")
        elf_path = output_dir / "elf_model.pt"
        torch.save(
            {
                "state_dict": elf_state,
                "config": asdict(torch_config),
                "metadata": elf_metadata,
                "source": checkpoint_path,
            },
            elf_path,
        )
        manifest["elf_model"] = str(elf_path)
        manifest.update(elf_metadata)
        print(f"Wrote {elf_path}")

    save_json(output_dir / "metadata.json", manifest)
    print(f"Wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
