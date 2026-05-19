#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from elf_torch.config import SamplingConfig, apply_config_overrides, load_config_from_yaml
from elf_torch.checkpoint import load_model_state
from elf_torch.model import build_elf_from_config
from elf_torch.sampling import generate_conditional, generate_unconditional
from elf_torch.t5_encoder import T5Encoder, T5EncoderConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with converted PyTorch ELF weights.")
    parser.add_argument("--config", default="ELF-B-xsum/ELF-B-xsum.yml")
    parser.add_argument(
        "--elf-checkpoint",
        default="ELF-B-xsum/elf_model.pt",
        help="Converted PyTorch ELF checkpoint path or HF repo id.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default="t5_small_encoder/t5_encoder.pt",
        help="Converted PyTorch T5 encoder checkpoint. Required for conditional prompts.",
    )
    parser.add_argument("--prompt", action="append", default=[], help="Conditional prompt. Repeat for a batch.")
    parser.add_argument("--prompts-file", default=None, help="Optional text/jsonl file with prompts.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto default.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--sampling-index", type=int, default=0)
    parser.add_argument("--method", choices=("ode", "sde"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--self-cond-cfg", type=float, default=None)
    parser.add_argument("--sde-gamma", type=float, default=None)
    parser.add_argument("--time-schedule", choices=("uniform", "logit_normal"), default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--model-key",
        choices=("ema_params1", "params", "state_dict", "raw_state_dict"),
        default="ema_params1",
        help="Checkpoint tensor tree to load when the file contains training-state keys.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT_DEFAULT / path).resolve()


def resolve_checkpoint_value(value: str) -> str | Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    repo_path = REPO_ROOT_DEFAULT / path
    if repo_path.exists():
        return repo_path.resolve()
    return value


def load_prompts(args) -> list[str]:
    prompts = list(args.prompt)
    if args.prompts_file:
        path = resolve_path(args.prompts_file)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    data = json.loads(line)
                    prompts.append(data.get("input") or data.get("prompt") or data.get("text") or "")
                else:
                    prompts.append(line)
    if prompts and len(prompts) == 1 and args.num_samples > 1:
        prompts = prompts * args.num_samples
    return prompts


def validate_checkpoint_config(config, checkpoint_payload: dict) -> None:
    checkpoint_config = checkpoint_payload.get("config", {}) if isinstance(checkpoint_payload, dict) else {}
    checkpoint_model = checkpoint_config.get("model")
    if checkpoint_model and checkpoint_model != config.model:
        raise RuntimeError(
            f"Config/checkpoint mismatch: --config builds {config.model}, "
            f"but --elf-checkpoint was saved from {checkpoint_model}. "
            "Use the YAML that belongs to this checkpoint."
        )
    for field in ("data_path", "eval_data_path", "max_length", "max_input_length", "latent_std"):
        checkpoint_value = checkpoint_config.get(field)
        current_value = getattr(config, field, None)
        if checkpoint_value is not None and checkpoint_value != current_value:
            raise RuntimeError(
                f"Config/checkpoint mismatch for {field}: --config has {current_value!r}, "
                f"but --elf-checkpoint was saved with {checkpoint_value!r}. "
                "Use the YAML that belongs to this checkpoint."
            )


def selected_sampling_config(config, args) -> SamplingConfig:
    if not config.sampling_configs:
        sc = SamplingConfig()
    else:
        sc = config.sampling_configs[args.sampling_index]
    sc = SamplingConfig(**vars(sc))
    if args.method is not None:
        sc.sampling_method = args.method
    if args.steps is not None:
        sc.num_sampling_steps = [args.steps]
    if args.cfg is not None:
        sc.cfgs = [args.cfg]
    if args.self_cond_cfg is not None:
        sc.self_cond_cfg_scales = [args.self_cond_cfg]
    if args.sde_gamma is not None:
        sc.sde_gamma = args.sde_gamma
    if args.time_schedule is not None:
        sc.time_schedule = args.time_schedule
    return sc


def apply_official_checkpoint_defaults(config, checkpoint: str) -> None:
    checkpoint = checkpoint.lower()
    common = {
        "encoder_model_name": "t5-small",
        "latent_mean": 0.0,
        "latent_std": 0.2,
        "bottleneck_dim": 128,
        "num_time_tokens": 4,
        "num_self_cond_cfg_tokens": 4,
        "num_model_mode_tokens": 4,
        "denoiser_p_mean": -1.5,
        "denoiser_p_std": 0.8,
        "denoiser_noise_scale": 2.0,
        "t_eps": 0.05,
        "time_schedule": "logit_normal",
        "decoder_prob": 0.2,
        "decoder_p_mean": 0.8,
        "decoder_p_std": 0.8,
        "self_cond_prob": 0.5,
    }
    for key, value in common.items():
        setattr(config, key, value)

    if "elf-m-owt-torch" in checkpoint:
        config.model = "ELF-M"
    elif "elf-l-owt-torch" in checkpoint:
        config.model = "ELF-L"
    else:
        config.model = "ELF-B"

    if "owt-torch" in checkpoint:
        config.data_path = "embedded-language-flows/openwebtext-t5"
        config.eval_data_path = None
        config.max_length = 1024
        config.max_input_length = None
        config.pad_token = "pad"
        config.decoder_noise_scale = 5.0
        config.sampling_configs = [
            SamplingConfig(
                sampling_method="sde",
                num_sampling_steps=[32],
                cfgs=[1],
                self_cond_cfg_scales=[3],
                time_schedule="logit_normal",
                sde_gamma=1.5,
            )
        ]
    elif "xsum-torch" in checkpoint:
        config.data_path = "embedded-language-flows/xsum_train_t5"
        config.eval_data_path = "embedded-language-flows/xsum_validation_t5"
        config.max_length = 1088
        config.max_input_length = 1024
        config.pad_token = "eos"
        config.decoder_noise_scale = 1.0
        config.label_drop_prob = 0.1
        config.sampling_configs = [
            SamplingConfig(
                sampling_method="ode",
                num_sampling_steps=[64],
                cfgs=[2],
                self_cond_cfg_scales=[1],
                time_schedule="logit_normal",
            )
        ]
    elif "de-en-torch" in checkpoint:
        config.data_path = "embedded-language-flows/wmt14_de-en_train_t5"
        config.eval_data_path = "embedded-language-flows/wmt14_de-en_validation_t5"
        config.max_length = 128
        config.max_input_length = 64
        config.pad_token = "eos"
        config.decoder_noise_scale = 1.0
        config.label_drop_prob = 0.1
        config.sampling_configs = [
            SamplingConfig(
                sampling_method="ode",
                num_sampling_steps=[64],
                cfgs=[2],
                self_cond_cfg_scales=[1],
                time_schedule="logit_normal",
            )
        ]


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config_from_yaml(config_path)
    if not config_path.is_file():
        apply_official_checkpoint_defaults(config, args.elf_checkpoint)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    sampling_config = selected_sampling_config(config, args)
    print(
        "sampling:",
        {
            "method": sampling_config.sampling_method,
            "steps": sampling_config.num_sampling_steps,
            "cfgs": sampling_config.cfgs,
            "self_cond_cfg_scales": sampling_config.self_cond_cfg_scales,
            "time_schedule": sampling_config.time_schedule,
            "sde_gamma": sampling_config.sde_gamma,
        },
        file=sys.stderr,
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    elf_state, elf_metadata, elf_path = load_model_state(
        resolve_checkpoint_value(args.elf_checkpoint),
        model_key=args.model_key,
    )
    validate_checkpoint_config(config, {"config": elf_metadata.get("config", {}), "metadata": elf_metadata})
    vocab_size = int(elf_metadata.get("vocab_size", elf_state["unembed_bias"].shape[0]))
    final_norm = float(elf_state["final_layer.linear.weight"].float().norm())
    if final_norm < 1e-8:
        raise RuntimeError(
            "The converted ELF checkpoint looks untrained: final_layer.linear.weight has near-zero norm. "
            "Reconvert from a complete ELF checkpoint, preferably `--checkpoint embedded-language-flows/ELF-B-xsum`."
        )
    print(f"loaded ELF checkpoint: {elf_path}", file=sys.stderr)

    encoder_state = None
    encoder_payload = {}
    if args.prompt or args.prompts_file:
        encoder_state, encoder_payload, encoder_path = load_model_state(
            resolve_checkpoint_value(args.encoder_checkpoint)
        )
        print(f"loaded encoder checkpoint: {encoder_path}", file=sys.stderr)
        t5_config = T5EncoderConfig(**encoder_payload.get("config", {}))
    else:
        t5_config = T5EncoderConfig.from_pretrained(config.encoder_model_name)

    model = build_elf_from_config(config, t5_config.d_model, vocab_size)
    missing, unexpected = model.load_state_dict(elf_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ELF checkpoint mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    model.to(device).eval()

    prompts = load_prompts(args)
    batch_size = args.batch_size or (len(prompts) if prompts else args.num_samples)
    batch_size = max(1, batch_size)

    if prompts:
        encoder = T5Encoder(t5_config)
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"T5 checkpoint mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
        encoder.to(device).eval()
        texts = generate_conditional(
            model,
            encoder,
            tokenizer,
            config,
            sampling_config,
            prompts=prompts,
            batch_size=batch_size,
            text_encoder_dim=t5_config.d_model,
            seed=args.seed,
            show_progress=not args.no_progress,
        )
    else:
        texts = generate_unconditional(
            model,
            tokenizer,
            config,
            sampling_config,
            num_samples=args.num_samples,
            batch_size=batch_size,
            text_encoder_dim=t5_config.d_model,
            seed=args.seed,
            show_progress=not args.no_progress,
        )

    rows = [{"id": i, "generated": text} for i, text in enumerate(texts)]
    if args.output:
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
