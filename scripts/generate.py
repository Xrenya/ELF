#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


THIS_FILE = Path(__file__).resolve()
PORT_ROOT = THIS_FILE.parents[1]


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pytorch_port").is_dir():
            return candidate
    return THIS_FILE.parents[2]


REPO_ROOT_DEFAULT = find_repo_root(THIS_FILE)
if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))

from elf_torch.config import SamplingConfig, apply_config_overrides, load_config_from_yaml
from elf_torch.model import build_elf_from_config
from elf_torch.sampling import generate_conditional, generate_unconditional
from elf_torch.t5_encoder import T5Encoder, T5EncoderConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with converted PyTorch ELF weights.")
    parser.add_argument("--config", default="ELF-B-xsum/ELF-B-xsum.yml")
    parser.add_argument(
        "--elf-checkpoint",
        default="pytorch_port/checkpoints/ELF-B-xsum/elf_model.pt",
        help="Converted PyTorch ELF checkpoint.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default="pytorch_port/checkpoints/ELF-B-xsum/t5_encoder.pt",
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
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT_DEFAULT / path).resolve()


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


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    return payload, {}


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


def main():
    args = parse_args()
    config = load_config_from_yaml(resolve_path(args.config))
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
    elf_state, elf_payload = load_state(resolve_path(args.elf_checkpoint))
    vocab_size = int(elf_payload.get("metadata", {}).get("vocab_size", elf_state["unembed_bias"].shape[0]))
    final_norm = float(elf_state["final_layer.linear.weight"].float().norm())
    if final_norm < 1e-8:
        raise RuntimeError(
            "The converted ELF checkpoint looks untrained: final_layer.linear.weight has near-zero norm. "
            "Reconvert from a complete ELF checkpoint, preferably `--checkpoint embedded-language-flows/ELF-B-xsum`."
        )

    encoder_state = None
    encoder_payload = {}
    if args.prompt or args.prompts_file:
        encoder_state, encoder_payload = load_state(resolve_path(args.encoder_checkpoint))
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
