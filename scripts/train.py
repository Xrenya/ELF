#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))


from elf_torch.config import apply_config_overrides, load_config_from_yaml
from elf_torch.data import add_label_drop_mask, get_dataloader, load_train_eval_datasets, move_batch_to_device
from elf_torch.model import build_elf_from_config
from elf_torch.t5_encoder import T5Encoder, T5EncoderConfig
from elf_torch.train_utils import (
    clone_for_ema,
    compute_train_loss,
    config_to_dict,
    load_training_checkpoint,
    make_lr_scheduler,
    make_optimizer,
    save_training_checkpoint,
    update_ema,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF with the PyTorch port.")
    parser.add_argument("--config", required=True, help="ELF YAML config.")
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Converted PyTorch T5 encoder checkpoint. Defaults to config.encoder_checkpoint.",
    )
    parser.add_argument("--output-dir", default=None, help="Override config.output_dir.")
    parser.add_argument("--resume", default=None, help="Resume from a PyTorch checkpoint or output dir.")
    parser.add_argument("--batch-size", type=int, default=None, help="Single-process micro-batch size.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many micro-steps.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto default.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Model compute dtype. Use float32 for safest CPU training.",
    )
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--config-override", "--config_override", action="append", default=[])
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--save-freq", type=float, default=None, help="Epoch frequency; supports fractional values.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT_DEFAULT / path).resolve()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_encoder(path: Path, fallback_model_name: str) -> tuple[T5Encoder, T5EncoderConfig]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = T5EncoderConfig(**payload.get("config", {}))
    else:
        state_dict = payload
        config = T5EncoderConfig.from_pretrained(fallback_model_name)
    encoder = T5Encoder(config)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"T5 checkpoint mismatch: missing={missing[:20]} unexpected={unexpected[:20]}")
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, config


def maybe_init_wandb(config):
    if not getattr(config, "use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("config.use_wandb=True but wandb is not installed.") from exc
    tags = config.wandb_tag.split(",") if config.wandb_tag else None
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        id=config.wandb_run_name,
        resume=config.wandb_resume,
        tags=tags,
        config=config_to_dict(config),
    )
    return wandb


def main():
    args = parse_args()
    config = load_config_from_yaml(resolve_path(args.config))
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.log_freq is not None:
        config.log_freq = args.log_freq
    if args.save_freq is not None:
        config.save_freq = args.save_freq

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    dtype = torch_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU training uses float32; ignoring --dtype for CPU.", file=sys.stderr)
        dtype = torch.float32

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("=" * 60)
    print("ELF PyTorch Training")
    print("=" * 60)
    print(f"Model: {config.model}")
    print(f"Data: {config.data_path}")
    print(f"Output dir: {config.output_dir}")
    print(f"Device: {device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    encoder_checkpoint = resolve_path(args.encoder_checkpoint or config.encoder_checkpoint)
    if encoder_checkpoint is None:
        raise ValueError(
            "--encoder-checkpoint is required for PyTorch training. "
            "Pass the converted t5_encoder.pt, not the original JAX pickle."
        )
    encoder, t5_config = load_encoder(encoder_checkpoint, config.encoder_model_name)
    encoder.to(device=device, dtype=dtype).eval()

    train_dataset, eval_dataset = load_train_eval_datasets(
        config,
        tokenizer,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    if eval_dataset is not None:
        print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")
    else:
        print(f"Train size: {len(train_dataset)}")

    batch_size = args.batch_size or config.batch_size or config.global_batch_size
    if not batch_size:
        raise ValueError("Set --batch-size or config.batch_size/global_batch_size.")
    config.batch_size = int(batch_size)
    config.global_batch_size = int(batch_size)

    steps_per_epoch = max(len(train_dataset) // batch_size, 1)
    total_micro_steps = steps_per_epoch * config.epochs
    if args.max_steps is not None:
        total_micro_steps = min(total_micro_steps, args.max_steps)
    grad_accum_steps = max(int(config.grad_accum_steps), 1)
    total_optimizer_steps = max(
        (total_micro_steps + grad_accum_steps - 1) // grad_accum_steps,
        1,
    )
    if config.warmup_steps >= 0:
        warmup_micro_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        warmup_micro_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        warmup_micro_steps = 0
    warmup_optimizer_steps = warmup_micro_steps // grad_accum_steps
    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * (batch_size * grad_accum_steps) / 256

    model = build_elf_from_config(config, t5_config.d_model, vocab_size)
    model.to(device=device, dtype=dtype).train()
    ema_model = clone_for_ema(model)
    ema_model.to(device=device, dtype=dtype)

    if config.optimizer.lower() == "muon":
        print("PyTorch port currently maps optimizer=muon to AdamW.", file=sys.stderr)
    optimizer = make_optimizer(model, config)
    scheduler = make_lr_scheduler(
        optimizer,
        config,
        total_optimizer_steps=total_optimizer_steps,
        warmup_optimizer_steps=warmup_optimizer_steps,
    )

    resume_path = args.resume or config.resume
    start_step = 0
    start_epoch_float = 0.0
    if resume_path:
        start_step, start_epoch_float = load_training_checkpoint(
            resolve_path(resume_path),
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        print(f"Resumed from step {start_step} epoch {start_epoch_float:.3f}")

    output_dir = resolve_path(config.output_dir) or Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)

    train_loader = get_dataloader(
        train_dataset,
        tokenizer,
        config,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    wandb = maybe_init_wandb(config)

    print(
        f"Batch={batch_size}, grad_accum={grad_accum_steps}, "
        f"steps/epoch={steps_per_epoch}, lr={config.lr:.2e}"
    )
    print("=" * 60)

    global_step = start_step
    start_epoch = int(start_epoch_float)
    skip_in_epoch = max(start_step - start_epoch * steps_per_epoch, 0)
    last_log_time = time.time()
    last_log_step = global_step
    last_save_epoch = start_epoch_float
    accum_metrics: list[dict[str, float]] = []
    pending_grad = False
    optimizer.zero_grad(set_to_none=True)

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)
    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(
            train_loader,
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            disable=args.no_progress,
        )
        for step_in_epoch, batch in enumerate(progress):
            if step_in_epoch >= steps_per_epoch:
                break
            if epoch == start_epoch and step_in_epoch < skip_in_epoch:
                continue
            if args.max_steps is not None and global_step >= args.max_steps:
                break

            batch = move_batch_to_device(batch, device)
            batch = add_label_drop_mask(batch, config.label_drop_prob)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                loss, metrics = compute_train_loss(model, encoder, batch, config)
                scaled_loss = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            pending_grad = True
            accum_metrics.append(metrics)

            global_step += 1
            is_optimizer_step = global_step % grad_accum_steps == 0
            if is_optimizer_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                update_ema(ema_model, model, config.ema_decay1)
                optimizer.zero_grad(set_to_none=True)
                pending_grad = False

            if global_step % config.log_freq == 0:
                elapsed = max(time.time() - last_log_time, 1e-8)
                steps_per_sec = (global_step - last_log_step) / elapsed
                avg = {
                    key: sum(m[key] for m in accum_metrics) / max(len(accum_metrics), 1)
                    for key in ("loss", "l2_loss", "ce_loss", "branch")
                }
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"step={global_step} loss={avg['loss']:.4f} "
                    f"l2={avg['l2_loss']:.4f} ce={avg['ce_loss']:.4f} "
                    f"decoder_rate={avg['branch']:.2f} lr={lr:.2e} sps={steps_per_sec:.2f}"
                )
                print(msg)
                if not args.no_progress:
                    progress.set_postfix(loss=f"{avg['loss']:.4f}", lr=f"{lr:.2e}")
                if wandb is not None:
                    wandb.log(
                        {
                            "train_loss": avg["loss"],
                            "train_l2_loss": avg["l2_loss"],
                            "train_ce_loss": avg["ce_loss"],
                            "decoder_rate": avg["branch"],
                            "lr": lr,
                            "epoch": epoch + (step_in_epoch + 1) / steps_per_epoch,
                        },
                        step=global_step,
                    )
                accum_metrics = []
                last_log_time = time.time()
                last_log_step = global_step

            if 0 < config.save_freq < 1:
                epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                if epoch_progress - last_save_epoch >= config.save_freq:
                    path = save_training_checkpoint(
                        output_dir,
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        step=global_step,
                        epoch=epoch_progress,
                        vocab_size=vocab_size,
                        text_encoder_dim=t5_config.d_model,
                    )
                    print(f"Saved {path}")
                    last_save_epoch = epoch_progress

        current_epoch = epoch + 1
        if config.save_freq >= 1 and current_epoch % int(config.save_freq) == 0:
            path = save_training_checkpoint(
                output_dir,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                step=global_step,
                epoch=float(current_epoch),
                vocab_size=vocab_size,
                text_encoder_dim=t5_config.d_model,
            )
            print(f"Saved {path}")

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if pending_grad:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        update_ema(ema_model, model, config.ema_decay1)
        optimizer.zero_grad(set_to_none=True)

    path = save_training_checkpoint(
        output_dir,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        step=global_step,
        epoch=min(float(config.epochs), global_step / max(steps_per_epoch, 1)),
        vocab_size=vocab_size,
        text_encoder_dim=t5_config.d_model,
        is_final=True,
    )
    print(f"Final checkpoint saved to {path}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
