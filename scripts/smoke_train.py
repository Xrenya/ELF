#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


THIS_FILE = Path(__file__).resolve()
PORT_ROOT = THIS_FILE.parents[1]
if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))

from elf_torch.data import collate_elf_batch
from elf_torch.model import ELF
from elf_torch.t5_encoder import T5Encoder, T5EncoderConfig
from elf_torch.train_utils import (
    clone_for_ema,
    compute_train_loss,
    load_training_checkpoint,
    make_lr_scheduler,
    make_optimizer,
    save_training_checkpoint,
    update_ema,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny ELF PyTorch training smoke test.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def tiny_config(decoder_prob: float) -> SimpleNamespace:
    return SimpleNamespace(
        model="ELF-TINY",
        data_path="synthetic",
        eval_data_path=None,
        max_length=8,
        max_input_length=3,
        pad_token="pad",
        latent_mean=0.0,
        latent_std=1.0,
        denoiser_p_mean=0.0,
        denoiser_p_std=1.0,
        denoiser_noise_scale=1.0,
        t_eps=0.05,
        time_schedule="uniform",
        decoder_prob=decoder_prob,
        decoder_noise_scale=1.0,
        decoder_p_mean=0.0,
        decoder_p_std=1.0,
        label_drop_prob=0.5,
        self_cond_prob=0.5,
        self_cond_cfg_min=0.5,
        self_cond_cfg_max=2.0,
        num_self_cond_cfg_tokens=2,
        optimizer="adamw",
        lr=1e-4,
        adam_b1=0.9,
        adam_b2=0.95,
        weight_decay=0.0,
        lr_schedule="constant",
        min_lr=0.0,
        ema_decay1=0.9,
        batch_size=2,
        global_batch_size=2,
        grad_accum_steps=1,
    )


def make_encoder(device: torch.device) -> tuple[T5Encoder, int]:
    config = T5EncoderConfig(
        vocab_size=32,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        is_gated_act=False,
    )
    encoder = T5Encoder(config).to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, config.d_model


def make_model(device: torch.device) -> ELF:
    return ELF(
        text_encoder_dim=8,
        max_length=8,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=4,
        num_time_tokens=2,
        num_self_cond_cfg_tokens=2,
        num_model_mode_tokens=1,
        vocab_size=32,
        use_self_cond_proj=True,
    ).to(device)


def make_batch(device: torch.device) -> dict[str, torch.Tensor]:
    examples = [
        {"condition_input_ids": [2, 3], "input_ids": [4, 5, 6, 1]},
        {"condition_input_ids": [7], "input_ids": [8, 9, 1]},
    ]
    batch = collate_elf_batch(
        examples,
        max_seq_length=8,
        pad_token_id=0,
        max_input_seq_length=3,
    )
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["label_drop_mask"] = torch.tensor([False, True], device=device)
    return batch


def assert_has_finite_grad(model: torch.nn.Module) -> None:
    found = False
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise AssertionError("Found non-finite gradient.")
        if param.grad.abs().sum().item() > 0:
            found = True
    if not found:
        raise AssertionError("No non-zero gradient found.")


def run_branch(decoder_prob: float, device: torch.device) -> None:
    torch.manual_seed(123)
    config = tiny_config(decoder_prob)
    encoder, _ = make_encoder(device)
    model = make_model(device).train()
    ema_model = clone_for_ema(model).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_lr_scheduler(
        optimizer,
        config,
        total_optimizer_steps=2,
        warmup_optimizer_steps=0,
    )
    batch = make_batch(device)

    loss, metrics = compute_train_loss(model, encoder, batch, config)
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite loss for decoder_prob={decoder_prob}: {loss}")
    loss.backward()
    assert_has_finite_grad(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    update_ema(ema_model, model, config.ema_decay1)
    optimizer.zero_grad(set_to_none=True)
    branch = "decoder" if decoder_prob == 1.0 else "denoiser"
    print(f"{branch}: loss={metrics['loss']:.4f}")


def run_checkpoint_roundtrip(device: torch.device) -> None:
    config = tiny_config(decoder_prob=0.0)
    model = make_model(device)
    ema_model = clone_for_ema(model).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_lr_scheduler(
        optimizer,
        config,
        total_optimizer_steps=2,
        warmup_optimizer_steps=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_training_checkpoint(
            tmpdir,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=1,
            epoch=0.5,
            vocab_size=32,
            text_encoder_dim=8,
        )
        inference_payload = torch.load(Path(tmpdir) / "elf_model.pt", map_location="cpu")
        if "state_dict" not in inference_payload or "optimizer" in inference_payload:
            raise AssertionError("elf_model.pt is not a lean generation checkpoint.")

        model2 = make_model(device)
        ema2 = clone_for_ema(model2).to(device)
        optimizer2 = make_optimizer(model2, config)
        scheduler2 = make_lr_scheduler(
            optimizer2,
            config,
            total_optimizer_steps=2,
            warmup_optimizer_steps=0,
        )
        step, epoch = load_training_checkpoint(
            tmpdir,
            model=model2,
            ema_model=ema2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            map_location=device,
        )
        if step != 1 or abs(epoch - 0.5) > 1e-6:
            raise AssertionError(f"Bad resume metadata: step={step}, epoch={epoch}")
    print("checkpoint: roundtrip ok")


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_branch(decoder_prob=0.0, device=device)
    run_branch(decoder_prob=0.5, device=device)
    run_branch(decoder_prob=1.0, device=device)
    run_checkpoint_roundtrip(device=device)
    print("smoke_train: ok")


if __name__ == "__main__":
    main()
