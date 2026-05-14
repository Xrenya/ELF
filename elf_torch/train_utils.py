from __future__ import annotations

import copy
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .sampling import encode_text, net_out_to_v_x, restore_cond


def config_to_dict(config) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def sample_timesteps(
    batch_size: int,
    *,
    p_mean: float,
    p_std: float,
    time_schedule: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if time_schedule == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device, dtype=dtype) * p_std + p_mean
        return torch.sigmoid(z)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(
    batch_size: int,
    *,
    cfg_min: float,
    cfg_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low = 1.0 + cfg_min
    high = 1.0 + cfg_max
    u = torch.rand(batch_size, device=device, dtype=dtype)
    return low * torch.exp(u * math.log(high / low)) - 1.0


def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    config,
    *,
    cond_seq_mask: torch.Tensor,
) -> torch.Tensor:
    t = t.reshape(-1, 1, 1)
    z = t * x0 + (1.0 - t) * noise * config.denoiser_noise_scale
    return restore_cond(z, x0, cond_seq_mask)


def reduce_token_loss(per_token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    loss_mask = loss_mask.to(dtype=per_token_loss.dtype)
    return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def _deterministic_forward(model: nn.Module, *args, **kwargs):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(*args, **kwargs)
    finally:
        model.train(was_training)


def _self_conditioned_input(
    model: nn.Module,
    denoiser_z: torch.Tensor,
    denoiser_t: torch.Tensor,
    x0: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor | None,
    use_self_cond_mask: torch.Tensor | None,
    config,
) -> torch.Tensor:
    if config.self_cond_prob == 0:
        return denoiser_z

    z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
    z_with_zeros = torch.cat([denoiser_z, z_uncond], dim=-1)
    net_out_init, _ = _deterministic_forward(
        model,
        z_with_zeros,
        denoiser_t,
        self_cond_cfg_scale=self_cond_cfg_scale,
    )
    _, x_pred_init = net_out_to_v_x(net_out_init, denoiser_z, denoiser_t, config.t_eps)
    x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
    x_pred_cond = x_pred_init
    if use_self_cond_mask is not None:
        x_pred_cond = x_pred_cond * use_self_cond_mask.to(dtype=x_pred_cond.dtype)
    x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
    return torch.cat([denoiser_z, x_pred_cond], dim=-1)


def _self_condition_guidance(
    model: nn.Module,
    denoiser_z: torch.Tensor,
    denoiser_t: torch.Tensor,
    x0: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    self_cond_cfg_scale: torch.Tensor,
    use_self_cond_mask: torch.Tensor | None,
    base_v_target: torch.Tensor,
    config,
) -> torch.Tensor:
    if config.num_self_cond_cfg_tokens <= 0:
        return base_v_target

    with torch.no_grad():
        if config.self_cond_prob == 0:
            net_out_uncond, _ = _deterministic_forward(
                model,
                denoiser_z,
                denoiser_t,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
            v_uncond, _ = net_out_to_v_x(net_out_uncond, denoiser_z, denoiser_t, config.t_eps)
            v_cond = v_uncond
        else:
            z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
            z_input_uncond = torch.cat([denoiser_z, z_uncond], dim=-1)
            net_out_uncond, _ = _deterministic_forward(
                model,
                z_input_uncond,
                denoiser_t,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
            v_uncond, x_uncond = net_out_to_v_x(
                net_out_uncond, denoiser_z, denoiser_t, config.t_eps
            )
            x_uncond = restore_cond(x_uncond, x0, cond_seq_mask)

            z_input_cond = torch.cat([denoiser_z, x_uncond], dim=-1)
            net_out_cond, _ = _deterministic_forward(
                model,
                z_input_cond,
                denoiser_t,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
            v_cond, _ = net_out_to_v_x(net_out_cond, denoiser_z, denoiser_t, config.t_eps)

        sc_w = self_cond_cfg_scale.reshape(-1, 1, 1).clamp_min(1e-6)
        sc_guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
        if use_self_cond_mask is not None:
            sc_guidance = torch.where(
                use_self_cond_mask,
                sc_guidance,
                torch.zeros_like(sc_guidance),
            )
    return base_v_target + sc_guidance


def compute_train_loss(
    model: nn.Module,
    encoder: nn.Module,
    batch: dict[str, torch.Tensor],
    config,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"]
    batch_size, seq_length = input_ids.shape
    device = input_ids.device
    model_dtype = next(model.parameters()).dtype

    encoder_attention_mask = batch["encoder_attention_mask"]
    cond_seq_mask_2d = batch["cond_seq_mask"]
    cond_seq_mask = cond_seq_mask_2d[:, :, None]

    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]
        block_mask = (1.0 - cond_seq_mask_2d)[:, :, None] * cond_seq_mask_2d[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1.0 - drop.to(block_mask.dtype) * block_mask)

    with torch.no_grad():
        x0 = encode_text(
            encoder,
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
        )
    x0 = x0.to(dtype=model_dtype)

    t = sample_timesteps(
        batch_size,
        p_mean=config.denoiser_p_mean,
        p_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device,
        dtype=model_dtype,
    )
    noise = torch.randn_like(x0)
    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    if config.label_drop_prob > 0:
        drop_tokens = batch["label_drop_mask"][:, None, None] & (cond_seq_mask > 0)
        denoiser_z = torch.where(drop_tokens, torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop_tokens, torch.zeros_like(x0), x0)

    if config.pad_token == "pad":
        loss_mask = batch["attention_mask"]
    else:
        loss_mask = torch.ones_like(batch["attention_mask"])
    loss_mask = loss_mask * (1.0 - cond_seq_mask_2d)

    decoder_targets = input_ids
    decoder_step_active = torch.rand((), device=device) < config.decoder_prob

    if config.self_cond_prob > 0:
        use_self_cond_mask = (
            torch.rand(batch_size, device=device) < config.self_cond_prob
        ).reshape(-1, 1, 1)
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min,
            cfg_max=config.self_cond_cfg_max,
            device=device,
            dtype=model_dtype,
        )
    else:
        self_cond_cfg_scale = None

    if bool(decoder_step_active.item()):
        lambda_t = torch.sigmoid(
            torch.randn(batch_size * seq_length, device=device, dtype=model_dtype)
            * config.decoder_p_std
            + config.decoder_p_mean
        ).reshape(batch_size, seq_length, 1)
        decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
        decoder_z = lambda_t * x0 + (1.0 - lambda_t) * decoder_noise
        decoder_input = (
            torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
            if config.self_cond_prob > 0
            else decoder_z
        )
        decoder_t = torch.ones(batch_size, device=device, dtype=model_dtype)
        _, decoder_logits = model(
            decoder_input,
            decoder_t,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=True,
        )
        ce = F.cross_entropy(
            decoder_logits.float().reshape(-1, decoder_logits.shape[-1]),
            decoder_targets.reshape(-1),
            reduction="none",
        ).reshape(batch_size, seq_length)
        ce_loss = reduce_token_loss(ce, loss_mask)
        l2_loss = torch.zeros((), device=device, dtype=ce_loss.dtype)
        loss = ce_loss
    else:
        denoiser_input = _self_conditioned_input(
            model,
            denoiser_z,
            t,
            x0,
            cond_seq_mask,
            self_cond_cfg_scale,
            use_self_cond_mask,
            config,
        )
        net_out, _ = model(
            denoiser_input,
            t,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
        v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
        t_reshaped = t.reshape(-1, 1, 1)
        v_target = (x0 - denoiser_z) / torch.maximum(
            1.0 - t_reshaped,
            torch.full_like(t_reshaped, config.t_eps),
        )
        v_target = _self_condition_guidance(
            model,
            denoiser_z,
            t,
            x0,
            cond_seq_mask,
            self_cond_cfg_scale,
            use_self_cond_mask,
            v_target,
            config,
        )
        l2 = (v_pred - v_target).pow(2).mean(dim=-1)
        l2_loss = reduce_token_loss(l2, loss_mask)
        ce_loss = torch.zeros((), device=device, dtype=l2_loss.dtype)
        loss = l2_loss

    denoiser_prob = max(1.0 - float(config.decoder_prob), 1e-8)
    decoder_prob = max(float(config.decoder_prob), 1e-8)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "l2_loss": float((l2_loss.detach() / denoiser_prob).cpu())
        if not bool(decoder_step_active.item())
        else 0.0,
        "ce_loss": float((ce_loss.detach() / decoder_prob).cpu())
        if bool(decoder_step_active.item())
        else 0.0,
        "branch": 1.0 if bool(decoder_step_active.item()) else 0.0,
    }
    return loss, metrics


def make_optimizer(model: nn.Module, config):
    if config.optimizer.lower() not in {"adamw", "adam", "muon"}:
        raise ValueError(f"Unsupported optimizer in PyTorch training port: {config.optimizer}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.adam_b1, config.adam_b2),
        weight_decay=config.weight_decay,
    )


def make_lr_scheduler(optimizer, config, *, total_optimizer_steps: int, warmup_optimizer_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_optimizer_steps > 0 and step < warmup_optimizer_steps:
            return step / warmup_optimizer_steps
        if config.lr_schedule == "constant":
            return 1.0
        if config.lr_schedule == "cosine":
            denom = max(total_optimizer_steps - warmup_optimizer_steps, 1)
            progress = min(max((step - warmup_optimizer_steps) / denom, 0.0), 1.0)
            min_ratio = config.min_lr / config.lr if config.lr else 0.0
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr_schedule: {config.lr_schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, ema_param in ema_params.items():
        ema_param.mul_(decay).add_(model_params[name], alpha=1.0 - decay)
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(model_buffer)


def clone_for_ema(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def _final_layer_norm(state_dict: dict[str, torch.Tensor]) -> float:
    weight = state_dict.get("final_layer.linear.weight")
    if weight is None:
        return 0.0
    return float(weight.float().norm().cpu())


def save_training_checkpoint(
    output_dir: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer,
    scheduler,
    config,
    step: int,
    epoch: float,
    vocab_size: int,
    text_encoder_dim: int,
    is_final: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in ema_model.state_dict().items()}
    raw_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    metadata = {
        "step": int(step),
        "epoch": float(epoch),
        "model_key": "ema_params1",
        "vocab_size": int(vocab_size),
        "text_encoder_dim": int(text_encoder_dim),
        "final_layer_kernel_norm": _final_layer_norm(state_dict),
    }
    train_payload = {
        "state_dict": state_dict,
        "raw_state_dict": raw_state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config_to_dict(config),
        "metadata": metadata,
    }
    inference_payload = {
        "state_dict": state_dict,
        "config": config_to_dict(config),
        "metadata": metadata,
    }
    checkpoint_path = output_dir / f"checkpoint_{step}.pt"
    torch.save(train_payload, checkpoint_path)
    torch.save(inference_payload, output_dir / "elf_model.pt")
    if is_final:
        torch.save(inference_payload, output_dir / "final.pt")
    return checkpoint_path


def resolve_resume_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(
        path.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else -1,
    )
    if candidates:
        return candidates[-1]
    elf_model = path / "elf_model.pt"
    if elf_model.is_file():
        return elf_model
    raise FileNotFoundError(f"No PyTorch checkpoint found in {path}")


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> tuple[int, float]:
    ckpt_path = resolve_resume_checkpoint(path)
    payload = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        payload = {"state_dict": payload}
    raw_state = payload.get("raw_state_dict") or payload["state_dict"]
    ema_state = payload.get("state_dict") or raw_state
    model.load_state_dict(raw_state)
    ema_model.load_state_dict(ema_state)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    metadata = payload.get("metadata", {})
    return int(metadata.get("step", 0)), float(metadata.get("epoch", 0.0))
