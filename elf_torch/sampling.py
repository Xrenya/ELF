from __future__ import annotations

import itertools
from typing import Iterable

import torch
from tqdm import tqdm


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id/eos_token_id for the requested padding mode.")
    return token_id


def mask_after_eos(
    predicted_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    eos_mask = predicted_ids == eos_token_id
    keep_mask = torch.cumsum(eos_mask.to(torch.int64), dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(
    x: torch.Tensor,
    shift_per_sample: torch.Tensor,
    pad_value: int | float = 0,
    axis: int = 1,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    axis = axis if axis >= 0 else x.ndim + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    shift_per_sample = shift_per_sample.to(device=x.device, dtype=torch.long)
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None] + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.ndim == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [*gather_idx.shape, *([1] * (x.ndim - 2))]
        gather_idx_expanded = gather_idx.reshape(expand_shape).expand(
            *gather_idx.shape, *x.shape[2:]
        )
        shifted = torch.gather(x, 1, gather_idx_expanded)
        valid_expanded = valid.reshape(expand_shape).expand_as(shifted)
        shifted = torch.where(valid_expanded, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


def build_self_attn_cond_masks(
    is_cond: torch.Tensor,
    is_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :])
        | (~is_cond[:, :, None] & is_valid[:, None, :])
    ).float()
    attention_mask = is_valid.float()
    cond_seq_mask = is_cond.float()
    return encoder_attention_mask, attention_mask, cond_seq_mask


def encode_text(
    encoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_mean: float,
    latent_std: float,
) -> torch.Tensor:
    latents = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return (latents - latent_mean) / latent_std


def get_sampling_steps(
    n_steps: int,
    time_schedule: str,
    p_mean: float,
    p_std: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        steps = torch.randn(n_steps - 1, device=device, generator=generator) * p_std + p_mean
        steps = torch.sigmoid(steps).sort().values
        return torch.cat([torch.zeros(1, device=device), steps, torch.ones(1, device=device)])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def restore_cond(
    z_updated: torch.Tensor,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> torch.Tensor:
    mask = cond_seq_mask
    while mask.ndim < max(z_updated.ndim, cond_seq.ndim):
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(
    v: torch.Tensor,
    x: torch.Tensor,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = restore_cond(x, cond_seq, cond_seq_mask)
    v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def net_out_to_v_x(
    net_out,
    z: torch.Tensor,
    t: torch.Tensor,
    t_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.maximum(
        1.0 - t_reshaped,
        torch.full_like(t_reshaped, t_eps),
    )
    return v, x


def _forward_sample_self_cond(
    model,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full(
            (z.shape[0],), self_cond_cfg_scale, device=z.device, dtype=z.dtype
        )
        net_out_cond = model(
            z_input_cond,
            t_batch,
            self_cond_cfg_scale=self_cond_scale_batch,
        )
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, config.t_eps)
        return restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)

    if config.self_cond_prob == 0:
        net_out = model(z, t_batch)
        v, x = net_out_to_v_x(net_out, z, t_batch, config.t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, config.t_eps)
        v_uncond, x_uncond = restore_vx(v_uncond, x_uncond, cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, config.t_eps)
    v_cond, x_cond = restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _forward_sample(
    model,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_cond, x_cond = _forward_sample_self_cond(
        model,
        z,
        t_batch,
        x_pred_prev,
        config,
        self_cond_cfg_scale,
        cond_seq,
        cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None
        if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model,
        z_uncond,
        t_batch,
        x_pred_prev_uncond,
        config,
        self_cond_cfg_scale,
        torch.zeros_like(cond_seq),
        cond_seq_mask,
    )
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def ode_step(
    model,
    z: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_batch = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model,
        z,
        t_batch,
        x_pred_prev,
        config,
        cfg_scale,
        self_cond_cfg_scale,
        cond_seq,
        cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


def sde_step(
    model,
    z: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
    gamma: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = t_next - t
    alpha = torch.clamp(1.0 - gamma * h, 0.0, 1.0)
    t_back = alpha * t
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
    eps = eps * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model,
        z_back,
        t_batch,
        x_pred_prev,
        config,
        cfg_scale,
        self_cond_cfg_scale,
        cond_seq,
        cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred


@torch.no_grad()
def sample_latents(
    model,
    config,
    sampling_config,
    batch_size: int,
    text_encoder_dim: int,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor | None = None,
    cond_seq_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    show_progress: bool = True,
) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    z = torch.randn(
        batch_size,
        config.max_length,
        text_encoder_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * config.denoiser_noise_scale

    if cond_seq is None:
        cond_seq = torch.zeros_like(z)
        cond_seq_mask = torch.zeros(batch_size, config.max_length, device=device, dtype=dtype)
    else:
        cond_seq = cond_seq.to(device=device, dtype=dtype)
        cond_seq_mask = cond_seq_mask.to(device=device, dtype=dtype)
        z = restore_cond(z, cond_seq, cond_seq_mask)

    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
    steps = sampling_config.num_sampling_steps[0]
    t_steps = get_sampling_steps(
        steps,
        sampling_config.time_schedule,
        config.denoiser_p_mean,
        config.denoiser_p_std,
        device=device,
        generator=generator,
    ).to(dtype=dtype)

    pairs = list(zip(t_steps[:-2], t_steps[1:-1]))
    iterator: Iterable = tqdm(pairs, desc="sampling", leave=False) if show_progress else pairs
    for t, t_next in iterator:
        if sampling_config.sampling_method == "sde":
            z, x_pred = sde_step(
                model,
                z,
                t,
                t_next,
                x_pred,
                config,
                cfg_scale,
                self_cond_cfg_scale,
                cond_seq,
                cond_seq_mask,
                getattr(sampling_config, "sde_gamma", 0.0),
                generator=generator,
            )
        elif sampling_config.sampling_method == "ode":
            z, x_pred = ode_step(
                model,
                z,
                t,
                t_next,
                x_pred,
                config,
                cfg_scale,
                self_cond_cfg_scale,
                cond_seq,
                cond_seq_mask,
            )
        else:
            raise ValueError(f"Invalid sampling method: {sampling_config.sampling_method}")

    z, _ = ode_step(
        model,
        z,
        t_steps[-2],
        t_steps[-1],
        x_pred,
        config,
        cfg_scale,
        self_cond_cfg_scale,
        cond_seq,
        cond_seq_mask,
    )
    return z


@torch.no_grad()
def decode_ids(
    model,
    z: torch.Tensor,
    config,
    self_cond_cfg_scale: float,
    t_final: float = 1.0,
) -> torch.Tensor:
    if config.self_cond_prob > 0:
        z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    else:
        z_input = z
    t_batch = torch.full((z.shape[0],), t_final, device=z.device, dtype=z.dtype)
    scale_batch = (
        torch.full((z.shape[0],), self_cond_cfg_scale, device=z.device, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0
        else None
    )
    _, decoder_logits = model(
        z_input,
        t_batch,
        self_cond_cfg_scale=scale_batch,
        decoder_step_active=True,
    )
    return torch.argmax(decoder_logits, dim=-1)


def build_condition_batch(
    prompts: list[str],
    tokenizer,
    config,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    max_input = config.max_input_length or config.max_length
    seqs = []
    cond_lens = []
    for prompt in prompts:
        cond = tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_input]
        cond_lens.append(min(len(cond), config.max_length))
        padded = cond[: config.max_length]
        padded = padded + [pad_token_id] * (config.max_length - len(padded))
        seqs.append(padded)

    input_ids = torch.tensor(seqs, device=device, dtype=torch.long)
    cond_lens_tensor = torch.tensor(cond_lens, device=device, dtype=torch.long)
    pos = torch.arange(config.max_length, device=device)[None, :]
    is_cond = pos < cond_lens_tensor[:, None]
    is_valid = is_cond
    encoder_attention_mask, attention_mask, cond_seq_mask = build_self_attn_cond_masks(
        is_cond, is_valid
    )
    return {
        "input_ids": input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "attention_mask": attention_mask,
        "cond_seq_mask": cond_seq_mask,
        "cond_lens": cond_lens_tensor,
        "prompts": prompts,
    }


@torch.no_grad()
def generate_unconditional(
    model,
    tokenizer,
    config,
    sampling_config,
    num_samples: int,
    batch_size: int,
    text_encoder_dim: int,
    seed: int = 42,
    show_progress: bool = True,
) -> list[str]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    texts = []
    for start in range(0, num_samples, batch_size):
        current = min(batch_size, num_samples - start)
        z = sample_latents(
            model,
            config,
            sampling_config,
            current,
            text_encoder_dim,
            cfg_scale=1.0,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
            generator=generator,
            show_progress=show_progress,
        )
        ids = decode_ids(
            model,
            z,
            config,
            self_cond_cfg_scale=sampling_config.self_cond_cfg_scales[0],
        )
        ids = mask_after_eos(ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        texts.extend(tokenizer.batch_decode(ids.cpu().numpy(), skip_special_tokens=True))
    return texts


@torch.no_grad()
def generate_conditional(
    model,
    encoder,
    tokenizer,
    config,
    sampling_config,
    prompts: list[str],
    batch_size: int,
    text_encoder_dim: int,
    seed: int = 42,
    show_progress: bool = True,
) -> list[str]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch = build_condition_batch(batch_prompts, tokenizer, config, device=device)
        cond_seq = encode_text(
            encoder,
            batch["input_ids"],
            batch["encoder_attention_mask"],
            config.latent_mean,
            config.latent_std,
        )
        cond_seq_mask = batch["cond_seq_mask"]
        cfg_values = sampling_config.cfgs if getattr(sampling_config, "cfgs", None) else [1.0]
        for cfg_scale, self_cond_cfg_scale in itertools.product(
            cfg_values,
            sampling_config.self_cond_cfg_scales,
        ):
            z = sample_latents(
                model,
                config,
                sampling_config,
                len(batch_prompts),
                text_encoder_dim,
                cfg_scale=cfg_scale,
                self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq,
                cond_seq_mask=cond_seq_mask,
                generator=generator,
                show_progress=show_progress,
            )
            ids = decode_ids(
                model,
                z,
                config,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
            gen_length = (
                config.max_length - config.max_input_length
                if config.max_input_length is not None
                else config.max_length
            )
            ids = shift_left(ids, batch["cond_lens"], 0)[:, :gen_length]
            ids = mask_after_eos(ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            outputs.extend(tokenizer.batch_decode(ids.cpu().numpy(), skip_special_tokens=True))
    return outputs
