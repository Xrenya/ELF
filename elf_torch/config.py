from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SamplingConfig:
    sampling_method: str = "ode"
    num_sampling_steps: list[int] = field(default_factory=lambda: [50])
    cfgs: list[float] = field(default_factory=lambda: [1])
    self_cond_cfg_scales: list[float] = field(default_factory=lambda: [1.0])
    time_schedule: str = "logit_normal"
    sde_gamma: float = 0.0


@dataclass
class Config:
    data_path: str | None = None
    eval_data_path: str | None = None
    max_length: int = 128
    max_input_length: int | None = None
    pad_token: str = "pad"

    tokenizer_name: str | None = None

    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str | None = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    model: str = "ELF-B"
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"

    decoder_prob: float = 0.5
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    epochs: int = 200
    warmup_epochs: float | None = None
    warmup_steps: int = 5000
    batch_size: int | None = None
    global_batch_size: int = 512
    lr: float | None = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1

    ema_decay1: float = 0.9999

    sampling_configs_path: str | None = None
    sampling_configs: list[SamplingConfig] = field(default_factory=lambda: [SamplingConfig()])
    num_samples: int = 100

    online_eval: bool = True
    eval_ppl_model: str = "gpt2-large"
    eval_ppl_batch_size: int = 64
    eval_ppl_max_length: int = 1024

    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100

    output_dir: str = "./output_dir"
    hf_repo_id: str | None = None
    resume: str | None = None

    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tag: str | None = None
    wandb_resume: str = "allow"

    seed: int = 0
    num_workers: int = 0


def load_sampling_configs(path: str | Path) -> list[SamplingConfig]:
    with open(path, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    return [SamplingConfig(**entry) for entry in entries]


def load_config_from_yaml(path: str | Path) -> Config:
    config = Config()
    path = Path(path)
    if not path.is_file():
        return config

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        sampling_path = Path(config.sampling_configs_path)
        candidates = [sampling_path] if sampling_path.is_absolute() else []
        if not sampling_path.is_absolute():
            search_roots = [path.parent, *path.parents, Path.cwd(), Path.cwd() / "src"]
            candidates.extend(root / sampling_path for root in search_roots)
            candidates.extend(root / "src" / sampling_path for root in path.parents)
        for candidate in candidates:
            if candidate.exists():
                config.sampling_configs = load_sampling_configs(candidate)
                break

    return config


def _convert_override_value(original: Any, value_str: str, field_type: Any) -> Any:
    if value_str.lower() == "none":
        return None
    if original is None:
        if field_type is int:
            return int(value_str)
        if field_type is float:
            return float(value_str)
        if field_type is bool:
            return value_str.lower() in ("true", "1", "yes")
        return value_str
    if isinstance(original, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(original, int) and not isinstance(original, bool):
        return int(value_str)
    if isinstance(original, float):
        return float(value_str)
    return value_str


def apply_config_overrides(config: Config, overrides: list[str]) -> Config:
    annotations = getattr(config, "__annotations__", {})
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected field=value")
        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()
        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named {field_name!r}")
        value = _convert_override_value(
            getattr(config, field_name), value_str, annotations.get(field_name)
        )
        setattr(config, field_name, value)
    return config
