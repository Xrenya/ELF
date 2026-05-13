from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def unwrap_params(tree: Any) -> Any:
    if isinstance(tree, Mapping) and "params" in tree:
        return tree["params"]
    return tree


def to_plain_dict(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {str(k): to_plain_dict(v) for k, v in tree.items()}
    return tree


def to_torch(array: Any, transpose: bool = False) -> torch.Tensor:
    try:
        import jax

        array = jax.device_get(array)
    except Exception:
        pass
    arr = np.asarray(array)
    if transpose:
        if arr.ndim != 2:
            raise ValueError(f"Expected a rank-2 kernel to transpose, got shape {arr.shape}")
        arr = arr.T
    return torch.from_numpy(np.array(arr, copy=True))


def _require(tree: Mapping[str, Any], path: str) -> Any:
    cur = tree
    for part in path.split("/"):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"Missing Flax parameter path: {path}")
        cur = cur[part]
    return cur


def _copy_param(out: dict[str, torch.Tensor], torch_key: str, tree: Mapping[str, Any], flax_path: str):
    out[torch_key] = to_torch(_require(tree, flax_path))


def _copy_linear(
    out: dict[str, torch.Tensor],
    torch_prefix: str,
    tree: Mapping[str, Any],
    flax_path: str,
    bias: bool = True,
):
    parent = _require(tree, flax_path)
    out[f"{torch_prefix}.weight"] = to_torch(parent["kernel"], transpose=True)
    if bias and "bias" in parent:
        out[f"{torch_prefix}.bias"] = to_torch(parent["bias"])


def convert_elf_params(params: Mapping[str, Any], config) -> dict[str, torch.Tensor]:
    params = to_plain_dict(unwrap_params(params))
    out: dict[str, torch.Tensor] = {}

    if "self_cond_proj" in params:
        _copy_linear(out, "self_cond_proj", params, "self_cond_proj")
    _copy_linear(out, "text_proj.proj1", params, "text_proj/proj1", bias=False)
    _copy_linear(out, "text_proj.proj2", params, "text_proj/proj2")

    _copy_param(out, "t_emb_tokens", params, "t_emb_tokens")
    _copy_linear(out, "t_embedder.mlp.0", params, "t_embedder/mlp_0")
    _copy_linear(out, "t_embedder.mlp.2", params, "t_embedder/mlp_2")

    if config.num_self_cond_cfg_tokens > 0 and "self_cond_cfg_embedder" in params:
        _copy_param(out, "self_cond_cfg_tokens", params, "self_cond_cfg_tokens")
        _copy_linear(out, "self_cond_cfg_embedder.mlp.0", params, "self_cond_cfg_embedder/mlp_0")
        _copy_linear(out, "self_cond_cfg_embedder.mlp.2", params, "self_cond_cfg_embedder/mlp_2")

    if config.num_model_mode_tokens > 0 and "mode_tokens" in params:
        _copy_param(out, "mode_tokens", params, "mode_tokens")

    depth = {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}.get(config.model)
    if depth is None:
        raise ValueError(f"Unsupported ELF model name: {config.model}")

    for i in range(depth):
        flax_block = f"blocks_{i}"
        torch_block = f"blocks.{i}"
        _copy_param(out, f"{torch_block}.norm1.weight", params, f"{flax_block}/norm1/weight")
        _copy_linear(out, f"{torch_block}.attn.qkv", params, f"{flax_block}/attn/qkv")
        _copy_param(out, f"{torch_block}.attn.q_norm.weight", params, f"{flax_block}/attn/q_norm/weight")
        _copy_param(out, f"{torch_block}.attn.k_norm.weight", params, f"{flax_block}/attn/k_norm/weight")
        _copy_linear(out, f"{torch_block}.attn.proj", params, f"{flax_block}/attn/proj")
        _copy_param(out, f"{torch_block}.norm2.weight", params, f"{flax_block}/norm2/weight")
        _copy_linear(out, f"{torch_block}.mlp.w12", params, f"{flax_block}/mlp/w12")
        _copy_linear(out, f"{torch_block}.mlp.w3", params, f"{flax_block}/mlp/w3")

    out["proj_kernel"] = to_torch(_require(params, "proj_kernel"))
    out["proj_bias"] = to_torch(_require(params, "proj_bias"))
    out["unembed_kernel"] = to_torch(_require(params, "unembed_kernel"))
    out["unembed_bias"] = to_torch(_require(params, "unembed_bias"))
    _copy_param(out, "final_layer.norm_final.weight", params, "final_layer/norm_final/weight")
    _copy_linear(out, "final_layer.linear", params, "final_layer/linear")
    return out


def convert_t5_encoder_params(params: Mapping[str, Any], t5_config) -> dict[str, torch.Tensor]:
    params = to_plain_dict(unwrap_params(params))
    out: dict[str, torch.Tensor] = {}

    _copy_param(out, "shared.weight", params, "shared/embedding")
    for i in range(t5_config.num_layers):
        flax_block = f"encoder/block_{i}"
        torch_block = f"encoder.blocks.{i}"
        _copy_param(out, f"{torch_block}.layer_0.layer_norm.weight", params, f"{flax_block}/layer_0/layer_norm/weight")
        _copy_linear(out, f"{torch_block}.layer_0.SelfAttention.q", params, f"{flax_block}/layer_0/SelfAttention/q", bias=False)
        _copy_linear(out, f"{torch_block}.layer_0.SelfAttention.k", params, f"{flax_block}/layer_0/SelfAttention/k", bias=False)
        _copy_linear(out, f"{torch_block}.layer_0.SelfAttention.v", params, f"{flax_block}/layer_0/SelfAttention/v", bias=False)
        _copy_linear(out, f"{torch_block}.layer_0.SelfAttention.o", params, f"{flax_block}/layer_0/SelfAttention/o", bias=False)
        if i == 0:
            _copy_param(
                out,
                f"{torch_block}.layer_0.SelfAttention.relative_attention_bias.weight",
                params,
                f"{flax_block}/layer_0/SelfAttention/relative_attention_bias/rel_embedding",
            )
        _copy_param(out, f"{torch_block}.layer_1.layer_norm.weight", params, f"{flax_block}/layer_1/layer_norm/weight")
        dense_path = f"{flax_block}/layer_1/DenseReluDense"
        if t5_config.is_gated_act:
            _copy_linear(out, f"{torch_block}.layer_1.DenseReluDense.wi_0", params, f"{dense_path}/wi_0", bias=False)
            _copy_linear(out, f"{torch_block}.layer_1.DenseReluDense.wi_1", params, f"{dense_path}/wi_1", bias=False)
        else:
            _copy_linear(out, f"{torch_block}.layer_1.DenseReluDense.wi", params, f"{dense_path}/wi", bias=False)
        _copy_linear(out, f"{torch_block}.layer_1.DenseReluDense.wo", params, f"{dense_path}/wo", bias=False)

    _copy_param(out, "encoder.final_layer_norm.weight", params, "encoder/final_layer_norm/weight")
    return out


def load_strict(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], label: str):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:20]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError(f"{label} state dict does not match Torch module: {'; '.join(details)}")
