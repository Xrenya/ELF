from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(dtype=input_dtype)


class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.weight = nn.Parameter(torch.empty(num_buckets, num_heads))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    @staticmethod
    def _compute_relative_position(query_length: int, key_length: int, device) -> torch.Tensor:
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        return memory_position - context_position

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.minimum(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.minimum(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets.to(torch.long)

    def forward(self, query_length: int, key_length: int, device) -> torch.Tensor:
        relative_position = self._compute_relative_position(query_length, key_length, device)
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.weight[relative_position_bucket]
        return values.permute(2, 0, 1).unsqueeze(0)


class T5Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.has_relative_attention_bias = has_relative_attention_bias
        inner_dim = num_heads * d_kv
        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = T5RelativePositionBias(num_heads=num_heads)
        else:
            self.relative_attention_bias = None

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = states.shape
        return states.reshape(bsz, seq_len, self.num_heads, self.d_kv).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_length, _ = hidden_states.shape
        q = self._shape(self.q(hidden_states))
        k = self._shape(self.k(hidden_states))
        v = self._shape(self.v(hidden_states))

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)

        if position_bias is None and self.relative_attention_bias is not None:
            position_bias = self.relative_attention_bias(seq_length, seq_length, hidden_states.device)
        if position_bias is not None:
            scores = scores + position_bias
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores.float(), dim=-1).to(dtype=hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_length, -1)
        return self.o(attn_output), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        has_relative_attention_bias: bool = False,
    ):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.SelfAttention = T5Attention(
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias


class T5DenseGatedActDense(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.0):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = F.gelu(self.wi_0(hidden_states), approximate="tanh")
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_gelu * hidden_linear)
        return self.wo(hidden_states)


class T5DenseActDense(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.0):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.relu(self.wi(hidden_states))
        hidden_states = self.dropout(hidden_states)
        return self.wo(hidden_states)


class T5LayerFF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        is_gated_act: bool = True,
    ):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        dense_cls = T5DenseGatedActDense if is_gated_act else T5DenseActDense
        self.DenseReluDense = dense_cls(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ff_output = self.DenseReluDense(self.layer_norm(hidden_states))
        return hidden_states + self.dropout(ff_output)


class T5EncoderOnlyBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        has_relative_attention_bias: bool = False,
        is_gated_act: bool = True,
    ):
        super().__init__()
        self.layer_0 = T5LayerSelfAttention(
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        self.layer_1 = T5LayerFF(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            is_gated_act=is_gated_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, position_bias = self.layer_0(
            hidden_states, attention_mask=attention_mask, position_bias=position_bias
        )
        hidden_states = self.layer_1(hidden_states)
        return hidden_states, position_bias


class T5EncoderLikeStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_kv: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        is_gated_act: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                T5EncoderOnlyBlock(
                    d_model=d_model,
                    d_kv=d_kv,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    layer_norm_epsilon=layer_norm_epsilon,
                    has_relative_attention_bias=(i == 0),
                    is_gated_act=is_gated_act,
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, Any]:
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            else:
                extended_attention_mask = attention_mask
            extended_attention_mask = (
                (1.0 - extended_attention_mask.to(dtype=inputs_embeds.dtype))
                * torch.finfo(inputs_embeds.dtype).min
            )
        else:
            extended_attention_mask = None

        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        all_hidden_states = [] if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, position_bias = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
            )
        hidden_states = self.dropout(self.final_layer_norm(hidden_states))
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": tuple(all_hidden_states) if output_hidden_states else None,
        }


@dataclass
class T5EncoderConfig:
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True

    @classmethod
    def from_pretrained(cls, model_name: str) -> "T5EncoderConfig":
        configs = {
            "t5-small": dict(
                vocab_size=32128,
                d_model=512,
                d_kv=64,
                d_ff=2048,
                num_layers=6,
                num_decoder_layers=6,
                num_heads=8,
                is_gated_act=False,
            ),
            "t5-base": dict(
                vocab_size=32128,
                d_model=768,
                d_kv=64,
                d_ff=3072,
                num_layers=12,
                num_decoder_layers=12,
                num_heads=12,
                is_gated_act=False,
            ),
            "t5-large": dict(
                vocab_size=32128,
                d_model=1024,
                d_kv=64,
                d_ff=4096,
                num_layers=24,
                num_decoder_layers=24,
                num_heads=16,
                is_gated_act=False,
            ),
        }
        return cls(**configs.get(model_name, configs["t5-small"]))


class T5Encoder(nn.Module):
    def __init__(self, config: T5EncoderConfig):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5EncoderLikeStack(
            num_layers=config.num_layers,
            d_model=config.d_model,
            d_kv=config.d_kv,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            layer_norm_epsilon=config.layer_norm_epsilon,
            is_gated_act=config.is_gated_act,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        inputs_embeds = self.shared(input_ids)
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            return outputs
        return outputs["last_hidden_state"]
