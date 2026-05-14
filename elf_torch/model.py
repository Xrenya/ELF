from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 512,
        ft_seq_len: int | None = None,
        theta: float = 10000.0,
        num_empty_token: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        main_len = max(seq_len - self.num_empty_token, 0)
        device = t.device
        calc_dtype = torch.float32

        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=device, dtype=calc_dtype)[: self.dim // 2] / self.dim)
        )
        pos = torch.arange(self.ft_seq_len, device=device, dtype=calc_dtype)
        pos = pos / self.ft_seq_len * self.pt_seq_len
        freqs_main = torch.einsum("n,f->nf", pos, freqs)
        freqs_main = freqs_main.repeat_interleave(2, dim=-1)[:main_len]

        parts_cos = []
        parts_sin = []
        if self.num_empty_token > 0:
            empty = min(self.num_empty_token, seq_len)
            parts_cos.append(torch.ones(empty, self.dim, device=device, dtype=calc_dtype))
            parts_sin.append(torch.zeros(empty, self.dim, device=device, dtype=calc_dtype))
        if main_len > 0:
            parts_cos.append(torch.cos(freqs_main))
            parts_sin.append(torch.sin(freqs_main))

        cos = torch.cat(parts_cos, dim=0).to(dtype=t.dtype)
        sin = torch.cat(parts_sin, dim=0).to(dtype=t.dtype)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
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


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    attn_weight = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            mask = attn_mask[:, None, None, :]
        elif attn_mask.ndim == 3:
            mask = attn_mask[:, None, :, :]
        else:
            mask = attn_mask
        attn_weight = attn_weight.masked_fill(mask == 0, -1e9)
    attn_weight = torch.softmax(attn_weight, dim=-1).to(dtype=value.dtype)
    return torch.matmul(attn_weight, value)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        head_dim = self.dim // self.num_heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        x = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        inner_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.drop = nn.Dropout(drop)
        self.w3 = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        hidden = self.drop(F.silu(x1) * x2)
        return self.w3(hidden)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class ELFBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ELF(nn.Module):
    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        use_self_cond_proj: bool = True,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.self_cond_proj = (
            nn.Linear(text_encoder_dim * 2, text_encoder_dim) if use_self_cond_proj else None
        )
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size)
            )
        else:
            self.self_cond_cfg_embedder = None
            self.register_parameter("self_cond_cfg_tokens", None)
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
        else:
            self.register_parameter("mode_tokens", None)

        q1, q3 = depth // 4, depth // 4 * 3
        blocks = []
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            blocks.append(
                ELFBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if in_drop_range else 0.0,
                    proj_drop=proj_drop if in_drop_range else 0.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, text_encoder_dim))
        self.proj_bias = nn.Parameter(torch.empty(text_encoder_dim))
        self.unembed_kernel = nn.Parameter(torch.empty(text_encoder_dim, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        self.final_layer = FinalLayer(hidden_size, 1, text_encoder_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.t_emb_tokens, mean=0.0, std=0.02)
        if self.self_cond_cfg_tokens is not None:
            nn.init.normal_(self.self_cond_cfg_tokens, mean=0.0, std=0.02)
        if self.mode_tokens is not None:
            nn.init.normal_(self.mode_tokens, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.proj_kernel)
        nn.init.zeros_(self.proj_bias)
        nn.init.xavier_uniform_(self.unembed_kernel)
        nn.init.zeros_(self.unembed_bias)

    def _make_prefix(self, emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.expand(emb.shape[0], -1, -1) + emb[:, None, :]

    def build_context(
        self, t: torch.Tensor, self_cond_cfg_scale: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        prefix_tokens = [
            self._make_prefix(self.t_embedder(t), self.t_emb_tokens)
        ]
        if self_cond_cfg_scale is not None and self.self_cond_cfg_embedder is not None:
            prefix_tokens.append(
                self._make_prefix(
                    self.self_cond_cfg_embedder(self_cond_cfg_scale),
                    self.self_cond_cfg_tokens,
                )
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: bool | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if self_cond_cfg_scale is not None and self_cond_cfg_scale.ndim == 0:
            self_cond_cfg_scale = self_cond_cfg_scale.expand(x.shape[0])

        bsz = x.shape[0]
        if self.self_cond_proj is not None and x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)

        x = self.text_proj(x)

        model_mode_offset = 0
        if self.mode_tokens is not None:
            mode_tokens = self.mode_tokens.expand(bsz, -1, -1)
            active_gate = False if decoder_step_active is None else decoder_step_active
            if isinstance(active_gate, torch.Tensor):
                active_gate = bool(active_gate.detach().item())
            if not active_gate:
                mode_tokens = torch.zeros_like(mode_tokens)
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(
                    bsz, self.num_model_mode_tokens, device=x.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    bsz, prefix_len, device=x.device, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        head_dim = self.hidden_size // self.num_heads
        rope_fn = TextRotaryEmbeddingFast(
            dim=head_dim,
            pt_seq_len=self.max_length,
            num_empty_token=prefix_len + model_mode_offset,
        ).to(device=x.device)

        for block in self.blocks:
            x = block(x, rope_fn=rope_fn, attention_mask=attention_mask)

        x = x[:, prefix_len + model_mode_offset :]

        decoder_logits = None
        if decoder_step_active is not None:
            active = decoder_step_active
            if isinstance(active, torch.Tensor):
                active = bool(active.detach().item())
            if active:
                hidden = F.gelu(x @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            else:
                decoder_logits = torch.zeros(
                    *x.shape[:2], self.vocab_size, device=x.device, dtype=x.dtype
                )

        return self.final_layer(x), decoder_logits


def ELF_B(**kwargs) -> ELF:
    return ELF(depth=12, hidden_size=768, num_heads=12, **kwargs)


def ELF_M(**kwargs) -> ELF:
    return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)


def ELF_L(**kwargs) -> ELF:
    return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)


ELF_MODELS = {
    "ELF-B": ELF_B,
    "ELF-M": ELF_M,
    "ELF-L": ELF_L,
}


def build_elf_from_config(config, text_encoder_dim: int, vocab_size: int) -> ELF:
    return ELF_MODELS[config.model](
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        use_self_cond_proj=config.self_cond_prob > 0,
    )
