"""
Core transformer model for LWLM.

Architecture: decoder-only transformer with RoPE, RMSNorm, and SwiGLU FFN.
Targets CPU inference on 8th-gen i7 / 32 GB RAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LWLMConfig:
    vocab_size: int = 8192
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ffn: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Pre-compute RoPE cosine/sine frequencies."""
    positions = torch.arange(seq_len, device=device)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(positions, freqs)
    # shape: [seq_len, head_dim/2, 2] — last dim is (cos, sin)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensor."""
    # x: [B, n_heads, T, head_dim]
    # rope: [T, head_dim/2, 2]
    B, H, T, D = x.shape
    x_r = x.float().reshape(B, H, T, D // 2, 2)
    cos = rope[:T, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
    sin = rope[:T, :, 1].unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack(
        [x_r[..., 0] * cos - x_r[..., 1] * sin,
         x_r[..., 0] * sin + x_r[..., 1] * cos],
        dim=-1,
    )
    return x_rot.flatten(-2).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: LWLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim()
        self.d_model = cfg.d_model

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, S]

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), new_kv_cache


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: LWLMConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)
        self.w2 = nn.Linear(cfg.d_ffn, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ffn, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LWLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), rope, mask, kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class LWLM(nn.Module):
    def __init__(self, cfg: LWLMConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        device = input_ids.device

        rope = build_rope_cache(T, self.cfg.head_dim(), self.cfg.rope_theta, device)

        # Causal mask
        mask = torch.full((T, T), float("-inf"), device=device).triu(1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        x = self.token_emb(input_ids)

        if kv_caches is None:
            kv_caches = [None] * self.cfg.n_layers

        new_kv_caches = []
        for layer, kv in zip(self.layers, kv_caches):
            x, new_kv = layer(x, rope, mask, kv)
            new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
