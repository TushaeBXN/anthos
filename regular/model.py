"""A plain decoder-only transformer: RMSNorm, rotary positions, grouped-query attention, SwiGLU, tied embeddings.

    from regular import Regular, PRESETS
    model = Regular(PRESETS["138m"])
    logits = model(input_ids)            # (B, T, vocab)
"""
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RegularConfig:
    vocab_size:   int   = 100352      # cl100k_base plus the chat/tool special tokens (same ids as Anthos)
    dim:          int   = 768
    n_layers:     int   = 12
    n_heads:      int   = 12
    n_kv_heads:   int   = 4
    ffn_dim:      int   = 1536
    max_seq_len:  int   = 1024
    rope_theta:   float = 10000.0
    norm_eps:     float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


PRESETS = {
    "tiny": RegularConfig(vocab_size=1024, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=128, max_seq_len=128),
    "138m": RegularConfig(),                                           # 138,431,232 parameters with the 100K vocabulary
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def rope_table(head_dim: int, max_len: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_len).float(), inv)               # (T, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim); rotates consecutive pairs."""
    T = x.shape[2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:T][None, None], sin[:T][None, None]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: RegularConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.q = nn.Linear(cfg.dim, cfg.n_heads * hd, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: RegularConfig):
        super().__init__()
        self.gate, self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False), nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: RegularConfig):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.dim, cfg.norm_eps), Attention(cfg)
        self.ffn_norm, self.ffn = RMSNorm(cfg.dim, cfg.norm_eps), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class Regular(nn.Module):
    def __init__(self, cfg: RegularConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = rope_table(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:                                            # residual-output layers start smaller
            nn.init.normal_(b.attn.o.weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)
            nn.init.normal_(b.ffn.down.weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert input_ids.shape[1] <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x, self.rope_cos, self.rope_sin)
        return F.linear(self.norm(x), self.embed.weight)                 # tied output embedding

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def expected_params(cfg: RegularConfig) -> int:
        hd = cfg.head_dim
        attn = cfg.dim * cfg.n_heads * hd * 2 + cfg.dim * cfg.n_kv_heads * hd * 2
        ffn = 3 * cfg.dim * cfg.ffn_dim
        return cfg.vocab_size * cfg.dim + cfg.n_layers * (attn + ffn + 2 * cfg.dim) + cfg.dim


def with_(cfg: RegularConfig, **kw) -> RegularConfig:
    return replace(cfg, **kw)
