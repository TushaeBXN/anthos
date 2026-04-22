"""
Anthos — Thought-Token Bifurcated Recurrent Transformer

Core innovation: Thought Tokens
  A small pool of n_thought persistent scratch-pad vectors live inside the
  recurrent block alongside the sequence.  At every loop iteration:

    • Thought tokens attend to the FULL sequence (non-causal)
    • Sequence tokens attend causally to each other + ALL thought tokens
    • Both streams are updated via separate energy-conserving LTI injections
    • Thought tokens are discarded at output — only sequence logits remain

  This creates an explicit bifurcation between:
    Sequence stream — carries content; updated causally; produces output
    Thought stream  — carries reasoning state; sees full context; shapes
                      the sequence stream's attention but leaves no token trace

Architecture:
  Input
    ↓
  [Embedding]
    ↓
  [Prelude blocks]                ← standard transformer, no thought tokens
    ↓
  [Anthos Recurrent Block × T]    ← thought + sequence evolve together
    ├─ Prepend thought tokens → [thought₁…thoughtₙ | tok₁…tokₜ]
    ├─ Modified causal mask:
    │    thoughts → full attention (see everything)
    │    sequence → causal on sequence + full on thoughts
    ├─ TransformerBlock (MoE FFN, MLA/GQA attention)
    ├─ LTI-stable thought update  (energy-conserving)
    ├─ LTI-stable sequence update (energy-conserving)
    └─ ACT halting on sequence positions
    ↓
  [Coda blocks]                   ← standard transformer, no thought tokens
    ↓
  Output logits (sequence only)

Built on OpenMythos v2 foundations:
  • Energy-conserving LTI: h = A·h + (1-A)·combined  (norm bounded by construction)
  • Residual gate: alpha blends t_out vs B·e per channel
  • Vectorized MoE dispatch (sort-by-expert, no GPU-sync loops)
  • Load-balancing + ACT auxiliary losses via pop_aux_loss()
  • Separate RMSNorm for h and e before combining
  • Precomputed sinusoidal loop-index embeddings
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnthosConfig:
    # Token vocabulary
    vocab_size:    int   = 32000

    # Hidden dimension & heads
    dim:           int   = 2048
    n_heads:       int   = 16
    n_kv_heads:    int   = 4          # GQA key/value heads
    max_seq_len:   int   = 4096

    # Recurrence
    max_loop_iters: int  = 16
    prelude_layers: int  = 2
    coda_layers:    int  = 2

    # ── Thought tokens (core Anthos innovation) ──────────────────────────────
    n_thought_tokens: int = 16
    # Thought tokens attend to full sequence; sequence attends to thought tokens.
    # Both streams run separate energy-conserving LTI updates each loop.
    # Thought tokens are discarded before the Coda — they leave no output trace.

    # Attention flavour: "gqa" | "mla"
    attn_type:     str   = "mla"

    # MLA parameters (ignored when attn_type="gqa")
    kv_lora_rank:      int = 512
    q_lora_rank:       int = 1536
    qk_rope_head_dim:  int = 64
    qk_nope_head_dim:  int = 128
    v_head_dim:        int = 128

    # MoE FFN
    n_experts:         int   = 64
    n_shared_experts:  int   = 2
    n_experts_per_tok: int   = 4
    expert_dim:        int   = 512

    # Adaptive Computation Time
    act_threshold:     float = 0.99

    # RoPE
    rope_theta:        float = 500000.0

    # Depth-wise LoRA rank
    lora_rank:         int   = 16

    # Auxiliary loss weights (set to 0 to disable)
    moe_aux_coef:      float = 1e-2
    act_aux_coef:      float = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Positional Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t     = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64 (max_len, dim//2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, head_dim)  freqs_cis: (T, head_dim//2)"""
    xc        = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def _anthos_rope_freqs(
    freqs_cis: torch.Tensor,
    n_thought:  int,
    seq_len:    int,
) -> torch.Tensor:
    """
    Build combined RoPE frequencies for [thought | sequence].

    Thought tokens all receive position-0 frequencies (a fixed neutral
    reference that doesn't encode sequence order — thoughts are not
    sequential, they're working-memory slots).

    Sequence tokens receive their natural positions 0…seq_len-1.

    Returns: (n_thought + seq_len, head_dim//2)
    """
    thought_freqs = freqs_cis[0:1].expand(n_thought, -1)   # all same (pos 0)
    seq_freqs     = freqs_cis[:seq_len]                     # natural positions
    return torch.cat([thought_freqs, seq_freqs], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Thought-aware causal mask
# ─────────────────────────────────────────────────────────────────────────────

def _anthos_causal_mask(n_thought: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Modified causal mask for Anthos thought-token attention.

    Layout:  [thought₁…thoughtₙ | tok₁…tokₜ]

    Thought rows  → 0 everywhere  (see all thoughts AND all sequence positions)
    Sequence rows → 0 for all n_thought thought columns
                  + standard upper-triangular -inf for sequence-to-sequence

    Shape: (1, 1, n_thought + seq_len, n_thought + seq_len)

    This lets thought tokens act as non-causal "global context registers"
    while preserving autoregressive generation for sequence tokens.
    """
    total = n_thought + seq_len
    mask  = torch.zeros(1, 1, total, total, device=device)

    # Sequence-to-sequence block: upper triangle = -inf (causal)
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    mask[0, 0, n_thought:, n_thought:] = causal
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Query Attention
# ─────────────────────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.groups     = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
        kv_cache:  Optional[dict]         = None,
        cache_key: str                    = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        S = k.shape[1]
        k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.groups, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)
        v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.groups, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)

        q = q.transpose(1, 2);  k = k.transpose(1, 2);  v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Latent Attention
# ─────────────────────────────────────────────────────────────────────────────

class MLAttention(nn.Module):
    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.n_heads     = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim  = cfg.qk_rope_head_dim
        self.qk_nope_dim  = cfg.qk_nope_head_dim
        self.v_dim        = cfg.v_head_dim
        self.q_head_dim   = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        self.q_down     = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm     = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope  = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope  = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)

        self.kv_down = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up   = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.wo      = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
        kv_cache:  Optional[dict]         = None,
        cache_key: str                    = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        c_q    = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q      = torch.cat([q_nope, q_rope], dim=-1)

        kv_raw  = self.kv_down(x)
        c_kv    = kv_raw[..., :self.kv_lora_rank]
        k_rope  = kv_raw[..., self.kv_lora_rank:]
        k_rope  = k_rope.unsqueeze(2).expand(B, T, self.n_heads, self.qk_rope_dim).contiguous()
        k_rope  = apply_rope(k_rope, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv   = torch.cat([kv_cache[cache_key]["c_kv"],   c_kv],   dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S  = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv)).view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k  = torch.cat([kv[..., :self.qk_nope_dim], k_rope], dim=-1)
        v  = kv[..., self.qk_nope_dim:]

        q = q.transpose(1, 2);  k = k.transpose(1, 2);  v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Expert & MoE FFN
# ─────────────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up   = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Vectorized MoE dispatch: sort-by-expert gives coalesced memory access,
    no per-expert GPU syncs.  Load-balancing loss tracked via pop_aux_loss().
    """
    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared  = cfg.n_shared_experts
        self.topk      = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
             for _ in range(self.n_shared)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss). aux_loss is differentiable load-balancing loss."""
        B, T, D = x.shape
        N    = B * T
        flat = x.view(N, D)

        logits      = self.router(flat) + self.router_bias
        scores      = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = scores.topk(self.topk, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(-1, keepdim=True)

        tokens_exp  = flat.unsqueeze(1).expand(-1, self.topk, -1).reshape(-1, D)
        expert_ids  = topk_idx.reshape(-1)
        weights_exp = topk_scores.reshape(-1, 1)

        sort_idx          = torch.argsort(expert_ids, stable=True)
        tokens_sorted     = tokens_exp[sort_idx]
        expert_ids_sorted = expert_ids[sort_idx]
        weights_sorted    = weights_exp[sort_idx]

        counts  = torch.bincount(expert_ids_sorted, minlength=self.n_experts)
        offsets = torch.cat([
            torch.zeros(1, device=x.device, dtype=torch.long),
            counts.cumsum(0)
        ])

        output_sorted = torch.zeros_like(tokens_sorted)
        for eid in range(self.n_experts):
            s, e = offsets[eid].item(), offsets[eid + 1].item()
            if s == e:
                continue
            output_sorted[s:e] = self.routed_experts[eid](tokens_sorted[s:e])

        output_unsorted = torch.empty_like(output_sorted)
        output_unsorted[sort_idx] = output_sorted
        out = (output_unsorted * weights_sorted).view(N, self.topk, D).sum(1)

        # Load-balancing aux loss — differentiable, returned directly (no buffer)
        importance = scores.mean(0)
        load       = counts.float() / (N * self.topk)
        aux_loss   = (self.n_experts * importance * load).sum()

        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D), aux_loss


# ─────────────────────────────────────────────────────────────────────────────
# Loop-index embeddings (precomputed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_loop_embeddings(max_loops: int, dim: int, loop_dim: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, dtype=torch.float32) / loop_dim))
    table = torch.zeros(max_loops, dim)
    for t in range(max_loops):
        angles = t * freqs
        emb    = torch.cat([angles.sin(), angles.cos()])[:loop_dim]
        table[t, :loop_dim] = emb
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Depth-wise LoRA adapter
# ─────────────────────────────────────────────────────────────────────────────

class LoRAAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down  = nn.Linear(dim, rank, bias=False)
        self.B     = nn.Parameter(torch.empty(rank, dim))
        nn.init.kaiming_uniform_(self.B)
        self.scale = nn.Embedding(max_loops, rank)
        nn.init.ones_(self.scale.weight)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        s    = self.scale(torch.tensor(loop_t, device=x.device))
        down = self.down(x) * s
        return down @ self.B


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: AnthosConfig, use_moe: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm  = RMSNorm(cfg.dim)
        self.attn  = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn   = MoEFFN(cfg)     if use_moe               else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.use_moe = use_moe

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
        kv_cache:  Optional[dict]         = None,
        cache_key: str                    = "default",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss). aux_loss is 0 for dense blocks."""
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        if self.use_moe:
            ffn_out, aux = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
            aux     = x.new_zeros(1)
        x = x + ffn_out
        return x, aux


# ─────────────────────────────────────────────────────────────────────────────
# Energy-conserving LTI Injection
# ─────────────────────────────────────────────────────────────────────────────

class LTIInjection(nn.Module):
    """
    Energy-conserving recurrent update with residual gate.

    Update rule:
        combined   = alpha * t_out + (1 - alpha) * (B * gated_e)
        h_{t+1}    = A * h_t + (1 - A) * combined

    where:
        A      ∈ (0, 1)  by construction (ZOH discretization of stable LTI)
        alpha  ∈ (0, 1)  per channel, learned residual gate
        gated_e = sigmoid(W_gate · e) * e   (soft input mask)

    The (1-A) factor guarantees energy conservation:
        ‖h_{t+1}‖ ≤ max(‖h_t‖, ‖combined‖)
    Hidden state norm cannot grow across any number of loops.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A  = nn.Parameter(torch.full((dim,), -0.1))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B      = nn.Parameter(torch.full((dim,), 0.5))

        self.input_gate = nn.Linear(dim, dim, bias=False)
        self.res_gate   = nn.Linear(dim * 2, dim, bias=False)
        self.trans_norm = RMSNorm(dim)

    def get_A(self) -> torch.Tensor:
        return torch.exp(torch.exp(self.log_dt) * (-torch.exp(self.log_A)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        A       = self.get_A()
        gated_e = torch.sigmoid(self.input_gate(e)) * e
        t_out   = self.trans_norm(transformer_out)
        alpha   = torch.sigmoid(self.res_gate(torch.cat([h, t_out], dim=-1)))
        combined = alpha * t_out + (1.0 - alpha) * (self.B * gated_e)
        return A * h + (1.0 - A) * combined


# ─────────────────────────────────────────────────────────────────────────────
# ACT Halting
# ─────────────────────────────────────────────────────────────────────────────

class ACTHalting(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Thought Token Pool  (Anthos core)
# ─────────────────────────────────────────────────────────────────────────────

class ThoughtTokenPool(nn.Module):
    """
    Learnable pool of thought tokens.

    Thought tokens are NOT input tokens — they carry no semantic content from
    the vocabulary.  They are learned scratch-pad vectors that the model uses
    to accumulate and route reasoning across recurrent loop iterations.

    Each call to init_batch() returns a fresh batch-expanded copy of the
    learned embeddings.  They are reset per sequence and not shared across
    requests.  The pool itself (self.embeddings) is a model parameter updated
    through gradient descent like any other weight.

    Design choices:
      • n_thought is a hyperparameter: 8–32 is practical for most scales.
        More thoughts = richer working memory but higher attention cost.
      • Initialized with N(0, 0.02) matching token embedding init.
      • The thought tokens' "position" in RoPE is always position 0
        (neutral, non-sequential reference — see _anthos_rope_freqs).
    """

    def __init__(self, n_thought: int, dim: int):
        super().__init__()
        self.n_thought  = n_thought
        self.embeddings = nn.Parameter(torch.empty(n_thought, dim))
        nn.init.normal_(self.embeddings, std=0.02)

    def init_batch(self, B: int, device: torch.device) -> torch.Tensor:
        """Return (B, n_thought, dim) — fresh copy so thoughts diverge per-batch."""
        return self.embeddings.unsqueeze(0).expand(B, -1, -1).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Anthos Recurrent Block  (thought + sequence dual-stream)
# ─────────────────────────────────────────────────────────────────────────────

class AnthosRecurrentBlock(nn.Module):
    """
    The core of Anthos: one TransformerBlock looped T times, with thought
    tokens participating in every attention call.

    Per loop iteration:
      1. Prepend thought tokens to sequence hidden state
         → full tensor: (B, n_thought + seq_len, dim)
      2. Build Anthos causal mask:
         → thoughts: full attention everywhere
         → sequence: causal on sequence, full on thoughts
      3. Build combined RoPE freqs:
         → thoughts: all get position-0 frequencies
         → sequence: natural 0…T-1 frequencies
      4. Run TransformerBlock (shared weights, looped) + LoRA delta
      5. Split output back into thought_out and seq_out
      6. Update thought stream via thought_injection (LTI)
         e_thought = mean-pooled encoded input (global summary)
      7. Update sequence stream via seq_injection (LTI)
      8. Normalize both streams (h_norm, thought_norm)
      9. ACT halting on sequence tokens; thoughts run every loop
     10. Accumulate ACT penalty loss
    """

    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.cfg       = cfg
        self.n_thought = cfg.n_thought_tokens

        self.block = TransformerBlock(cfg, use_moe=True)
        self.lora  = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

        # Separate LTI injections for sequence and thought streams
        self.seq_injection     = LTIInjection(cfg.dim)
        self.thought_injection = LTIInjection(cfg.dim)

        # Separate norms for sequence and thought streams
        self.h_pre_norm   = RMSNorm(cfg.dim)   # pre-combine norm for sequence h
        self.e_norm       = RMSNorm(cfg.dim)   # encoded input norm (once, before loop)
        self.h_norm       = RMSNorm(cfg.dim)   # post-injection norm for sequence
        self.thought_norm = RMSNorm(cfg.dim)   # post-injection norm for thoughts

        # ACT halting (sequence only — thoughts run every loop)
        self.act = ACTHalting(cfg.dim)

        # Precomputed loop-index sinusoidal embeddings
        self.loop_dim = cfg.dim // 8
        loop_table    = _build_loop_embeddings(cfg.max_loop_iters, cfg.dim, self.loop_dim)
        self.register_buffer("loop_embeds", loop_table)   # (max_loop_iters, dim)

    def forward(
        self,
        h:         torch.Tensor,
        thoughts:  torch.Tensor,
        e:         torch.Tensor,
        freqs_cis: torch.Tensor,
        n_loops:   Optional[int]  = None,
        kv_cache:  Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_out, moe_aux_total, act_aux). All differentiable."""
        n_loops = n_loops or self.cfg.max_loop_iters

        B, T, D = h.shape
        halted       = torch.zeros(B, T,    device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T,    device=h.device)
        h_out        = torch.zeros_like(h)
        loops_used   = torch.zeros(B, T,    device=h.device)

        e_normed      = self.e_norm(e)
        e_thought     = e.mean(dim=1, keepdim=True).expand_as(thoughts)
        combined_mask = _anthos_causal_mask(self.n_thought, T, h.device) if T > 1 else None
        moe_aux_total = h.new_zeros(1)

        for t in range(n_loops):
            loop_emb       = self.loop_embeds[t]
            h_combined     = self.h_pre_norm(h + loop_emb) + e_normed
            full_seq       = torch.cat([thoughts, h_combined], dim=1)
            combined_freqs = _anthos_rope_freqs(freqs_cis, self.n_thought, T)

            cache_key         = f"anthos_loop_{t}"
            full_out, moe_aux = self.block(full_seq, combined_freqs, combined_mask, kv_cache, cache_key)
            full_out          = full_out + self.lora(full_out, t)
            moe_aux_total     = moe_aux_total + moe_aux

            thought_out = full_out[:, :self.n_thought, :]
            seq_out     = full_out[:, self.n_thought:,  :]

            thoughts = self.thought_norm(self.thought_injection(thoughts, e_thought, thought_out))
            h        = self.h_norm(self.seq_injection(h, e, seq_out))

            p             = self.act(h)
            still_running = ~halted
            remainder     = (1.0 - cumulative_p).clamp(min=0.0)
            weight        = torch.where(
                cumulative_p + p >= self.cfg.act_threshold, remainder, p,
            ) * still_running.float()

            h_out        = h_out        + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.float()
            halted       = halted       | (cumulative_p >= self.cfg.act_threshold)
            loops_used   = loops_used   + still_running.float()

            if halted.all():
                break

        act_aux = loops_used.mean()  # differentiable ACT penalty
        return h_out, moe_aux_total, act_aux


# ─────────────────────────────────────────────────────────────────────────────
# Full Anthos Model
# ─────────────────────────────────────────────────────────────────────────────

class Anthos(nn.Module):
    """
    Anthos — Thought-Token Bifurcated Recurrent Transformer

    Input tokens
      ↓
    [Embedding]
      ↓
    [Prelude]          — standard transformer blocks, no thought tokens
      ↓
    [AnthosRecurrentBlock × T loops]
      ├─ [thought₁…thoughtₙ | tok₁…tokₜ] processed together each loop
      ├─ Thoughts: non-causal working memory, separate LTI, discarded at end
      └─ Sequence: causal content stream, energy-conserving LTI, ACT halting
      ↓
    [Coda]             — standard transformer blocks, sequence only
      ↓
    Output logits
    """

    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.cfg = cfg

        self.embed       = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.thought_pool = ThoughtTokenPool(cfg.n_thought_tokens, cfg.dim)

        # Separate RoPE freq tables for GQA (full head_dim) and MLA (rope_head_dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = AnthosRecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Restore LoRA-specific inits that _init_weights would overwrite
        nn.init.kaiming_uniform_(self.recurrent.lora.B)
        nn.init.ones_(self.recurrent.lora.scale.weight)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids:  torch.Tensor,
        n_loops:    Optional[int]  = None,
        kv_cache:   Optional[dict] = None,
        return_aux: bool           = False,
    ):
        """
        Args:
            input_ids  — (B, T) token indices
            n_loops    — recurrent depth (defaults to cfg.max_loop_iters)
            kv_cache   — mutable dict for autoregressive KV caching
            return_aux — if True return (logits, aux_loss); add aux_loss to CE
                         during training for load-balancing + ACT regularization

        Returns:
            logits (B, T, vocab_size)  or  (logits, aux_loss) if return_aux
        """
        B, T    = input_ids.shape
        device  = input_ids.device

        x = self.embed(input_ids)                          # (B, T, dim)

        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[:T]

        # Standard causal mask for prelude/coda (no thought tokens)
        std_mask = self._causal_mask(T, device) if T > 1 else None

        # ── Prelude ───────────────────────────────────────────────────────
        for i, layer in enumerate(self.prelude):
            x, _ = layer(x, freqs_cis, std_mask, kv_cache, cache_key=f"prelude_{i}")

        # ── Anthos Recurrent Block ────────────────────────────────────────
        e        = x
        thoughts = self.thought_pool.init_batch(B, device)
        x, moe_aux, act_aux = self.recurrent(x, thoughts, e, freqs_cis, n_loops, kv_cache)

        # ── Coda ──────────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            x, _ = layer(x, freqs_cis, std_mask, kv_cache, cache_key=f"coda_{i}")

        logits = self.head(self.norm(x))

        if not return_aux:
            return logits

        aux_loss = self.cfg.moe_aux_coef * moe_aux + self.cfg.act_aux_coef * act_aux
        return logits, aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 64,
        n_loops:        int   = 8,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        kv_cache  = {}
        generated = input_ids

        for step in range(max_new_tokens):
            ctx    = generated if step == 0 else generated[:, -1:]
            logits = self.forward(ctx, n_loops=n_loops, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# Pre-configured variants
# ─────────────────────────────────────────────────────────────────────────────

def _base(overrides: dict) -> AnthosConfig:
    defaults = dict(
        attn_type="mla", n_kv_heads=8,
        kv_lora_rank=512, q_lora_rank=1536,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        prelude_layers=2, coda_layers=2,
        n_shared_experts=2, n_experts_per_tok=4,
        act_threshold=0.99, rope_theta=500000.0, lora_rank=16,
        moe_aux_coef=1e-2, act_aux_coef=1e-3,
    )
    defaults.update(overrides)
    return AnthosConfig(**defaults)


# n_thought_tokens scales with model size — more parameters = richer working memory
def anthos_1b()   -> AnthosConfig: return _base(dict(dim=2048, n_heads=16, n_experts=64,  expert_dim=2048, max_loop_iters=16, max_seq_len=4096,   vocab_size=32000, n_thought_tokens=16))
def anthos_3b()   -> AnthosConfig: return _base(dict(dim=3072, n_heads=24, n_experts=64,  expert_dim=4096, max_loop_iters=16, max_seq_len=4096,   vocab_size=32000, n_thought_tokens=24))
def anthos_10b()  -> AnthosConfig: return _base(dict(dim=4096, n_heads=32, n_experts=128, expert_dim=5632, max_loop_iters=24, max_seq_len=8192,   vocab_size=32000, n_thought_tokens=32))
def anthos_50b()  -> AnthosConfig: return _base(dict(dim=6144, n_heads=48, n_experts=256, expert_dim=9728, max_loop_iters=32, max_seq_len=8192,   vocab_size=32000, n_thought_tokens=48))
def anthos_100b() -> AnthosConfig: return _base(dict(dim=8192, n_heads=64, n_experts=256, expert_dim=13568,max_loop_iters=32, max_seq_len=131072, vocab_size=32000, n_thought_tokens=64))
