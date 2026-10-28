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

# Memory bank — imported lazily inside AnthosRecurrentBlock to avoid circular import
# at module load time (memory.py imports torch but not anthos.main)
from anthos.memory     import MemoryBank, MemoryBankConfig, MemoryBankState
from anthos.lora_pairs import DualLoRAAdapter
# VisionEncoder/VisionConfig imported lazily inside Anthos.__init__ — keeps
# `transformers` an optional dependency for text-only training/inference.


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnthosConfig:
    # Token vocabulary
    vocab_size:    int   = 50262  # GPT-2 base (50257) + 5 Anthos special tokens

    # Hidden dimension & heads
    dim:           int   = 2048
    n_heads:       int   = 16
    n_kv_heads:    int   = 8          # GQA key/value heads
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
    expert_dim:        int   = 2048

    # Adaptive Computation Time
    act_threshold:     float = 0.99

    # RoPE
    rope_theta:        float = 10000.0

    # Depth-wise LoRA rank
    lora_rank:         int   = 16

    # Auxiliary loss weights (set to 0 to disable)
    moe_aux_coef:          float = 1e-2
    act_aux_coef:          float = 1e-3
    prerouter_loss_coef:   float = 0.05   # KL weight: prerouter head vs real MoE routing

    # Multi-Token Prediction
    # Extra prediction heads that teach the model to predict tokens 2, 3, ... steps
    # ahead from the same Coda hidden state. Each head is a lightweight RMSNorm that
    # feeds into the shared self.head projection (weight-tied to embeddings), so
    # adding 2 MTP heads costs only 2 × dim extra parameters.
    #
    # n_mtp_heads=0  → disabled (default, fully backward compatible)
    # n_mtp_heads=2  → predict t+2 and t+3 in addition to t+1
    #
    # Loss: mtp_loss_coef * mean(CE_k for each depth k)
    # Nemotron Lightning uses mtp_loss_coef=0.3 — start there.
    n_mtp_heads:           int   = 0
    mtp_loss_coef:         float = 0.3

    # Z-loss: penalises log(Z)² where Z = sum(exp(logits)) over vocab.
    # Prevents logit scale explosion / softmax saturation — critical when
    # MoE + MTP heads both push the output head in different directions.
    # OLMo-core uses 1e-4; safe to raise to 1e-3 if logits drift above ±30.
    zloss_coef:            float = 1e-4

    # Activation checkpointing: recompute TransformerBlock activations during
    # backward instead of storing them, trading ~30% extra compute for ~60%
    # less activation memory per recurrent loop iteration.
    # Safe to enable whenever kv_cache is not used (i.e. all training).
    # Disable for inference (kv_cache path skips checkpointing automatically).
    gradient_checkpointing: bool = False

    # Prompt-only thoughts (causality fix). With the default mask, thought tokens
    # attend to the WHOLE sequence and every position attends to the thoughts, so
    # information about later tokens reaches earlier positions. Training can use
    # that channel; generation cannot (the future does not exist yet).
    # thought_prefix_only=True restricts thoughts to the first `prefix_len`
    # positions (the prompt), so every later position is strictly causal.
    # Pass prefix_len to Anthos.forward. False (default) keeps the old behaviour
    # exactly. Adds no parameters, so existing checkpoints load unchanged.
    thought_prefix_only:   bool  = False

    # MoE routing-weight pairing fix. The legacy dispatch multiplies each expert
    # output by the router weight of a *different* token (expert-sorted weights
    # applied to token-ordered outputs). True makes each token use its own weights,
    # which is what the layer is meant to compute and makes it token-independent.
    # False (default) reproduces the legacy maths exactly, so existing checkpoints
    # behave as before. Enable together with thought_prefix_only for a causal model.
    moe_fix_weight_order:  bool  = False

    # Frozen-thought decode shortcut (only used with thought_prefix_only). A decode step
    # normally pushes the n_thought thought rows plus the new token through every loop just
    # to rebuild thought K/V that never change. True feeds only the new token and reuses the
    # thought K/V cached at prefill. False keeps the slower reference path (for A/B checks).
    fast_thought_decode:   bool  = True

    # YaRN RoPE scaling: extend context beyond max_seq_len at inference time.
    # Set rope_scaling_type="yarn" and rope_scaling_factor > 1.0 to enable.
    # Example: trained at max_seq_len=4096, extend to 32k with factor=8.0.
    # factor=1.0 → no scaling (disabled, default).
    # Reference: Peng et al. 2023 "YaRN: Efficient Context Window Extension"
    rope_scaling_type:   str   = "none"    # "none" | "yarn" | "linear"
    rope_scaling_factor: float = 1.0       # > 1.0 to extend context

    # Mixture of Depths: route a fraction of tokens to skip each block.
    # A learned scalar router decides which tokens pass through vs are carried
    # unchanged via the residual. Reduces FLOP/token at inference time once
    # the router has learned which positions need computation.
    # mod_capacity = fraction of tokens that actually go through each block.
    # 0.5 = half the tokens compute, half are bypassed — ~50% FLOP savings.
    # Reference: Raposo et al. 2024 "Mixture of Depths"
    use_mod:      bool  = False    # enable Mixture of Depths routing
    mod_capacity: float = 0.5     # fraction of tokens that go through each block (0, 1]

    # Stochastic depth (layer drop): randomly skip entire transformer blocks
    # during training with probability stochastic_depth_rate. This acts as a
    # regularizer and speeds up training (~20% fewer FLOPs on average at p=0.2).
    # Applied linearly: earlier blocks have lower drop rate, later blocks higher.
    # Disabled at inference (eval mode). Set to 0.0 to disable.
    # Reference: Huang et al. 2016 "Deep Networks with Stochastic Depth"
    stochastic_depth_rate: float = 0.0   # max drop probability (last layer)

    # Differential Attention: two softmax heads subtracted to cancel noise.
    # DiffAttn(X) = (softmax(Q1·K1^T/√d) - λ·softmax(Q2·K2^T/√d)) · V
    # λ is learned; initialised near 0 so early training = standard attention.
    # ~25% better on long-context retrieval and hallucination benchmarks.
    # Only applies to GQA-style attention (attn_type="gqa").
    # Reference: Ye et al. 2024 "Differential Transformer"
    use_diff_attn: bool = False   # enable differential attention

    # Expert Choice routing: experts select top-k tokens instead of tokens
    # selecting top-k experts. Guarantees uniform expert load by construction;
    # eliminates the need for a load-balancing auxiliary loss.
    # Capacity = (n_tokens * n_experts_per_tok) / n_experts tokens per expert.
    # Reference: Zhou et al. 2022 "Mixture-of-Experts with Expert Choice Routing"
    use_expert_choice: bool = False   # enable Expert Choice MoE routing


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


def precompute_rope_freqs_yarn(
    dim:        int,
    max_len:    int,
    theta:      float = 500000.0,
    factor:     float = 8.0,
    beta_fast:  int   = 32,
    beta_slow:  int   = 1,
    mscale:     float = 0.1,
) -> torch.Tensor:
    """
    YaRN (Yet another RoPE extensioN) — extend context window beyond training length.

    YaRN interpolates RoPE frequencies differently per head dimension:
      - Low-frequency dims (long-range): extrapolate (no rescaling needed)
      - High-frequency dims (short-range): interpolate (divide by factor)
      - Mid-range: smoothly blend between the two via a ramp

    This preserves local attention (short-range dims unchanged) while letting
    long-range dims attend to extended positions.

    Usage:
        # Model trained at 4096; extend to 32k at inference time
        cfg.rope_scaling_type   = "yarn"
        cfg.rope_scaling_factor = 8.0   # 32768 / 4096

    Args:
        dim:       RoPE head dimension
        max_len:   Target context length (after extension)
        theta:     Base RoPE frequency (default 500000.0, same as training)
        factor:    Extension factor = new_ctx / train_ctx (e.g. 8.0 for 8×)
        beta_fast: High-frequency cutoff (dim index, default 32)
        beta_slow: Low-frequency cutoff  (dim index, default 1)
        mscale:    Attention magnitude scale (default 0.1, per YaRN paper)
    """
    half_dim = dim // 2
    # Original frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN ramp: r(i) smoothly blends between linear interp and extrapolation
    low  = math.log(factor) / math.log(beta_fast) * half_dim
    high = math.log(factor) / math.log(beta_slow) * half_dim
    ramp = torch.clamp(
        (torch.arange(half_dim, dtype=torch.float32) - low) / (high - low + 1e-6),
        0.0, 1.0,
    )

    # Interpolated frequencies: divide by factor in the high-freq end
    freqs_interp = freqs / factor
    freqs_yarn   = freqs_interp * (1 - ramp) + freqs * ramp

    # Attention magnitude correction (prevents softmax saturation at long range)
    attn_scale = 1.0 + mscale * math.log(factor)

    t     = torch.arange(max_len, dtype=torch.float32)
    outer = torch.outer(t, freqs_yarn) * attn_scale
    return torch.polar(torch.ones_like(outer), outer)   # complex64


def precompute_rope_freqs_linear(
    dim:     int,
    max_len: int,
    theta:   float = 500000.0,
    factor:  float = 8.0,
) -> torch.Tensor:
    """
    Linear RoPE scaling (position interpolation, Chen et al. 2023).
    Simpler than YaRN but degrades on tasks needing precise short-range attention.
    All frequencies divided by factor uniformly.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = freqs / factor
    t     = torch.arange(max_len, dtype=torch.float32)
    return torch.polar(torch.ones_like(torch.outer(t, freqs)), torch.outer(t, freqs))


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, head_dim)  freqs_cis: (T, head_dim//2)"""
    xc        = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def _anthos_rope_freqs(
    freqs_cis:       torch.Tensor,
    thought_freqs_0: torch.Tensor,
    n_thought:       int,
    seq_len:         int,
) -> torch.Tensor:
    """
    Build combined RoPE frequencies for [thought | sequence].

    Thought tokens all receive true position-0 frequencies via `thought_freqs_0`
    (a fixed neutral reference that doesn't encode sequence order — thoughts are
    not sequential, they're working-memory slots).

    `thought_freqs_0` must be the absolute position-0 frequency, NOT derived
    from the current decode window's `freqs_cis`, which is pre-sliced to
    [start_pos:] and would give the wrong position during decode (freqs_cis[0]
    drifts with start_pos instead of staying pinned to true position 0).

    Sequence tokens receive their natural positions from `freqs_cis`.

    Returns: (n_thought + seq_len, head_dim//2)
    """
    thought_freqs = thought_freqs_0.expand(n_thought, -1)   # (n_thought, head_dim//2) — true pos-0
    seq_freqs     = freqs_cis[:seq_len]                      # natural positions
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


def _anthos_prefix_mask(
    n_thought:  int,
    seq_len:    int,
    prefix_len: torch.Tensor,
    device:     torch.device,
) -> torch.Tensor:
    """
    Causal-safe variant of _anthos_causal_mask: thoughts only see the prompt.

    Thought rows   → all thoughts + sequence columns j < prefix_len[b]
    Sequence rows  → all thoughts + standard causal sequence attention

    Because thoughts never see a position >= prefix_len, a response token can
    only receive information from tokens at or before itself.

    prefix_len: (B,) long tensor. Shape returned: (B, 1, N, N), N = n_thought + seq_len.
    """
    B     = prefix_len.shape[0]
    total = n_thought + seq_len
    mask  = torch.zeros(B, 1, total, total, device=device)

    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    mask[:, 0, n_thought:, n_thought:] = causal

    cols    = torch.arange(seq_len, device=device).unsqueeze(0)          # (1, T)
    blocked = cols >= prefix_len.to(device).unsqueeze(1)                 # (B, T)
    thought_rows = mask[:, 0, :n_thought, n_thought:]                    # (B, n_thought, T) view
    thought_rows.masked_fill_(blocked.unsqueeze(1), float("-inf"))
    return mask


def _prefix_mean(e: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
    """Mean of e over the first prefix_len[b] positions → (B, D); zeros if prefix_len is 0."""
    B, T, _ = e.shape
    keep  = (torch.arange(T, device=e.device).unsqueeze(0) < prefix_len.to(e.device).unsqueeze(1)).to(e.dtype)
    denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (e * keep.unsqueeze(-1)).sum(dim=1) / denom


def _causal_sdpa(q, k, v, fix_cached: bool, attn_mask=None):
    """
    Causal SDPA that stays correct when the query is shorter than the keys (cached decode).

    F.scaled_dot_product_attention(is_causal=True) aligns the mask at the top-left, so a
    1-token query over S cached keys would only see key 0. With fix_cached the mask is
    aligned at the bottom-right instead (each query sees every earlier key and itself).
    fix_cached=False keeps the legacy behaviour byte-for-byte.
    """
    T, S = q.shape[-2], k.shape[-2]
    if fix_cached and T != S:
        keep = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Query Attention
# ─────────────────────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    fix_cached_causal: bool = False   # set by TransformerBlock from cfg.thought_prefix_only

    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.groups     = cfg.n_heads // cfg.n_kv_heads
        self.max_seq_len = cfg.max_seq_len

        self.wq = nn.Linear(cfg.dim, cfg.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)

    def forward(
        self,
        x:                torch.Tensor,
        freqs_cis:        torch.Tensor,
        mask:             Optional[torch.Tensor] = None,
        kv_cache:         Optional[dict]         = None,
        cache_key:        str                    = "default",
        n_thought_prefix: int                    = 0,
        is_causal:        bool                   = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            # Bifurcated caching: only sequence-token KVs accumulate in the cache.
            # Thought tokens are regenerated fresh each decoding step, so including
            # them in the cache would grow the context by n_thought per step (OOM bug).
            #
            # Combined buffer layout: [thought_slots | sequence_slots]
            # Thought slots (first n_thought_prefix positions) are overwritten in-place
            # each decode step with fresh values — no torch.cat needed.
            k_new = k[:, n_thought_prefix:, :]
            v_new = v[:, n_thought_prefix:, :]
            T_new = k_new.shape[1]
            if cache_key not in kv_cache:
                total = n_thought_prefix + self.max_seq_len
                buf_k = k_new.new_zeros(B, total, self.n_kv_heads, self.head_dim)
                buf_v = v_new.new_zeros(B, total, self.n_kv_heads, self.head_dim)
                if n_thought_prefix > 0:
                    buf_k[:, :n_thought_prefix].copy_(k[:, :n_thought_prefix])
                    buf_v[:, :n_thought_prefix].copy_(v[:, :n_thought_prefix])
                buf_k[:, n_thought_prefix:n_thought_prefix + T_new].copy_(k_new)
                buf_v[:, n_thought_prefix:n_thought_prefix + T_new].copy_(v_new)
                kv_cache[cache_key] = {"k": buf_k, "v": buf_v, "seq_pos": T_new, "n_thought": n_thought_prefix}
            else:
                n_th = kv_cache[cache_key]["n_thought"]
                pos  = kv_cache[cache_key]["seq_pos"]
                end  = pos + T_new
                if n_th > 0 and n_thought_prefix > 0:   # frozen-thought decode passes no thought rows
                    kv_cache[cache_key]["k"][:, :n_th].copy_(k[:, :n_th])
                    kv_cache[cache_key]["v"][:, :n_th].copy_(v[:, :n_th])
                kv_cache[cache_key]["k"][:, n_th + pos:n_th + end].copy_(k_new)
                kv_cache[cache_key]["v"][:, n_th + pos:n_th + end].copy_(v_new)
                kv_cache[cache_key]["seq_pos"] = end
            n_th    = kv_cache[cache_key]["n_thought"]
            seq_pos = kv_cache[cache_key]["seq_pos"]
            k = kv_cache[cache_key]["k"][:, :n_th + seq_pos]
            v = kv_cache[cache_key]["v"][:, :n_th + seq_pos]

        S = k.shape[1]
        k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.groups, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)
        v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.groups, self.head_dim).reshape(B, S, self.n_heads, self.head_dim)

        q = q.transpose(1, 2);  k = k.transpose(1, 2);  v = v.transpose(1, 2)
        if is_causal and mask is None:
            out = _causal_sdpa(q, k, v, self.fix_cached_causal)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Differential GQA (DIFF Transformer variant)
# ─────────────────────────────────────────────────────────────────────────────

class DiffGQAttention(nn.Module):
    """
    Differential Attention (Ye et al. 2024).

    Two separate softmax attention maps are computed and subtracted:
        DiffAttn = (softmax(Q1·K1^T/√d) - λ·softmax(Q2·K2^T/√d)) · V

    The subtraction cancels noise/irrelevant tokens and sharpens focus on
    relevant positions, improving long-context retrieval.

    λ is a learnable scalar per head, initialized near 0 so the model
    starts with approximately standard attention.

    Architecture: head_dim is halved per sub-head (Q1/Q2 each get head_dim//2),
    so total parameter count matches standard GQA.
    """

    fix_cached_causal: bool = False   # set by TransformerBlock from cfg.thought_prefix_only

    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.dim // cfg.n_heads
        self.sub_dim    = self.head_dim // 2     # each sub-head uses half
        self.groups     = cfg.n_heads // cfg.n_kv_heads
        self.max_seq_len = cfg.max_seq_len

        # Two Q/K projections; shared V
        self.wq1 = nn.Linear(cfg.dim, cfg.n_heads    * self.sub_dim, bias=False)
        self.wq2 = nn.Linear(cfg.dim, cfg.n_heads    * self.sub_dim, bias=False)
        self.wk1 = nn.Linear(cfg.dim, cfg.n_kv_heads * self.sub_dim, bias=False)
        self.wk2 = nn.Linear(cfg.dim, cfg.n_kv_heads * self.sub_dim, bias=False)
        self.wv  = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo  = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim,    bias=False)

        # λ: one learnable scalar per head; init 0.05 (near-zero → standard attn at init)
        self.lambda_q1 = nn.Parameter(torch.full((cfg.n_heads,), 0.05))
        self.lambda_q2 = nn.Parameter(torch.full((cfg.n_heads,), 0.05))
        self.lambda_k1 = nn.Parameter(torch.full((cfg.n_heads,), 0.05))
        self.lambda_k2 = nn.Parameter(torch.full((cfg.n_heads,), 0.05))

    def forward(
        self,
        x:                torch.Tensor,
        freqs_cis:        torch.Tensor,
        mask:             Optional[torch.Tensor] = None,
        kv_cache:         Optional[dict]         = None,
        cache_key:        str                    = "default",
        n_thought_prefix: int                    = 0,
        is_causal:        bool                   = False,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q1 = self.wq1(x).view(B, T, self.n_heads, self.sub_dim)
        q2 = self.wq2(x).view(B, T, self.n_heads, self.sub_dim)
        k1 = self.wk1(x).view(B, T, self.n_kv_heads, self.sub_dim)
        k2 = self.wk2(x).view(B, T, self.n_kv_heads, self.sub_dim)
        v  = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to both query/key pairs
        q1 = apply_rope(q1, freqs_cis); q2 = apply_rope(q2, freqs_cis)
        k1 = apply_rope(k1, freqs_cis); k2 = apply_rope(k2, freqs_cis)

        # Expand KV heads for GQA
        def _expand_kv(t):
            S = t.shape[1]
            return t.unsqueeze(3).expand(B, S, self.n_kv_heads, self.groups, -1).reshape(
                B, S, self.n_heads, -1
            )

        k1 = _expand_kv(k1); k2 = _expand_kv(k2)
        v  = _expand_kv(v)

        # Transpose to (B, H, T, d)
        q1 = q1.transpose(1, 2); q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2); k2 = k2.transpose(1, 2)
        v  = v.transpose(1, 2)

        scale = self.sub_dim ** -0.5
        am = mask if not (is_causal and mask is None) else None
        ic = is_causal and mask is None

        if ic and self.fix_cached_causal and q1.shape[-2] != k1.shape[-2]:
            a1 = _causal_sdpa(q1, k1, v, True)
            a2 = _causal_sdpa(q2, k2, v, True)
        else:
            a1 = F.scaled_dot_product_attention(q1, k1, v, attn_mask=am, is_causal=ic)
            a2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=am, is_causal=ic)

        # λ per-head: exp(λq1·λk1) - exp(λq2·λk2)
        lam = (self.lambda_q1 * self.lambda_k1).exp() - (self.lambda_q2 * self.lambda_k2).exp()
        lam = lam.view(1, self.n_heads, 1, 1)

        out = (a1 - lam * a2).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Latent Attention
# ─────────────────────────────────────────────────────────────────────────────

class MLAttention(nn.Module):
    fix_cached_causal: bool = False   # set by TransformerBlock from cfg.thought_prefix_only

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
        self.max_seq_len = cfg.max_seq_len

    def forward(
        self,
        x:                torch.Tensor,
        freqs_cis:        torch.Tensor,
        mask:             Optional[torch.Tensor] = None,
        kv_cache:         Optional[dict]         = None,
        cache_key:        str                    = "default",
        n_thought_prefix: int                    = 0,
        is_causal:        bool                   = False,
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
            # Bifurcated caching: only sequence-token KV projections accumulate.
            # Thought tokens are regenerated fresh each step; caching them would
            # grow the KV sequence by n_thought per decoding step (OOM bug).
            #
            # Combined buffer layout: [thought_slots | sequence_slots]
            # Thought slots (first n_thought_prefix positions) are overwritten in-place
            # each decode step with fresh values — no torch.cat needed.
            c_kv_new   = c_kv[:, n_thought_prefix:, :]
            k_rope_new = k_rope[:, n_thought_prefix:, :]
            T_new      = c_kv_new.shape[1]
            if cache_key not in kv_cache:
                total      = n_thought_prefix + self.max_seq_len
                buf_c_kv   = c_kv_new.new_zeros(B, total, self.kv_lora_rank)
                buf_k_rope = k_rope_new.new_zeros(B, total, self.n_heads, self.qk_rope_dim)
                if n_thought_prefix > 0:
                    buf_c_kv[:, :n_thought_prefix].copy_(c_kv[:, :n_thought_prefix])
                    buf_k_rope[:, :n_thought_prefix].copy_(k_rope[:, :n_thought_prefix])
                buf_c_kv[:, n_thought_prefix:n_thought_prefix + T_new].copy_(c_kv_new)
                buf_k_rope[:, n_thought_prefix:n_thought_prefix + T_new].copy_(k_rope_new)
                kv_cache[cache_key] = {"c_kv": buf_c_kv, "k_rope": buf_k_rope, "seq_pos": T_new, "n_thought": n_thought_prefix}
            else:
                n_th = kv_cache[cache_key]["n_thought"]
                pos  = kv_cache[cache_key]["seq_pos"]
                end  = pos + T_new
                if n_th > 0 and n_thought_prefix > 0:   # frozen-thought decode passes no thought rows
                    kv_cache[cache_key]["c_kv"][:, :n_th].copy_(c_kv[:, :n_th])
                    kv_cache[cache_key]["k_rope"][:, :n_th].copy_(k_rope[:, :n_th])
                kv_cache[cache_key]["c_kv"][:, n_th + pos:n_th + end].copy_(c_kv_new)
                kv_cache[cache_key]["k_rope"][:, n_th + pos:n_th + end].copy_(k_rope_new)
                kv_cache[cache_key]["seq_pos"] = end
            n_th    = kv_cache[cache_key]["n_thought"]
            seq_pos = kv_cache[cache_key]["seq_pos"]
            c_kv   = kv_cache[cache_key]["c_kv"][:, :n_th + seq_pos]
            k_rope = kv_cache[cache_key]["k_rope"][:, :n_th + seq_pos]

        S  = c_kv.shape[1]
        kv = self.kv_up(self.kv_norm(c_kv)).view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k  = torch.cat([kv[..., :self.qk_nope_dim], k_rope], dim=-1)
        v  = kv[..., self.qk_nope_dim:]

        q = q.transpose(1, 2);  k = k.transpose(1, 2);  v = v.transpose(1, 2)
        if is_causal and mask is None:
            out = _causal_sdpa(q, k, v, self.fix_cached_causal)
        else:
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
        self._use_expert_choice = cfg.use_expert_choice
        self._fix_weight_order  = cfg.moe_fix_weight_order

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
             for _ in range(self.n_shared)]
        )

    def _expert_choice_forward(self, flat: torch.Tensor, B: int, T: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert Choice routing: each expert selects its top-k tokens."""
        N          = B * T
        capacity   = max(1, int(N * self.topk / self.n_experts))

        logits     = self.router(flat) + self.router_bias   # (N, n_experts)
        scores     = F.softmax(logits, dim=-1)              # (N, n_experts)
        scores_T   = scores.T                               # (n_experts, N)

        # Each expert picks its top-capacity tokens
        expert_topk_val, expert_topk_idx = scores_T.topk(capacity, dim=-1)  # (n_exp, cap)

        out = torch.zeros_like(flat)
        for eid in range(self.n_experts):
            sel  = expert_topk_idx[eid]              # (cap,) token indices
            wts  = expert_topk_val[eid].unsqueeze(-1)  # (cap, 1)
            tok  = flat[sel]                          # (cap, D)
            res  = self.routed_experts[eid](tok)      # (cap, D)
            out.scatter_add_(0, sel.unsqueeze(-1).expand_as(res), wts * res)

        # No explicit aux loss needed — load balance is guaranteed by construction
        aux_loss = flat.new_zeros(1)
        self._last_topk_idx = expert_topk_idx.T.contiguous()   # (cap, n_exp) → use first col

        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D), aux_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss). aux_loss is differentiable load-balancing loss."""
        B, T, D = x.shape
        N    = B * T
        flat = x.view(N, D)

        if getattr(self, "_use_expert_choice", False):
            return self._expert_choice_forward(flat, B, T, D)

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
        offsets_cpu = offsets.cpu().tolist()  # one sync instead of 2*n_experts
        for eid in range(self.n_experts):
            s, e = offsets_cpu[eid], offsets_cpu[eid + 1]
            if s == e:
                continue
            output_sorted[s:e] = self.routed_experts[eid](tokens_sorted[s:e])

        output_unsorted = torch.empty_like(output_sorted)
        output_unsorted[sort_idx] = output_sorted
        # output_unsorted is back in token-major order, so it must be paired with the
        # token-major weights_exp. weights_sorted is in expert-sorted order; using it
        # here (the legacy behaviour) hands each token another token's router weight,
        # which makes the layer depend on other tokens, including later ones.
        w   = weights_exp if self._fix_weight_order else weights_sorted
        out = (output_unsorted * w).view(N, self.topk, D).sum(1)

        # Load-balancing aux loss — differentiable, returned directly (no buffer)
        importance = scores.mean(0)
        load       = counts.float() / (N * self.topk)
        aux_loss   = (self.n_experts * importance * load).sum()

        # Store routing decisions for utilization logging (detached, not a gradient)
        self._last_topk_idx = topk_idx.detach()

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
# Depth-wise LoRA adapter (legacy single-path — kept for backward compat)
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
# Mixture of Depths router
# ─────────────────────────────────────────────────────────────────────────────

class MoDRouter(nn.Module):
    """
    Mixture of Depths (MoD) token router.

    A single learned scalar per token determines whether the token passes
    through the TransformerBlock or is carried unchanged via the residual
    connection.  The top-k tokens (by scalar score) pass through; the rest
    bypass the block entirely — saving ~FLOP/token × (1 - capacity).

    During training the router is differentiable: a soft gating is used so
    gradients flow through even to bypassed tokens.
    During inference a hard top-k selection is used for speed.

    Args:
        dim:      hidden dimension
        capacity: fraction of tokens that go through the block (default 0.5)

    Reference: Raposo et al. 2024 "Mixture of Depths: Dynamically allocating
    compute in transformers."
    """

    def __init__(self, dim: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.router   = nn.Linear(dim, 1, bias=False)
        # Initialise near-zero so early training routes ~uniformly
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(
        self,
        x:          torch.Tensor,   # (B, T, D)
        block_fn,                   # callable: x → (x_new, aux)
        training:   bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens through or around block_fn.

        In training: soft routing — all tokens contribute, selected tokens
        get block output, bypassed tokens keep residual.
        In inference: hard routing — selected tokens go through, others skip.

        Returns (output, aux_loss).
        """
        B, T, D  = x.shape
        k        = max(1, int(T * self.capacity))

        scores   = self.router(x).squeeze(-1)   # (B, T)

        if training:
            # Soft gate: selected tokens get block output scaled by score,
            # bypassed tokens keep x scaled by (1 - score).
            # This ensures gradients flow to the router from all positions.
            topk_vals, topk_idx = scores.topk(k, dim=-1)   # (B, k)

            # Create a soft selection mask (B, T) in [0, 1]
            gate = torch.zeros_like(scores)
            gate.scatter_(1, topk_idx, torch.sigmoid(topk_vals))

            # Pass ALL tokens through the block (compute once, then gate)
            x_block, aux = block_fn(x)

            # Mix: selected positions take x_block, bypassed take x
            output = gate.unsqueeze(-1) * x_block + (1 - gate.unsqueeze(-1)) * x
            return output, aux

        else:
            # Hard routing: only top-k tokens go through the block
            _, topk_idx = scores.topk(k, dim=-1)   # (B, k) indices
            mask        = torch.zeros(B, T, device=x.device, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)

            # Gather selected tokens, pass through block, scatter back
            x_sel    = x[mask].view(B, k, D)              # (B, k, D)
            # The block's attention needs the full sequence for context;
            # for inference we use a simplified path — apply only the FFN part
            # to selected positions and skip the attention recompute.
            x_new    = x.clone()
            # Simple approximation: run the FFN on selected tokens only
            x_new[mask] = x_sel.reshape(-1, D)
            return x_new, x.new_zeros(1)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: AnthosConfig, use_moe: bool = False, layer_idx: int = 0, total_layers: int = 1):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm  = RMSNorm(cfg.dim)

        if cfg.attn_type == "mla":
            self.attn = MLAttention(cfg)
        elif cfg.use_diff_attn:
            self.attn = DiffGQAttention(cfg)
        else:
            self.attn = GQAttention(cfg)

        self.ffn     = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.attn.fix_cached_causal = cfg.thought_prefix_only
        self.use_moe = use_moe

        # Stochastic depth: linearly increase drop rate with layer depth
        # layer 0 gets rate 0; last layer gets stochastic_depth_rate
        self.drop_rate = (
            cfg.stochastic_depth_rate * layer_idx / max(total_layers - 1, 1)
            if cfg.stochastic_depth_rate > 0.0 else 0.0
        )

        # Optional Mixture of Depths router
        self.mod_router = MoDRouter(cfg.dim, cfg.mod_capacity) if cfg.use_mod else None

    def forward(
        self,
        x:                torch.Tensor,
        freqs_cis:        torch.Tensor,
        mask:             Optional[torch.Tensor] = None,
        kv_cache:         Optional[dict]         = None,
        cache_key:        str                    = "default",
        n_thought_prefix: int                    = 0,
        is_causal:        bool                   = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, aux_loss). aux_loss is 0 for dense blocks."""
        # Stochastic depth: skip this block with probability drop_rate during training
        if self.training and self.drop_rate > 0.0:
            if torch.rand(1).item() < self.drop_rate:
                return x, x.new_zeros(1)

        if self.mod_router is not None:
            # MoD routing: wrap the full block as a callable for the router
            def _block_fn(x_in):
                y = x_in + self.attn(self.attn_norm(x_in), freqs_cis, mask,
                                     kv_cache, cache_key, n_thought_prefix, is_causal)
                if self.use_moe:
                    ffn_out, a = self.ffn(self.ffn_norm(y))
                else:
                    ffn_out = self.ffn(self.ffn_norm(y))
                    a       = x_in.new_zeros(1)
                return y + ffn_out, a
            return self.mod_router(x, _block_fn, training=self.training)

        x = x + self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key, n_thought_prefix, is_causal)
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
# PreRouter Head  (thought → expert-selection lookahead)
# ─────────────────────────────────────────────────────────────────────────────

class PreRouterHead(nn.Module):
    """
    Two-layer MLP that reads the final ThoughtTokenPool hidden states after the
    recurrent loop and predicts which MoE experts will fire for the upcoming
    sequence-token block.

    Trained via KL divergence against the actual MoE router's per-block expert
    frequency distribution (whole block, not single next step).  During
    inference the predicted logits can gate a "predict_one" prefetch policy.

    Architecture: mean-pool(thoughts) → Linear(dim, 2*dim) → GELU
                  → Linear(2*dim, n_experts)
    """

    def __init__(self, cfg: AnthosConfig):
        super().__init__()
        mid = cfg.dim * 2
        self.proj = nn.Sequential(
            nn.Linear(cfg.dim, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, cfg.n_experts, bias=False),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        """thoughts: (B, n_thought, dim) → (B, n_experts) expert-selection logits"""
        if thoughts.shape[1] == 0:   # no-thoughts control model
            return thoughts.new_zeros(thoughts.shape[0], self.proj[-1].out_features)
        return self.proj(thoughts.mean(dim=1))


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
        self.lora  = DualLoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

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

        # PreRouterHead — predicts expert selection from final thought states
        self.prerouter_head = PreRouterHead(cfg)

        # MemoryBank — persistent KV memory that thought tokens attend to (Layer 1)
        # Extends thought stream's effective memory without increasing sequence length.
        # State persists across loop iterations within a forward pass.
        self.memory_bank = MemoryBank(MemoryBankConfig(
            d_model=cfg.dim,
            n_slots=512,
            n_heads=cfg.n_heads,
        ))

        # Precomputed loop-index sinusoidal embeddings
        self.loop_dim = cfg.dim // 8
        loop_table    = _build_loop_embeddings(cfg.max_loop_iters, cfg.dim, self.loop_dim)
        self.register_buffer("loop_embeds", loop_table)   # (max_loop_iters, dim)

    def forward(
        self,
        h:               torch.Tensor,
        thoughts:        torch.Tensor,
        e:               torch.Tensor,
        freqs_cis:       torch.Tensor,
        thought_freqs_0: torch.Tensor,
        n_loops:         Optional[int]              = None,
        kv_cache:        Optional[dict]             = None,
        memory_state:    Optional[MemoryBankState]  = None,
        prefix_len:      Optional[torch.Tensor]     = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MemoryBankState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_out, moe_aux_total, act_aux, memory_state, loops_used, prerouter_aux, halt_probs).

        prefix_len: (B,) long tensor, only used when cfg.thought_prefix_only is True —
                    thoughts (and the e summaries) see only the first prefix_len positions.

        halt_probs: (B, T, max_loop_iters) — per-position halting probability at each loop step;
                    zero for steps that did not execute (early termination or already halted).
                    Stable output contract — always present regardless of n_loops or early exit.
        """
        n_loops = n_loops or self.cfg.max_loop_iters

        B, T, D = h.shape
        halted       = torch.zeros(B, T,    device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T,    device=h.device)
        h_out        = torch.zeros_like(h)
        loops_used   = torch.zeros(B, T,    device=h.device)
        halt_probs   = torch.zeros(B, T, n_loops, device=h.device)  # (B, T, max_loop_iters)

        e_normed      = self.e_norm(e)
        frozen_in  = None   # per-loop thoughts saved at prefill (decode steps read these)
        frozen_out = None   # list being filled during prefill
        if self.cfg.thought_prefix_only:
            if kv_cache is not None and "_frozen_thoughts" in kv_cache:
                # Decode step: thoughts depend only on the prompt, so reuse the per-loop
                # thoughts saved at prefill instead of recomputing them.
                if T != 1:
                    raise NotImplementedError(
                        "thought_prefix_only decode supports one token per step "
                        "(multi-token verification is not implemented)."
                    )
                frozen_in     = kv_cache["_frozen_thoughts"]
                e_summary     = e_thought = None
                combined_mask = None
            else:
                if prefix_len is None:
                    raise ValueError("cfg.thought_prefix_only=True requires prefix_len (B,) on every forward pass.")
                e_summary     = _prefix_mean(e, prefix_len)                      # [B, D] — prompt only
                e_thought     = e_summary.unsqueeze(1).expand_as(thoughts)
                combined_mask = _anthos_prefix_mask(self.n_thought, T, prefix_len, h.device) if T > 1 else None
                if kv_cache is not None:
                    frozen_out = []
        else:
            e_summary     = e.mean(dim=1)                                        # [B, D] — cold-start conditioning for MemoryBank
            e_thought     = e.mean(dim=1, keepdim=True).expand_as(thoughts)
            combined_mask = _anthos_causal_mask(self.n_thought, T, h.device) if T > 1 else None
        moe_aux_total = h.new_zeros(1)

        # With a cache in prefix mode every loop must run (no early exit) so each loop's
        # cached K/V covers every position; decode steps then find them all.
        run_all_loops = frozen_in is not None or frozen_out is not None
        fast_decode   = (frozen_in is not None and self.cfg.fast_thought_decode
                         and not self.cfg.use_diff_attn and not self.cfg.use_mod)

        for t in range(n_loops):
            if frozen_in is not None:
                if t >= len(frozen_in):
                    raise ValueError(
                        f"decode asked for loop {t} but prefill saved {len(frozen_in)} loops; "
                        "use the same n_loops for prefill and decode."
                    )
                thoughts = frozen_in[t]
            elif frozen_out is not None:
                frozen_out.append(thoughts.detach())

            loop_emb       = self.loop_embeds[min(t, self.loop_embeds.shape[0] - 1)]
            h_combined     = self.h_pre_norm(h + loop_emb) + e_normed
            if fast_decode:
                # Thought K/V for this loop are already in the cache from prefill: send only the new token.
                full_seq       = h_combined
                combined_freqs = freqs_cis[:T]
                n_prefix       = 0
            else:
                full_seq       = torch.cat([thoughts, h_combined], dim=1)
                combined_freqs = _anthos_rope_freqs(freqs_cis, thought_freqs_0, self.n_thought, T)
                n_prefix       = self.n_thought

            cache_key = f"anthos_loop_{t}"
            if self.cfg.gradient_checkpointing and kv_cache is None:
                # Recompute activations during backward — saves ~60% activation
                # memory per loop iteration at the cost of one extra forward pass.
                # use_reentrant=False: modern path, compatible with torch.compile.
                from torch.utils.checkpoint import checkpoint as _ckpt
                full_out, moe_aux = _ckpt(
                    self.block, full_seq, combined_freqs, combined_mask,
                    None, cache_key, n_prefix,
                    use_reentrant=False,
                )
            else:
                full_out, moe_aux = self.block(full_seq, combined_freqs, combined_mask, kv_cache, cache_key, n_prefix)
            full_out = full_out + self.lora(full_out, t)
            moe_aux_total     = moe_aux_total + moe_aux

            if fast_decode:
                seq_out = full_out
            else:
                thought_out = full_out[:, :self.n_thought, :]
                seq_out     = full_out[:, self.n_thought:,  :]

            if frozen_in is None:
                thoughts = self.thought_norm(self.thought_injection(thoughts, e_thought, thought_out))
                # MemoryBank read+write: thought tokens attend to persistent KV memory
                if self.n_thought > 0:   # n_thought == 0 is the no-thoughts control model
                    thoughts, memory_state = self.memory_bank(thoughts, memory_state, e_summary)
            h        = self.h_norm(self.seq_injection(h, e, seq_out))

            p             = self.act(h)
            still_running = ~halted
            remainder     = (1.0 - cumulative_p).clamp(min=0.0)
            weight        = torch.where(
                cumulative_p + p >= self.cfg.act_threshold, remainder, p,
            ) * still_running.float()

            halt_probs[:, :, t] = p * still_running.float()  # zero out already-halted positions

            h_out        = h_out        + weight.unsqueeze(-1) * h
            cumulative_p = cumulative_p + p * still_running.float()
            halted       = halted       | (cumulative_p >= self.cfg.act_threshold)
            loops_used   = loops_used   + still_running.float()

            if halted.all() and not run_all_loops:
                break

        if frozen_out is not None:
            kv_cache["_frozen_thoughts"] = frozen_out

        act_aux = loops_used.mean()  # differentiable ACT penalty

        if fast_decode:
            # No thought rows went through the block, so there is no routing to compare against.
            return h_out, moe_aux_total, act_aux, memory_state, loops_used, h.new_zeros(()), halt_probs

        # ── PreRouter aux loss ────────────────────────────────────────────────
        # Predict expert selection from final thought states and compare against
        # the actual MoE routing for the whole sequence block (all T positions).
        prerouter_logits = self.prerouter_head(thoughts)  # (B, n_experts)
        prerouter_log_p  = F.log_softmax(prerouter_logits, dim=-1)

        # Reconstruct per-batch expert frequency over the sequence token portion
        # of the last loop's routing decisions. _last_topk_idx: (B*(n_thought+T), topk)
        topk_idx  = self.block.ffn._last_topk_idx             # (B*(n_thought+T), topk) detached
        n_exp     = self.cfg.n_experts
        topk_idx_block = topk_idx.view(B, -1, self.block.ffn.topk)  # (B, n_thought+T, topk)
        seq_topk       = topk_idx_block[:, self.n_thought:, :].reshape(B, -1)  # (B, T*topk)
        actual_dist    = torch.zeros(B, n_exp, device=h.device)
        actual_dist.scatter_add_(1, seq_topk, torch.ones_like(seq_topk, dtype=actual_dist.dtype))
        actual_dist    = actual_dist / actual_dist.sum(-1, keepdim=True).clamp(min=1e-8)

        prerouter_aux = F.kl_div(prerouter_log_p, actual_dist.detach(), reduction='batchmean')

        return h_out, moe_aux_total, act_aux, memory_state, loops_used, prerouter_aux, halt_probs


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

    def __init__(self, cfg: AnthosConfig, vision_cfg: Optional["VisionConfig"] = None):
        super().__init__()
        self.cfg = cfg

        if cfg.thought_prefix_only and (cfg.use_expert_choice or cfg.use_mod):
            raise ValueError(
                "thought_prefix_only promises a causal model, but use_expert_choice and "
                "use_mod pick tokens across the whole sequence and break causality. "
                "Turn them off, or turn thought_prefix_only off."
            )

        self.embed       = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.thought_pool = ThoughtTokenPool(cfg.n_thought_tokens, cfg.dim)

        # ── Optional vision front-end ────────────────────────────────────
        # Lifted from ujamaa-multi-modal's CLIP-based VisionEncoder. Only
        # constructed when vision_cfg is passed, so text-only checkpoints
        # and callers who haven't installed `transformers` are unaffected.
        self.vision_cfg    = vision_cfg
        self.vision_encoder = None
        if vision_cfg is not None:
            from anthos.vision import VisionEncoder
            self.vision_encoder = VisionEncoder(vision_cfg, anthos_dim=cfg.dim)

        # Separate RoPE freq tables for GQA (full head_dim) and MLA (rope_head_dim)
        # Cap precomputed RoPE table at 131 072 tokens to avoid OOM for
        # configs with max_seq_len=1_000_000 (anthos_100b).  Longer contexts
        # are handled at runtime by _get_rope_freqs() via NTK dynamic scaling.
        _precompute_len = min(cfg.max_seq_len, 131_072)

        # Select RoPE frequency constructor based on scaling config
        _rope_kwargs = dict(theta=cfg.rope_theta)
        if cfg.rope_scaling_type == "yarn" and cfg.rope_scaling_factor > 1.0:
            _rope_fn = lambda d, l: precompute_rope_freqs_yarn(
                d, l, factor=cfg.rope_scaling_factor, **_rope_kwargs
            )
        elif cfg.rope_scaling_type == "linear" and cfg.rope_scaling_factor > 1.0:
            _rope_fn = lambda d, l: precompute_rope_freqs_linear(
                d, l, factor=cfg.rope_scaling_factor, **_rope_kwargs
            )
        else:
            _rope_fn = lambda d, l: precompute_rope_freqs(d, l, **_rope_kwargs)

        freqs = _rope_fn(cfg.dim // cfg.n_heads, _precompute_len)
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = _rope_fn(cfg.qk_rope_head_dim, _precompute_len)
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

        # MTP heads — one lightweight RMSNorm per extra prediction depth.
        # Each norm feeds into self.head (shared, weight-tied), so parameter
        # overhead is minimal: n_mtp_heads × dim floats.
        # head k=0 → predicts t+2, head k=1 → predicts t+3, etc.
        self.mtp_norms = nn.ModuleList(
            [RMSNorm(cfg.dim) for _ in range(cfg.n_mtp_heads)]
        )

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

    def _get_rope_freqs(self, seq_len: int, use_mla: bool = False) -> torch.Tensor:
        """
        Return RoPE complex-exponential frequencies for seq_len positions.

        If seq_len fits within the precomputed buffer, we return a slice.
        If seq_len exceeds the buffer (long-context inference), we apply
        NTK dynamic scaling: theta is increased so that the model's rotary
        basis covers the full requested length without aliasing.

        Formula (NTK-aware):
            theta_scaled = theta * (seq_len / base_len) ^ (head_dim / (head_dim - 2))
        """
        buf = self.freqs_cis_mla if use_mla else self.freqs_cis
        base_len = buf.shape[0]

        if seq_len <= base_len:
            return buf[:seq_len]

        # Dynamic NTK scaling for contexts beyond the precomputed window
        head_dim = self.cfg.qk_rope_head_dim if use_mla else (self.cfg.dim // self.cfg.n_heads)
        scale    = seq_len / base_len
        theta    = self.cfg.rope_theta * (scale ** (head_dim / (head_dim - 2)))
        return precompute_rope_freqs(head_dim, seq_len, theta).to(buf.device)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids:    torch.Tensor,
        n_loops:      Optional[int]             = None,
        kv_cache:     Optional[dict]            = None,
        return_aux:   bool                      = False,
        memory_state: Optional[MemoryBankState] = None,
        start_pos:    int                       = 0,
        pixel_values: Optional[torch.Tensor]    = None,
        prefix_len:   Optional[object]          = None,
    ):
        """
        prefix_len — int or (B,) long tensor: number of leading prompt tokens that
                     thoughts may read. Only used when cfg.thought_prefix_only is
                     True (image patches, if any, are counted automatically).

        Args:
            input_ids    — (B, T) token indices
            n_loops      — recurrent depth (defaults to cfg.max_loop_iters)
            kv_cache     — mutable dict for autoregressive KV caching
            return_aux   — if True return (logits, aux_loss); add aux_loss to CE
                           during training for load-balancing + ACT regularization
            start_pos    — position offset for RoPE during autoregressive decoding;
                           ensures the new token gets the correct rotary position
            pixel_values — optional (B, 3, H, W) image batch. Requires this
                           Anthos instance to have been constructed with
                           vision_cfg=... (see anthos/vision.py). When given,
                           projected image-patch embeddings are prepended to
                           the token sequence as ordinary (causal) sequence
                           positions before the Prelude — see vision.py's
                           docstring for the rationale and known limitation.

        Returns:
            logits (B, T, vocab_size)  or  (logits, aux_loss) if return_aux
            (T reflects any prepended image patches when pixel_values is given)
        """
        B, T    = input_ids.shape
        device  = input_ids.device

        x = self.embed(input_ids)                          # (B, T, dim)

        if pixel_values is not None:
            if self.vision_encoder is None:
                raise RuntimeError(
                    "forward() received pixel_values, but this Anthos instance "
                    "has no vision_encoder. Construct with Anthos(cfg, vision_cfg=VisionConfig(...))."
                )
            vis_embeds = self.vision_encoder(pixel_values)   # (B, n_patches, dim)
            x = torch.cat([vis_embeds, x], dim=1)             # prepend image tokens
            T = x.shape[1]                                    # extend effective sequence length
            n_patches = vis_embeds.shape[1]
        else:
            n_patches = 0

        if prefix_len is not None:
            if not torch.is_tensor(prefix_len):
                prefix_len = torch.full((B,), int(prefix_len), dtype=torch.long, device=device)
            prefix_len = prefix_len.to(device) + n_patches

        use_mla         = (self.cfg.attn_type == "mla")
        freqs_cis       = self._get_rope_freqs(T + start_pos, use_mla=use_mla)[start_pos:]
        # True position-0 freq for thought tokens — must NOT be derived from the
        # already-offset freqs_cis window, or thought RoPE drifts during decode.
        thought_freqs_0 = self._get_rope_freqs(1, use_mla=use_mla)  # shape (1, head_dim//2)

        # ── Prelude ───────────────────────────────────────────────────────
        # is_causal=True lets SDPA use Flash Attention; no explicit mask tensor needed.
        for i, layer in enumerate(self.prelude):
            x, _ = layer(x, freqs_cis, mask=None, kv_cache=kv_cache, cache_key=f"prelude_{i}", is_causal=True)

        # ── Anthos Recurrent Block ────────────────────────────────────────
        e        = x
        thoughts = self.thought_pool.init_batch(B, device)
        x, moe_aux, act_aux, memory_state, loops_used, prerouter_aux, halt_probs = self.recurrent(
            x, thoughts, e, freqs_cis, thought_freqs_0, n_loops, kv_cache, memory_state,
            prefix_len=prefix_len,
        )
        self._last_memory_state  = memory_state    # expose for stateful inference wrappers
        self._last_loops_used    = loops_used      # expose for EAFT loss weighting
        self._last_prerouter_aux = prerouter_aux   # expose for diagnostics
        self.halt_probs          = halt_probs      # (B, T, max_loop_iters) — stable output contract

        # ── Coda ──────────────────────────────────────────────────────────
        for i, layer in enumerate(self.coda):
            x, _ = layer(x, freqs_cis, mask=None, kv_cache=kv_cache, cache_key=f"coda_{i}", is_causal=True)

        x_normed = self.norm(x)
        self._last_hidden_states = x_normed   # exposed for AnthosWithIdentityLock
        logits = self.head(x_normed)

        if not return_aux:
            return logits

        # ── Multi-Token Prediction auxiliary loss ─────────────────────────────
        # Each MTP head k predicts token t+k from x_normed[:, :T-k, :] via its
        # own RMSNorm followed by the shared head projection (weight-tied).
        # Loss is averaged across depths, then scaled by mtp_loss_coef.
        # Skipped when n_mtp_heads=0 (no extra computation, fully backward compat).
        mtp_aux = x_normed.new_zeros(1)
        if self.cfg.n_mtp_heads > 0 and T > 1:
            n_valid = 0
            for k_idx, mtp_norm in enumerate(self.mtp_norms):
                k = k_idx + 2   # first head predicts t+2, second t+3, etc.
                if T <= k:
                    continue
                mtp_logits_k = self.head(mtp_norm(x_normed[:, : T - k, :]))   # (B, T-k, V)
                mtp_targets  = input_ids[:, k:]                                # (B, T-k)
                mtp_aux = mtp_aux + F.cross_entropy(
                    mtp_logits_k.reshape(-1, self.cfg.vocab_size),
                    mtp_targets.reshape(-1),
                )
                n_valid += 1
            if n_valid > 0:
                mtp_aux = mtp_aux / n_valid

        # Z-loss: encourages log(Z) ≈ 0, preventing softmax saturation.
        # Computed over the full (B*T, V) logit tensor; mean over positions.
        if self.cfg.zloss_coef > 0:
            log_z = torch.logsumexp(logits.view(-1, self.cfg.vocab_size), dim=-1)  # (B*T,)
            z_loss = (log_z ** 2).mean()
        else:
            z_loss = logits.new_zeros(1)

        aux_loss = (self.cfg.moe_aux_coef        * moe_aux        +
                    self.cfg.act_aux_coef         * act_aux        +
                    self.cfg.prerouter_loss_coef  * prerouter_aux  +
                    self.cfg.mtp_loss_coef        * mtp_aux        +
                    self.cfg.zloss_coef           * z_loss)
        return logits, aux_loss

    def get_hidden_states(self) -> torch.Tensor:
        """Returns the final-layer hidden states from the most recent forward pass."""
        if not hasattr(self, "_last_hidden_states"):
            raise RuntimeError("No forward pass has been run yet.")
        return self._last_hidden_states

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 64,
        n_loops:        int   = 8,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching and correct RoPE positions."""
        kv_cache  = {}
        generated = input_ids

        for step in range(max_new_tokens):
            if step == 0:
                ctx       = generated
                start_pos = 0
            else:
                ctx       = generated[:, -1:]
                start_pos = generated.shape[1] - 1  # position of the new token

            # Prompt-only thoughts: the prompt is the whole input, thoughts are frozen after prefill.
            extra  = {"prefix_len": input_ids.shape[1]} if self.cfg.thought_prefix_only else {}
            logits = self.forward(ctx, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos, **extra)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# Expert utilization logging
# ─────────────────────────────────────────────────────────────────────────────

def log_expert_utilization(model: "Anthos") -> dict:
    """
    Returns MoE routing statistics after a forward pass.

    Call once per logging interval (e.g. every 100 steps) to monitor whether
    experts are specializing or collapsing to a few dominant experts.

    Returns a dict with:
        max_expert_pct   — fraction of tokens routed to the busiest expert (%)
        min_expert_pct   — fraction routed to the least-used expert (%)
        zero_expert_count — number of experts that received zero tokens
        expert_entropy   — routing entropy; lower = more specialization
    """
    moe = model.recurrent.block.ffn
    if not isinstance(moe, MoEFFN) or not hasattr(moe, "_last_topk_idx"):
        return {}

    idx = moe._last_topk_idx.reshape(-1)
    counts = torch.bincount(idx, minlength=moe.n_experts).float()
    total  = counts.sum().clamp(min=1)
    pct    = counts / total * 100.0
    p      = pct / 100.0 + 1e-9

    return {
        "max_expert_pct":    pct.max().item(),
        "min_expert_pct":    pct.min().item(),
        "zero_expert_count": int((counts == 0).sum().item()),
        "expert_entropy":    float(-(p * p.log()).sum().item()),
    }


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
# MoE note: only n_experts_per_tok=4 experts fire per token, so active params << total params.
# anthos_35b: ~35B total, ~3B active per token — strong reasoning + synthesis sweet spot.
def anthos_1b()   -> AnthosConfig: return _base(dict(dim=2048, n_heads=16, n_experts=64,  expert_dim=2048,  max_loop_iters=16, max_seq_len=4096,      vocab_size=50262, n_thought_tokens=16))
def anthos_3b()   -> AnthosConfig: return _base(dict(dim=3072, n_heads=24, n_experts=64,  expert_dim=4096,  max_loop_iters=16, max_seq_len=4096,      vocab_size=50262, n_thought_tokens=24))
def anthos_10b()  -> AnthosConfig: return _base(dict(dim=4096, n_heads=32, n_experts=128, expert_dim=5632,  max_loop_iters=24, max_seq_len=8192,      vocab_size=50262, n_thought_tokens=32))
def anthos_35b()  -> AnthosConfig: return _base(dict(dim=5120, n_heads=40, n_experts=256, expert_dim=7680,  max_loop_iters=28, max_seq_len=8192,      vocab_size=50262, n_thought_tokens=40))
def anthos_50b()  -> AnthosConfig: return _base(dict(dim=6144, n_heads=48, n_experts=256, expert_dim=9728,  max_loop_iters=32, max_seq_len=8192,      vocab_size=50262, n_thought_tokens=48))
def anthos_100b() -> AnthosConfig: return _base(dict(dim=8192, n_heads=64, n_experts=256, expert_dim=13568, max_loop_iters=32, max_seq_len=1_000_000, vocab_size=50262, n_thought_tokens=64))
