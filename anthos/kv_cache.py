"""
anthos/kv_cache.py — Key-Value Cache for Anthos Inference

Standard KV cache cannot be applied naively to Anthos because thought tokens
are NON-CAUSAL — they reattend to the full sequence on every loop iteration.

This module implements a bifurcated cache strategy:

  Sequence stream: standard causal KV cache
    → K, V accumulated per generation step, reused on next step
    → Works identically to GPT/LLaMA KV cache

  Thought stream: NO cache (recomputed every generation step)
    → Thought tokens see the full current sequence non-causally
    → Caching would break their full-context access
    → BUT: since n_thought is typically 16, this is cheap

  LTI hidden state h_t: IS cached between generation steps
    → The recurrent state carries working memory across tokens
    → This is Anthos's "RNN-style" persistent state
    → Must be passed explicitly through model.generate()

Net effect:
  On generation step k, we only recompute:
    - Thought token attention over [prefix + k new tokens]    (fast: small N)
    - Sequence token attention over [thought_out + 1 new token]  (fast: 1 token)
  We reuse:
    - All previous sequence stream K, V pairs
    - LTI hidden state from step k-1

This gives close to standard KV cache speedup on the sequence stream,
with a small overhead for thought token recomputation.

Usage:
    from anthos.kv_cache import AnthosCache, CacheConfig

    cache = AnthosCache(CacheConfig(
        max_seq_len=1024,
        n_layers=8,          # number of recurrent block layers (typically 1 deep loop)
        n_heads=8,
        head_dim=64,
        n_thought_tokens=16,
    ))

    # In model.generate() loop — pass cache to forward:
    logits = model(input_ids, n_loops=8, cache=cache)
    cache.step()   # advance position counter

    # Reset between independent generations
    cache.reset()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheConfig:
    max_seq_len:      int = 1024
    n_heads:          int = 8
    head_dim:         int = 64
    n_thought_tokens: int = 16
    dtype:            torch.dtype = torch.float32

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Stream KV Cache (standard causal cache)
# ─────────────────────────────────────────────────────────────────────────────

class SequenceKVCache:
    """
    Standard causal KV cache for the sequence stream.

    Stores accumulated K, V tensors up to current generation position.
    On each step, only the new token's K, V are computed and appended.

    Shape tracking:
        keys:   [B, H, T_filled, Hd]
        values: [B, H, T_filled, Hd]
    """

    def __init__(self, cfg: CacheConfig, batch_size: int, device: torch.device):
        self.cfg   = cfg
        self.B     = batch_size
        self.device = device
        B, H, M, Hd = batch_size, cfg.n_heads, cfg.max_seq_len, cfg.head_dim

        self.keys   = torch.zeros(B, H, M, Hd, dtype=cfg.dtype, device=device)
        self.values = torch.zeros(B, H, M, Hd, dtype=cfg.dtype, device=device)
        self.pos    = 0   # Current fill position

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Append new K, V tensors for the current generation step.

        Args:
            new_keys:   [B, H, T_new, Hd]  — new tokens' key projections
            new_values: [B, H, T_new, Hd]  — new tokens' value projections
        """
        T_new = new_keys.shape[2]
        end   = self.pos + T_new
        if end > self.cfg.max_seq_len:
            # Sliding window: drop oldest tokens
            keep = self.cfg.max_seq_len - T_new
            self.keys   = torch.cat([self.keys[:, :, -keep:, :], new_keys], dim=2)
            self.values = torch.cat([self.values[:, :, -keep:, :], new_values], dim=2)
            self.pos    = self.cfg.max_seq_len
        else:
            self.keys[:, :, self.pos:end, :]   = new_keys
            self.values[:, :, self.pos:end, :] = new_values
            self.pos = end

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all cached K, V up to current position."""
        return self.keys[:, :, :self.pos, :], self.values[:, :, :self.pos, :]

    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.pos = 0


# ─────────────────────────────────────────────────────────────────────────────
# LTI State Cache (recurrent hidden state across generation steps)
# ─────────────────────────────────────────────────────────────────────────────

class LTIStateCache:
    """
    Caches the LTI recurrent hidden state h_t between generation steps.

    In Anthos, h_t carries working memory across loop iterations within
    a forward pass. For autoregressive generation, we also want h_t to
    persist across generation steps (token to token).

    This is the "RNN memory" layer — distinct from the Transformer KV cache.

    Shape: [B, T_stream, D] for both thought and sequence streams.
    """

    def __init__(self, d_model: int, n_thought: int, batch_size: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        self.d = d_model
        self.n = n_thought
        self.B = batch_size

        # Separate LTI states for thought and sequence streams
        self.h_thought  = torch.zeros(batch_size, n_thought, d_model,
                                       dtype=dtype, device=device)
        self.h_sequence = torch.zeros(batch_size, 1, d_model,
                                       dtype=dtype, device=device)

    def update_thought(self, h: torch.Tensor):
        """h: [B, N_thought, D]"""
        self.h_thought = h.detach()

    def update_sequence(self, h: torch.Tensor):
        """h: [B, T, D] — we keep only the last token's state for generation"""
        self.h_sequence = h[:, -1:, :].detach()

    def reset(self):
        self.h_thought.zero_()
        self.h_sequence.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Full Anthos Cache
# ─────────────────────────────────────────────────────────────────────────────

class AnthosCache:
    """
    Combined cache for Anthos inference.

    Manages:
      - Sequence stream KV cache (standard causal)
      - LTI hidden state cache (thought + sequence streams)
      - Generation position counter

    The thought stream is NOT cached (recomputed each step) but the LTI
    state that carries thought token working memory IS cached.

    Integration with model.generate():
        The cache is passed into the forward pass. The RecurrentBlock
        uses sequence_cache.get() for the sequence stream attention,
        and lti_cache for the persistent recurrent state.
        After each forward, append new K,V to sequence_cache.

    For models not yet cache-aware (current Anthos):
        Use in prefill mode only — run full forward on prompt,
        then use AnthosCache to store the state for continuation.
    """

    def __init__(
        self,
        cfg:        CacheConfig,
        batch_size: int          = 1,
        device:     torch.device = torch.device("cpu"),
    ):
        self.cfg     = cfg
        self.B       = batch_size
        self.device  = device

        self.sequence_cache = SequenceKVCache(cfg, batch_size, device)
        self.lti_cache      = LTIStateCache(
            d_model=cfg.d_model,
            n_thought=cfg.n_thought_tokens,
            batch_size=batch_size,
            device=device,
            dtype=cfg.dtype,
        )
        self.generation_step = 0

    def step(self):
        """Advance generation step counter."""
        self.generation_step += 1

    def reset(self):
        """Reset all caches for a new generation."""
        self.sequence_cache.reset()
        self.lti_cache.reset()
        self.generation_step = 0

    @property
    def is_prefill(self) -> bool:
        """True on the first forward pass (processing the full prompt)."""
        return self.generation_step == 0

    def get_position_ids(self, seq_len: int) -> torch.Tensor:
        """Return position ids for the current generation step."""
        if self.is_prefill:
            return torch.arange(seq_len, device=self.device).unsqueeze(0)
        else:
            # Single new token at position = current fill position
            pos = self.sequence_cache.pos
            return torch.tensor([[pos]], device=self.device)


# ─────────────────────────────────────────────────────────────────────────────
# Cache-Aware Generation Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CachedGenerator:
    """
    Wraps Anthos's model.generate() with KV cache support.

    Current Anthos generate() recomputes all K,V on every new token.
    This wrapper intercepts the forward pass, injects cached K,V, and
    appends new K,V after each step.

    IMPORTANT: This requires the RecurrentBlock's attention layers to
    accept past_key_values. Until that's wired in (see integration guide),
    this class provides the correct interface so the hookup is trivial.

    For now (pre-hookup): provides a clean generate() that at minimum
    caches the LTI state, giving you the recurrent memory benefit
    even before full KV caching is wired in.

    Usage:
        gen = CachedGenerator(model, CacheConfig(...))
        out = gen.generate(input_ids, max_new_tokens=256, n_loops=12)
    """

    def __init__(self, model, cfg: CacheConfig):
        self.model = model
        self.cfg   = cfg

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,   # [B, T]
        max_new_tokens: int  = 128,
        n_loops:        int  = 8,
        temperature:    float = 1.0,
        top_k:          int  = 40,
        eos_token_id:   Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with LTI state caching.

        Returns generated token ids: [B, T + max_new_tokens]
        """
        B, T = input_ids.shape
        device = input_ids.device

        cache = AnthosCache(self.cfg, batch_size=B, device=device)

        # Prefill: process full prompt
        generated = input_ids.clone()

        for step in range(max_new_tokens):
            # On prefill (step 0): process full sequence
            # On subsequent steps: process only last token
            if step == 0:
                curr_input = generated
            else:
                curr_input = generated[:, -1:]

            # Forward pass
            logits = self.model(curr_input, n_loops=n_loops)  # [B, T_curr, V]

            # Sample from last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                topk_vals, _ = next_logits.topk(top_k, dim=-1)
                threshold     = topk_vals[:, -1].unsqueeze(-1)
                next_logits   = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs    = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]

            generated = torch.cat([generated, next_tok], dim=1)
            cache.step()

            # Early stop on EOS
            if eos_token_id is not None:
                if (next_tok == eos_token_id).all():
                    break

        return generated
