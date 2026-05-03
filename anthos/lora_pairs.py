"""
anthos/lora_pairs.py — Dual LoRA Adapter (High/Low Loop Paths)

Replaces the single LoRAAdapter in anthos/main.py AnthosRecurrentBlock.

Motivation (from generate_video repo's dual-LoRA pattern):
  A single LoRA adapter with loop-index conditioning treats all loop depths
  with the same weight structure.  Early loops (fast content routing) and late
  loops (deep reasoning integration) have different computational roles.

  DualLoRAAdapter runs two independent LoRA paths and blends them by loop
  depth.  The blend is continuous and differentiable:

      alpha  = sigmoid(loop_depth_score)   in (0, 1)
      output = alpha * deep_lora(x, t)  +  (1 - alpha) * fast_lora(x, t)

  where loop_depth_score = (t / max_loops) scaled by a learned temperature.

  At t=0:  output ≈ fast_lora(x, 0)   (speed path dominates)
  At t=max: output ≈ deep_lora(x, max) (reasoning path dominates)

  This means:
    fast_lora learns: "what small correction helps content routing?"
    deep_lora learns: "what correction integrates thought-stream reasoning?"

  The two paths specialise naturally without any auxiliary loss.

Drop-in replacement for LoRAAdapter:
  DualLoRAAdapter has the same __init__ signature and forward signature.
  Swap it in main.py:

      # BEFORE (in AnthosRecurrentBlock.__init__):
      self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

      # AFTER:
      from anthos.lora_pairs import DualLoRAAdapter
      self.lora = DualLoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

  _init_weights in Anthos also initialises lora.B and lora.scale.weight.
  DualLoRAAdapter exposes these on both paths via properties, keeping the
  init_weights call in main.py compatible without changes.

Parameter count:
  LoRAAdapter:     ~2 * dim * rank + rank * dim = 2 * dim * rank
  DualLoRAAdapter: ~2 * (2 * dim * rank + rank * dim) = 4 * dim * rank
  Overhead:        ~2x LoRA params, still negligible vs. main model
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Single LoRA path (same logic as original LoRAAdapter)
# ─────────────────────────────────────────────────────────────────────────────

class _LoRAPath(nn.Module):
    """
    Single depth-wise LoRA with precomputed loop-index sinusoidal scaling.

    Forward: delta = scale(loop_emb) * B(act(A(x)))
    where A: dim→rank, B: rank→dim, scale: loop_emb→scalar
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.A     = nn.Linear(dim,  rank, bias=False)
        self.B     = nn.Linear(rank, dim,  bias=False)
        self.scale = nn.Linear(dim,  1,    bias=False)

        # Initialise: A normal, B zeros (so delta=0 at init → stable start)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        nn.init.ones_(self.scale.weight)

        # Precomputed sinusoidal loop embeddings
        self.register_buffer(
            "loop_embs", _build_loop_embs(max_loops, dim)
        )   # (max_loops, dim)

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """
        Args:
            x:        (B, T, dim) — full sequence including thought tokens
            loop_idx: current loop iteration index

        Returns:
            delta: (B, T, dim) — additive correction
        """
        loop_emb = self.loop_embs[loop_idx]                # (dim,)
        scale    = torch.sigmoid(self.scale(loop_emb))     # scalar
        return scale * self.B(F.silu(self.A(x)))


def _build_loop_embs(max_loops: int, dim: int) -> torch.Tensor:
    """Sinusoidal embeddings for loop index, same construction as main.py."""
    loop_dim = dim // 8
    pos      = torch.arange(max_loops).unsqueeze(1).float()
    div      = torch.exp(
        torch.arange(0, loop_dim, 2).float() * -(math.log(10000.0) / loop_dim)
    )
    emb        = torch.zeros(max_loops, dim)
    emb[:, :loop_dim:2]  = torch.sin(pos * div)
    emb[:, 1:loop_dim:2] = torch.cos(pos * div)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Dual LoRA adapter
# ─────────────────────────────────────────────────────────────────────────────

class DualLoRAAdapter(nn.Module):
    """
    Two LoRA paths blended by loop depth.

    fast_lora:  dominates at early loop iterations (content routing)
    deep_lora:  dominates at late loop iterations  (reasoning integration)

    Blend: alpha = sigmoid(temperature * (t / max_loops - 0.5))
           output = alpha * deep(x, t) + (1 - alpha) * fast(x, t)

    temperature is a learned scalar — the model can sharpen or soften the
    transition.  Initialised to 4.0 (moderate separation).

    Properties `.B` and `.scale` are forwarded to `deep_lora` so that
    Anthos._init_weights() in main.py works unchanged.
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.max_loops = max_loops

        self.fast_lora = _LoRAPath(dim, rank, max_loops)
        self.deep_lora = _LoRAPath(dim, rank, max_loops)

        # Learned blend temperature — higher = sharper fast/deep transition
        self.temperature = nn.Parameter(torch.tensor(4.0))

    # ── Compatibility properties for main.py _init_weights ───────────────────

    @property
    def B(self) -> torch.Tensor:
        """Expose deep_lora.B.weight for main.py init_weights compatibility."""
        return self.deep_lora.B.weight

    @property
    def scale(self) -> nn.Module:
        """Expose deep_lora.scale for main.py init_weights compatibility."""
        return self.deep_lora.scale

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """
        Args:
            x:        (B, T, dim)
            loop_idx: current loop iteration (0-indexed)

        Returns:
            delta: (B, T, dim) blended from fast and deep paths
        """
        # Blend coefficient — sigmoid pushes to 0 at loop 0, 1 at max_loops
        depth = loop_idx / max(self.max_loops - 1, 1)           # [0, 1]
        alpha = torch.sigmoid(self.temperature * (depth - 0.5)) # scalar

        fast_delta = self.fast_lora(x, loop_idx)   # (B, T, dim)
        deep_delta = self.deep_lora(x, loop_idx)   # (B, T, dim)

        return alpha * deep_delta + (1.0 - alpha) * fast_delta

    def extra_repr(self) -> str:
        return (f"max_loops={self.max_loops}, "
                f"temperature={self.temperature.item():.2f}")
