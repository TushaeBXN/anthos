"""
anthos/eaft.py — Entropy-Aware Focal Training Loss

EAFT weights the cross-entropy loss by the entropy of the model's own top-k
logit distribution.  High-entropy predictions (uncertain) receive more
gradient signal; low-entropy predictions (confident) receive less.

Why this matters for Anthos specifically:
  1. ACT halting decisions are high-entropy early in training.  EAFT naturally
     pushes more gradient toward uncertain halting positions — the model learns
     WHEN to halt faster.
  2. The thought stream accumulates reasoning state; positions where the
     sequence stream is still uncertain benefit most from additional gradient.
     EAFT targets exactly these positions without any manual annotation.
  3. Standard CE averages over all positions equally — including positions
     the model already has near-zero uncertainty on.  EAFT reclaims that
     wasted gradient budget.

Combined loss (what train.py should use):
    loss = eaft_loss + cfg.moe_aux_coef * moe_aux + cfg.act_aux_coef * act_aux

Pack-awareness:
  EAFT is fully compatible with Multipack — just pass the attention_mask
  from MultipackDataset and it correctly ignores padding positions.

Integration in train.py:
    from anthos.eaft import EAFTLoss

    criterion = EAFTLoss(
        vocab_size  = cfg.vocab_size,
        top_k       = 50,            # entropy over top-50 logits
        focal_gamma = 1.0,           # entropy weight strength (0 = standard CE)
        act_gamma   = 0.5,           # extra weight for high-ponder positions
        label_smoothing = 0.1,
    )

    # In training loop:
    logits, aux_loss = model(input_ids, return_aux=True)
    loss = criterion(logits, labels, attention_mask, loops_used=loops_used)
    total_loss = loss + aux_loss
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EAFTLoss(nn.Module):
    """
    Entropy-Aware Focal Training loss for Anthos.

    Standard cross-entropy:
        CE(y, ŷ) = -log(ŷ[y])

    EAFT:
        H(ŷ, k)  = entropy of top-k softmax distribution
        w(pos)   = 1 + focal_gamma * H(ŷ, k) / log(k)   [0, 1+focal_gamma]
        EAFT     = mean(w * CE)                            [normalised]

    ACT extension (optional, needs loops_used from AnthosRecurrentBlock):
        loops_used: (B, T) — number of loops each position used before halting
        act_weight = 1 + act_gamma * (loops_used / max_loops)
        final_w    = w * act_weight

    Args:
        vocab_size:      model vocab size (for pre-allocating buffers)
        top_k:           entropy computed over top-k logits (default 50)
        focal_gamma:     entropy weight strength; 0 = standard CE, 2 = strong
        act_gamma:       extra weight for high-loop-usage positions; 0 = off
        max_loops:       max_loop_iters from AnthosConfig (for normalising)
        label_smoothing: standard label smoothing (default 0.1)
        reduction:       "mean" | "sum" | "none"
    """

    def __init__(
        self,
        vocab_size:      int,
        top_k:           int   = 50,
        focal_gamma:     float = 1.0,
        act_gamma:       float = 0.5,
        max_loops:       int   = 16,
        label_smoothing: float = 0.1,
        reduction:       str   = "mean",
    ):
        super().__init__()
        self.vocab_size      = vocab_size
        self.top_k           = min(top_k, vocab_size)
        self.focal_gamma     = focal_gamma
        self.act_gamma       = act_gamma
        self.max_loops       = max_loops
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

        # log(k) for normalising entropy to [0, 1]
        self._log_k = math.log(self.top_k)

    def _entropy_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-position entropy weight from top-k logit distribution.

        Args:
            logits: (B, T, V)
        Returns:
            weights: (B, T)  in [1, 1+focal_gamma]
        """
        B, T, V = logits.shape

        # Top-k logits only — cheaper than full softmax over vocab
        topk_logits, _ = logits.topk(self.top_k, dim=-1)          # (B, T, k)
        topk_probs      = F.softmax(topk_logits, dim=-1)            # (B, T, k)

        # Shannon entropy, normalised to [0, 1]
        entropy = -(topk_probs * (topk_probs + 1e-10).log()).sum(-1) # (B, T)
        entropy = (entropy / self._log_k).clamp(0.0, 1.0)

        return 1.0 + self.focal_gamma * entropy                     # (B, T)

    def _act_weights(
        self,
        loops_used: torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """
        Extra weight for positions that needed more loop iterations.
        These positions had high uncertainty in the recurrent stream —
        they deserve more gradient, same reasoning as entropy weighting.

        Returns: (B, T)  in [1, 1+act_gamma]
        """
        normalised = (loops_used / self.max_loops).clamp(0.0, 1.0)
        return 1.0 + self.act_gamma * normalised

    def forward(
        self,
        logits:         torch.Tensor,              # (B, T, V)
        labels:         torch.Tensor,              # (B, T)  -100 = ignore
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) 1=real, 0=pad
        loops_used:     Optional[torch.Tensor] = None,  # (B, T) from ACT
    ) -> torch.Tensor:
        """
        Compute EAFT loss.

        Labels should already have -100 at padding positions (MultipackDataset
        sets this automatically).  attention_mask is used as a belt-and-
        suspenders check — any position with mask=0 is zeroed out of the loss
        regardless of label value.

        Returns:
            scalar loss (if reduction="mean") or (B, T) tensor
        """
        B, T, V = logits.shape

        # ── Entropy weights ────────────────────────────────────────────────
        with torch.no_grad():
            w = self._entropy_weights(logits)                        # (B, T)

            # ACT extension
            if loops_used is not None and self.act_gamma > 0:
                w = w * self._act_weights(loops_used)

        # ── Per-position cross-entropy ─────────────────────────────────────
        # Use label smoothing via F.cross_entropy with reduction="none"
        logits_2d = logits.view(B * T, V)
        labels_1d = labels.view(B * T)

        ce = F.cross_entropy(
            logits_2d,
            labels_1d,
            reduction       = "none",
            label_smoothing = self.label_smoothing,
            ignore_index    = -100,
        ).view(B, T)                                                 # (B, T)

        # ── Apply weights ──────────────────────────────────────────────────
        weighted = ce * w.detach()                                   # (B, T)

        # ── Mask out padding ──────────────────────────────────────────────
        if attention_mask is not None:
            weighted = weighted * attention_mask.float()

        # ── Reduce ────────────────────────────────────────────────────────
        if self.reduction == "none":
            return weighted

        # Mean over non-ignored positions only
        valid_positions = (labels != -100)
        if attention_mask is not None:
            valid_positions = valid_positions & attention_mask.bool()

        n_valid = valid_positions.float().sum().clamp(min=1.0)

        if self.reduction == "mean":
            return weighted.sum() / n_valid
        return weighted.sum()   # "sum"

    def extra_repr(self) -> str:
        return (f"top_k={self.top_k}, focal_gamma={self.focal_gamma}, "
                f"act_gamma={self.act_gamma}, label_smoothing={self.label_smoothing}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: standard CE wrapped in same interface (for ablation)
# ─────────────────────────────────────────────────────────────────────────────

class StandardLoss(nn.Module):
    """
    Standard cross-entropy with the same interface as EAFTLoss.
    Use this for ablation: swap with EAFTLoss and compare loss curves.
    """

    def __init__(self, label_smoothing: float = 0.1, **kwargs):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits:         torch.Tensor,
        labels:         torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loops_used:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.view(B * T, V),
            labels.view(B * T),
            reduction       = "none",
            label_smoothing = self.label_smoothing,
            ignore_index    = -100,
        ).view(B, T)

        if attention_mask is not None:
            ce = ce * attention_mask.float()

        valid    = (labels != -100)
        n_valid  = valid.float().sum().clamp(min=1.0)
        return ce.sum() / n_valid
