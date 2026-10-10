"""
merge.py — Model merging and LoRA bake-in utilities

Three merging methods:

SLERP (Spherical Linear Interpolation)
  Interpolates between two checkpoints along the unit-sphere geodesic.
  Smoother than LERP for weight matrices; preserves norms.
  Best for fine-tuned variants of the same base model.

TIES (Trim, Elect, Merge)
  Resolves sign conflicts across multiple models:
    1. Trim small deltas (noise)
    2. Elect dominant sign per parameter
    3. Average only same-sign deltas
  Best for merging task-specific models with conflicting updates.

DARE (Drop And REscale)
  Randomly drop each delta weight with probability p, then rescale
  by 1/(1-p) to preserve expected magnitude. Reduces interference
  between merged models without reducing coverage.
  Often combined with TIES: DARE+TIES.

LoRA bake-in
  Absorbs a LoRA adapter back into the base weights:
    W_merged = W_base + (lora_B @ lora_A) * alpha / rank
  Produces a single checkpoint with no LoRA overhead at inference.

Usage:
    from anthos.merge import slerp, ties_merge, dare_merge, bake_lora

    # SLERP between two checkpoints
    merged_sd = slerp("ckpt_a.pt", "ckpt_b.pt", t=0.5)

    # TIES across multiple checkpoints
    merged_sd = ties_merge(["ckpt_a.pt", "ckpt_b.pt", "ckpt_c.pt"],
                           base_path="ckpt_base.pt", density=0.2)

    # Bake LoRA adapter into base weights
    merged_sd = bake_lora("ckpt_base.pt", lora_state_dict, scale=1.0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"] if "model" in ckpt else ckpt


def _save(state_dict: dict, path: str, extra: Optional[dict] = None) -> None:
    payload = {"model": state_dict}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Saved merged checkpoint → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SLERP
# ─────────────────────────────────────────────────────────────────────────────

def _slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    norm_a = a_f.norm()
    norm_b = b_f.norm()
    if norm_a < 1e-9 or norm_b < 1e-9:
        return ((1 - t) * a + t * b).to(a.dtype)
    a_n   = a_f / norm_a
    b_n   = b_f / norm_b
    dot   = torch.clamp((a_n * b_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        return ((1 - t) * a + t * b).to(a.dtype)
    sin_o = torch.sin(omega)
    out   = (torch.sin((1 - t) * omega) / sin_o * a_f +
             torch.sin(t * omega)       / sin_o * b_f)
    return out.view_as(a).to(a.dtype)


def slerp(
    path_a:   str,
    path_b:   str,
    t:        float = 0.5,
    output:   Optional[str] = None,
) -> dict:
    """
    Merge two checkpoints via SLERP at interpolation coefficient t.
    t=0 → model A,  t=1 → model B,  t=0.5 → midpoint.
    """
    sd_a = _load(path_a)
    sd_b = _load(path_b)

    merged = {}
    for key in sd_a:
        if key not in sd_b:
            merged[key] = sd_a[key]
            continue
        a, b = sd_a[key], sd_b[key]
        if a.shape != b.shape or not a.is_floating_point():
            merged[key] = a if t < 0.5 else b
            continue
        merged[key] = _slerp_tensor(a, b, t)

    if output:
        _save(merged, output, {"merge_method": "slerp", "t": t})
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# TIES
# ─────────────────────────────────────────────────────────────────────────────

def ties_merge(
    paths:      list[str],
    base_path:  str,
    density:    float        = 0.2,   # keep top (1-density) fraction of deltas
    scale:      float        = 1.0,   # output scale coefficient
    output:     Optional[str] = None,
) -> dict:
    """
    TIES merge: Trim, Elect sign, Merge.

    Args:
        paths:   checkpoints to merge (fine-tuned variants)
        base_path: base model (used to compute deltas)
        density:   fraction of delta weights to DROP (trim by magnitude)
        scale:     scale factor applied to final merged delta
    """
    base   = _load(base_path)
    models = [_load(p) for p in paths]

    merged = {}
    for key in base:
        b = base[key]
        if not b.is_floating_point():
            merged[key] = b
            continue

        deltas = []
        for sd in models:
            if key in sd and sd[key].shape == b.shape:
                deltas.append(sd[key].float() - b.float())

        if not deltas:
            merged[key] = b
            continue

        delta_stack = torch.stack(deltas, dim=0)   # (n_models, ...)

        # Step 1: Trim — zero out smallest |delta| values
        flat  = delta_stack.view(len(deltas), -1)
        mag   = flat.abs()
        k     = max(1, int(flat.shape[-1] * (1 - density)))
        threshold, _ = mag.kthvalue(flat.shape[-1] - k + 1, dim=-1, keepdim=True)
        flat  = flat * (mag >= threshold)
        delta_stack = flat.view_as(delta_stack)

        # Step 2: Elect — majority sign per parameter
        signs = delta_stack.sign()
        elected_sign = signs.sum(0).sign()
        elected_sign[elected_sign == 0] = 1

        # Step 3: Merge — average same-sign deltas
        same_sign = (signs == elected_sign.unsqueeze(0)).float()
        counts    = same_sign.sum(0).clamp(min=1)
        merged_delta = (delta_stack * same_sign).sum(0) / counts

        merged[key] = (b.float() + scale * merged_delta).to(b.dtype)

    if output:
        _save(merged, output, {"merge_method": "ties", "density": density, "n_models": len(paths)})
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DARE
# ─────────────────────────────────────────────────────────────────────────────

def dare_merge(
    paths:      list[str],
    base_path:  str,
    drop_rate:  float        = 0.9,   # probability of dropping each delta weight
    scale:      float        = 1.0,
    output:     Optional[str] = None,
    seed:       int          = 42,
) -> dict:
    """
    DARE merge: Drop And REscale deltas before merging.

    Each delta weight is randomly zeroed with probability drop_rate,
    then rescaled by 1/(1-drop_rate) to preserve expected magnitude.
    Reduces inter-model interference.  Combine with TIES for best results.
    """
    torch.manual_seed(seed)
    base   = _load(base_path)
    models = [_load(p) for p in paths]

    merged = {}
    for key in base:
        b = base[key]
        if not b.is_floating_point():
            merged[key] = b
            continue

        scaled_deltas = []
        for sd in models:
            if key in sd and sd[key].shape == b.shape:
                delta = sd[key].float() - b.float()
                mask  = torch.bernoulli(
                    torch.full_like(delta, 1.0 - drop_rate)
                )
                scaled = delta * mask / max(1 - drop_rate, 1e-9)
                scaled_deltas.append(scaled)

        if not scaled_deltas:
            merged[key] = b
            continue

        avg_delta = torch.stack(scaled_deltas, dim=0).mean(0)
        merged[key] = (b.float() + scale * avg_delta).to(b.dtype)

    if output:
        _save(merged, output, {"merge_method": "dare", "drop_rate": drop_rate, "n_models": len(paths)})
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# LoRA bake-in
# ─────────────────────────────────────────────────────────────────────────────

def bake_lora(
    base_path:         str,
    lora_state_dict:   dict,
    rank:              int   = 16,
    alpha:             float = 16.0,
    output:            Optional[str] = None,
) -> dict:
    """
    Absorb LoRA weights into base model weights.

    Looks for pairs of keys matching:
        {prefix}.lora_A  (rank × d_in)
        {prefix}.lora_B  (d_out × rank)
    and computes: W_merged = W_base + (B @ A) * (alpha / rank)

    Args:
        base_path:       path to base checkpoint
        lora_state_dict: state dict containing lora_A / lora_B keys
        rank:            LoRA rank used during training
        alpha:           LoRA alpha scaling factor
    """
    scale  = alpha / rank
    base   = _load(base_path)
    merged = {k: v.clone() for k, v in base.items()}

    # Find LoRA A/B pairs
    lora_a_keys = [k for k in lora_state_dict if k.endswith(".lora_A")]
    for a_key in lora_a_keys:
        b_key  = a_key.replace(".lora_A", ".lora_B")
        w_key  = a_key.replace(".lora_A", ".weight")
        if b_key not in lora_state_dict:
            continue
        if w_key not in merged:
            # Try stripping a common "base_model.model." prefix
            w_key_alt = w_key.replace("base_model.model.", "")
            if w_key_alt in merged:
                w_key = w_key_alt
            else:
                continue

        A = lora_state_dict[a_key].float()   # (rank, d_in)
        B = lora_state_dict[b_key].float()   # (d_out, rank)
        W = merged[w_key].float()
        merged[w_key] = (W + scale * (B @ A)).to(merged[w_key].dtype)

    if output:
        _save(merged, output, {"merge_method": "lora_bake", "rank": rank, "alpha": alpha})
    return merged
