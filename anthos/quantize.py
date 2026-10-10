"""
anthos/quantize.py — Post-Training Quantization (GPTQ-style + AWQ-style)

Reduces model size and inference latency by representing weights in INT4 or INT8
instead of BF16/FP32, with minimal accuracy loss.

Two approaches:
  GPTQ (Frantar et al. 2022):
    Layer-by-layer weight quantization using the Hessian of the layer's output.
    Minimises E[||W·X - Q(W)·X||²] per block, where Q(W) is the quantized weight.
    Each row's quantization error is propagated to remaining columns via Cholesky.
    → 4-bit quantization with near-bf16 accuracy on most tasks.

  AWQ (Lin et al. 2023, Activation-Aware Weight Quantization):
    Scales weights channel-wise based on activation magnitudes before quantizing.
    Key insight: only ~1% of weights (those with large activations) matter for
    accuracy. Scaling them up before quantizing preserves their precision.
    → Simpler than GPTQ, no Hessian needed, but slightly less accurate.

Usage:
    from anthos.quantize import quantize_model_gptq, quantize_model_awq

    # Quantize to INT4 (GPTQ-style)
    quant_model = quantize_model_gptq(
        model,
        calibration_loader,   # DataLoader of calibration batches
        bits=4,
        group_size=128,        # quantize in groups of 128 columns
        device="cuda",
    )

    # Save quantized model
    torch.save({"model": quant_model.state_dict()}, "checkpoints/anthos-q4.pt")

    # AWQ (faster, no calibration loop)
    quant_model = quantize_model_awq(model, bits=4, group_size=128)

Note:
    These are pure-PyTorch reference implementations.
    For production inference speed, use bitsandbytes or llama.cpp after export.
    See export_gguf.py for GGUF export.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Quantization primitives
# ─────────────────────────────────────────────────────────────────────────────

def quantize_weight(
    W:          torch.Tensor,   # (out_features, in_features)
    bits:       int   = 4,
    group_size: int   = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetric group-wise quantization of a weight matrix.

    Returns:
        W_quant: (out_features, in_features) — dequantized weights (float)
        scales:  (out_features, n_groups)    — per-group scale factors
        zeros:   (out_features, n_groups)    — per-group zero points (symmetric → 0)
    """
    out_features, in_features = W.shape
    n_groups  = math.ceil(in_features / group_size)
    maxq      = 2 ** (bits - 1) - 1     # 7 for INT4, 127 for INT8

    W_float = W.float()
    scales  = []
    quanted = []

    for g in range(n_groups):
        start = g * group_size
        end   = min(start + group_size, in_features)
        block = W_float[:, start:end]              # (out, block)

        # Symmetric scale: max(|w|) / max_q
        scale = block.abs().max(dim=1, keepdim=True).values / maxq
        scale = scale.clamp(min=1e-8)
        scales.append(scale)

        # Quantize and dequantize in float (for calibration error measurement)
        w_q     = (block / scale).round().clamp(-maxq, maxq)
        w_deq   = w_q * scale
        quanted.append(w_deq)

    W_quant = torch.cat(quanted, dim=1).to(W.dtype)
    scales  = torch.cat(scales, dim=1)             # (out, n_groups)
    zeros   = torch.zeros_like(scales)
    return W_quant, scales, zeros


def dequantize_weight(
    W_int:      torch.Tensor,   # (out, in) — INT quantized
    scales:     torch.Tensor,   # (out, n_groups)
    group_size: int = 128,
) -> torch.Tensor:
    """Reconstruct float weight from INT representation and scales."""
    out_features, in_features = W_int.shape
    n_groups = scales.shape[1]
    blocks   = []
    for g in range(n_groups):
        start = g * group_size
        end   = min(start + group_size, in_features)
        block = W_int[:, start:end].float()
        block = block * scales[:, g : g + 1]
        blocks.append(block)
    return torch.cat(blocks, dim=1).to(W_int.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# AWQ: Activation-Aware Weight Quantization
# ─────────────────────────────────────────────────────────────────────────────

def _collect_activation_stats(
    model:     nn.Module,
    loader,
    n_batches: int = 32,
    device:    str = "cuda",
) -> dict[str, torch.Tensor]:
    """
    Collect per-channel activation magnitudes (max) by running calibration batches.
    Returns {layer_name: max_activation_per_channel}.
    """
    stats:    dict[str, torch.Tensor] = {}
    handles:  list = []

    def _hook(name):
        def fn(module, input, output):
            x = input[0].detach().float()
            # Max over batch and sequence dims → per-channel magnitude
            mag = x.abs().reshape(-1, x.shape[-1]).max(dim=0).values
            if name in stats:
                stats[name] = torch.maximum(stats[name], mag)
            else:
                stats[name] = mag
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            else:
                ids = batch.to(device)
            try:
                model(ids[:, :-1])
            except Exception:
                pass

    for h in handles:
        h.remove()

    return stats


def quantize_model_awq(
    model:        nn.Module,
    loader=None,
    bits:         int  = 4,
    group_size:   int  = 128,
    n_calib_batches: int = 32,
    device:       str  = "cuda",
) -> nn.Module:
    """
    AWQ-style quantization: scale channels by activation magnitude before quantizing.

    If loader is None, falls back to simple symmetric quantization without
    activation-aware scaling.

    Modifies the model in-place and returns it.
    """
    if loader is not None:
        print("[quantize] AWQ: collecting activation statistics...", flush=True)
        act_stats = _collect_activation_stats(model, loader, n_calib_batches, device)
    else:
        act_stats = {}

    n_quantized = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.data   # (out, in)

        # AWQ scaling: scale = act_mag^0.5, scale weight up to preserve large-act channels
        if name in act_stats:
            act_mag = act_stats[name].to(W.device)
            # Clamp to avoid extreme scales
            scale   = act_mag.pow(0.5).clamp(min=1e-4)
            scale   = scale / scale.mean()                   # normalise to unit mean
            W_scaled = W * scale.unsqueeze(0)               # (out, in) * (in,)
            # After quantization, the scale is absorbed back into the weight
            W_quant, _, _ = quantize_weight(W_scaled, bits=bits, group_size=group_size)
            W_quant = W_quant / scale.unsqueeze(0)
        else:
            W_quant, _, _ = quantize_weight(W, bits=bits, group_size=group_size)

        module.weight.data = W_quant.to(W.dtype)
        n_quantized += 1

    print(f"[quantize] AWQ {bits}-bit: quantized {n_quantized} linear layers")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GPTQ: Hessian-based layer-wise quantization
# ─────────────────────────────────────────────────────────────────────────────

class GPTQQuantizer:
    """
    GPTQ quantizer for a single Linear layer.

    Accumulates Hessian H = 2 X Xᵀ from calibration batches, then uses
    the inverse Cholesky of H to compensate quantization error column-by-column.
    """

    def __init__(self, layer: nn.Linear, bits: int = 4, group_size: int = 128):
        self.layer      = layer
        self.bits       = bits
        self.group_size = group_size
        W               = layer.weight.data.clone().float()
        self.rows, self.cols = W.shape
        self.H          = torch.zeros(self.cols, self.cols, device=W.device)
        self.n_samples  = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate Hessian from a calibration input batch."""
        inp = inp.float()
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])   # (B*T, C_in)
        self.H         += 2 * inp.T @ inp
        self.n_samples += inp.shape[0]

    @torch.no_grad()
    def quantize(self) -> torch.Tensor:
        """
        Run GPTQ quantization. Returns the quantized weight matrix.
        The layer's weight is updated in-place.
        """
        W   = self.layer.weight.data.clone().float()
        H   = self.H / (self.n_samples + 1e-6)
        H   = H + 1e-2 * torch.eye(self.cols, device=H.device)   # dampening

        # Cholesky decomposition of H for efficient triangular solve
        try:
            L = torch.linalg.cholesky(H)
            H_inv_diag = torch.cholesky_solve(
                torch.eye(self.cols, device=H.device), L
            ).diag()
        except Exception:
            H_inv_diag = (1.0 / H.diag()).clamp(max=1e6)

        maxq   = 2 ** (self.bits - 1) - 1
        W_quant = W.clone()

        for col in range(self.cols):
            w = W[:, col]                            # (out,)
            g = col // self.group_size
            start = g * self.group_size
            end   = min(start + self.group_size, self.cols)

            scale = W[:, start:end].abs().max(dim=1).values / maxq
            scale = scale.clamp(min=1e-8)

            w_q   = (w / scale).round().clamp(-maxq, maxq) * scale
            err   = (w - w_q) * H_inv_diag[col]

            # Propagate error to remaining columns (simplified 1-D Cholesky correction)
            if col + 1 < self.cols:
                W[:, col + 1 :] -= err.unsqueeze(1) * H[col, col + 1:].unsqueeze(0) / (H[col, col] + 1e-8)

            W_quant[:, col] = w_q

        self.layer.weight.data = W_quant.to(self.layer.weight.dtype)
        return W_quant


def quantize_model_gptq(
    model:        nn.Module,
    loader,
    bits:         int  = 4,
    group_size:   int  = 128,
    n_calib_batches: int = 128,
    device:       str  = "cuda",
) -> nn.Module:
    """
    GPTQ-style quantization with calibration data.

    Collects Hessian information from the calibration loader, then quantizes
    each Linear layer layer-by-layer.

    Args:
        model:              Anthos model (on `device`)
        loader:             DataLoader yielding calibration batches
        bits:               Quantization bits (4 or 8)
        group_size:         Columns per quantization group (64 or 128)
        n_calib_batches:    How many batches to use for Hessian estimation
        device:             "cuda" or "cpu"

    Returns:
        Quantized model (modified in-place).
    """
    print(f"[quantize] GPTQ {bits}-bit: setting up quantizers...", flush=True)

    quantizers: dict[str, GPTQQuantizer] = {}
    handles: list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            q = GPTQQuantizer(module, bits=bits, group_size=group_size)
            quantizers[name] = q

            def _hook(q_ref):
                def fn(module, input, output):
                    q_ref.add_batch(input[0].detach())
                return fn

            handles.append(module.register_forward_hook(_hook(q)))

    # Collect calibration data
    model.eval()
    print(f"[quantize] GPTQ: collecting Hessians from {n_calib_batches} batches...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_calib_batches:
                break
            if isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)
            else:
                ids = batch.to(device)
            try:
                model(ids[:, :-1])
            except Exception:
                pass

    for h in handles:
        h.remove()

    # Quantize each layer
    n_layers = len(quantizers)
    for i, (name, q) in enumerate(quantizers.items()):
        q.quantize()
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_layers}] quantized {name}", flush=True)

    print(f"[quantize] GPTQ done: {n_layers} layers quantized to {bits}-bit")
    return model
