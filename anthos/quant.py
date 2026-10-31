"""
anthos/quant.py — Quantization Utilities (FP8 / FP16 / BF16)

FP8 quantization cuts memory roughly in half vs FP16 with minimal quality
loss on MoE models.  Required for anthos_1b+ on consumer hardware.

Device awareness:
  CUDA (H100+): FP8 via torch.float8_e4m3fn  — full support
  CUDA (older): FP16 / BF16 fallback
  MPS (Apple):  BF16 (M1/M2/M3 support bfloat16 via MPS)
  CPU:          FP32

FP8 is NOT used during training — gradients need higher precision.
Use FP8 for inference loading and KV cache compression only.

Integration:
  At serving time (serve.py), load the checkpoint with:
      model = load_quantized(model, checkpoint_path, device)

  During training, use standard FP32/BF16 — quant.py is irrelevant there.

Usage:
    from anthos.quant import load_quantized, get_dtype, QuantConfig

    cfg   = QuantConfig(mode="fp8")          # "fp8" | "bf16" | "fp16" | "fp32"
    dtype = get_dtype(cfg, device)
    model = load_quantized(model, ckpt_path, device, quant_cfg=cfg)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantConfig:
    mode:            str   = "auto"   # "auto" | "fp8" | "bf16" | "fp16" | "fp32"
    fp8_dtype:       str   = "e4m3"   # "e4m3" | "e5m2"
    skip_modules:    list  = None     # module name substrings to skip (e.g. ["norm"])
    moe_only:        bool  = False    # quantise only MoE expert weights
    per_channel:     bool  = True     # per-channel scaling for experts

    def __post_init__(self):
        if self.skip_modules is None:
            # Always keep norms + embeddings in high precision
            self.skip_modules = ["norm", "embed", "head", "act", "gate_proj"]


# ─────────────────────────────────────────────────────────────────────────────
# Device / dtype detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fp8_available() -> bool:
    """FP8 requires CUDA + torch >= 2.1 + H100/Ada architecture."""
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    # Check compute capability
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 9   # H100 = sm_90


def get_dtype(cfg: QuantConfig, device: torch.device) -> torch.dtype:
    """
    Resolve the best available dtype for the given device and config.

    Auto-selection rules:
      CUDA + H100+ + fp8 mode  → float8_e4m3fn  (inference only)
      CUDA + any              → bfloat16
      MPS                     → bfloat16 (M1/M2/M3 support BF16)
      CPU                     → float32
    """
    mode = cfg.mode

    if mode == "auto":
        if fp8_available() and device.type == "cuda":
            return torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16
        if device.type in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32

    dtype_map = {
        "fp8":   torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.bfloat16,
        "bf16":  torch.bfloat16,
        "fp16":  torch.float16,
        "fp32":  torch.float32,
    }
    resolved = dtype_map.get(mode, torch.float32)

    # Fallback: if fp8 requested but not available, use bf16
    if mode == "fp8" and not fp8_available():
        print("[quant] FP8 not available — falling back to BF16")
        return torch.bfloat16

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Per-tensor FP8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def _quantize_tensor_fp8(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to FP8 with absmax scaling.

    Returns:
        (quantized_tensor, scale_factor)
    """
    if per_channel and tensor.dim() >= 2:
        # Per output-channel scaling
        amax   = tensor.abs().amax(dim=tuple(range(1, tensor.dim())), keepdim=True)
    else:
        amax   = tensor.abs().amax()

    amax   = amax.clamp(min=1e-12)
    scale  = amax / 448.0   # FP8 e4m3 max value
    scaled = (tensor / scale).clamp(-448.0, 448.0)

    return scaled.to(fp8_dtype), scale.to(torch.float32)


def _should_quantize(name: str, cfg: QuantConfig, module: nn.Module) -> bool:
    """Decide if this module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False
    for skip in cfg.skip_modules:
        if skip in name:
            return False
    if cfg.moe_only and "routed_experts" not in name:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FP8 Linear wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FP8Linear(nn.Module):
    """
    FP8-quantized linear layer for inference.

    Stores weight in FP8, dequantizes to BF16 for matmul.
    Saves ~50% memory vs BF16 weights.
    """

    def __init__(
        self,
        weight_fp8:  torch.Tensor,   # quantized weight
        scale:       torch.Tensor,   # per-channel scale
        bias:        Optional[torch.Tensor] = None,
        fp8_dtype:   torch.dtype            = None,
    ):
        super().__init__()
        self.fp8_dtype = fp8_dtype or torch.float8_e4m3fn
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(
        cls,
        linear:      nn.Linear,
        fp8_dtype:   torch.dtype = None,
        per_channel: bool        = True,
    ) -> "FP8Linear":
        fp8_dtype = fp8_dtype or (
            torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn")
            else torch.bfloat16
        )
        w_fp8, scale = _quantize_tensor_fp8(
            linear.weight.data, fp8_dtype, per_channel
        )
        bias = linear.bias.data if linear.bias is not None else None
        return cls(w_fp8, scale, bias, fp8_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize to BF16 for the actual matmul
        w = (self.weight_fp8.to(torch.bfloat16) * self.scale.to(torch.bfloat16))
        return nn.functional.linear(x, w, self.bias)


# ─────────────────────────────────────────────────────────────────────────────
# Model quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(
    model:  nn.Module,
    cfg:    QuantConfig,
    device: torch.device,
) -> nn.Module:
    """
    Quantize eligible Linear layers in-place to FP8 (inference only).

    For MoE models (like Anthos), quantize expert weights first — they
    comprise the bulk of parameters and benefit most.
    """
    if not fp8_available() or cfg.mode not in ("fp8", "auto"):
        print(f"[quant] Skipping FP8 quantization (mode={cfg.mode}, "
              f"fp8_available={fp8_available()})")
        return model.to(device)

    fp8_dtype     = (torch.float8_e4m3fn if cfg.fp8_dtype == "e4m3"
                     else torch.float8_e5m2)
    n_quantized   = 0
    n_skipped     = 0
    param_savings = 0.0

    for name, module in list(model.named_modules()):
        if _should_quantize(name, cfg, module):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model

            original_params = module.weight.numel()
            fp8_layer = FP8Linear.from_linear(
                module, fp8_dtype=fp8_dtype, per_channel=cfg.per_channel
            )
            setattr(parent, child_name, fp8_layer)

            n_quantized   += 1
            param_savings  += original_params * (2 - 1) / (1024 ** 2)  # MB saved
        elif isinstance(module, nn.Linear):
            n_skipped += 1

    print(f"[quant] Quantized {n_quantized} layers, skipped {n_skipped} "
          f"| Est. savings: {param_savings:.1f} MB")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading with quantization
# ─────────────────────────────────────────────────────────────────────────────

def load_quantized(
    model:      nn.Module,
    ckpt_path:  str,
    device:     Optional[torch.device] = None,
    quant_cfg:  Optional[QuantConfig]  = None,
) -> nn.Module:
    """
    Load a checkpoint and optionally quantize for inference.

    Args:
        model:     Anthos model instance (uninitialised weights)
        ckpt_path: path to .pt checkpoint
        device:    target device (auto-detected if None)
        quant_cfg: quantization config (default: auto mode)

    Returns:
        model ready for inference
    """
    device    = device or detect_device()
    quant_cfg = quant_cfg or QuantConfig(mode="auto")

    print(f"[quant] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[quant] Loaded step {ckpt.get('step', '?')} "
          f"| loss {ckpt.get('loss', '?'):.4f}")

    dtype = get_dtype(quant_cfg, device)
    print(f"[quant] Target dtype: {dtype} on {device}")

    if quant_cfg.mode == "fp8" and fp8_available():
        model = quantize_model(model, quant_cfg, device)
    else:
        model = model.to(device=device, dtype=dtype)

    model.eval()
    return model
