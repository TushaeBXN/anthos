"""
anthos/telemetry.py — Honest checkpoint telemetry harness

Records decode tok/s, prefill tok/s, peak process RSS, and KV-cache size at
two representative context lengths (3 k and 100 k tokens) every time a
training checkpoint is saved.  Results are written as a JSON sidecar file
alongside each checkpoint so performance history is tied directly to the
model weights.

Usage (in training loop — call after torch.save):
    from anthos.telemetry import TelemetryHarness

    harness = TelemetryHarness(cfg, hardware="A6000_48GB")
    # … training loop …
    torch.save(model.state_dict(), ckpt_path)
    harness.record(model, tokenizer, ckpt_path, step)

Sidecar written to: <ckpt_path>.telemetry.json

Hardware labels:
    "A6000_48GB"        — RunPod RTX A6000 (training-side)
    "apple_m4_16GB"     — Mac mini M4 16 GB (local inference target)
    "apple_m3pro_18GB"  — MacBook Pro M3 Pro 18 GB
    "cpu_only"          — CPU-only environment (MacBook Intel, CI, etc.)
"""

from __future__ import annotations

import gc
import json
import os
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TELEMETRY_CONTEXT_LENGTHS = (3_000, 100_000)   # tokens — two representative points


@dataclass
class TelemetryConfig:
    """Controls what the harness measures.  Set fields to False to skip."""
    measure_prefill:        bool  = True   # tok/s on a fresh forward pass
    measure_decode:         bool  = True   # tok/s for autoregressive generation
    measure_rss:            bool  = True   # peak process RSS in GB
    measure_kv_cache:       bool  = True   # KV-cache size in MB at each ctx length
    context_lengths:        tuple = TELEMETRY_CONTEXT_LENGTHS
    prefill_batch_size:     int   = 1
    decode_new_tokens:      int   = 64     # tokens to generate for decode throughput
    decode_n_loops:         int   = 4      # recurrent depth during decode measurement
    warmup_steps:           int   = 1      # throwaway forward passes before timing


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _peak_rss_gb() -> float:
    """Peak RSS in GB.  Works on Linux (rusage) and macOS (getrusage)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: bytes; macOS: bytes (man 2 getrusage)
    return rss / (1024 ** 3)


def _kv_cache_mb(kv_cache: dict) -> float:
    """Sum of all tensors in the kv_cache dict, in MB."""
    total = 0
    for v in kv_cache.values():
        if isinstance(v, torch.Tensor):
            total += v.nelement() * v.element_size()
        elif isinstance(v, (list, tuple)):
            for t in v:
                if isinstance(t, torch.Tensor):
                    total += t.nelement() * t.element_size()
    return total / (1024 ** 2)


def _sync() -> None:
    """Flush CUDA/MPS queues for accurate wall-clock timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry harness
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryHarness:
    """
    Checkpoint-triggered performance measurement.

    Attach one instance to your training loop.  After every checkpoint save,
    call harness.record(model, tokenizer, ckpt_path, step) to:
      1. Run prefill + decode benchmarks at each configured context length
      2. Capture peak RSS
      3. Capture KV-cache size
      4. Write <ckpt_path>.telemetry.json next to the checkpoint

    The JSON is human-readable and can be aggregated across checkpoints to
    track whether performance regresses as the model trains.
    """

    def __init__(
        self,
        model_cfg,                      # AnthosConfig — used for device/dim
        hardware:   str            = "cpu_only",
        tel_cfg:    TelemetryConfig = None,
        stage_tag:  str            = "",
    ):
        self.model_cfg  = model_cfg
        self.hardware   = hardware
        self.tel_cfg    = tel_cfg or TelemetryConfig()
        self.stage_tag  = stage_tag   # e.g. "stage1_40pct_195M_tokens"

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        model,
        tokenizer,
        ckpt_path: str,
        step:      int,
    ) -> dict:
        """
        Benchmark model and write a JSON sidecar next to ckpt_path.

        Returns the telemetry dict (also written to disk).
        """
        sidecar = Path(ckpt_path).with_suffix("").with_suffix(".telemetry.json")
        device  = next(model.parameters()).device

        results = {
            "step":          step,
            "hardware":      self.hardware,
            "stage_tag":     self.stage_tag,
            "ckpt_path":     str(ckpt_path),
            "torch_version": torch.__version__,
            "device":        str(device),
            "measurements":  {},
        }

        model.eval()
        with torch.no_grad():
            for ctx_len in self.tel_cfg.context_lengths:
                tag = f"ctx_{ctx_len}"
                results["measurements"][tag] = self._measure_context(
                    model, tokenizer, ctx_len, device
                )

        if self.tel_cfg.measure_rss:
            results["peak_rss_gb"] = _peak_rss_gb()

        sidecar.write_text(json.dumps(results, indent=2))
        return results

    # ── Measurement helpers ───────────────────────────────────────────────────

    def _make_prompt(self, ctx_len: int, device: torch.device, vocab_size: int) -> torch.Tensor:
        """Synthetic random prompt of exactly ctx_len tokens."""
        return torch.randint(0, min(vocab_size, 1000), (1, ctx_len), device=device)

    def _measure_context(
        self,
        model,
        tokenizer,
        ctx_len: int,
        device: torch.device,
    ) -> dict:
        """Run prefill and decode benchmarks for a single context length."""
        cfg    = self.tel_cfg
        vocab  = self.model_cfg.vocab_size
        result = {}

        # Cap context to model's max_seq_len — 100k may exceed it
        safe_ctx = min(ctx_len, self.model_cfg.max_seq_len)
        prompt   = self._make_prompt(safe_ctx, device, vocab)

        # ── Prefill ───────────────────────────────────────────────────────
        if cfg.measure_prefill:
            for _ in range(cfg.warmup_steps):
                model(prompt)

            gc.collect()
            _sync()
            t0 = time.perf_counter()
            model(prompt)
            _sync()
            elapsed = time.perf_counter() - t0

            result["prefill_toks_per_sec"] = safe_ctx / elapsed if elapsed > 0 else 0.0
            result["prefill_latency_ms"]   = elapsed * 1_000

        # ── KV-cache size at this context ─────────────────────────────────
        if cfg.measure_kv_cache:
            kv_cache = {}
            model(prompt, kv_cache=kv_cache)
            result["kv_cache_mb"] = _kv_cache_mb(kv_cache)
            del kv_cache

        # ── Decode throughput ─────────────────────────────────────────────
        if cfg.measure_decode and safe_ctx <= self.model_cfg.max_seq_len - cfg.decode_new_tokens:
            # Use a short prefix so generate() can run decode_new_tokens steps
            decode_prompt = self._make_prompt(
                min(safe_ctx, self.model_cfg.max_seq_len // 2), device, vocab
            )

            for _ in range(cfg.warmup_steps):
                model.generate(
                    decode_prompt,
                    max_new_tokens=cfg.decode_new_tokens,
                    n_loops=cfg.decode_n_loops,
                )

            gc.collect()
            _sync()
            t0 = time.perf_counter()
            model.generate(
                decode_prompt,
                max_new_tokens=cfg.decode_new_tokens,
                n_loops=cfg.decode_n_loops,
            )
            _sync()
            elapsed = time.perf_counter() - t0

            result["decode_toks_per_sec"] = cfg.decode_new_tokens / elapsed if elapsed > 0 else 0.0
            result["decode_latency_ms"]   = elapsed * 1_000
            result["decode_new_tokens"]   = cfg.decode_new_tokens
        else:
            # Context too long for decode test — record skipped reason
            result["decode_toks_per_sec"] = None
            result["decode_skipped"]      = f"ctx_len {safe_ctx} too close to max_seq_len"

        result["ctx_len_requested"] = ctx_len
        result["ctx_len_used"]      = safe_ctx
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI — measure a saved checkpoint directly
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from anthos.main import Anthos, AnthosConfig
    from anthos.configs import get_training_config

    parser = argparse.ArgumentParser(description="Run telemetry on a saved checkpoint")
    parser.add_argument("checkpoint",  help="Path to .pt checkpoint file")
    parser.add_argument("--tier",      default="sft", help="Config tier (default: sft)")
    parser.add_argument("--hardware",  default="cpu_only", help="Hardware label")
    parser.add_argument("--step",      type=int, default=0)
    args = parser.parse_args()

    model_cfg, _ = get_training_config(args.tier)
    model        = Anthos(model_cfg)
    state        = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key in ("model", "model_state_dict"):
        if key in state:
            state = state[key]
            break
    model.load_state_dict(state, strict=False)
    model.eval()

    harness = TelemetryHarness(model_cfg, hardware=args.hardware)
    results = harness.record(model, tokenizer=None, ckpt_path=args.checkpoint, step=args.step)

    sidecar = Path(args.checkpoint).with_suffix("").with_suffix(".telemetry.json")
    print(f"Telemetry written → {sidecar}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
