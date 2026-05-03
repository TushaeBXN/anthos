"""
anthos/serve.py — RunPod Serverless Worker

Deploys Anthos as a RunPod serverless endpoint with an OpenAI-compatible
/v1/completions interface.  Integrates the full memory stack (Layer 1 + 2)
and FP8 quantization for efficient inference.

Architecture at serve time:
    Request
        ↓
    ExternalMemoryReader (Engram retrieval → prepend to prompt)
        ↓
    Anthos.generate() (with MemoryBank active for thought stream)
        ↓
    Response (OpenAI-compatible JSON)

RunPod deployment:
  1. Build Docker image from Dockerfile.serve (see below)
  2. Push to your container registry
  3. Create RunPod Serverless Endpoint pointing to your image
  4. Set env vars: HF_TOKEN, MODEL_CHECKPOINT, ENGRAM_WING (optional)

Environment variables:
    MODEL_CHECKPOINT     path or HF hub id of the checkpoint (required)
    MODEL_CONFIG         anthos config variant: "1b" | "3b" | "10b" (default: auto)
    QUANT_MODE           "auto" | "fp8" | "bf16" | "fp32" (default: auto)
    N_LOOPS              recurrent loop depth for inference (default: 12)
    ENGRAM_WING          Engram wing for memory retrieval (optional)
    MAX_MEMORY_TOKENS    max memory prefix tokens (default: 170)
    HF_TOKEN             HuggingFace token if needed

Local testing:
    python -m anthos.serve --test

OpenAI-compatible request format:
    POST /run  (RunPod) or  /v1/completions  (OpenAI-compat)
    {
        "input": {
            "prompt": "What is the Anthos architecture?",
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_k": 50,
            "n_loops": 12,
            "memory_query": "transformer architecture"   // optional Engram query
        }
    }
"""

from __future__ import annotations

import os
import sys
import time
import json
import base64
import logging
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [serve] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — avoid loading torch at import time for fast cold starts
# ─────────────────────────────────────────────────────────────────────────────

_model  = None
_tokenizer = None
_reader = None   # ExternalMemoryReader


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load model, tokenizer, and memory reader on first request."""
    global _model, _tokenizer, _reader

    if _model is not None:
        return

    from anthos.main   import Anthos, anthos_1b, anthos_3b, anthos_10b
    from anthos.quant  import load_quantized, QuantConfig, detect_device
    from anthos.memory import ExternalMemoryReader

    ckpt_path   = os.environ.get("MODEL_CHECKPOINT", "")
    config_name = os.environ.get("MODEL_CONFIG", "auto")
    quant_mode  = os.environ.get("QUANT_MODE", "auto")
    engram_wing = os.environ.get("ENGRAM_WING", "anthos")
    tok_path    = os.environ.get("TOKENIZER_PATH", "data/anthos_tokenizer")

    if not ckpt_path:
        raise RuntimeError("MODEL_CHECKPOINT environment variable is required")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(tok_path)
        log.info(f"Tokenizer loaded: vocab_size={len(_tokenizer)}")
    except Exception as e:
        log.warning(f"Could not load tokenizer from {tok_path}: {e}")
        _tokenizer = None

    # Detect config from checkpoint if "auto"
    device = detect_device()
    if config_name == "auto":
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            embed_shape = ckpt["model"]["embed.weight"].shape
            dim = embed_shape[1]
            cfg_map = {128: anthos_1b, 2048: anthos_1b, 3072: anthos_3b, 4096: anthos_10b}
            cfg_fn  = cfg_map.get(dim, anthos_1b)
            cfg     = cfg_fn()
            cfg.vocab_size = embed_shape[0]
            log.info(f"Auto-detected config: dim={dim}, vocab_size={cfg.vocab_size}")
        except Exception:
            cfg = anthos_1b()
            log.warning("Could not auto-detect config, defaulting to anthos_1b")
    else:
        cfg_fn = {"1b": anthos_1b, "3b": anthos_3b, "10b": anthos_10b}[config_name]
        cfg    = cfg_fn()

    # Load with quantization
    model = Anthos(cfg)
    quant_cfg = QuantConfig(mode=quant_mode)
    _model = load_quantized(model, ckpt_path, device=device, quant_cfg=quant_cfg)
    log.info(f"Model loaded on {device}")

    # Memory reader (optional — graceful if Engram not installed)
    if _tokenizer is not None:
        try:
            _reader = ExternalMemoryReader(
                tokenizer         = _tokenizer,
                engram_wing       = engram_wing,
                max_memory_tokens = int(os.environ.get("MAX_MEMORY_TOKENS", 170)),
            )
            log.info(f"Memory reader ready (Engram: {_reader._engram_available})")
        except Exception as e:
            log.warning(f"Memory reader init failed: {e}")
            _reader = None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(job_input: dict) -> dict:
    """
    Core inference function.

    Input keys:
        prompt          str   — required
        max_new_tokens  int   — default 256
        temperature     float — default 0.7
        top_k           int   — default 50
        n_loops         int   — default from env or 12
        memory_query    str   — optional Engram retrieval query

    Returns OpenAI-compatible completion dict.
    """
    _load_model()

    prompt         = job_input.get("prompt", "")
    max_new_tokens = int(job_input.get("max_new_tokens", 256))
    temperature    = float(job_input.get("temperature", 0.7))
    top_k          = int(job_input.get("top_k", 50))
    n_loops        = int(job_input.get("n_loops", os.environ.get("N_LOOPS", 12)))
    memory_query   = job_input.get("memory_query", None)

    if not prompt:
        return {"error": "prompt is required", "status": "FAILED"}

    if _tokenizer is None:
        return {"error": "tokenizer not loaded", "status": "FAILED"}

    t0 = time.time()

    # Tokenise prompt
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(
        next(_model.parameters()).device
    )

    # Prepend Engram memories if requested
    if memory_query and _reader is not None:
        input_ids = _reader.prepend_memories(input_ids, query=memory_query)
        log.info(f"Memory prefix applied (query: '{memory_query[:40]}')")

    # Generate
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens = max_new_tokens,
            n_loops        = n_loops,
            temperature    = temperature,
            top_k          = top_k,
        )

    # Decode completion only (strip prompt)
    completion_ids = output_ids[0, input_ids.shape[1]:]
    completion     = _tokenizer.decode(completion_ids, skip_special_tokens=True)

    elapsed = time.time() - t0
    n_tokens = completion_ids.shape[0]
    tok_per_s = n_tokens / max(elapsed, 1e-6)

    log.info(f"Generated {n_tokens} tokens in {elapsed:.2f}s ({tok_per_s:.1f} tok/s)")

    return {
        "id":      f"anthos-{int(t0)}",
        "object":  "text_completion",
        "model":   "anthos",
        "choices": [
            {
                "text":          completion,
                "index":         0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     input_ids.shape[1],
            "completion_tokens": n_tokens,
            "total_tokens":      input_ids.shape[1] + n_tokens,
        },
        "meta": {
            "elapsed_s":  round(elapsed, 3),
            "tok_per_s":  round(tok_per_s, 1),
            "n_loops":    n_loops,
        },
        "status": "COMPLETED",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RunPod handler
# ─────────────────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    RunPod calls this function with:
        job = {"id": "...", "input": {...}}

    Returns the result dict which RunPod wraps in standard response envelope.
    """
    try:
        job_input = job.get("input", {})
        return _run_inference(job_input)
    except Exception as e:
        log.exception("Inference error")
        return {"error": str(e), "status": "FAILED"}


# ─────────────────────────────────────────────────────────────────────────────
# Local test runner
# ─────────────────────────────────────────────────────────────────────────────

def _local_test():
    """Quick sanity check — run locally before deploying."""
    test_job = {
        "input": {
            "prompt":         "Explain the Anthos thought token architecture.",
            "max_new_tokens": 64,
            "temperature":    0.7,
            "n_loops":        8,
        }
    }
    print("Running local test...")
    result = handler(test_job)
    print(json.dumps(result, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        _local_test()
    else:
        # RunPod serverless entrypoint
        try:
            import runpod
            runpod.serverless.start({"handler": handler})
        except ImportError:
            print("runpod package not installed. Install with: pip install runpod")
            print("Running local test instead...")
            _local_test()
