"""
serve.py — OpenAI-compatible HTTP inference server for Anthos

Provides a drop-in replacement for the OpenAI API, letting you use any
OpenAI-compatible client (openai-python, LangChain, LlamaIndex, etc.) with
a locally running Anthos model.

Implements:
  POST /v1/completions       — text completions
  POST /v1/chat/completions  — chat completions (applies ChatML template)
  GET  /v1/models            — model listing
  GET  /health               — liveness probe

Usage:
    # Single GPU
    python serve.py --checkpoint checkpoints/sft/step_000500.pt --tier sft

    # Custom port / host
    python serve.py --checkpoint checkpoints/proof/step_010000.pt \
                    --host 0.0.0.0 --port 8000 --n-loops 12

    # With torch.compile for max throughput
    python serve.py --checkpoint checkpoints/proof/step_010000.pt --compile

Then connect with any OpenAI client:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

    # Completions
    resp = client.completions.create(
        model="anthos",
        prompt="def fibonacci(n):",
        max_tokens=128,
        temperature=0.8,
    )

    # Chat
    resp = client.chat.completions.create(
        model="anthos",
        messages=[{"role": "user", "content": "Explain recursion."}],
        max_tokens=256,
    )

Requires:
    pip install fastapi uvicorn

For high-throughput production serving, consider vLLM with a custom model
registration (requires implementing vllm.model_executor.models.ModelRegistry
with Anthos's custom architecture).
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config


# ─────────────────────────────────────────────────────────────────────────────
# ChatML template
# ─────────────────────────────────────────────────────────────────────────────

def _apply_chatml(messages: list[dict]) -> str:
    """Convert OpenAI message list to ChatML prompt string."""
    parts = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_model:      Optional[Anthos] = None
_tokenizer                    = None
_device:     str              = "cpu"
_n_loops:    int              = 8
_model_name: str              = "anthos"


def _load_model(
    checkpoint_path: str,
    tier:            str  = "proof",
    device:          str  = "cpu",
    n_loops:         int  = 8,
    compile_model:   bool = False,
) -> None:
    global _model, _tokenizer, _device, _n_loops, _model_name

    _device  = device
    _n_loops = n_loops

    model_cfg, _ = get_training_config(tier)
    _model = Anthos(model_cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _model.load_state_dict(ckpt["model"], strict=False)
    _model.eval()

    if compile_model:
        try:
            _model = torch.compile(_model, mode="reduce-overhead", dynamic=True)
            print("  ✓ torch.compile enabled")
        except Exception as e:
            print(f"  ⚠ torch.compile skipped: {e}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    _tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    step = ckpt.get("step", 0)
    _model_name = f"anthos-{tier}-step{step}"
    print(f"  ✓ Model loaded: {_model_name} on {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate(
    prompt:        str,
    max_tokens:    int   = 128,
    temperature:   float = 1.0,
    top_p:         float = 1.0,
    top_k:         int   = 0,
    stop:          Optional[list[str]] = None,
    n_loops:       Optional[int]       = None,
) -> tuple[str, int, int]:
    """
    Generate text for a prompt string.
    Returns (completion_text, prompt_tokens, completion_tokens).
    """
    n_loops = n_loops or _n_loops

    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
    n_prompt  = input_ids.shape[1]

    _topk = top_k if top_k > 0 else 50

    out = _model.generate(
        input_ids,
        max_new_tokens = max_tokens,
        n_loops        = n_loops,
        temperature    = max(temperature, 1e-6),
        top_k          = _topk,
    )

    completion_ids  = out[0, n_prompt:]
    completion_text = _tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)

    if stop:
        for s in stop:
            if s in completion_text:
                completion_text = completion_text[:completion_text.index(s)]

    return completion_text, n_prompt, len(completion_ids)


@torch.no_grad()
def _generate_stream(
    prompt:      str,
    max_tokens:  int   = 128,
    temperature: float = 1.0,
    top_k:       int   = 0,
    stop:        Optional[list[str]] = None,
    n_loops:     Optional[int]       = None,
):
    """
    Token-by-token streaming generator.
    Yields decoded string fragments as each token is produced.
    """
    import torch.nn.functional as _F

    n_loops   = n_loops or _n_loops
    _topk     = top_k if top_k > 0 else 50
    kv_cache  = {}
    generated = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
    n_prompt  = generated.shape[1]
    accumulated = ""

    for step in range(max_tokens):
        ctx       = generated if step == 0 else generated[:, -1:]
        start_pos = 0 if step == 0 else generated.shape[1] - 1

        logits = _model(ctx, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if _topk > 0:
            v, _ = torch.topk(logits, _topk)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs      = _F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated  = torch.cat([generated, next_token], dim=1)

        token_text  = _tokenizer.decode([next_token.item()], skip_special_tokens=True)
        accumulated += token_text

        # Check stop sequences on accumulated text
        if stop:
            hit = False
            for s in stop:
                if s in accumulated:
                    token_text = accumulated[:accumulated.index(s)]
                    hit        = True
                    yield token_text
                    return
            if not hit:
                yield token_text
        else:
            yield token_text


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

def _build_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "fastapi and uvicorn are required for the inference server.\n"
            "Install: pip install fastapi uvicorn"
        )

    app = FastAPI(title="Anthos Inference Server", version="1.0")

    # ── Request / response models ─────────────────────────────────────────────

    class CompletionRequest(BaseModel):
        model:       str   = "anthos"
        prompt:      str
        max_tokens:  int   = 128
        temperature: float = 1.0
        top_p:       float = 1.0
        top_k:       int   = 0
        stop:        Optional[list[str]] = None
        n_loops:     Optional[int]       = None
        stream:      bool  = False

    class ChatMessage(BaseModel):
        role:    str
        content: str

    class ChatRequest(BaseModel):
        model:       str   = "anthos"
        messages:    list[ChatMessage]
        max_tokens:  int   = 256
        temperature: float = 1.0
        top_p:       float = 1.0
        top_k:       int   = 0
        stop:        Optional[list[str]] = None
        n_loops:     Optional[int]       = None
        stream:      bool  = False

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok", "model": _model_name}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id":       _model_name,
                "object":   "model",
                "created":  int(time.time()),
                "owned_by": "anthos-intelligence",
            }],
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if req.stream:
            cid = f"cmpl-{uuid.uuid4().hex[:12]}"

            def _sse():
                for fragment in _generate_stream(
                    req.prompt, req.max_tokens, req.temperature,
                    req.top_k, req.stop, req.n_loops,
                ):
                    chunk = json.dumps({
                        "id": cid, "object": "text_completion",
                        "model": _model_name,
                        "choices": [{"text": fragment, "index": 0, "finish_reason": None}],
                    })
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        t0 = time.time()
        text, n_prompt, n_comp = _generate(
            req.prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.n_loops,
        )
        return {
            "id":      f"cmpl-{uuid.uuid4().hex[:12]}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   _model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens":     n_prompt,
                "completion_tokens": n_comp,
                "total_tokens":      n_prompt + n_comp,
            },
            "_latency_ms": int((time.time() - t0) * 1000),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        prompt = _apply_chatml([m.dict() for m in req.messages])

        if req.stream:
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

            def _sse():
                for fragment in _generate_stream(
                    prompt, req.max_tokens, req.temperature,
                    req.top_k, req.stop, req.n_loops,
                ):
                    chunk = json.dumps({
                        "id": cid, "object": "chat.completion.chunk",
                        "model": _model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": fragment},
                            "finish_reason": None,
                        }],
                    })
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        t0     = time.time()
        text, n_prompt, n_comp = _generate(
            prompt, req.max_tokens, req.temperature,
            req.top_p, req.top_k, req.stop, req.n_loops,
        )
        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   _model_name,
            "choices": [{
                "index":   0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens":     n_prompt,
                "completion_tokens": n_comp,
                "total_tokens":      n_prompt + n_comp,
            },
            "_latency_ms": int((time.time() - t0) * 1000),
        }

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anthos OpenAI-compatible inference server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--tier",       default="proof",
                        help="Config tier used to build the model")
    parser.add_argument("--host",       default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose externally)")
    parser.add_argument("--port",       type=int, default=8000,
                        help="Bind port (default: 8000)")
    parser.add_argument("--n-loops",    type=int, default=8,
                        help="Recurrent loop depth for inference (default: 8)")
    parser.add_argument("--compile",    action="store_true",
                        help="torch.compile the model before serving (~2min startup, then faster)")
    parser.add_argument("--device",     default=None,
                        help="Device: cuda, cpu, mps (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'─'*60}")
    print(f"  Anthos Inference Server")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Tier       : {args.tier}")
    print(f"  Device     : {device}")
    print(f"  n_loops    : {args.n_loops}")
    print(f"  URL        : http://{args.host}:{args.port}/v1")
    print(f"{'─'*60}\n")

    _load_model(
        checkpoint_path = args.checkpoint,
        tier            = args.tier,
        device          = device,
        n_loops         = args.n_loops,
        compile_model   = args.compile,
    )

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required: pip install fastapi uvicorn")

    app = _build_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
