"""
bench_decode.py — measure prefill + decode throughput for Anthos on the current device.

Run this on your Mac mini M4 to get a before/after number for the KV-cache,
MoE-sync, and RoPE fixes. To get a real "before/after", run it once on your
pre-fix commit and once on the current one (git stash / git checkout the old
main.py, or just diff the numbers against whatever you remember pre-fix
generation feeling like).

Usage:
    python3 bench_decode.py --tier smoke        # tiny model, fast sanity check
    python3 bench_decode.py --tier proof         # 44M-param scale, closer to real use
    python3 bench_decode.py --prompt-len 128 --gen-len 256

Reports:
    - prefill time + tokens/sec
    - decode time + tokens/sec (steady-state, excludes prefill)
    - peak memory (if available on this backend)
"""

import argparse
import time

import torch

from anthos import Anthos, AnthosConfig


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Force the device to finish pending work before timing — required for
    accurate wall-clock measurement on both MPS and CUDA, which are async."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_config(tier: str) -> AnthosConfig:
    if tier == "smoke":
        return AnthosConfig(
            vocab_size=1024, dim=256, n_heads=4, n_kv_heads=2,
            max_seq_len=512, max_loop_iters=4, n_thought_tokens=8,
            attn_type="gqa", n_experts=8, expert_dim=256,
            prelude_layers=1, coda_layers=1,
        )
    if tier == "proof":
        return AnthosConfig(
            vocab_size=50262, dim=512, n_heads=8, n_kv_heads=4,
            max_seq_len=1024, max_loop_iters=8, n_thought_tokens=16,
            attn_type="gqa", n_experts=16, expert_dim=512,
            prelude_layers=2, coda_layers=2,
        )
    raise ValueError(f"unknown tier: {tier}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["smoke", "proof"], default="smoke")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--n-loops", type=int, default=None,
                     help="override recurrent loop count; defaults to cfg.max_loop_iters")
    ap.add_argument("--warmup", type=int, default=1,
                     help="warmup generate() calls before timing (MPS/CUDA kernel compile)")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    cfg = build_config(args.tier)
    n_loops = args.n_loops or cfg.max_loop_iters
    model = Anthos(cfg).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"tier={args.tier}  params={n_params:,}  n_loops={n_loops}  "
          f"n_thought_tokens={cfg.n_thought_tokens}  attn_type={cfg.attn_type}")

    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)

    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=8, n_loops=n_loops)
        sync(device)

    # ── Prefill timing ──────────────────────────────────────────────────────
    kv_cache = {}
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(prompt, n_loops=n_loops, kv_cache=kv_cache, start_pos=0)
    sync(device)
    prefill_s = time.perf_counter() - t0
    prefill_tok_s = args.prompt_len / prefill_s

    print(f"\nprefill: {prefill_s*1000:.1f} ms  ({prefill_tok_s:.1f} tok/s over {args.prompt_len} tokens)")

    # ── Decode timing (steady state) ────────────────────────────────────────
    next_token = torch.multinomial(
        torch.softmax(logits[:, -1, :], dim=-1), num_samples=1
    )
    generated = torch.cat([prompt, next_token], dim=1)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(args.gen_len):
            start_pos = generated.shape[1] - 1
            logits = model(generated[:, -1:], n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
    sync(device)
    decode_s = time.perf_counter() - t0
    decode_tok_s = args.gen_len / decode_s

    print(f"decode:  {decode_s*1000:.1f} ms  ({decode_tok_s:.1f} tok/s over {args.gen_len} tokens, "
          f"{decode_s/args.gen_len*1000:.2f} ms/token)")

    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        print(f"mps allocated memory: {peak_mb:.1f} MB")
    elif device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"cuda peak memory: {peak_mb:.1f} MB")

    print(f"\nsummary: prefill {prefill_tok_s:.1f} tok/s | decode {decode_tok_s:.1f} tok/s | "
          f"{decode_s/args.gen_len*1000:.2f} ms/token")


if __name__ == "__main__":
    main()
