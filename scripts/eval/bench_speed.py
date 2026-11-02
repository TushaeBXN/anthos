"""
bench_speed.py — prefill / decode speed for Anthos (prompt-only-thoughts mode).

Measures, on the device you run it on:
  * prefill time and tokens/s for a prompt of --prompt tokens
  * decode tokens/s, one token per step, using the KV cache
  * how both change with the number of recurrent loops (the latency/quality dial)
  * the frozen-thought shortcut (fast_thought_decode) against the reference path

Random weights are fine: speed does not depend on what the weights learned.

Usage:
    python3 bench_speed.py                        # small config, loops 2/4/8/16
    python3 bench_speed.py --dim 512 --prompt 256 --new 64 --loops 4 8 16
"""

import argparse
import statistics
import time

import torch

from anthos.main import Anthos, AnthosConfig


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build(args, fast: bool) -> Anthos:
    cfg = AnthosConfig(
        vocab_size=args.vocab, dim=args.dim, n_heads=args.heads, n_kv_heads=max(1, args.heads // 2),
        max_seq_len=args.prompt + args.new + 8, max_loop_iters=max(args.loops),
        prelude_layers=1, coda_layers=1, n_thought_tokens=args.thoughts, attn_type="gqa",
        n_experts=args.experts, n_shared_experts=1, n_experts_per_tok=2, expert_dim=args.dim // 2,
        thought_prefix_only=True, moe_fix_weight_order=True, fast_thought_decode=fast,
    )
    torch.manual_seed(0)
    return Anthos(cfg).to(args.device).eval()


@torch.no_grad()
def run_once(model: Anthos, args, n_loops: int):
    ids = torch.randint(0, args.vocab, (1, args.prompt), device=args.device)
    cache = {}
    sync(args.device)
    t0 = time.perf_counter()
    logits = model(ids, n_loops=n_loops, kv_cache=cache, prefix_len=args.prompt)
    sync(args.device)
    prefill = time.perf_counter() - t0

    tok = logits[:, -1:].argmax(-1)
    t0 = time.perf_counter()
    for i in range(args.new):
        logits = model(tok, n_loops=n_loops, kv_cache=cache, start_pos=args.prompt + i, prefix_len=args.prompt)
        tok = logits[:, -1:].argmax(-1)
    sync(args.device)
    decode = time.perf_counter() - t0
    return prefill, decode


def bench(model, args, n_loops):
    run_once(model, args, n_loops)                       # warm-up
    runs = [run_once(model, args, n_loops) for _ in range(args.repeats)]
    pre = statistics.median(r[0] for r in runs)
    dec = statistics.median(r[1] for r in runs)
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--thoughts", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=50262)
    ap.add_argument("--prompt", type=int, default=128, help="prompt length in tokens")
    ap.add_argument("--new", type=int, default=32, help="tokens to decode")
    ap.add_argument("--loops", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    ref, fast = build(args, False), build(args, True)
    params = sum(p.numel() for p in fast.parameters()) / 1e6
    print(f"device={args.device} params={params:.1f}M dim={args.dim} thoughts={args.thoughts} "
          f"prompt={args.prompt} new={args.new}\n")
    print(f"{'loops':>5} | {'prefill tok/s':>13} | {'decode tok/s (ref)':>18} | {'decode tok/s (fast)':>19} | {'speedup':>7}")
    print("-" * 74)
    for L in args.loops:
        pre_r, dec_r = bench(ref, args, L)
        pre_f, dec_f = bench(fast, args, L)
        print(f"{L:>5} | {args.prompt / pre_f:>13,.0f} | {args.new / dec_r:>18,.1f} | {args.new / dec_f:>19,.1f} | {dec_r / dec_f:>6.2f}x")


if __name__ == "__main__":
    main()
