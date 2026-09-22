"""
bench_coherence.py — qualitative check for the thought-token RoPE fix.

The KV-cache / MoE-sync fixes show up in tokens/sec. The RoPE position-0 fix
does NOT — it's a correctness fix, not a speed fix. The bug was: thought
tokens were supposed to get a fixed, position-independent RoPE reference
frame ("position 0" always), but during decode they were drifting to
whatever the current absolute position was. Since thought tokens shape the
non-causal working-memory stream that every sequence token attends to, a
drifting reference frame should show up as accumulating incoherence the
longer generation runs — not as a crash, just gradually worse text.

This script generates the SAME prompt, with the SAME seed, for a long-ish
sequence (default 200 tokens — long enough for drift to compound), and
prints raw output so you can read it side by side.

Usage:
    # On current (fixed) code:
    python3 bench_coherence.py --tag post-fix --out post_fix.txt

    # Then: git stash (back to pre-fix main.py), same benchmark script kept
    python3 bench_coherence.py --tag pre-fix --out pre_fix.txt

    # Then: git stash pop (restore fixes), diff the two output files
    diff pre_fix.txt post_fix.txt

Note: this uses random weights (no trained checkpoint), so don't expect
either output to be fluent English — the point isn't fluency, it's whether
the post-fix run stays more STABLE (less repetition collapse, less token-
level degeneration) over the back half of the generation than the pre-fix
run, since that's the signature of a drifting attention reference.

If you have a real checkpoint (--checkpoint path), pass it — the effect
will be far more visible with trained weights than random ones, since a
trained model has actually learned to rely on a stable thought-token frame.
"""

import argparse

import torch

from anthos import Anthos, AnthosConfig


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


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def repetition_stats(ids: list[int], window: int = 20) -> dict:
    """Cheap degeneration signal: fraction of repeated tokens in sliding windows,
    computed separately for the first half and second half of generation.
    A drifting reference frame should show MORE repetition in the second half
    (as position keeps moving further from what the model was ever trained near)."""
    def frac_repeated(chunk: list[int]) -> float:
        if len(chunk) < 2:
            return 0.0
        repeats = sum(1 for i in range(1, len(chunk)) if chunk[i] == chunk[i - 1])
        return repeats / (len(chunk) - 1)

    mid = len(ids) // 2
    return {
        "first_half_repeat_frac": frac_repeated(ids[:mid]),
        "second_half_repeat_frac": frac_repeated(ids[mid:]),
        "unique_tokens": len(set(ids)),
        "total_tokens": len(ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["smoke", "proof"], default="smoke")
    ap.add_argument("--tag", default="run", help="label for this run, printed in output")
    ap.add_argument("--out", default=None, help="write raw token ids to this file for diffing")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--gen-len", type=int, default=200,
                     help="long enough for position drift to compound — don't go below ~150")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-loops", type=int, default=None)
    ap.add_argument("--checkpoint", default=None,
                     help="path to a trained checkpoint; without one, weights are random init "
                          "and only the repetition-stat signal is meaningful, not fluency")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)

    cfg = build_config(args.tier)
    n_loops = args.n_loops or cfg.max_loop_iters
    model = Anthos(cfg).to(device).eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: no --checkpoint given, using random init weights. "
              "Fluency will look like noise either way — only the repetition-fraction "
              "delta between first half / second half, and between runs, is meaningful.")

    torch.manual_seed(args.seed)  # re-seed so prompt + sampling are identical across pre/post runs
    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=args.gen_len, n_loops=n_loops)

    ids = generated[0].tolist()
    stats = repetition_stats(ids[args.prompt_len:])  # exclude prompt from repetition stats

    print(f"\n=== {args.tag} ===")
    print(f"tier={args.tier}  n_loops={n_loops}  seed={args.seed}  gen_len={args.gen_len}")
    print(f"first-half repeat fraction:  {stats['first_half_repeat_frac']:.3f}")
    print(f"second-half repeat fraction: {stats['second_half_repeat_frac']:.3f}")
    print(f"delta (second - first):      {stats['second_half_repeat_frac'] - stats['first_half_repeat_frac']:+.3f}")
    print(f"unique tokens used: {stats['unique_tokens']} / {stats['total_tokens']}")
    print(f"\ntoken ids: {ids}")

    if args.out:
        with open(args.out, "w") as f:
            f.write(f"tag={args.tag}\n")
            f.write(f"first_half_repeat_frac={stats['first_half_repeat_frac']:.4f}\n")
            f.write(f"second_half_repeat_frac={stats['second_half_repeat_frac']:.4f}\n")
            f.write(f"delta={stats['second_half_repeat_frac'] - stats['first_half_repeat_frac']:+.4f}\n")
            f.write(f"ids={ids}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
