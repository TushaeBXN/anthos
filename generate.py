"""
generate.py — Native Python inference for the Anthos pretrained checkpoint

Runs the full architecture: 8 recurrent loops, all MoE experts, thought tokens.
No quality loss — this is the real model, not a GGUF approximation.

Usage:
    # Interactive REPL (completion mode)
    python generate.py

    # Single prompt
    python generate.py --prompt "The capital of France is"

    # Specify checkpoint explicitly
    python generate.py --checkpoint checkpoints/pod-anthos-pt-768/step_030000.pt
"""

import argparse
import sys
import torch
import tiktoken
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — auto-detects architecture from checkpoint weights
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    ckpt_path = Path(checkpoint_path)
    print(f"Loading {ckpt_path.name} ...", end=" ", flush=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd   = ckpt["model"]

    model_cfg, _ = get_training_config("proof")

    # Auto-detect architecture from weight shapes (same logic as export_gguf.py)
    if "embed.weight" in sd:
        model_cfg.vocab_size = sd["embed.weight"].shape[0]
        model_cfg.dim        = sd["embed.weight"].shape[1]
    if "recurrent.loop_embeds" in sd:
        model_cfg.max_loop_iters = sd["recurrent.loop_embeds"].shape[0]
    if "recurrent.block.ffn.routed_experts.0.gate.weight" in sd:
        model_cfg.expert_dim = sd["recurrent.block.ffn.routed_experts.0.gate.weight"].shape[0]
    if "freqs_cis" in sd:
        head_dim = sd["freqs_cis"].shape[1] * 2
        model_cfg.n_heads = model_cfg.dim // head_dim
    if "prelude.0.attn.wk.weight" in sd:
        kv_dim = sd["prelude.0.attn.wk.weight"].shape[0]
        head_dim = model_cfg.dim // model_cfg.n_heads
        model_cfg.n_kv_heads = kv_dim // head_dim

    model = Anthos(model_cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"\n  ⚠  {len(missing)} missing keys (expected for pretrained checkpoint)")

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"done  ({params/1e6:.1f}M params, {model_cfg.max_loop_iters} loops)")
    return model, model_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    enc,
    prompt:           str,
    max_new_tokens:   int   = 200,
    temperature:      float = 0.8,
    top_k:            int   = 50,
    top_p:            float = 0.95,
    rep_penalty:      float = 1.3,
    rep_window:       int   = 64,
    n_loops:          int   = 8,
    stream:           bool  = True,
) -> str:
    eot_id    = enc._special_tokens.get("<|endoftext|>", 100257)
    ids       = enc.encode(prompt)
    x         = torch.tensor([ids], dtype=torch.long)

    generated = []
    for _ in range(max_new_tokens):
        # Full-context forward each step — no KV-cache incremental path.
        # Slower but architecturally correct for Anthos's recurrent loops.
        logits      = model.forward(x, n_loops=n_loops)
        next_logits = logits[0, -1, :].clone()

        # Repetition penalty over the last rep_window tokens
        if rep_penalty != 1.0:
            recent = x[0, -rep_window:].tolist()
            for tok in set(recent):
                if next_logits[tok] > 0:
                    next_logits[tok] /= rep_penalty
                else:
                    next_logits[tok] *= rep_penalty

        next_logits = next_logits / max(temperature, 1e-8)

        # top-k
        if top_k > 0:
            kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[-1]
            next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))

        # top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

        probs   = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == eot_id:
            break

        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)

        if stream:
            print(enc.decode([next_id]), end="", flush=True)

    if stream:
        print()
    return enc.decode(generated)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anthos native inference")
    parser.add_argument("--checkpoint", default="checkpoints/pod-anthos-pt-768/step_030000.pt")
    parser.add_argument("--prompt",     default=None,  help="Single prompt (omit for interactive REPL)")
    parser.add_argument("--max-tokens", type=int,   default=200)
    parser.add_argument("--temp",       type=float, default=0.8)
    parser.add_argument("--top-k",      type=int,   default=50)
    parser.add_argument("--top-p",       type=float, default=0.95)
    parser.add_argument("--rep-penalty", type=float, default=1.3,
                        help="Repetition penalty (1.0 = off, 1.3 default)")
    parser.add_argument("--loops",      type=int,   default=8,
                        help="Recurrent iterations (default 8, max from checkpoint)")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint)
    enc        = tiktoken.get_encoding("cl100k_base")
    n_loops    = min(args.loops, cfg.max_loop_iters)

    print(f"  vocab={cfg.vocab_size}  dim={cfg.dim}  loops={n_loops}")
    print(f"  temp={args.temp}  top_k={args.top_k}  top_p={args.top_p}")

    gen_kwargs = dict(
        max_new_tokens = args.max_tokens,
        temperature    = args.temp,
        top_k          = args.top_k,
        top_p          = args.top_p,
        rep_penalty    = args.rep_penalty,
        n_loops        = n_loops,
        stream         = True,
    )

    if args.prompt:
        print(f"\n{args.prompt}", end="")
        generate(model, enc, args.prompt, **gen_kwargs)
        return

    # Interactive REPL
    print("\nAnthos pretrained — completion mode")
    print("Type a prompt and press Enter. Ctrl-C or 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        print(prompt, end="")
        generate(model, enc, prompt, **gen_kwargs)
        print()


if __name__ == "__main__":
    main()
