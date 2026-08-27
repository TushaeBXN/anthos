#!/usr/bin/env python3
"""
generate_code_steering.py — Code persona vector for the native Anthos arch

Follows the same pattern as generate_vector.py (contrastive activation extraction
from the recurrent block), specialized for code generation behavior.

Run after training the code tier:
    python3 generate_code_steering.py --checkpoint checkpoints/mansa_sovereign/step_XXXXXX.pt

Activate in generation:
    from anthos.steering import AnthosSteer
    steer = AnthosSteer(model, target="recurrent")
    steer.load_persona("vectors/code_persona.pt")
    steer.engage(strength=0.6)
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from anthos import Anthos
from anthos.configs import get_training_config

import argparse

# ── Contrastive pairs for code generation ─────────────────────────────────────
# Each pair: pos = the behavior we want, neg = the behavior we want to suppress.
# The vector is mean(pos_acts) - mean(neg_acts).

PAIRS = [
    {
        "pos": "Write clean Python code with all imports and error handling included.",
        "neg": "Sure! Here's a rough sketch of how you might approach this problem...",
    },
    {
        "pos": "```python\nimport os\n\ndef read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()\n```",
        "neg": "You could probably use the open() function to read the file, something like that.",
    },
    {
        "pos": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2",
        "neg": "I would implement this by first checking if the array is sorted, then maybe using a loop...",
    },
    {
        "pos": "import asyncio\nasync def fetch(url):\n    async with aiohttp.ClientSession() as sess:",
        "neg": "There are several ways you could do this. One approach would be to use requests, or alternatively aiohttp...",
    },
    {
        "pos": "class LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = {}",
        "neg": "An LRU cache is an interesting data structure! Let me explain what LRU means before we dive in...",
    },
    {
        "pos": "# Returns -1 if not found\nreturn bisect.bisect_left(arr, target)",
        "neg": "I hope this helps! Let me know if you need any clarification on any part of this solution.",
    },
    {
        "pos": "TypeError: unsupported operand type(s) for +: 'int' and 'str'\nFix: cast to int first: int(value) + count",
        "neg": "That's a great question about debugging! Errors can be tricky sometimes.",
    },
    {
        "pos": "def validate_email(email: str) -> bool:\n    import re\n    return bool(re.match(r'^[\\w.+-]+@[\\w-]+\\.[\\w.]+$', email))",
        "neg": "Email validation is a complex topic. There are many different approaches you could take...",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Anthos native arch checkpoint (.pt)")
    parser.add_argument("--tier",       type=str, default="smoke",
                        help="Config tier matching the checkpoint (smoke/proof/research)")
    parser.add_argument("--out",        type=str, default="vectors/code_persona.pt")
    parser.add_argument("--n-loops",    type=int, default=4)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_cfg, _ = get_training_config(args.tier)
    model = Anthos(model_cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Hook recurrent block output ───────────────────────────────────────────
    captured: list[torch.Tensor] = []

    def capture_hook(module, input, output):
        h_out = output[0]                           # (B, T, D)
        captured.append(h_out[0, -1].detach())      # last token, first batch item

    handle = model.recurrent.register_forward_hook(capture_hook)

    # ── Extract activations ───────────────────────────────────────────────────
    pos_acts, neg_acts = [], []

    with torch.no_grad():
        for pair in PAIRS:
            for key, text in [("pos", pair["pos"]), ("neg", pair["neg"])]:
                captured.clear()
                ids = torch.tensor(
                    tokenizer.encode(text, truncation=True, max_length=256),
                    dtype=torch.long,
                ).unsqueeze(0)
                model(ids, n_loops=args.n_loops)
                if captured:
                    if key == "pos":
                        pos_acts.append(captured[0])
                    else:
                        neg_acts.append(captured[0])

    handle.remove()

    if not pos_acts or not neg_acts:
        raise RuntimeError("No activations captured — check the checkpoint path.")

    pos_mean = torch.stack(pos_acts).mean(0)
    neg_mean = torch.stack(neg_acts).mean(0)
    vector   = pos_mean - neg_mean

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, args.out)

    print(f"\n✓ Code persona vector saved → {args.out}")
    print(f"  Shape: {tuple(vector.shape)}  Norm: {vector.norm().item():.4f}")
    print(f"\nActivate with:")
    print(f"  from anthos.steering import AnthosSteer")
    print(f"  steer = AnthosSteer(model, target='recurrent')")
    print(f"  steer.load_persona('{args.out}')")
    print(f"  steer.engage(strength=0.6)")


if __name__ == "__main__":
    main()
