"""
Anthos — Chat / Instruction inference
──────────────────────────────────────
Interactive prompt loop using a fine-tuned instruct checkpoint.
Formats your input as an Alpaca instruction and generates a response.

Usage:
    # After training with --tier instruct:
    python3 examples/chat.py --checkpoint checkpoints/anthos-instruct/final.pt

    # Or try with the smoke checkpoint (responses will be rough — needs instruct training):
    python3 examples/chat.py --checkpoint checkpoints/anthos-smoke/final.pt
"""

import argparse
import torch
from transformers import AutoTokenizer

from anthos import Anthos
from anthos.configs import get_training_config

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="checkpoints/anthos-instruct/final.pt",
                    help="Path to checkpoint file")
parser.add_argument("--tier",       type=str, default="instruct",
                    choices=["smoke", "ethnic", "proof", "instruct"],
                    help="Config tier matching the checkpoint")
parser.add_argument("--loops",      type=int, default=16,
                    help="Number of recurrent loops at inference (more = deeper thinking)")
parser.add_argument("--max-tokens", type=int, default=200,
                    help="Max new tokens to generate per response")
parser.add_argument("--temp",       type=float, default=0.7,
                    help="Sampling temperature (lower = more focused)")
args = parser.parse_args()

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\nLoading Anthos from {args.checkpoint}...")
ckpt      = torch.load(args.checkpoint, map_location="cpu")
model_cfg, _ = get_training_config(args.tier)
model     = Anthos(model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 2048

total = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total:,}")
print(f"Loops      : {args.loops}")
print(f"Tier       : {args.tier}")
print("\nType your message. Ctrl+C to quit.\n")
print("─" * 60)

# ── Alpaca prompt template ────────────────────────────────────────────────────

SYSTEM = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

def build_prompt(instruction: str, context: str = "") -> str:
    if context.strip():
        return (
            f"{SYSTEM}\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{context.strip()}\n\n"
            f"### Response:\n"
        )
    return (
        f"{SYSTEM}\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n"
    )

# ── Chat loop ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def respond(instruction: str, context: str = "") -> str:
    prompt = build_prompt(instruction, context)
    ids    = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=torch.long
    ).unsqueeze(0)

    out = model.generate(
        ids,
        max_new_tokens = args.max_tokens,
        n_loops        = args.loops,
        temperature    = args.temp,
        top_k          = 50,
    )

    # Decode only the newly generated tokens
    new_tokens = out[0, ids.shape[1]:]
    response   = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    # Stop at the next ### marker if the model keeps going
    for stop in ["### Instruction:", "### Input:", "### Response:", "\n\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


try:
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        print("\nAnthos: ", end="", flush=True)
        reply = respond(user_input)
        print(reply)
        print("─" * 60)

except KeyboardInterrupt:
    print("\n\nAnthos: Goodbye.")
