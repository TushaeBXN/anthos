"""
Anthos — Chat Interface
────────────────────────
Interactive conversation loop using Anthos special tokens.

Two modes:
  sft      — uses <|user|> / <|thought|> / <|assistant|> template (after SFT training)
  legacy   — uses Alpaca prompt format (instruct/smoke/ethnic checkpoints)

Usage:
    # SFT checkpoint (best):
    python3 examples/chat.py --checkpoint checkpoints/anthos-sft/final.pt --tier sft

    # Ethnic checkpoint (rough but works today):
    python3 examples/chat.py --checkpoint checkpoints/anthos-ethnic/final.pt --tier ethnic
"""

import argparse
import torch
from pathlib import Path

from anthos import Anthos
from anthos.configs import get_training_config

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,   default="checkpoints/anthos-sft/final.pt")
parser.add_argument("--tier",       type=str,   default="sft",
                    choices=["smoke", "ethnic", "proof", "instruct", "sft", "convo_smoke"])
parser.add_argument("--loops",      type=int,   default=16)
parser.add_argument("--max-tokens", type=int,   default=200)
parser.add_argument("--temp",       type=float, default=0.7)
parser.add_argument("--tokenizer",  type=str,   default="data/anthos_tokenizer")
args = parser.parse_args()

# ── Load tokenizer ────────────────────────────────────────────────────────────

from transformers import AutoTokenizer

sft_mode = (args.tier in ("sft", "convo_smoke"))
tok_path = args.tokenizer if (sft_mode and Path(args.tokenizer).exists()) else "gpt2"

print(f"\nLoading tokenizer from: {tok_path}")
tok = AutoTokenizer.from_pretrained(tok_path)
tok.model_max_length = 2048

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading Anthos from {args.checkpoint}...")
ckpt      = torch.load(args.checkpoint, map_location="cpu")
model_cfg, _ = get_training_config(args.tier)
model     = Anthos(model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

total = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total:,}")
print(f"Mode       : {'SFT chat' if sft_mode else 'Legacy Alpaca'}")
print(f"Loops      : {args.loops}")
print("\nType your message. Ctrl+C to quit.\n")
print("─" * 60)

# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are Anthos. You think carefully before you speak. "
    "You are helpful, honest, and direct."
)

def build_sft_prompt(user_msg: str) -> str:
    """Anthos native chat format with thought stream."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_msg.strip()}<|end|>\n"
        f"<|thought|>\n<|end|>\n"   # model fills the thought stream
        f"<|assistant|>\n"
    )

def build_legacy_prompt(user_msg: str) -> str:
    """Alpaca format for smoke/ethnic/instruct checkpoints."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{user_msg.strip()}\n\n"
        f"### Response:\n"
    )

STOP_TOKENS = ["<|user|>", "<|system|>", "### Instruction:", "\n\n\n"]

# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def respond(user_msg: str) -> str:
    prompt  = build_sft_prompt(user_msg) if sft_mode else build_legacy_prompt(user_msg)
    ids     = torch.tensor(
        tok.encode(prompt, add_special_tokens=False),
        dtype=torch.long
    ).unsqueeze(0)

    out = model.generate(
        ids,
        max_new_tokens = args.max_tokens,
        n_loops        = args.loops,
        temperature    = args.temp,
        top_k          = 50,
    )

    new_tokens = out[0, ids.shape[1]:]
    response   = tok.decode(new_tokens.tolist(), skip_special_tokens=True)

    for stop in STOP_TOKENS:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()

# ── Chat loop ─────────────────────────────────────────────────────────────────

try:
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        print("\nAnthos:", end=" ", flush=True)
        print(respond(user_input))
        print("─" * 60)

except KeyboardInterrupt:
    print("\n\nAnthos: Goodbye.")
