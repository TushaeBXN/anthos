"""
Anthos — Minimal usage example
Think in Streams.

Loads your trained smoke checkpoint and generates stories —
first in default mode, then with the Sovereign Rogue persona engaged.
"""

import torch
from transformers import AutoTokenizer

from anthos import Anthos
from anthos.configs import get_training_config
from anthos.steering import AnthosSteer

# ── Load trained model ────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/anthos-smoke/final.pt"

print("Loading Anthos smoke checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu")

model_cfg, _ = get_training_config("smoke")
model = Anthos(model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}\n")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ── Helper ────────────────────────────────────────────────────────────────────

def generate(prompt: str, n_loops: int = 16, max_new: int = 80) -> str:
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, n_loops=n_loops,
                             temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())

# ── Default generation ────────────────────────────────────────────────────────

prompts = [
    "Once upon a time",
    "The small robot looked at",
    "In a world where",
]

print("=" * 60)
print("DEFAULT MODE")
print("=" * 60)
for p in prompts:
    print(f"\n> {p}")
    print(generate(p))

# ── Sovereign Rogue persona ───────────────────────────────────────────────────

steer = AnthosSteer(model, target="recurrent")
steer.load_persona("vectors/tars_rogue.pt")
steer.engage(strength=0.75)

print("\n" + "=" * 60)
print("SOVEREIGN ROGUE MODE  (strength=0.75)")
print("=" * 60)
for p in prompts:
    print(f"\n> {p}")
    print(generate(p))

steer.disengage()
