"""
generate_vector.py — Sovereign Rogue Persona Vector Factory
────────────────────────────────────────────────────────────
Reads data/persona_pairs.json, runs both sides through your trained
Anthos model, and saves the contrastive difference vector to
vectors/tars_rogue.pt.

Run once after training:
    python3 generate_vector.py

Then use it in generation:
    from anthos.steering import AnthosSteer
    steer = AnthosSteer(model, target="recurrent")
    steer.load_persona("vectors/tars_rogue.pt")
    steer.engage(strength=0.75)
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from anthos import Anthos, AnthosConfig
from anthos.configs import get_training_config


# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT   = "checkpoints/anthos-smoke/final.pt"
PAIRS_FILE   = "data/persona_pairs.json"
OUT_FILE     = "vectors/tars_rogue.pt"
HOOK_TARGET  = "recurrent"   # layer to extract activations from

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location="cpu")

model_cfg, _ = get_training_config("smoke")
model = Anthos(model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ── Hook to capture recurrent block output ────────────────────────────────────

captured: list[torch.Tensor] = []

def capture_hook(module, input, output):
    # RecurrentBlock returns (h_out, moe_aux, act_aux)
    # h_out shape: (B, T, D) — take last token of first batch item
    h_out = output[0]
    captured.append(h_out[0, -1].detach().clone())   # (D,)

layer = getattr(model, HOOK_TARGET)
handle = layer.register_forward_hook(capture_hook)

# ── Extract activations ───────────────────────────────────────────────────────

with open(PAIRS_FILE) as f:
    pairs = json.load(f)

pos_acts, neg_acts = [], []

with torch.no_grad():
    for pair in pairs:
        for key, text in [("pos", pair["pos"]), ("neg", pair["neg"])]:
            captured.clear()
            ids = torch.tensor(
                tokenizer.encode(text), dtype=torch.long
            ).unsqueeze(0)          # (1, T)

            _ = model(ids, n_loops=4)   # 4 loops is enough for extraction

            if captured:
                act = captured[0]   # (D,)
                if key == "pos":
                    pos_acts.append(act)
                else:
                    neg_acts.append(act)
            else:
                print(f"  [!] No activation captured for: {text[:40]}")

handle.remove()

# ── Calculate contrastive vector ──────────────────────────────────────────────

if not pos_acts or not neg_acts:
    raise RuntimeError("No activations collected — check your checkpoint path.")

pos_mean = torch.stack(pos_acts).mean(0)   # (D,)
neg_mean = torch.stack(neg_acts).mean(0)   # (D,)
vector   = pos_mean - neg_mean             # (D,)  the "TARS/Hacker direction"

# ── Save ──────────────────────────────────────────────────────────────────────

Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
torch.save(vector, OUT_FILE)

print(f"\n✓ Sovereign Rogue vector saved → {OUT_FILE}")
print(f"  Shape : {tuple(vector.shape)}")
print(f"  Norm  : {vector.norm().item():.4f}")
print(f"\nActivate with:")
print(f"  from anthos.steering import AnthosSteer")
print(f"  steer = AnthosSteer(model, target='recurrent')")
print(f"  steer.load_persona('{OUT_FILE}')")
print(f"  steer.engage(strength=0.75)")
