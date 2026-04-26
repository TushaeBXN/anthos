"""
Anthos — Checkpoint Migration: proof → sft

Proof tier:  vocab_size=50257, max_loop_iters=8
SFT tier:    vocab_size=50262, max_loop_iters=16

Run this ONCE on the pod between proof and SFT training:

    python3 migrate_checkpoint.py \
        --src  checkpoints/mansa_sovereign/step_001700.pt \
        --dst  checkpoints/mansa_sovereign/step_001700_sft_ready.pt
"""

import argparse
import torch

def migrate(src_path: str, dst_path: str):
    print(f"Loading: {src_path}")
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]

    # ── 1. Vocab: 50257 → 50262 (5 new special tokens) ──────────────────────
    OLD_V, NEW_V = 50257, 50262
    extra = NEW_V - OLD_V   # 5

    for key in ("embed.weight", "head.weight"):
        if key in sd:
            old = sd[key]                         # (OLD_V, dim)
            dim = old.shape[1]                    # auto-detect: 128 (smoke) or 512 (proof)
            new_rows = torch.zeros(extra, dim)
            torch.nn.init.normal_(new_rows, std=0.02)
            sd[key] = torch.cat([old, new_rows], dim=0)
            print(f"  {key}: {old.shape} → {sd[key].shape}")

    # ── 2. loop_embeds buffer: existing → 16 ────────────────────────────────
    NEW_L = 16
    key = "recurrent.loop_embeds"
    if key in sd:
        old   = sd[key]                            # (OLD_L, dim)
        OLD_L = old.shape[0]
        if OLD_L < NEW_L:
            pad = torch.zeros(NEW_L - OLD_L, old.shape[1])
            sd[key] = torch.cat([old, pad], dim=0)
            print(f"  {key}: {old.shape} → {sd[key].shape}")
        else:
            print(f"  {key}: already {old.shape}, no change needed")

    # ── 3. LoRA scale embedding: existing → 16 ───────────────────────────────
    key = "recurrent.lora.scale.weight"
    if key in sd:
        old   = sd[key]                            # (OLD_L, rank)
        OLD_L = old.shape[0]
        if OLD_L < NEW_L:
            pad = torch.ones(NEW_L - OLD_L, old.shape[1])
            sd[key] = torch.cat([old, pad], dim=0)
            print(f"  {key}: {old.shape} → {sd[key].shape}")
        else:
            print(f"  {key}: already {old.shape}, no change needed")

    # ── Save ──────────────────────────────────────────────────────────────────
    ckpt["model"] = sd
    ckpt["step"]  = 0   # Reset step count — SFT budget starts fresh
    torch.save(ckpt, dst_path)
    print(f"\n✓ Migrated checkpoint saved → {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source proof checkpoint")
    parser.add_argument("--dst", required=True, help="Destination SFT-ready checkpoint")
    args = parser.parse_args()
    migrate(args.src, args.dst)
