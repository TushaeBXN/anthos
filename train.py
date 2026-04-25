"""
Anthos — Sovereign Training Script (Mansa Edition)
Think in Streams.
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.data    import get_dataloader, get_instruct_dataloader, get_chat_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# MANSA CONFIGURATION (HARD-CODED FOR STABILITY)
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS = 10000       # Your new 10k goal
MAX_LR = 1.0e-4         # Lowered from 3.0e-4 to stop the stuttering
MIN_LR = 1.0e-5         # Steady floor
WARMUP_STEPS = 2000     # Gentler start
PHASE1_STEPS = 3000     # First 3k steps focus on simple language
PHASE1_LOOPS = 4        # Start simple (4 loops)
PHASE2_LOOPS = 16       # Scale to complex thinking (16 loops)
SEQ_LEN = 1024          # Fixed to prevent indexing errors

# ─────────────────────────────────────────────────────────────────────────────
# Logic Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

def get_n_loops(step: int) -> int:
    return PHASE1_LOOPS if step < PHASE1_STEPS else PHASE2_LOOPS

@torch.no_grad()
def generate_samples(model: Anthos, device: str, n_loops: int, n_samples: int = 3) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = SEQ_LEN
    model.eval()
    prompts = ["Once upon a time", "The small robot looked at", "In a world where"][:n_samples]
    samples = []
    for prompt in prompts:
        ids = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(ids, max_new_tokens=80, n_loops=n_loops, temperature=0.6, top_k=40)
        samples.append(tok.decode(out[0].tolist()))
    model.train()
    return samples

def save_checkpoint(path: Path, model: Anthos, optimizer: AdamW, step: int, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(tier: str = "proof", resume: str | None = None):
    model_cfg, train_cfg = get_training_config(tier)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/mansa_sovereign")

    model = Anthos(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"  Anthos — Sovereign Training (Mansa Edition)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Max Steps:  {MAX_STEPS:,} | Warmup: {WARMUP_STEPS}")
    print(f"  Max LR:     {MAX_LR} | Loops: {PHASE1_LOOPS}->{PHASE2_LOOPS}")
    print(f"{'─'*60}\n")

    optimizer = AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), fused=True)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    is_sft = (tier == "sft")
    if is_sft:
        loader = get_chat_dataloader(
            seq_len        = train_cfg.seq_len,
            batch_size     = train_cfg.batch_size,
            num_workers    = 4,
            tokenizer_path = "data/anthos_tokenizer",
        )
    else:
        loader = get_dataloader(
            dataset_name = train_cfg.dataset,
            split        = "train",
            seq_len      = SEQ_LEN,
            batch_size   = train_cfg.batch_size,
            num_workers  = 4,
        )
    data_iter = iter(loader)

    model.train()
    step = start_step
    loss_accum = aux_accum = 0.0
    t0 = time.time()

    while step < MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        n_loops = get_n_loops(step)

        optimizer.zero_grad()
        for _ in range(train_cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if is_sft:
                input_ids = batch[0].to(device)
                labels    = batch[1].to(device)
            else:
                batch     = batch.to(device)
                input_ids = batch[:, :-1]
                labels    = batch[:, 1:]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)
                ce = F.cross_entropy(logits.reshape(-1, model_cfg.vocab_size), labels.reshape(-1))
                loss = (ce + aux) / train_cfg.grad_accum

            loss.backward()
            loss_accum += ce.item()
            aux_accum += aux.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 100 == 0:
            t1 = time.time()
            avg_loss, avg_aux = loss_accum / (100 * train_cfg.grad_accum), aux_accum / (100 * train_cfg.grad_accum)
            tok_sec = (100 * train_cfg.batch_size * train_cfg.grad_accum * SEQ_LEN) / (t1 - t0)
            print(f"step {step:6d} | loss {avg_loss:.4f} | ponder {avg_aux:.5f} | loops {n_loops} | lr {lr:.2e} | {tok_sec:,.0f} tok/s")
            loss_accum = aux_accum = 0.0
            t0 = t1

        if step % 1000 == 0:
            print("\n── Sample outputs ─────────────────────────────────────")
            for sample in generate_samples(model, device, n_loops):
                print(f"  {sample[:200]}\n")
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, avg_loss)

    print(f"\n✓ Sovereign Training complete — {step} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",   type=str, default="proof",
                        choices=["smoke", "proof", "research", "ethnic", "instruct", "sft"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(tier=args.tier, resume=args.resume)
