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
# These defaults are for proof/smoke tier. SFT overrides are applied below
# inside train() after the tier is known.
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS = 10000       # Your new 10k goal  (SFT → 500)
MAX_LR = 1.0e-4         # Lowered from 3.0e-4 to stop the stuttering  (SFT → 3e-5)
MIN_LR = 1.0e-5         # Steady floor  (SFT → 3e-6)
WARMUP_STEPS = 2000     # Gentler start  (SFT → 50)
PHASE1_STEPS = 3000     # First 3k steps focus on simple language
PHASE1_LOOPS = 4        # Start simple (4 loops)
PHASE2_LOOPS = 16       # Scale to complex thinking (16 loops)
SEQ_LEN = 512           # Fixed to prevent indexing errors

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
    # CRITICAL FIX: Match the model's SEQ_LEN and enable truncation
    tok.model_max_length = SEQ_LEN
    model.eval()
    prompts = ["Once upon a time", "The small robot looked at", "In a world where"][:n_samples]
    samples = []
    for prompt in prompts:
        # Fixed: Ensure truncation is handled if a prompt is somehow too long
        enc = tok(prompt, truncation=True, max_length=SEQ_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=80, n_loops=n_loops, temperature=0.3, top_k=40)
        samples.append(tok.decode(out[0].tolist()))
    model.train()
    return samples

def save_checkpoint(path: Path, model: Anthos, optimizer: AdamW, step: int, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Security update: Using weights_only=True where possible in future, but for now standard save
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
    global MAX_STEPS, MAX_LR, MIN_LR, WARMUP_STEPS, SEQ_LEN

    model_cfg, train_cfg = get_training_config(tier)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path("checkpoints/mansa_sovereign")

    # ── Tier-specific overrides ───────────────────────────────────────────────
    if tier in ("sft", "instruct"):
        # The Lily Exorcism settings: gentle LR, short run, no catastrophic forgetting.
        MAX_STEPS    = 500
        MAX_LR       = 3e-5
        MIN_LR       = 3e-6
        WARMUP_STEPS = 50
    elif tier == "convo_smoke":
        # Patient CPU run — same philosophy as smoke but on conversations.
        MAX_STEPS    = 10_000
        MAX_LR       = 5e-5
        MIN_LR       = 5e-6
        WARMUP_STEPS = 500
        SEQ_LEN      = 256   # match convo_smoke model's max_seq_len

    model = Anthos(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"  Anthos — Sovereign Training (Mansa Edition)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Max Steps:  {MAX_STEPS:,} | Warmup: {WARMUP_STEPS}")
    print(f"  Max LR:     {MAX_LR} | Loops: {PHASE1_LOOPS}->{PHASE2_LOOPS}")
    print(f"{'─'*60}\n")

    # Fixed: Using standard AdamW for stability; fused=True only if CUDA is available
    use_fused = True if device == "cuda" else False
    optimizer = AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), fused=use_fused)

    start_step = 0
    if resume:
        # Security: weights_only=False kept for loading old pickles, but warnings expected
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if tier in ("sft", "instruct", "convo_smoke"):
            # Fresh optimizer — stale momentum from previous tier corrupts the new LR.
            start_step = 0
            print(f"  ✓ {tier} mode: optimizer state reset (fresh Adam at {MAX_LR})")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]

    is_sft = (tier in ("sft", "instruct", "convo_smoke"))
    if is_sft:
        # Use the Anthos tokenizer (50262 tokens) so special tokens are real IDs,
        # not split into unknown pieces by GPT-2's BPE.
        tok_path = "data/anthos_tokenizer"
        # convo_smoke: 5k high-quality FineTome conversations, loops indefinitely
        # sft/instruct: full SlimOrca (517k)
        if tier == "convo_smoke":
            # Use local teacher-generated data if available, else fall back to FineTome
            local_data = "data/teacher_conversations.jsonl"
            if Path(local_data).exists():
                dataset_name = local_data
                max_samples  = 0   # local file loops itself
                print(f"  ✓ Using local teacher data: {local_data}")
            else:
                dataset_name = "mlabonne/FineTome-100k"
                max_samples  = 5000
                print(f"  ✓ Using FineTome-100k (run generate_teacher_data.py for better results)")
        else:
            max_samples  = 0
            dataset_name = "Open-Orca/SlimOrca"
        # num_workers=0 for CPU runs (collate fn can't be pickled for multiprocessing)
        n_workers = 0 if tier == "convo_smoke" else 1
        loader = get_chat_dataloader(
            seq_len        = SEQ_LEN,
            batch_size     = train_cfg.batch_size,
            num_workers    = n_workers,
            tokenizer_path = tok_path,
            max_samples    = max_samples,
            dataset_name   = dataset_name,
        )
    else:
        loader = get_dataloader(
            dataset_name = train_cfg.dataset,
            split        = "train",
            seq_len      = SEQ_LEN,
            batch_size   = train_cfg.batch_size,
            num_workers  = 1, # Fixed: Forced to 1 to match single shard datasets
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
                input_ids = batch[0].to(device)[:, :SEQ_LEN]
                labels    = batch[1].to(device)[:, :SEQ_LEN]
            else:
                batch     = batch.to(device)
                # CRITICAL FIX: Ensure batch is truncated to SEQ_LEN to prevent indexing errors
                input_ids = batch[:, :SEQ_LEN-1]
                labels    = batch[:, 1:SEQ_LEN]

            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
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
                        choices=["smoke", "proof", "research", "ethnic", "instruct", "sft", "convo_smoke"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(tier=args.tier, resume=args.resume)
