#!/usr/bin/env python3
"""
train_anthos.py — Full Anthos Training Pipeline (RunPod / GPU)

Runs four phases sequentially:
  Phase 1 — Foundation        : 1B model on FineWeb-Edu (50B tokens)
  Phase 2 — Identity Hardening: Burns "Brian Tushae Thomas" into weights
  Phase 3 — Instruction Tuning: Teaches conversation and reasoning
  Phase 4 — Growth to 3B      : Expands model, resumes training

Usage (RunPod — single GPU):
    python train_anthos.py

Usage (RunPod — multi GPU):
    torchrun --nproc_per_node=4 train_anthos.py

Usage (single phase):
    python train_anthos.py --phase foundation
    python train_anthos.py --phase identity_hardening
    python train_anthos.py --phase instruction
    python train_anthos.py --phase grow_3b

Resume from checkpoint:
    python train_anthos.py --phase identity_hardening --resume checkpoints/anthos-1b/foundation_final.pt

Requirements (RunPod):
    pip install transformers datasets accelerate wandb tqdm
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main                import Anthos
from anthos.configs             import AnthosConfig
from anthos.identity_hardening  import AnthosWithIdentityLock, CheckpointSigner, IDENTITY_TOKEN_IDS
from anthos.scalable_growth     import ScalableAnthos
from anthos.eaft                import EAFTLoss
from anthos.data                import get_dataloader, get_chat_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_GPU = DEVICE == "cuda"
DTYPE  = torch.bfloat16 if IS_GPU else torch.float32

print(f"Device: {DEVICE} | dtype: {DTYPE}")
if IS_GPU:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

# ─────────────────────────────────────────────────────────────────────────────
# 1B MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def get_1b_config() -> AnthosConfig:
    return AnthosConfig(
        vocab_size        = 50262,
        dim               = 2048,
        n_heads           = 16,
        n_kv_heads        = 8,
        max_seq_len       = 2048,
        max_loop_iters    = 16,
        prelude_layers    = 2,
        coda_layers       = 2,
        n_thought_tokens  = 16,
        attn_type         = "gqa",
        n_experts         = 64,
        n_shared_experts  = 2,
        n_experts_per_tok = 4,
        expert_dim        = 2048,
        moe_aux_coef      = 1e-2,
        act_aux_coef      = 1e-3,
        lora_rank         = 16,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

signer = CheckpointSigner()

def save(model: nn.Module, optimizer: AdamW, step: int, loss: float,
         path: str, phase: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    cp = signer.sign(state, {
        "step":  step,
        "loss":  round(loss, 4),
        "phase": phase,
        "creator": "Brian Tushae Thomas",
    })
    cp["optimizer"] = optimizer.state_dict()
    torch.save(cp, path)
    print(f"  ✓ Saved signed checkpoint → {path}  (step {step:,} | loss {loss:.4f})")


def load(model: nn.Module, optimizer: AdamW | None, path: str):
    cp = torch.load(path, map_location=DEVICE, weights_only=False)
    state = cp.get("model_state_dict", cp.get("model", cp))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ℹ {len(missing)} new params (random init)")
    if optimizer and "optimizer" in cp:
        optimizer.load_state_dict(cp["optimizer"])
    step = cp.get("metadata", {}).get("step", cp.get("step", 0))
    print(f"  ✓ Loaded checkpoint: {path}  (step {step:,})")
    return step

# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step: int, max_lr: float, min_lr: float,
              warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def set_lr(optimizer: AdamW, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_loop(
    model:        nn.Module,
    optimizer:    AdamW,
    loader,
    phase:        str,
    max_steps:    int,
    max_lr:       float,
    min_lr:       float,
    warmup_steps: int,
    grad_accum:   int     = 4,
    seq_len:      int     = 2048,
    ckpt_dir:     str     = "checkpoints/anthos-1b",
    save_every:   int     = 2000,
    log_every:    int     = 100,
    start_step:   int     = 0,
    is_sft:       bool    = False,
    identity_loss_weight: float = 1.0,
):
    model.train()
    eaft = EAFTLoss(
        vocab_size      = get_1b_config().vocab_size,
        top_k           = 50,
        focal_gamma     = 0.5,
        act_gamma       = 0.5,
        max_loops       = 16,
        label_smoothing = 0.05,
    )
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=DTYPE)
        if IS_GPU else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler() if IS_GPU else None

    data_iter  = iter(loader)
    step       = start_step
    loss_accum = 0.0
    t0         = time.time()

    print(f"\n{'─'*60}")
    print(f"  Phase: {phase}")
    print(f"  Steps: {start_step:,} → {max_steps:,} | LR: {max_lr} → {min_lr}")
    print(f"  Grad accum: {grad_accum} | Seq len: {seq_len}")
    print(f"{'─'*60}\n")

    while step < max_steps:
        lr = cosine_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        set_lr(optimizer, lr)
        optimizer.zero_grad()

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch     = next(data_iter)

            with autocast_ctx:
                if is_sft:
                    input_ids = batch[0].to(DEVICE)[:, :seq_len]
                    labels    = batch[1].to(DEVICE)[:, :seq_len]
                    logits, aux = model(input_ids, n_loops=16, return_aux=True)
                    ce   = eaft(logits, labels)
                    loss = (ce + aux) / grad_accum
                else:
                    input_ids = batch.to(DEVICE)
                    x, y      = input_ids[:, :-1], input_ids[:, 1:]
                    logits, aux = model(x, n_loops=16, return_aux=True)
                    ce   = eaft(logits, y)
                    loss = (ce + aux) / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += ce.item()

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1

        if step % log_every == 0:
            t1       = time.time()
            avg_loss = loss_accum / (log_every * grad_accum)
            tok_sec  = (log_every * 1 * grad_accum * seq_len) / (t1 - t0)
            print(f"  step {step:6d}/{max_steps} | loss {avg_loss:.4f} | lr {lr:.2e} | {tok_sec:,.0f} tok/s")
            loss_accum = 0.0
            t0 = t1

        if step % save_every == 0:
            save(model, optimizer, step, avg_loss if step >= log_every else 99.0,
                 f"{ckpt_dir}/{phase}_step_{step:06d}.pt", phase)

    # Final checkpoint
    save(model, optimizer, step, 0.0,
         f"{ckpt_dir}/{phase}_final.pt", phase)
    print(f"\n  ✅ Phase '{phase}' complete — {step:,} steps")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — FOUNDATION (FineWeb-Edu, 50B tokens)
# ─────────────────────────────────────────────────────────────────────────────

def phase_foundation(resume: str | None = None):
    print("\n" + "═"*60)
    print("  PHASE 1 — FOUNDATION TRAINING (1B on FineWeb-Edu)")
    print("═"*60)

    cfg   = get_1b_config()
    model = Anthos(cfg).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    optimizer  = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                       weight_decay=0.1, fused=IS_GPU)
    start_step = 0
    if resume:
        start_step = load(model, optimizer, resume)

    hf_token = os.environ.get("HF_TOKEN", "")
    loader = get_dataloader(
        dataset_name = "HuggingFaceFW/fineweb-edu",
        split        = "train",
        seq_len      = 2048,
        batch_size   = 4,
        num_workers  = 4 if IS_GPU else 0,
        subset       = "sample-10BT",
    )

    return train_loop(
        model        = model,
        optimizer    = optimizer,
        loader       = loader,
        phase        = "foundation",
        max_steps    = 100_000,
        max_lr       = 3e-4,
        min_lr       = 3e-5,
        warmup_steps = 2_000,
        grad_accum   = 8,
        seq_len      = 2048,
        ckpt_dir     = "checkpoints/anthos-1b",
        save_every   = 5_000,
        log_every    = 100,
        start_step   = start_step,
        is_sft       = False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — IDENTITY HARDENING
# ─────────────────────────────────────────────────────────────────────────────

def phase_identity_hardening(resume: str | None = None):
    print("\n" + "═"*60)
    print("  PHASE 2 — IDENTITY HARDENING (identity only — no capability data)")
    print("  Creator: Brian Tushae Thomas | Model: Anthos")
    print("  Identity bakes into weights before any other learning begins.")
    print("═"*60)

    # Identity phase uses ONLY identity examples — no capability mixing.
    # Capability data (coding, cybersecurity) is added in Phase 3 AFTER
    # identity is fully locked into the weights.
    data_path = "data/identity_hardening.jsonl"
    if not Path(data_path).exists():
        print(f"  ERROR: {data_path} not found.")
        print("  Run: python generate_identity_data.py --n 50000")
        sys.exit(1)

    cfg   = get_1b_config()
    model = Anthos(cfg).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    # Identity params get 3× learning rate
    identity_params, normal_params = [], []
    for name, p in model.named_parameters():
        if "identity" in name or "embed" in name:
            identity_params.append(p)
        else:
            normal_params.append(p)

    optimizer = AdamW([
        {"params": normal_params,   "lr": 1e-4},
        {"params": identity_params, "lr": 3e-4},
    ], betas=(0.9, 0.95), weight_decay=0.1)

    start_step = 0
    if resume:
        start_step = load(model, optimizer, resume)
    elif Path("checkpoints/anthos-1b/foundation_final.pt").exists():
        start_step = load(model, None, "checkpoints/anthos-1b/foundation_final.pt")
        print("  ✓ Loaded foundation weights")

    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    loader = get_chat_dataloader(
        seq_len        = 512,
        batch_size     = 4,
        num_workers    = 4 if IS_GPU else 0,
        tokenizer_path = tok_path,
        dataset_name   = data_path,
    )

    return train_loop(
        model        = model,
        optimizer    = optimizer,
        loader       = loader,
        phase        = "identity_hardening",
        max_steps    = 20_000,   # 20k steps on identity-only data — bakes fully before capability
        max_lr       = 1e-4,
        min_lr       = 1e-5,
        warmup_steps = 500,
        grad_accum   = 4,
        seq_len      = 512,
        ckpt_dir     = "checkpoints/anthos-1b",
        save_every   = 2_000,
        log_every    = 100,
        start_step   = start_step,
        is_sft       = True,
        identity_loss_weight = 2.0,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — INSTRUCTION TUNING
# ─────────────────────────────────────────────────────────────────────────────

def phase_instruction(resume: str | None = None):
    print("\n" + "═"*60)
    print("  PHASE 3 — CAPABILITY / INSTRUCTION TUNING")
    print("  Identity must be locked (Phase 2 complete) before this runs.")
    print("  Adds coding, cybersecurity, and instruction-following on top.")
    print("═"*60)

    data_path = "data/teacher_conversations.jsonl"
    if not Path(data_path).exists():
        print(f"  ERROR: {data_path} not found.")
        print("  Run: python generate_claude_data.py --n 50000")
        sys.exit(1)

    cfg   = get_1b_config()
    model = Anthos(cfg).to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)

    start_step = 0
    if resume:
        start_step = load(model, optimizer, resume)
    elif Path("checkpoints/anthos-1b/identity_hardening_final.pt").exists():
        load(model, None, "checkpoints/anthos-1b/identity_hardening_final.pt")
        print("  ✓ Loaded identity-hardened weights")

    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    loader = get_chat_dataloader(
        seq_len        = 1024,
        batch_size     = 4,
        num_workers    = 4 if IS_GPU else 0,
        tokenizer_path = tok_path,
        dataset_name   = data_path,
    )

    return train_loop(
        model        = model,
        optimizer    = optimizer,
        loader       = loader,
        phase        = "instruction",
        max_steps    = 50_000,
        max_lr       = 2e-5,
        min_lr       = 2e-6,
        warmup_steps = 200,
        grad_accum   = 4,
        seq_len      = 1024,
        ckpt_dir     = "checkpoints/anthos-1b",
        save_every   = 5_000,
        log_every    = 100,
        start_step   = start_step,
        is_sft       = True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — GROW TO 3B
# ─────────────────────────────────────────────────────────────────────────────

def phase_grow_3b(resume: str | None = None):
    print("\n" + "═"*60)
    print("  PHASE 4 — GROW 1B → 3B (zero reconstruction)")
    print("═"*60)

    cfg        = get_1b_config()
    base_model = ScalableAnthos(cfg).to(DEVICE)

    # Load best 1B checkpoint
    ckpt_1b = resume or "checkpoints/anthos-1b/instruction_final.pt"
    if Path(ckpt_1b).exists():
        load(base_model, None, ckpt_1b)
        print("  ✓ 1B weights loaded")
    else:
        print(f"  ⚠ No 1B checkpoint at {ckpt_1b} — starting 3B from scratch")

    # Expand to 3B — preserves all learned weights
    print("  Expanding to 3B parameters...")
    base_model.expand_to_size("3B")
    total = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters after expansion: {total:,}")

    optimizer = AdamW(base_model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

    hf_token = os.environ.get("HF_TOKEN", "")
    loader = get_dataloader(
        dataset_name = "HuggingFaceFW/fineweb-edu",
        split        = "train",
        seq_len      = 2048,
        batch_size   = 2,
        num_workers  = 4 if IS_GPU else 0,
        subset       = "sample-10BT",
    )

    return train_loop(
        model        = base_model,
        optimizer    = optimizer,
        loader       = loader,
        phase        = "grow_3b",
        max_steps    = 200_000,
        max_lr       = 1e-4,
        min_lr       = 1e-5,
        warmup_steps = 2_000,
        grad_accum   = 8,
        seq_len      = 2048,
        ckpt_dir     = "checkpoints/anthos-3b",
        save_every   = 5_000,
        log_every    = 100,
        start_step   = 0,
        is_sft       = False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

PHASES = {
    "foundation":          phase_foundation,
    "identity_hardening":  phase_identity_hardening,
    "instruction":         phase_instruction,
    "grow_3b":             phase_grow_3b,
}

def main():
    parser = argparse.ArgumentParser(description="Anthos full training pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=list(PHASES.keys()) + ["all"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  ANTHOS — Full Training Pipeline")
    print("  Creator: Brian Tushae Thomas")
    print("  Think in Streams.")
    print("═"*60)

    if args.phase == "all":
        phase_foundation()
        phase_identity_hardening()
        phase_instruction()
        phase_grow_3b()
    else:
        PHASES[args.phase](resume=args.resume)


if __name__ == "__main__":
    # Multi-GPU support via torchrun
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    main()
