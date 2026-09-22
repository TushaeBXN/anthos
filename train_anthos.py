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
from anthos.identity_hardening  import (
    AnthosWithIdentityLock, CheckpointSigner,
    IDENTITY_TOKEN_IDS, REQUIRED_IDENTITY_SEQUENCE,
)
from anthos.scalable_growth     import ScalableAnthos
from anthos.eaft                import EAFTLoss
from anthos.data                import get_dataloader, get_chat_dataloader

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

# Identity token row indices in the embedding table (rows 32000-32007)
IDENTITY_ROWS            = list(IDENTITY_TOKEN_IDS.values())  # [32000..32007]
IDENTITY_FREEZE_STEP_DEFAULT = 5000  # real run default; overridden by --freeze-at-step

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
IS_GPU = DEVICE in ("cuda", "mps")
DTYPE  = torch.bfloat16 if IS_GPU else torch.float32

print(f"Device: {DEVICE} | dtype: {DTYPE}")
if IS_GPU:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGS
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

def get_smoke_config() -> AnthosConfig:
    # Minimal config for local mechanism testing on M4.
    # vocab_size=32016 so identity token rows (32000-32007) exist in the embedding.
    # dim=256, 4 tiny experts, 2 loop iters — fits in <500MB; ~1s/step on MPS.
    return AnthosConfig(
        vocab_size        = 32016,
        dim               = 256,
        n_heads           = 4,
        n_kv_heads        = 2,
        max_seq_len       = 128,
        max_loop_iters    = 2,
        prelude_layers    = 1,
        coda_layers       = 1,
        n_thought_tokens  = 4,
        attn_type         = "gqa",
        n_experts         = 4,
        n_shared_experts  = 1,
        n_experts_per_tok = 2,
        expert_dim        = 128,
        moe_aux_coef      = 1e-2,
        act_aux_coef      = 1e-3,
        lora_rank         = 4,
    )

def get_config(tier: str) -> AnthosConfig:
    if tier == "smoke":
        return get_smoke_config()
    return get_1b_config()

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
    if "signature" in cp:
        signer.verify(cp)
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
    grad_accum:   int       = 4,
    seq_len:      int       = 2048,
    ckpt_dir:     str       = "checkpoints/anthos-1b",
    save_every:   int       = 2000,
    save_at:      set[int]  = frozenset(),
    log_every:    int       = 100,
    start_step:   int       = 0,
    is_sft:       bool      = False,
    identity_loss_weight: float = 1.0,
    freeze_at_step: int     = IDENTITY_FREEZE_STEP_DEFAULT,
    vocab_size:   int       = 50262,
    n_loops:      int       = 16,
):
    model.train()
    eaft = EAFTLoss(
        vocab_size      = vocab_size,
        top_k           = 50,
        focal_gamma     = 0.5,
        act_gamma       = 0.5,
        max_loops       = n_loops,
        label_smoothing = 0.05,
    )
    autocast_ctx = (
        torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
        if DEVICE == "cuda" else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

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
                # Route through base model when AnthosWithIdentityLock wrapper is present
                inner = model.base if hasattr(model, "base") else model
                if is_sft:
                    input_ids = batch[0].to(DEVICE)[:, :seq_len].clamp(max=vocab_size - 1)
                    labels    = batch[1].to(DEVICE)[:, :seq_len]
                    # clamp label IDs that aren't identity tokens and aren't -100
                    _lmask = (labels != -100) & ((labels < 32000) | (labels > 32007))
                    labels = labels.where(~_lmask, labels.clamp(max=vocab_size - 1))
                    logits, aux = inner(input_ids, n_loops=n_loops, return_aux=True)
                    loops_used = getattr(inner, "_last_loops_used", None)
                    ce   = eaft(logits, labels, loops_used=loops_used)
                    id_loss = torch.zeros(1, device=logits.device)
                    if hasattr(model, "identity_head"):
                        try:
                            hidden = inner.get_hidden_states()
                            id_logits = model.identity_head(hidden)
                            id_mask = (labels >= 32000) & (labels <= 32007)
                            if id_mask.any():
                                id_loss = F.cross_entropy(
                                    id_logits[id_mask], labels[id_mask] - 32000
                                )
                        except Exception:
                            pass
                    loss = (ce + aux + identity_loss_weight * id_loss) / grad_accum
                else:
                    input_ids = batch.to(DEVICE)
                    x, y      = input_ids[:, :-1], input_ids[:, 1:]
                    logits, aux = inner(x, n_loops=n_loops, return_aux=True)
                    loops_used = getattr(inner, "_last_loops_used", None)
                    ce   = eaft(logits, y, loops_used=loops_used)
                    loss = (ce + aux) / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += ce.item()

        # ── Identity gradient masking ────────────────────────────────────────
        # After IDENTITY_FREEZE_STEP optimizer steps, zero gradients for
        # identity embedding rows so Adam's momentum decays toward zero
        # instead of accumulating. The copy_-restore in
        # AnthosWithIdentityLock.forward acts as a secondary safety check
        # that undoes any residual weight-decay drift.
        if step >= freeze_at_step and hasattr(model, "base"):
            _embed = getattr(model.base, "embed", None)
            if _embed is not None and _embed.weight.grad is not None:
                _embed.weight.grad[IDENTITY_ROWS] = 0.0

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── Hard post-step restore ───────────────────────────────────────────
        # Grad masking causes momentum to decay over time (the slow fix).
        # This copy_ is the guaranteed immediate fix: even while momentum
        # is still non-zero and would drift the embedding, we restore
        # identity rows to the frozen snapshot after every optimizer.step().
        # Without this, checkpoints saved between now and full momentum
        # decay would contain drifted embedding values.
        if step >= freeze_at_step and hasattr(model, "identity_embedding_snapshot"):
            _embed = getattr(model.base, "embed", None)
            if _embed is not None:
                with torch.no_grad():
                    _embed.weight.data[IDENTITY_ROWS] = model.identity_embedding_snapshot

        step += 1

        # ── Momentum decay logging (every 500 steps after freeze) ───────────
        if (step >= freeze_at_step and step % 500 == 0
                and hasattr(model, "base")):
            _embed = getattr(model.base, "embed", None)
            if _embed is not None:
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p is _embed.weight and p in optimizer.state:
                            _st = optimizer.state[p]
                            if "exp_avg" in _st:
                                _mom = _st["exp_avg"][IDENTITY_ROWS].abs().mean().item()
                                print(f"  [identity freeze] step {step} | "
                                      f"embedding momentum (rows 32000-32007): {_mom:.6f}")

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
        elif step in save_at:
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
                       weight_decay=0.1, fused=(DEVICE == "cuda"))
    start_step = 0
    if resume:
        start_step = load(model, optimizer, resume)

    loader = get_dataloader(
        dataset_name = "HuggingFaceFW/fineweb-edu",
        split        = "train",
        seq_len      = 2048,
        batch_size   = 4,
        num_workers  = 4 if DEVICE == "cuda" else 0,
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

def phase_identity_hardening(
    resume:         str | None  = None,
    max_steps:      int | None  = None,
    save_at:        set[int]    = frozenset(),
    tier:           str         = "1b",
    freeze_at_step: int | None  = None,
):
    is_smoke = (tier == "smoke")
    cfg      = get_config(tier)
    ckpt_dir = f"checkpoints/{tier}"
    _freeze  = freeze_at_step if freeze_at_step is not None else IDENTITY_FREEZE_STEP_DEFAULT
    _max_steps = max_steps if max_steps is not None else (20_000 if not is_smoke else 20)
    _seq_len   = cfg.max_seq_len
    _grad_accum = 1 if is_smoke else 4
    _save_every = 100 if is_smoke else 2_000
    _log_every  = 1   if is_smoke else 100
    _n_loops    = cfg.max_loop_iters

    print("\n" + "═"*60)
    print("  PHASE 2 — IDENTITY HARDENING (identity only — no capability data)")
    print("  Creator: Brian Tushae Thomas | Model: Anthos")
    print(f"  Tier: {tier} | freeze_at_step: {_freeze}")
    print("  Identity bakes into weights before any other learning begins.")
    print("═"*60)

    data_path = "data/identity_hardening.jsonl"
    if not Path(data_path).exists():
        print(f"  ERROR: {data_path} not found.")
        print("  Run: python generate_identity_data.py --n 50000")
        sys.exit(1)

    _base = Anthos(cfg).to(DEVICE)
    model = AnthosWithIdentityLock(_base, hidden_dim=cfg.dim, freeze_after_steps=_freeze)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}  (base + identity_head)")

    # Identity params and embeddings get 3× learning rate.
    # model.named_parameters() includes both base.* and identity_head.* keys.
    #
    # TRADEOFF: weight_decay=0 applies to the ENTIRE embed tensor (all vocab rows),
    # not just the 8 identity rows. A row-level split is unsafe because Anthos ties
    # head.weight = embed.weight (main.py:884); splitting that tensor across param
    # groups would register it twice, doubling gradient accumulation. Accepting
    # zero decay on the full embedding is the correct tradeoff — the embed matrix
    # trains well without decay in practice, and the hard post-step restore already
    # prevents AdamW decay from drifting the 8 frozen identity rows.
    identity_params, normal_params = [], []
    seen_ids = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids:
            continue  # skip tied parameters already registered (head.weight = embed.weight)
        seen_ids.add(id(p))
        if "identity" in name or "embed" in name:
            identity_params.append(p)
        else:
            normal_params.append(p)

    optimizer = AdamW([
        {"params": normal_params,   "lr": 1e-4,  "weight_decay": 0.1},
        {"params": identity_params, "lr": 3e-4,  "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    start_step = 0
    if resume:
        start_step = load(model, optimizer, resume)
    elif not is_smoke and Path(f"checkpoints/1b/foundation_final.pt").exists():
        start_step = load(model, None, f"checkpoints/1b/foundation_final.pt")
        print("  ✓ Loaded foundation weights")

    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    loader = get_chat_dataloader(
        seq_len        = _seq_len,
        batch_size     = 1 if is_smoke else 4,
        num_workers    = 4 if DEVICE == "cuda" else 0,
        tokenizer_path = tok_path,
        dataset_name   = data_path,
    )

    return train_loop(
        model        = model,
        optimizer    = optimizer,
        loader       = loader,
        phase        = "identity_hardening",
        max_steps    = _max_steps,
        max_lr       = 1e-4,
        min_lr       = 1e-5,
        warmup_steps = max(1, _freeze // 10),
        grad_accum   = _grad_accum,
        seq_len      = _seq_len,
        ckpt_dir     = ckpt_dir,
        save_every   = _save_every,
        save_at      = save_at,
        log_every    = _log_every,
        start_step   = start_step,
        is_sft       = True,
        identity_loss_weight = 2.0,
        freeze_at_step = _freeze,
        vocab_size   = cfg.vocab_size,
        n_loops      = _n_loops,
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
    _base = Anthos(cfg).to(DEVICE)
    model = AnthosWithIdentityLock(_base, hidden_dim=cfg.dim, freeze_after_steps=IDENTITY_FREEZE_STEP_DEFAULT)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}  (base + identity_head)")

    # Mirror Phase 2's param groups so identity weights retain weight_decay=0
    # and gradient masking in train_loop fires at the same threshold.
    # seen_ids guard prevents double-registration of tied tensors (head.weight = embed.weight).
    identity_params, normal_params = [], []
    seen_ids = set()
    for name, p in model.named_parameters():
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        if "identity" in name or "embed" in name:
            identity_params.append(p)
        else:
            normal_params.append(p)
    optimizer = AdamW([
        {"params": normal_params,   "lr": 2e-5,  "weight_decay": 0.1},
        {"params": identity_params, "lr": 2e-6,  "weight_decay": 0.0},
    ], betas=(0.9, 0.95))

    start_step = 0
    _id_ckpt   = Path("checkpoints/anthos-1b/identity_hardening_final.pt")
    if resume:
        start_step = load(model, optimizer, resume)
    elif _id_ckpt.exists():
        _meta_cp  = torch.load(str(_id_ckpt), map_location="cpu", weights_only=False)
        _phase    = _meta_cp.get("metadata", {}).get("phase", "")
        if _phase not in ("identity_hardening", "instruction", "grow_3b"):
            raise RuntimeError(
                f"Phase 3 requires a checkpoint from phase 'identity_hardening' or later, "
                f"but {_id_ckpt.name} has phase='{_phase}'. Run Phase 2 first."
            )
        load(model, None, str(_id_ckpt))
        print("  ✓ Loaded identity-hardened weights")
    else:
        raise RuntimeError(
            f"Phase 3 (instruction) requires {_id_ckpt}. "
            "Run Phase 2 (identity_hardening) first, or pass --resume to load a specific checkpoint."
        )

    # Print trainable vs frozen param count so it's visible in logs
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params")

    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    loader = get_chat_dataloader(
        seq_len        = 1024,
        batch_size     = 4,
        num_workers    = 4 if DEVICE == "cuda" else 0,
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
    # NOTE: ScalableAnthos uses different parameter names than the main Anthos class
    # (token_embedding vs embed, recurrent_blocks vs recurrent, etc.). strict=False
    # means mismatched keys are silently skipped. Expect most params to re-initialize.
    # For full weight transfer, prefer exporting anthos_3b() from a trained Anthos model.
    ckpt_1b = resume or "checkpoints/anthos-1b/instruction_final.pt"
    if Path(ckpt_1b).exists():
        load(base_model, None, ckpt_1b)
        print("  ✓ 1B checkpoint loaded (see above for any new/missing param counts)")
    else:
        print(f"  ⚠ No 1B checkpoint at {ckpt_1b} — starting 3B from scratch")

    # Expand to 3B — preserves all learned weights
    print("  Expanding to 3B parameters...")
    base_model.expand_to_size("3B")
    total = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters after expansion: {total:,}")

    optimizer = AdamW(base_model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

    loader = get_dataloader(
        dataset_name = "HuggingFaceFW/fineweb-edu",
        split        = "train",
        seq_len      = 2048,
        batch_size   = 2,
        num_workers  = 4 if DEVICE == "cuda" else 0,
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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps for phase_identity_hardening "
                             "(useful for staged runs around the freeze boundary)")
    parser.add_argument("--save-at", type=str, default=None,
                        help="Comma-separated list of steps to force-save a checkpoint, "
                             "e.g. --save-at 5001,6000 (identity_hardening phase only)")
    parser.add_argument("--tier", type=str, default="1b", choices=["1b", "smoke"],
                        help="Model scale: '1b' (default, real run) or 'smoke' "
                             "(tiny local verification — dim=256, ~2s/step on M4)")
    parser.add_argument("--freeze-at-step", type=int, default=None,
                        help="Override AnthosWithIdentityLock freeze_after_steps and "
                             "train_loop's gradient-masking threshold. Default 5000 "
                             "for real runs; use --freeze-at-step 5 with --tier smoke "
                             "for fast local mechanism testing.")
    args = parser.parse_args()

    save_at_set: set[int] = set()
    if args.save_at:
        save_at_set = {int(s.strip()) for s in args.save_at.split(",")}

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
    elif args.phase == "identity_hardening":
        phase_identity_hardening(
            resume         = args.resume,
            max_steps      = args.max_steps,
            save_at        = save_at_set,
            tier           = args.tier,
            freeze_at_step = args.freeze_at_step,
        )
    else:
        PHASES[args.phase](resume=args.resume)


if __name__ == "__main__":
    # Multi-GPU support via torchrun
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    main()
