"""
Anthos — Training Script
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

# ── Allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.data    import get_dataloader, get_instruct_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule — linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Loop count — phased training
# ─────────────────────────────────────────────────────────────────────────────

def get_n_loops(step: int, phase1_steps: int, phase1_loops: int, phase2_loops: int) -> int:
    return phase1_loops if step < phase1_steps else phase2_loops


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model: Anthos, device: str, n_loops: int, n_samples: int = 3) -> list[str]:
    from transformers import AutoTokenizer
    tok    = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 2048 # Fixes sequence length warning
    model.eval()

    prompts = [
        "Once upon a time",
        "The small robot looked at",
        "In a world where",
    ][:n_samples]

    samples = []
    for prompt in prompts:
        ids = torch.tensor(
            tok.encode(prompt), dtype=torch.long, device=device
        ).unsqueeze(0)
        out = model.generate(ids, max_new_tokens=80, n_loops=n_loops, temperature=0.6, top_k=40)
        samples.append(tok.decode(out[0].tolist()))

    model.train()
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: Anthos,
    optimizer: AdamW,
    step: int,
    loss: float,
    cfg_dict: dict,
    train_cfg_dict: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":           step,
        "loss":           loss,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "cfg":            cfg_dict,
        "train_cfg":      train_cfg_dict,
    }, path)
    print(f"  ✓ Saved checkpoint → {path}")


def load_checkpoint(path: str, model: Anthos, optimizer: AdamW, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    print(f"  ✓ Resumed from step {step}  (loss={ckpt['loss']:.4f})")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(tier: str = "smoke", resume: str | None = None):
    model_cfg, train_cfg = get_training_config(tier)
    device  = train_cfg.device
    ckpt_dir = Path(train_cfg.checkpoint_dir) / train_cfg.run_name

    # ── Mixed precision context ───────────────────────────────────────────────
    dtype_map  = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    pt_dtype   = dtype_map[train_cfg.dtype]
    amp_ctx    = (
        torch.amp.autocast(device_type="cuda", dtype=pt_dtype)
        if device == "cuda" and train_cfg.dtype != "float32"
        else nullcontext()
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if device == "cuda" and train_cfg.dtype == "float16"
        else None
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Anthos(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"  Anthos — Think in Streams")
    print(f"  Tier:       {tier}")
    print(f"  Device:     {device}  ({train_cfg.dtype})")
    print(f"  Parameters: {total_params:,}")
    print(f"  Thoughts:   {model_cfg.n_thought_tokens} tokens")
    print(f"  Max loops:  {model_cfg.max_loop_iters}")
    print(f"  Dataset:    {train_cfg.dataset}")
    print(f"  Eff. batch: {train_cfg.batch_size * train_cfg.grad_accum} tokens×{train_cfg.seq_len}")
    print(f"  Max steps:  {train_cfg.max_steps:,}")
    print(f"{'─'*60}\n")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <  2]
    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr    = train_cfg.learning_rate,
        betas = (0.9, 0.95),
        fused = (device == "cuda"),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    is_instruct = (tier == "instruct")
    if is_instruct:
        loader = get_instruct_dataloader(
            seq_len     = train_cfg.seq_len,
            batch_size  = train_cfg.batch_size,
            num_workers = train_cfg.num_workers,
            mask_prompt = True,
        )
    else:
        subset = "sample-10BT" if "fineweb" in train_cfg.dataset.lower() else None
        loader = get_dataloader(
            dataset_name = train_cfg.dataset,
            split        = train_cfg.dataset_split,
            seq_len      = train_cfg.seq_len,
            batch_size   = train_cfg.batch_size,
            num_workers  = train_cfg.num_workers,
            subset       = subset,
        )
    data_iter = iter(loader)

    # ── Log file ──────────────────────────────────────────────────────────────
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step        = start_step
    loss_accum  = 0.0
    aux_accum   = 0.0
    t0          = time.time()

    optimizer.zero_grad()

    while step < train_cfg.max_steps:
        lr = get_lr(step, train_cfg.warmup_steps, train_cfg.max_steps,
                    train_cfg.learning_rate, train_cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        n_loops = get_n_loops(
            step,
            train_cfg.phase1_steps,
            train_cfg.phase1_loops,
            train_cfg.phase2_loops,
        )

        for micro_step in range(train_cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if is_instruct:
                # Instruct loader yields (input_ids, labels) with prompt masked
                input_ids, labels = batch[0].to(device), batch[1].to(device)
            else:
                batch     = batch.to(device)
                input_ids = batch[:, :-1]
                labels    = batch[:, 1:]

            with amp_ctx:
                # n_loops is the max ceiling; model returns ponder cost in aux
                logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)
                ce = F.cross_entropy(
                    logits.reshape(-1, model_cfg.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,   # masks prompt tokens in instruct mode
                )
                
                # Phase-4: Total Loss = Predictions + Efficiency Penalty
                loss = (ce + aux) / train_cfg.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum += ce.item()
            aux_accum  += aux.item()

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % train_cfg.log_every == 0:
            t1       = time.time()
            avg_loss = loss_accum / train_cfg.log_every
            avg_aux  = aux_accum  / train_cfg.log_every
            
            # Label based on current milestone
            phase = "phase-4 (elastic)" if step >= 12000 else "phase-2 (adaptive)"
            
            tokens_per_sec = (
                train_cfg.log_every * train_cfg.batch_size *
                train_cfg.grad_accum * train_cfg.seq_len / (t1 - t0)
            )
            print(
                f"step {step:6d} | loss {avg_loss:.4f} | ponder {avg_aux:.5f} | "
                f"loops {n_loops} | lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s | {phase}"
            )
            log_file.write(json.dumps({
                "step": step, "loss": avg_loss, "ponder": avg_aux,
                "lr": lr, "n_loops": n_loops,
            }) + "\n")
            log_file.flush()
            loss_accum = aux_accum = 0.0
            t0 = t1

        if step % train_cfg.sample_every == 0:
            print("\n── Sample outputs ─────────────────────────────────────")
            for sample in generate_samples(model, device, n_loops):
                print(f"  {sample[:200]}\n")
            print("───────────────────────────────────────────────────────\n")

        if step % train_cfg.save_every == 0:
            save_checkpoint(
                path           = ckpt_dir / f"step_{step:06d}.pt",
                model          = model,
                optimizer      = optimizer,
                step           = step,
                loss           = avg_loss if step % train_cfg.log_every == 0 else -1,
                cfg_dict       = model_cfg.__dict__,
                train_cfg_dict = train_cfg.__dict__,
            )

    # Final Save
    save_checkpoint(
        path           = ckpt_dir / "final.pt",
        model          = model,
        optimizer      = optimizer,
        step           = step,
        loss           = -1,
        cfg_dict       = model_cfg.__dict__,
        train_cfg_dict = train_cfg.__dict__,
    )
    log_file.close()
    print(f"\n✓ Training complete — {step} steps")

# ─────────────────────────────────────────────────────────────────────────────
# Generate-from-checkpoint mode
# ─────────────────────────────────────────────────────────────────────────────

def generate_from_checkpoint(ckpt_path: str, tier: str = "smoke"):
    model_cfg, train_cfg = get_training_config(tier)
    device = train_cfg.device

    model = Anthos(model_cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt['step']}  (loss={ckpt['loss']:.4f})\n")

    prompts = [
        "Once upon a time there was",
        "The little girl found a",
        "In a land far away",
        "The robot said to the child",
    ]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 2048 # Fixes sequence length warning

    for prompt in prompts:
        ids = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=120, n_loops=train_cfg.phase2_loops,
                                 temperature=0.6, top_k=40)
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT: {tok.decode(out[0].tolist())}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anthos training — Think in Streams")
    parser.add_argument("--tier",     type=str, default="smoke",
                        choices=["smoke", "proof", "research", "ethnic", "instruct"],
                        help="Hardware tier: smoke | proof | research | ethnic | instruct")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--generate", type=str, default=None,
                        help="Path to checkpoint for generation mode")
    args = parser.parse_args()

    if args.generate:
        generate_from_checkpoint(args.generate, tier=args.tier)
    else:
        train(tier=args.tier, resume=args.resume)
