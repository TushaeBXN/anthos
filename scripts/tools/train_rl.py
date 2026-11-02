#!/usr/bin/env python3
"""
train_rl.py — RLOO post-training for Anthos with executable unit-test reward.

Standalone new file. Does NOT modify train.py or any existing Anthos module.

Usage (from repo root):
    python train_rl.py \
        --checkpoint checkpoints/anthos-proof/step_002000.pt \
        --tier proof \
        --output checkpoints/anthos-rl-001 \
        --groups 64 --group-size 8

The script freezes all layers except the final coda block + norm + head, then
runs RLOO policy-gradient updates using pass/fail unit-test reward on synthetic
Python coding tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos_rl_tasks import RLTask, evaluate_task, get_tasks
from transformers import AutoTokenizer


# ── RLOO math ─────────────────────────────────────────────────────────────────

def leave_one_out(rewards: list[float]) -> list[float]:
    """Leave-one-out baseline: advantage[i] = reward[i] - mean(others)."""
    total = sum(rewards)
    n = len(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


# ── Model helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(
    model: Anthos,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    n_loops: int,
) -> list[int]:
    """Sample one completion from the model (no grad, uses built-in generate)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, n_loops=n_loops,
                         temperature=temperature, top_k=40)
    return out[0, len(prompt_ids):].tolist()


def completion_log_probs(
    model: Anthos,
    prompt_ids: list[int],
    completions: list[list[int]],
    device: torch.device,
    n_loops: int,
) -> torch.Tensor:
    """
    One forward pass over the whole group → scalar mean log-prob per completion.
    Gradients flow through this for the RLOO update.
    """
    if not completions or any(not c for c in completions):
        raise ValueError("empty completion in group")

    P       = len(prompt_ids)
    max_c   = max(len(c) for c in completions)
    G       = len(completions)

    # Pad completions to max_c
    rows = [prompt_ids + c + [0] * (max_c - len(c)) for c in completions]
    seq  = torch.tensor(rows, dtype=torch.long, device=device)  # (G, P+max_c)

    # Forward on [prompt + completion - last token] → predict completion tokens
    logits = model.forward(seq[:, :-1], n_loops=n_loops)        # (G, P+max_c-1, V)

    # Slice to completion positions: model predicts position P onward
    comp_logits = logits[:, P - 1 : P - 1 + max_c].float()     # (G, max_c, V)
    targets     = seq[:, P : P + max_c]                         # (G, max_c)

    token_lp = F.log_softmax(comp_logits, dim=-1) \
                 .gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (G, max_c)

    lengths = torch.tensor([len(c) for c in completions], device=device).float()
    mask    = torch.arange(max_c, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    return (token_lp * mask).sum(dim=1) / lengths               # (G,)


def load_model(checkpoint: Path, tier: str, device: torch.device) -> Anthos:
    model_cfg, _ = get_training_config(tier)
    model = Anthos(model_cfg)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded checkpoint {checkpoint} (step {ckpt.get('step', '?')})")
    return model


def freeze_except_head(model: Anthos) -> list[torch.nn.Parameter]:
    """Freeze everything except last coda block + norm + head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.coda[-1].parameters():
        p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n:,} (last coda block + norm + head)")
    return trainable


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",     type=Path, required=True,
                        help="Path to .pt checkpoint from train.py")
    parser.add_argument("--tier",           default="proof",
                        help="Training tier used to build the checkpoint (default: proof)")
    parser.add_argument("--output",         type=Path, required=True,
                        help="Directory to write RL checkpoints and log")
    parser.add_argument("--groups",         type=int,   default=64,
                        help="Total rollout groups (training steps that may update)")
    parser.add_argument("--group-size",     type=int,   default=8,
                        help="Sampled completions per prompt (>=2 for RLOO)")
    parser.add_argument("--max-new-tokens", type=int,   default=64)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--n-loops",        type=int,   default=8,
                        help="Anthos recurrent depth during RL (default: 8)")
    parser.add_argument("--families",       default="",
                        help="Comma-separated task families; empty = all")
    parser.add_argument("--device",
                        choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    if args.group_size < 2:
        parser.error("--group-size must be >= 2 (RLOO requires at least two samples)")

    # ── Setup ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    model     = load_model(args.checkpoint, args.tier, device)
    trainable = freeze_except_head(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    tok = AutoTokenizer.from_pretrained(tok_path)
    print(f"Tokenizer: {tok_path} (vocab {tok.vocab_size})")

    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    tasks    = get_tasks(families)
    print(f"Tasks: {len(tasks)} | Families: {sorted({t.family for t in tasks})}")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Task schedule (cycle through tasks until args.groups reached) ─────────
    rng = random.Random(args.seed)
    schedule: list[RLTask] = []
    while len(schedule) < args.groups:
        block = list(tasks)
        rng.shuffle(block)
        schedule.extend(block)
    schedule = schedule[: args.groups]

    # ── Training loop ─────────────────────────────────────────────────────────
    log_rows: list[dict] = []
    updates  = 0
    t0       = time.perf_counter()

    for step, task in enumerate(schedule, 1):
        prompt_ids = tok.encode(task.prompt)

        # 1. Rollout (no_grad — model.generate is already decorated)
        model.eval()
        comp_ids  = [rollout(model, prompt_ids, args.max_new_tokens,
                             args.temperature, device, args.n_loops)
                     for _ in range(args.group_size)]
        comp_strs = [tok.decode(c) for c in comp_ids]

        # 2. Reward
        rewards = []
        evals   = []
        for cs in comp_strs:
            ev = evaluate_task(task, cs)
            # binary: 1 for full pass, 0 for valid wrong, slight partial credit
            if ev["passed"]:
                r = 1.0
            elif ev["status"] == "invalid":
                r = -0.1
            else:
                r = ev["pass_fraction"] * 0.3   # partial credit, keep signal low
            rewards.append(r)
            evals.append(ev)

        spread     = max(rewards) - min(rewards)
        advantages = leave_one_out(rewards)
        n_passed   = sum(1 for e in evals if e["passed"])

        row = {
            "step":    step,
            "task_id": task.task_id,
            "family":  task.family,
            "rewards": rewards,
            "spread":  spread,
            "updated": False,
            "n_passed": n_passed,
        }

        # 3. RLOO update — only if there is reward variance and non-empty completions
        if spread > 1e-8 and all(c for c in comp_ids):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            log_p = completion_log_probs(model, prompt_ids, comp_ids, device, args.n_loops)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            loss  = -(adv_t * log_p).mean()

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                updates        += 1
                row["updated"]  = True
                row["loss"]     = float(loss.detach())

        log_rows.append(row)
        elapsed = time.perf_counter() - t0
        upd_str = f"loss={row['loss']:.4f}" if row["updated"] else "no-update"
        print(f"[{step:3d}/{args.groups}] {task.task_id:<20} "
              f"pass={n_passed}/{args.group_size}  "
              f"spread={spread:.3f}  {upd_str}  t={elapsed:.1f}s")

        # Save checkpoint every 32 steps (log saved separately to save disk)
        if step % 32 == 0 or step == args.groups:
            ckpt_path = args.output / f"step_{step:04d}.pt"
            try:
                torch.save({
                    "step":    step,
                    "updates": updates,
                    "model":   model.state_dict(),
                    "tier":    args.tier,
                }, ckpt_path)
                print(f"  → saved {ckpt_path}")
            except (RuntimeError, OSError) as e:
                print(f"  ! checkpoint save failed (disk full?): {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_passed = sum(r["n_passed"] for r in log_rows)
    total_trials = args.groups * args.group_size
    (args.output / "rl_log.json").write_text(json.dumps(log_rows, indent=2))
    print(f"\nDone. {updates}/{args.groups} groups updated.")
    print(f"Overall pass rate: {total_passed}/{total_trials} "
          f"({100 * total_passed / total_trials:.1f}%)")
    print(f"Log → {args.output / 'rl_log.json'}")


if __name__ == "__main__":
    main()
