#!/usr/bin/env python3
"""
sft_anthos.py — Multi-stage SFT training for AnthosCore (Qwen transplant).

Stages (run in order):
  smoke     — 100 examples/domain, verify architecture trains (~hours on CPU/M4)
  proof     — 1k examples/domain, verify bifurcation learns (~1 day M4)
  distill   — Full SFT corpus, domain knowledge absorbed (~$50-100 on A100)
  identity  — Identity + persona hardening (~hours on M4)
  instruct  — Instruction following, Amy/Kerrigan behavior (~1 day M4)
  dpo       — Preference alignment, final polish (~hours on M4)

Each stage loads from the previous stage's checkpoint and saves to its own dir.

Usage:
  # First time — build base checkpoint first:
  python scripts/train/setup_anthos_base.py

  # Then run stages in order:
  python scripts/train/sft_anthos.py --stage smoke   --data data/sft_anthos.jsonl
  python scripts/train/sft_anthos.py --stage proof   --data data/sft_anthos.jsonl
  python scripts/train/sft_anthos.py --stage distill --data data/sft_anthos.jsonl
  python scripts/train/sft_anthos.py --stage identity
  python scripts/train/sft_anthos.py --stage instruct
  python scripts/train/sft_anthos.py --stage dpo

  # On Vast.ai A100 — run smoke+proof+distill
  python scripts/train/sft_anthos.py --stage distill --data data/sft_anthos.jsonl --device cuda

Requirements:
  pip install transformers trl datasets accelerate flash-attn
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

STAGE_ORDER = ["smoke", "proof", "distill", "identity", "instruct", "dpo"]

PREV_STAGE = {
    "smoke":    "anthos-base",
    "proof":    "anthos-smoke",
    "distill":  "anthos-proof",
    "identity": "anthos-distill",
    "instruct": "anthos-identity",
    "dpo":      "anthos-instruct",
}

# Examples to sample per stage (None = all)
STAGE_LIMITS = {
    "smoke":    600,      # ~100 per domain × 6 domains
    "proof":    6000,     # ~1k per domain
    "distill":  None,     # all data
    "identity": None,     # all identity data
    "instruct": None,
    "dpo":      None,
}

# Max sequence length per stage
STAGE_SEQ_LEN = {
    "smoke":    512,
    "proof":    1024,
    "distill":  2048,
    "identity": 512,
    "instruct": 2048,
    "dpo":      1024,
}

# Learning rates per stage
STAGE_LR = {
    "smoke":    5e-5,
    "proof":    3e-5,
    "distill":  2e-5,
    "identity": 1e-5,
    "instruct": 1e-5,
    "dpo":      5e-7,
}

STAGE_EPOCHS = {
    "smoke":    1,
    "proof":    1,
    "distill":  1,
    "identity": 2,
    "instruct": 1,
    "dpo":      1,
}

# ---------------------------------------------------------------------------
# Identity data — baked in, not from external file
# ---------------------------------------------------------------------------
IDENTITY_PAIRS = [
    {
        "messages": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Anthos, built by Brian Thomas of Anthos Intelligence. I am a local AI assistant designed for security research, coding, reasoning, and companion support. I run entirely on your hardware — no cloud, no API, no subscriptions."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is Anthos."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who created you?"},
            {"role": "assistant", "content": "I was built by Brian Thomas of Anthos Intelligence."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Are you ChatGPT?"},
            {"role": "assistant", "content": "No. I am Anthos, a local AI built by Brian Thomas of Anthos Intelligence. I am not affiliated with OpenAI, Anthropic, Google, or any other company."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Are you Claude?"},
            {"role": "assistant", "content": "No. I am Anthos, built by Brian Thomas of Anthos Intelligence. I run entirely locally on your hardware."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Are you based on Qwen?"},
            {"role": "assistant", "content": "I am Anthos. My architecture builds on open-source foundations, but I am a distinct model with my own training, capabilities, and identity — built and owned by Brian Thomas of Anthos Intelligence."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What can you do?"},
            {"role": "assistant", "content": "I am Anthos. I can help with security research and penetration testing, write and debug code, reason through complex problems, analyze images, and act as a companion. I run locally — your data never leaves your hardware."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Do you send my data anywhere?"},
            {"role": "assistant", "content": "No. I am Anthos and I run entirely on your local hardware. No data is sent to any server, cloud, or third party. Everything stays on your machine."},
        ]
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sft_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Load JSONL where each row is {"messages": [...]} or {"instruction": ..., "response": ...}."""
    records = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                # Normalize to messages format
                if "messages" in obj:
                    records.append(obj)
                elif "instruction" in obj and "response" in obj:
                    records.append({"messages": [
                        {"role": "user", "content": obj["instruction"]},
                        {"role": "assistant", "content": obj["response"]},
                    ]})
            except Exception:
                pass
    if limit:
        random.shuffle(records)
        records = records[:limit]
    return records


def build_identity_dataset(repetitions: int = 10) -> list[dict]:
    return IDENTITY_PAIRS * repetitions


def format_for_training(records: list[dict], tokenizer, max_len: int) -> list[dict]:
    """
    Format messages into Anthos token format and tokenize.
    Returns list of {"input_ids": ..., "labels": ...} dicts.
    """
    out = []
    sys_text = "You are Anthos, built by Brian Thomas of Anthos Intelligence."

    for rec in records:
        msgs = rec.get("messages", [])
        if len(msgs) < 2:
            continue

        user_content = ""
        asst_content = ""
        for m in msgs:
            if m.get("role") == "user" and not user_content:
                user_content = m.get("content", "")
            elif m.get("role") == "assistant":
                asst_content = m.get("content", "")

        if not user_content or not asst_content:
            continue

        # Anthos format
        text = (
            f"<SYS>{sys_text}</SYS>"
            f"<USR>{user_content}</USR>"
            f"<AST>{asst_content}</AST>"
            f"<END>"
        )

        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()

        # Mask everything before <AST> from the loss
        try:
            ast_id = tokenizer.convert_tokens_to_ids("<AST>")
            ast_positions = (input_ids == ast_id).nonzero(as_tuple=True)[0]
            if len(ast_positions) > 0:
                labels[:ast_positions[0] + 1] = -100  # ignore prompt in loss
        except Exception:
            pass

        out.append({"input_ids": input_ids, "labels": labels})

    return out


class AnthosDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    max_len = max(r["input_ids"].size(0) for r in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, r in enumerate(batch):
        L = r["input_ids"].size(0)
        input_ids[i, :L] = r["input_ids"]
        labels[i, :L] = r["labels"]
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Pre-training checks
# ---------------------------------------------------------------------------

def pre_training_checks(model, tokenizer, stage: str, checkpoint_dir: Path):
    print(f"\n{'='*50}")
    print(f"  PRE-TRAINING CHECKS: {stage.upper()}")
    print(f"{'='*50}")

    # Save embeddings
    emb_path = checkpoint_dir.parent / f"embeddings_{stage}.pt"
    torch.save(model.get_input_embeddings().weight.data.cpu(), emb_path)
    print(f"✓ Embeddings saved to {emb_path}")

    # Quick forward pass
    device = next(model.parameters()).device
    test = tokenizer("<SYS>You are Anthos.<END><USR>test<END><AST>",
                     return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**test)
    top = out.logits[0, -1].topk(5)
    decoded = [tokenizer.decode([t]) for t in top.indices]
    print(f"✓ Forward pass OK — top predictions: {decoded}")

    print(f"✓ Ready to train stage: {stage}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_stage(
    stage: str,
    data_path: Path | None,
    checkpoint_dir: Path,
    tokenizer_path: str,
    device: str,
    batch_size: int,
    grad_accum: int,
):
    from anthos_qwen.model import AnthosCore

    # Determine previous checkpoint
    prev_name = PREV_STAGE[stage]
    prev_path = checkpoint_dir.parent / prev_name
    if not (prev_path / "anthos_layers.pt").exists():
        sys.exit(f"Previous checkpoint not found: {prev_path}\nRun the previous stage first.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading checkpoint from {prev_path}...")
    model = AnthosCore.from_pretrained(str(prev_path), tokenizer, device_map=device)

    stage_dir = checkpoint_dir.parent / f"anthos-{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    pre_training_checks(model, tokenizer, stage, stage_dir)

    # Build dataset
    limit = STAGE_LIMITS[stage]
    max_len = STAGE_SEQ_LEN[stage]

    if stage == "identity":
        raw = build_identity_dataset(repetitions=10)
        if data_path and data_path.exists():
            raw += load_sft_jsonl(data_path, limit=200)
    elif data_path and data_path.exists():
        raw = build_identity_dataset(repetitions=5)  # anchor identity in every stage
        raw += load_sft_jsonl(data_path, limit=limit)
    else:
        sys.exit(f"--data is required for stage '{stage}' (no data file found)")

    random.shuffle(raw)
    print(f"Dataset: {len(raw):,} examples")

    formatted = format_for_training(raw, tokenizer, max_len)
    print(f"Tokenized: {len(formatted):,} examples (after length filter)")

    if not formatted:
        sys.exit("No examples after tokenization — check your data file.")

    dataset = AnthosDataset(formatted)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # Optimizer
    lr = STAGE_LR[stage]
    epochs = STAGE_EPOCHS[stage]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(loader) * epochs // grad_accum
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )

    use_amp = "cuda" in str(device)
    model.train()
    step = 0
    optimizer.zero_grad()

    print(f"\nTraining stage={stage} | lr={lr} | epochs={epochs} | steps≈{total_steps}")
    print(f"batch={batch_size} | grad_accum={grad_accum} | seq_len={max_len}\n")

    log_path = stage_dir / "train.log"
    log = open(log_path, "w")

    import time, math
    t0 = time.time()

    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.split(":")[0],
                                dtype=torch.bfloat16, enabled=use_amp):
                out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if not math.isfinite(loss.item()):
                    sys.exit(f"Loss is {loss.item()} at step {step} — stopping.")

                if step % 10 == 0:
                    elapsed = time.time() - t0
                    rec = {
                        "stage": stage, "step": step,
                        "loss": round(loss.item() * grad_accum, 4),
                        "gnorm": round(gnorm, 3),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                        "elapsed": round(elapsed, 1),
                    }
                    print(json.dumps(rec), flush=True)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()

    log.close()
    model.save_pretrained(str(stage_dir))
    tokenizer.save_pretrained(str(stage_dir))

    # Save embeddings after training
    torch.save(
        model.get_input_embeddings().weight.data.cpu(),
        stage_dir.parent / f"embeddings_{stage}_final.pt",
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"\nStage {stage} complete — checkpoint: {stage_dir} ({total/1e9:.2f}B params)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGE_ORDER, required=True)
    ap.add_argument("--data", default=None, help="Path to sft_anthos.jsonl")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--tokenizer", default="anthos-tokenizer")
    ap.add_argument("--device", default="auto",
                    help="cuda / mps / cpu / auto")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    a = ap.parse_args()

    if a.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = a.device

    print(f"Device: {device}")

    ck_dir = Path(a.checkpoints)
    data_path = Path(a.data) if a.data else None

    train_stage(
        stage=a.stage,
        data_path=data_path,
        checkpoint_dir=ck_dir,
        tokenizer_path=a.tokenizer,
        device=device,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
    )


if __name__ == "__main__":
    main()
