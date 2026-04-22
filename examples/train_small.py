"""
Anthos — Minimal training loop
Trains a small Anthos model on streaming C4 data.
Demonstrates proper aux loss integration and phased training.

Usage:
    pip install datasets transformers
    python examples/train_small.py
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from anthos import Anthos, AnthosConfig

# ── Config (small enough to run on 1 GPU) ────────────────────────────────────
cfg = AnthosConfig(
    vocab_size       = 32000,
    dim              = 512,
    n_heads          = 8,
    n_kv_heads       = 4,
    max_seq_len      = 512,
    max_loop_iters   = 8,
    prelude_layers   = 2,
    coda_layers      = 2,
    n_thought_tokens = 16,
    attn_type        = "gqa",
    n_experts        = 16,
    n_shared_experts = 1,
    n_experts_per_tok = 2,
    expert_dim       = 256,
    lora_rank        = 8,
    moe_aux_coef     = 1e-2,
    act_aux_coef     = 1e-3,
)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN      = 512
BATCH_SIZE   = 4
MAX_STEPS    = 10_000
WARMUP_STEPS = 200
LOG_EVERY    = 100

# ── Training phases ───────────────────────────────────────────────────────────
# Phase 1 (0–2k steps): fixed loops, no ACT — verify loss decreases
# Phase 2 (2k+ steps):  enable ACT, increase loop count
def get_n_loops(step: int) -> int:
    if step < 2000:
        return 4    # fixed — stabilization phase
    return cfg.max_loop_iters  # adaptive — full recurrence


# ── Dataset ───────────────────────────────────────────────────────────────────
def get_loader():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("c4", "en", split="train", streaming=True)

        def tokenize(batch):
            enc = tokenizer(
                batch["text"],
                max_length=SEQ_LEN + 1,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return enc["input_ids"]

        return dataset, tokenize

    except ImportError:
        print("datasets / transformers not installed — using random token batches")
        return None, None


# ── Model + Optimizer ─────────────────────────────────────────────────────────
model     = Anthos(cfg).to(DEVICE)
optimizer = AdamW(
    model.parameters(),
    lr           = 3e-4,
    betas        = (0.9, 0.95),
    weight_decay = 0.1,
)
scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=3e-5)

total_params = sum(p.numel() for p in model.parameters())
print(f"Anthos training — {total_params:,} parameters on {DEVICE}")
print(f"Phases: fixed-4-loops until step 2000, then adaptive up to {cfg.max_loop_iters}")

dataset, tokenize = get_loader()

# ── Training loop ─────────────────────────────────────────────────────────────
model.train()
step       = 0
loss_accum = 0.0

while step < MAX_STEPS:
    # Build a batch (random if no dataset loaded)
    if dataset is None:
        ids = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN + 1), device=DEVICE)
    else:
        batch = next(iter(dataset))
        ids   = tokenize(batch).to(DEVICE)

    input_ids = ids[:, :-1]   # (B, T)
    labels    = ids[:, 1:]    # (B, T)

    n_loops = get_n_loops(step)

    logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)

    ce_loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        labels.reshape(-1),
        ignore_index=cfg.vocab_size - 1,  # skip pad tokens
    )
    loss = ce_loss + aux

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_accum += loss.item()
    step       += 1

    if step % LOG_EVERY == 0:
        avg = loss_accum / LOG_EVERY
        print(
            f"step {step:5d} | loss {avg:.4f} | ce {ce_loss.item():.4f} "
            f"| aux {aux.item():.6f} | loops {n_loops} | lr {scheduler.get_last_lr()[0]:.2e}"
        )
        loss_accum = 0.0

print("Training complete.")
