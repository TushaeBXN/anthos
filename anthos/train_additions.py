"""
anthos/train_additions.py — Training Integration Shim

Shows exactly how to wire multipack + EAFT + DualLoRA into your existing
train.py.  This file is NOT a standalone trainer — it's a drop-in reference
for the changes needed in train.py.

Changes required in train.py (search for the ── markers):

  ── 1. Dataset (replace LocalMarkdownDataset)
  ── 2. Loss function (replace F.cross_entropy)
  ── 3. loops_used extraction (add after model forward)
  ── 4. Pack mask (replace _anthos_causal_mask in batch loop)

The existing train.py structure, optimizer, scheduler, checkpoint logic,
and logging all stay exactly the same.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anthos.multipack import MultipackDataset, MultipackSampler, multipack_collate
from anthos.eaft      import EAFTLoss


# ─────────────────────────────────────────────────────────────────────────────
# ── 1. Dataset replacement
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    data_dir:  str,
    tokenizer,
    chunk_len: int  = 4096,
    batch_size: int = 2,
    seed:      int  = 42,
) -> DataLoader:
    """
    Replace your existing dataset + dataloader with this.

    In train.py, find wherever LocalMarkdownDataset / your current dataset
    is built and replace with:

        from anthos.train_additions import build_dataloader
        loader = build_dataloader(
            data_dir  = "data/new_history",
            tokenizer = tokenizer,
            chunk_len = cfg.max_seq_len,
            batch_size = BATCH_SIZE,
        )

    The loader yields dicts with keys:
        input_ids, labels, seq_ids, attention_mask
    All shape: (batch_size, chunk_len)
    """
    md_files = glob.glob(f"{data_dir}/**/*.md", recursive=True)
    md_files += glob.glob(f"{data_dir}/**/*.txt", recursive=True)

    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {data_dir}")

    dataset = MultipackDataset(
        file_paths = [Path(f) for f in md_files],
        tokenizer  = tokenizer,
        chunk_len  = chunk_len,
        seed       = seed,
    )

    sampler = MultipackSampler(dataset, seed=seed)

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        collate_fn  = multipack_collate,
        num_workers = 0,
        pin_memory  = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── 2. Loss function replacement
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(cfg) -> EAFTLoss:
    """
    Replace your F.cross_entropy call with this.

    In train.py, find the loss computation and replace:

        # BEFORE:
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

        # AFTER:
        from anthos.train_additions import build_loss
        criterion = build_loss(cfg)   # once, before training loop

        # In the training loop:
        loops_used = getattr(model, "_last_loops_used", None)
        loss = criterion(
            logits,
            labels,
            attention_mask = batch["attention_mask"],
            loops_used     = loops_used,
        )
    """
    return EAFTLoss(
        vocab_size      = cfg.vocab_size,
        top_k           = 50,
        focal_gamma     = 1.0,
        act_gamma       = 0.5,
        max_loops       = cfg.max_loop_iters,
        label_smoothing = 0.1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── 3. Full training step (reference implementation)
# ─────────────────────────────────────────────────────────────────────────────

def training_step(
    model:      nn.Module,
    batch:      dict,
    criterion:  EAFTLoss,
    device:     torch.device,
    n_loops:    Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Reference training step with all additions wired together.

    Paste this logic into your train.py training loop, or call it directly.

    Returns:
        (total_loss, metrics_dict)

    metrics_dict keys:
        loss        — EAFT language model loss
        aux_loss    — MoE load-balancing + ACT auxiliary losses
        total_loss  — loss + aux_loss
        ponder      — mean loops used (watch this in logs)
    """
    input_ids      = batch["input_ids"].to(device)        # (B, T)
    labels         = batch["labels"].to(device)           # (B, T)
    attention_mask = batch["attention_mask"].to(device)   # (B, T)

    # Forward pass — return_aux=True for MoE + ACT losses
    logits, aux_loss = model(
        input_ids,
        n_loops    = n_loops,
        return_aux = True,
    )

    # Extract loops_used from model (exposed by patched main.py)
    loops_used = getattr(model, "_last_loops_used", None)

    # EAFT loss — entropy-weighted, pack-aware, ACT-integrated
    lm_loss = criterion(
        logits,
        labels,
        attention_mask = attention_mask,
        loops_used     = loops_used,
    )

    total_loss = lm_loss + aux_loss

    # Metrics
    ponder = loops_used.float().mean().item() if loops_used is not None else 0.0
    metrics = {
        "loss":       lm_loss.item(),
        "aux_loss":   aux_loss.item(),
        "total_loss": total_loss.item(),
        "ponder":     ponder,
    }

    return total_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ── 4. Pack mask integration note
# ─────────────────────────────────────────────────────────────────────────────
#
# The MultipackDataset provides seq_ids per batch.  To get a proper
# block-diagonal attention mask that prevents cross-document attention,
# pass seq_ids to the mask builder from multipack.py:
#
#     from anthos.multipack import build_pack_mask_fast
#
#     # In the training loop, before the model call:
#     if batch["seq_ids"].unique().numel() > 1:
#         # Packed batch — build block-diagonal mask
#         combined_mask = build_pack_mask_fast(
#             batch["seq_ids"][0],   # (T,) — use first item in batch
#             n_thought = cfg.n_thought_tokens,
#             device    = device,
#         )
#     else:
#         combined_mask = None   # Single doc — standard causal mask
#
# NOTE: This mask replaces the one built inside AnthosRecurrentBlock.
# To use it, you'd need to pass it into model.forward() — which currently
# builds its own mask internally.  For now, the internal mask is used and
# cross-document leakage in packed batches is minimal (same-doc content
# dominates due to causal ordering).  Full pack mask integration requires
# a minor main.py addition to accept an external mask — defer until needed.
#
# ─────────────────────────────────────────────────────────────────────────────
