"""
anthos/prm.py — Process Reward Model (PRM)

A Process Reward Model assigns per-step rewards to multi-step reasoning chains
rather than a single outcome reward for the whole completion.

Why PRM for Anthos:
  Anthos generates explicit reasoning via thought tokens and <think>…</think>
  blocks.  A step-level reward signal is more informative than outcome-only:
    - Correct final answer with bad reasoning → low intermediate rewards
    - Wrong final answer with good reasoning  → partial credit per step
    - Direct path to correct answer           → high reward throughout

  PRM enables credit assignment at the thought-stream level — exactly aligned
  with Anthos's recurrent loop structure.

Architecture:
  A small value head (Linear → scalar) stacked on top of Anthos's hidden states,
  trained to predict step-by-step correctness labels.

  The PRM is trained separately from the policy:
    1. Collect completions with intermediate steps labelled correct/incorrect
    2. Train the PRM value head on these labels (supervised)
    3. Use PRM rewards in GRPO/REINFORCE++ to shape the policy

Data format (JSONL — each line is one reasoning chain):
    {
      "prompt":  "Solve: 2x + 3 = 11",
      "steps":   ["Let me isolate x.", "2x = 11 - 3 = 8", "x = 4"],
      "labels":  [1, 1, 1],   # 1=correct, 0=incorrect per step
      "answer":  "x = 4"
    }

Usage:
    # Train PRM
    prm = PRMTrainer(base_model, prm_config)
    for batch in DataLoader(prm_dataset, collate_fn=prm_collate):
        loss = prm.step(batch)
        loss.backward()
        optimizer.step()

    # Use PRM rewards in GRPO
    rewards = prm.score_steps(prompts, completions)   # list of per-step scores
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PRMConfig:
    n_loops:         int   = 8      # loop depth for hidden state extraction
    step_sep:        str   = "\n"   # token string that separates reasoning steps
    hidden_dropout:  float = 0.1    # dropout on the value head input
    loss_weight_pos: float = 2.0    # upweight correct steps (often rare)


# ─────────────────────────────────────────────────────────────────────────────
# Value head
# ─────────────────────────────────────────────────────────────────────────────

class PRMValueHead(nn.Module):
    """
    Lightweight scalar value head stacked on top of Anthos hidden states.
    Predicts correctness probability [0, 1] for each token position.

    The head is trained on step-separator positions (e.g. newlines).
    At inference, the score at each separator gives the per-step quality.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, 256)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(256, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, dim) → scores: (B, T) in [0, 1]"""
        x = F.gelu(self.proj(hidden))
        x = self.drop(x)
        return self.head(x).squeeze(-1).sigmoid()   # (B, T)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PRMDataset(Dataset):
    """
    Loads a JSONL dataset with per-step correctness labels.

    Each line: {"prompt": str, "steps": list[str], "labels": list[int]}
    labels[i] ∈ {0, 1}: 0 = incorrect step, 1 = correct step.
    """

    def __init__(
        self,
        path:           str,
        tokenizer_path: str   = "gpt2",
        max_length:     int   = 1024,
        step_sep:       str   = "\n",
    ):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.sep_id    = self.tok.encode(step_sep, add_special_tokens=False)
        self.max_length = max_length
        self.items: list[dict] = []

        with open(path) as f:
            for line in f:
                obj = json.loads(line.strip())
                if "prompt" in obj and "steps" in obj and "labels" in obj:
                    if len(obj["steps"]) == len(obj["labels"]):
                        self.items.append(obj)

        print(f"[PRMDataset] {len(self.items)} chains from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        prompt_ids = self.tok.encode(item["prompt"], add_special_tokens=False)

        input_ids: list[int] = prompt_ids[:]
        # -100 everywhere except at step-separator positions (where we predict)
        label_ids: list[int] = [-100] * len(prompt_ids)

        for step_text, step_label in zip(item["steps"], item["labels"]):
            step_ids = self.tok.encode(step_text + "\n", add_special_tokens=False)
            # Label goes on the last token of each step
            step_labels = [-100] * len(step_ids)
            step_labels[-1] = step_label
            input_ids.extend(step_ids)
            label_ids.extend(step_labels)

        # Truncate
        input_ids = input_ids[:self.max_length]
        label_ids = label_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(label_ids, dtype=torch.long),
        }


def prm_collate(batch: list[dict]) -> dict:
    """Pad a batch of PRM samples."""
    def _pad(seqs, pad):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([
            F.pad(s, (0, max_len - s.shape[0]), value=pad)
            for s in seqs
        ])
    return {
        "input_ids": _pad([b["input_ids"] for b in batch], 0),
        "labels":    _pad([b["labels"]    for b in batch], -100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PRMTrainer:
    """
    Trains the PRM value head on top of a frozen Anthos base model.

    The base model is frozen; only the value head receives gradients.
    After training, use score_steps() to get per-step rewards.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cfg:        PRMConfig = PRMConfig(),
    ):
        self.base  = base_model
        self.cfg   = cfg
        dim        = base_model.cfg.dim
        self.head  = PRMValueHead(dim, dropout=cfg.loss_weight_pos).to(
            next(base_model.parameters()).device
        )

        # Freeze the base model
        for p in self.base.parameters():
            p.requires_grad_(False)

    def parameters(self):
        return self.head.parameters()

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute PRM training loss for one batch.

        Returns (loss, metrics_dict).
        """
        device    = next(self.head.parameters()).device
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        with torch.no_grad():
            # Extract hidden states from Anthos (use last hidden layer)
            logits, aux = self.base(input_ids, n_loops=self.cfg.n_loops, return_aux=True)
            # Re-run to get hidden states (we need intermediate activations)
            # Attach a hook to capture the final hidden state
            hidden_out = {}
            def _hook(m, inp, out):
                hidden_out["x"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            h = self.base.head_norm.register_forward_hook(_hook) if hasattr(self.base, "head_norm") else None
            _ = self.base(input_ids, n_loops=self.cfg.n_loops, return_aux=True)
            if h is not None:
                h.remove()

        if "x" not in hidden_out:
            # Fallback: use logits as proxy hidden states (suboptimal but functional)
            hidden = logits
        else:
            hidden = hidden_out["x"]

        scores = self.head(hidden)   # (B, T) in [0, 1]

        # BCE loss only on labelled positions (where labels >= 0)
        mask     = (labels >= 0).float()
        targets  = labels.clamp(min=0).float()

        pos_weight = torch.tensor(self.cfg.loss_weight_pos, device=device)
        bce = F.binary_cross_entropy(
            scores * mask + (1 - mask) * 0.5,   # don't penalise masked positions
            targets * mask + (1 - mask) * 0.5,
            reduction="none",
        )
        # Apply positive-class upweighting
        weight = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
        loss   = (bce * weight * mask).sum() / (mask.sum().clamp(min=1))

        with torch.no_grad():
            pred_correct = ((scores > 0.5) == targets.bool())
            acc = (pred_correct * mask).sum() / mask.sum().clamp(min=1)

        return loss, {"prm_loss": loss.item(), "prm_acc": acc.item()}

    @torch.no_grad()
    def score_steps(
        self,
        prompt:     str,
        steps:      list[str],
        tokenizer_path: str = "gpt2",
    ) -> list[float]:
        """
        Score each reasoning step in [0, 1].

        Returns a list of floats, one per step, representing the predicted
        probability that each step is correct.
        """
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        device = next(self.head.parameters()).device

        input_ids = tok.encode(prompt, add_special_tokens=False)
        step_last_positions: list[int] = []

        for step in steps:
            step_ids = tok.encode(step + "\n", add_special_tokens=False)
            input_ids.extend(step_ids)
            step_last_positions.append(len(input_ids) - 1)

        ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)

        hidden_out: dict = {}
        def _hook(m, inp, out):
            hidden_out["x"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
        h = (self.base.head_norm.register_forward_hook(_hook)
             if hasattr(self.base, "head_norm") else None)
        logits, _ = self.base(ids_t, n_loops=self.cfg.n_loops, return_aux=True)
        if h is not None:
            h.remove()

        hidden = hidden_out.get("x", logits)
        scores = self.head(hidden)[0]   # (T,)

        return [scores[pos].item() for pos in step_last_positions]
