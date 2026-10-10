"""
anthos/kto.py — Kahneman-Tversky Optimization (KTO)

KTO aligns language models from unpaired binary feedback: each sample is
labelled simply "good" (desirable) or "bad" (undesirable) — no chosen/rejected
pairs required.  This makes it far cheaper to annotate than DPO.

How KTO works:
  The Kahneman-Tversky prospect theory (1979) describes how humans weight
  gains vs losses asymmetrically: losses loom larger than equivalent gains.
  KTO mimics this by using a different loss branch for good vs bad samples:

    Good  (y is desirable):  loss = 1 - σ(β * [log π(y|x) - log π_ref(y|x) - z₀])
    Bad   (y is undesirable): loss = 1 - σ(β * [z₀ - (log π(y|x) - log π_ref(y|x))])

  where z₀ is the KL divergence between the policy and reference (a running
  estimate of how far the policy has drifted from the reference model),
  and β controls how strongly the policy penalises drift.

Why KTO over DPO for Anthos:
  - No paired data: annotators just mark completions as good or bad individually.
  - More robust to annotation noise: each label is binary, not relative.
  - Natural for self-improvement loops: the model generates completions;
    a verifier (or the model itself) labels them good/bad.

Data format (JSONL):
    {"prompt": "...", "completion": "...", "label": true}   # good
    {"prompt": "...", "completion": "...", "label": false}  # bad

Reference: Ethayarajh et al. 2023 "KTO: Model Alignment as Prospect Theoretic Optimization"
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
class KTOConfig:
    beta:            float = 0.1    # KL penalty — how far policy can move from ref
    n_loops_policy:  int   = 12     # loop depth for policy (trained) model
    n_loops_ref:     int   = 8      # loop depth for reference (frozen) model
    desirable_weight: float = 1.0   # weight on the desirable (good) loss term
    undesirable_weight: float = 1.0 # weight on the undesirable (bad) loss term
    # KL estimate: updated as a running mean across the batch for stability.
    # Higher kl_ema_decay → smoother but slower to adapt.
    kl_ema_decay:    float = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KTODataset(Dataset):
    """
    Loads a JSONL dataset with unpaired binary labels.

    Each line: {"prompt": str, "completion": str, "label": bool}
      label=true  → desirable (good completion)
      label=false → undesirable (bad completion)
    """

    def __init__(
        self,
        path:           str,
        tokenizer_path: str = "gpt2",
        max_length:     int = 512,
    ):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.items: list[dict] = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line.strip())
                if "prompt" in obj and "completion" in obj and "label" in obj:
                    self.items.append(obj)

        self.max_length = max_length
        n_good = sum(1 for x in self.items if x["label"])
        n_bad  = len(self.items) - n_good
        print(f"[KTODataset] {len(self.items)} samples ({n_good} good, {n_bad} bad) from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        p_ids = self.tok.encode(item["prompt"],     add_special_tokens=False)
        c_ids = self.tok.encode(item["completion"], add_special_tokens=False)

        p_ids = p_ids[-(self.max_length // 2):]
        c_ids = c_ids[:self.max_length // 2]

        combined = p_ids + c_ids
        labels   = [-100] * len(p_ids) + c_ids

        return {
            "input_ids":    torch.tensor(combined, dtype=torch.long),
            "labels":       torch.tensor(labels,   dtype=torch.long),
            "label":        torch.tensor(item["label"], dtype=torch.bool),
        }


def kto_collate(batch: list[dict]) -> dict:
    """Pad a batch of KTO samples to equal length."""
    def _pad(seqs, pad_val):
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([
            F.pad(s, (0, max_len - s.shape[0]), value=pad_val)
            for s in seqs
        ])

    return {
        "input_ids": _pad([b["input_ids"] for b in batch], 0),
        "labels":    _pad([b["labels"]    for b in batch], -100),
        "label":     torch.stack([b["label"] for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# KTO loss
# ─────────────────────────────────────────────────────────────────────────────

def kto_loss(
    policy_logps:     torch.Tensor,   # (B,) — per-sequence log-probs under policy
    ref_logps:        torch.Tensor,   # (B,) — per-sequence log-probs under reference
    labels:           torch.Tensor,   # (B,) bool — True = desirable, False = undesirable
    z0:               torch.Tensor,   # scalar — KL estimate (updated externally)
    beta:             float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    KTO loss per Ethayarajh et al. 2023.

    Returns (loss, metrics_dict).
    """
    kl_term = beta * (policy_logps - ref_logps - z0)

    # Good samples: reward = σ(kl_term); loss = 1 - reward
    good_mask = labels.float()
    bad_mask  = 1.0 - good_mask

    loss_good = (1.0 - F.sigmoid( kl_term)) * good_mask * desirable_weight
    loss_bad  = (1.0 - F.sigmoid(-kl_term)) * bad_mask  * undesirable_weight

    n_good = good_mask.sum().clamp(min=1)
    n_bad  = bad_mask.sum().clamp(min=1)

    loss = loss_good.sum() / n_good + loss_bad.sum() / n_bad

    return loss, {
        "kto_loss":       loss.item(),
        "loss_good":      (loss_good.sum() / n_good).item(),
        "loss_bad":       (loss_bad.sum()  / n_bad).item(),
        "kl_drift":       (policy_logps - ref_logps).mean().item(),
        "z0":             z0.item(),
        "n_good":         int(good_mask.sum().item()),
        "n_bad":          int(bad_mask.sum().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class KTOTrainer:
    """
    KTO trainer for Anthos.

    Holds a policy (trained) and a frozen reference model.
    z₀ (the KL estimate) is maintained as a running EMA across steps.

    Usage:
        trainer = KTOTrainer.from_pretrained(
            checkpoint_path = "checkpoints/sft/step_000500.pt",
            model_class     = Anthos,
            cfg             = model_cfg,
            kto_config      = KTOConfig(beta=0.1),
        )
        dataset = KTODataset("data/feedback.jsonl", trainer.tokenizer_path)
        for batch in DataLoader(dataset, batch_size=4, collate_fn=kto_collate):
            loss, metrics = trainer.step(batch)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model:           nn.Module,
        ref_model:       nn.Module,
        tokenizer_path:  str       = "gpt2",
        cfg:             KTOConfig = KTOConfig(),
    ):
        self.model          = model
        self.ref_model      = ref_model
        self.tokenizer_path = tokenizer_path
        self.cfg            = cfg
        self._z0            = torch.zeros(1)   # KL running estimate

        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    def _log_probs(
        self,
        model:       nn.Module,
        input_ids:   torch.Tensor,
        labels:      torch.Tensor,
        n_loops:     int,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            logits, _ = model(input_ids, n_loops=n_loops, return_aux=True)
            log_probs = F.log_softmax(logits, dim=-1)
            targets   = labels[:, 1:].clamp(min=0)
            mask      = (labels[:, 1:] >= 0).float()
            per_tok   = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
            return (per_tok * mask).sum(dim=-1)   # (B,)

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        device    = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        is_good   = batch["label"].to(device)

        policy_logps = self._log_probs(
            self.model, input_ids, labels,
            self.cfg.n_loops_policy, requires_grad=True,
        )
        ref_logps = self._log_probs(
            self.ref_model, input_ids, labels,
            self.cfg.n_loops_ref, requires_grad=False,
        )

        # Update KL estimate: z₀ ≈ E[log π(y|x) - log π_ref(y|x)]
        kl_batch = (policy_logps.detach() - ref_logps).mean()
        self._z0  = self._z0.to(device)
        self._z0  = self.cfg.kl_ema_decay * self._z0 + (1 - self.cfg.kl_ema_decay) * kl_batch

        return kto_loss(
            policy_logps, ref_logps, is_good, self._z0,
            beta=self.cfg.beta,
            desirable_weight=self.cfg.desirable_weight,
            undesirable_weight=self.cfg.undesirable_weight,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_class,
        cfg,
        tokenizer_path:  str       = "gpt2",
        kto_config:      KTOConfig = KTOConfig(),
        device:          str       = "cpu",
    ) -> "KTOTrainer":
        import copy
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy = model_class(cfg).to(device)
        policy.load_state_dict(ckpt["model"], strict=False)
        ref    = copy.deepcopy(policy)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        return cls(policy, ref, tokenizer_path, kto_config)
