"""
orpo.py — ORPO: Odds Ratio Preference Optimization

Single-pass SFT + alignment: no reference model, no separate DPO step.
The odds ratio between chosen and rejected completions is used as a
preference signal directly inside the SFT cross-entropy loss.

Loss:
    L_ORPO = L_SFT - lambda * E[log σ(log OR(w, l))]

where OR(w, l) = (p_w/(1-p_w)) / (p_l/(1-p_l))

This is ~2× more compute-efficient than DPO because:
  - No reference model forward pass
  - Alignment happens during SFT, not in a separate stage
  - Works with any paired preference dataset

Reference: Hong et al. 2024 "ORPO: Monolithic Preference Optimization
without Reference Model"

Usage:
    from anthos.orpo import ORPOConfig, ORPODataset, ORPOTrainer

    cfg = ORPOConfig(beta=0.1)
    trainer = ORPOTrainer.from_pretrained(
        "checkpoints/base/step_001000.pt", tier="sft", cfg=cfg
    )
    dataset = ORPODataset("data/preferences.jsonl", "data/anthos_tokenizer")
    for batch in DataLoader(dataset, batch_size=2, collate_fn=orpo_collate):
        loss, metrics = trainer.step(batch)

Data format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}
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


@dataclass
class ORPOConfig:
    beta:         float = 0.1     # odds ratio coefficient (lambda in paper)
    n_loops:      int   = 8       # recurrent depth
    max_length:   int   = 2048
    lr:           float = 5e-6
    sft_coef:     float = 1.0     # weight on the SFT loss term
    or_coef:      float = 1.0     # weight on the odds-ratio loss term


class ORPODataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_length: int = 2048):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length
        self.data = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.data)

    def _encode(self, prompt: str, completion: str) -> dict:
        full  = prompt + completion
        p_ids = self.tok.encode(prompt)
        full_enc = self.tok(
            full, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()
        labels[:len(p_ids)] = -100  # mask the prompt tokens from loss
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "chosen":   self._encode(row["prompt"], row["chosen"]),
            "rejected": self._encode(row["prompt"], row["rejected"]),
        }


def orpo_collate(batch: list[dict]) -> dict:
    def _pad(items):
        max_len = max(x["input_ids"].shape[0] for x in items)
        ids = torch.zeros(len(items), max_len, dtype=torch.long)
        lbs = torch.full((len(items), max_len), -100, dtype=torch.long)
        for i, x in enumerate(items):
            L = x["input_ids"].shape[0]
            ids[i, :L] = x["input_ids"]
            lbs[i, :L] = x["labels"]
        return ids, lbs

    chosen_ids,   chosen_lbs   = _pad([b["chosen"]   for b in batch])
    rejected_ids, rejected_lbs = _pad([b["rejected"] for b in batch])
    return {
        "chosen_ids":    chosen_ids,
        "chosen_labels": chosen_lbs,
        "rejected_ids":    rejected_ids,
        "rejected_labels": rejected_lbs,
    }


def _sequence_log_probs(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)  with -100 for masked positions
) -> torch.Tensor:
    """Mean log-prob over non-masked response tokens."""
    B, T, V = logits.shape
    # Shift: logits[t] predicts labels[t+1]
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)             # (B, T-1, V)
    token_lp  = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    mask      = (shift_labels != -100).float()
    return (token_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)   # (B,) mean per seq


def orpo_loss(
    chosen_logits:   torch.Tensor,   # (B, T, V)
    rejected_logits: torch.Tensor,   # (B, T, V)
    chosen_labels:   torch.Tensor,   # (B, T)
    rejected_labels: torch.Tensor,   # (B, T)
    beta:   float = 0.1,
    sft_coef: float = 1.0,
    or_coef:  float = 1.0,
) -> tuple[torch.Tensor, dict]:
    # SFT loss on chosen completions only
    sft_loss = F.cross_entropy(
        chosen_logits[:, :-1].contiguous().reshape(-1, chosen_logits.shape[-1]),
        chosen_labels[:, 1:].contiguous().reshape(-1),
        ignore_index=-100,
    )

    # Sequence log-probs
    lp_w = _sequence_log_probs(chosen_logits,   chosen_labels)    # (B,)
    lp_l = _sequence_log_probs(rejected_logits, rejected_labels)  # (B,)

    # Odds ratio: OR(w, l) = (p_w / (1-p_w)) / (p_l / (1-p_l))
    # In log space: log OR = log p_w - log(1-p_w) - log p_l + log(1-p_l)
    # Since lp_w = log p_w (approximate, using mean token log-prob):
    log_or = lp_w - torch.log1p(-lp_w.exp().clamp(0, 1 - 1e-6)) \
           - lp_l + torch.log1p(-lp_l.exp().clamp(0, 1 - 1e-6))

    or_loss = -F.logsigmoid(beta * log_or).mean()

    loss = sft_coef * sft_loss + or_coef * or_loss
    return loss, {
        "loss":     loss.item(),
        "sft_loss": sft_loss.item(),
        "or_loss":  or_loss.item(),
        "log_or":   log_or.mean().item(),
    }


class ORPOTrainer:
    def __init__(self, model: nn.Module, cfg: ORPOConfig):
        self.model = model
        self.cfg   = cfg
        self.opt   = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, betas=(0.9, 0.95),
        )

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        device = next(self.model.parameters()).device
        c_ids = batch["chosen_ids"].to(device)
        c_lbs = batch["chosen_labels"].to(device)
        r_ids = batch["rejected_ids"].to(device)
        r_lbs = batch["rejected_labels"].to(device)

        c_logits = self.model(c_ids, n_loops=self.cfg.n_loops)
        r_logits = self.model(r_ids, n_loops=self.cfg.n_loops)

        loss, metrics = orpo_loss(
            c_logits, r_logits, c_lbs, r_lbs,
            self.cfg.beta, self.cfg.sft_coef, self.cfg.or_coef,
        )

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss, metrics

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tier:            str,
        cfg:             Optional[ORPOConfig] = None,
    ) -> "ORPOTrainer":
        from anthos.configs import get_training_config
        cfg = cfg or ORPOConfig()
        model_cfg, _ = get_training_config(tier)
        from anthos.main import Anthos
        model = Anthos(model_cfg)
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        return cls(model, cfg)
