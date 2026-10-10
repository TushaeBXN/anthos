"""
reward_model.py — Bradley-Terry Reward Model

Trains a scalar reward head on top of a frozen Anthos backbone.
Uses paired preference data (chosen > rejected) and the Bradley-Terry
ranking loss (equivalent to the reward model loss used in RLHF).

Loss: -log σ(r_chosen - r_rejected)
where r = scalar reward at [EOS] position (last non-pad token)

The learned reward model can then be used:
  - Directly in PPO/GRPO reward signal
  - As a reranker for best-of-N sampling
  - As a reward signal for RLHF data selection

Usage:
    from anthos.reward_model import RewardModelConfig, RewardModelTrainer

    trainer = RewardModelTrainer.from_pretrained(
        "checkpoints/sft/step_001000.pt", tier="sft"
    )
    dataset = RewardDataset("data/preferences.jsonl", "data/anthos_tokenizer")
    for batch in DataLoader(dataset, batch_size=4, collate_fn=rm_collate):
        loss, acc = trainer.step(batch)

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
class RewardModelConfig:
    n_loops:      int   = 8
    max_length:   int   = 2048
    dropout:      float = 0.1
    lr:           float = 1e-5
    margin:       float = 0.0   # optional margin: loss = -log σ(r_w - r_l - margin)


class RewardHead(nn.Module):
    """Two-layer MLP projecting hidden dim → scalar reward in ℝ."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class RewardDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_length: int = 2048):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length
        self.data = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.data)

    def _encode(self, prompt: str, completion: str) -> torch.Tensor:
        ids = self.tok.encode(
            prompt + completion,
            truncation=True,
            max_length=self.max_length,
        )
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "chosen":   self._encode(row["prompt"], row["chosen"]),
            "rejected": self._encode(row["prompt"], row["rejected"]),
        }


def rm_collate(batch: list[dict]) -> dict:
    def _pad(seqs):
        max_len = max(s.shape[0] for s in seqs)
        out = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s
        return out

    return {
        "chosen_ids":   _pad([b["chosen"]   for b in batch]),
        "rejected_ids": _pad([b["rejected"] for b in batch]),
    }


class RewardModelTrainer:
    def __init__(self, backbone: nn.Module, head: RewardHead, cfg: RewardModelConfig):
        self.backbone = backbone
        self.head     = head
        self.cfg      = cfg
        self.opt = torch.optim.AdamW(
            list(head.parameters()),
            lr=cfg.lr, betas=(0.9, 0.95),
        )

    def _reward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract scalar reward from the last non-padding token position."""
        logits = self.backbone(input_ids, n_loops=self.cfg.n_loops)
        hidden = self.backbone.get_hidden_states()   # (B, T, dim)

        # Use last non-pad position per sequence
        lengths = (input_ids != 0).sum(-1) - 1   # (B,)
        idx     = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.shape[-1])
        last    = hidden.gather(1, idx).squeeze(1)   # (B, dim)
        return self.head(last)                        # (B,)

    def step(self, batch: dict) -> tuple[torch.Tensor, float]:
        device = next(self.head.parameters()).device
        c_ids  = batch["chosen_ids"].to(device)
        r_ids  = batch["rejected_ids"].to(device)

        r_chosen   = self._reward(c_ids)
        r_rejected = self._reward(r_ids)

        loss = -F.logsigmoid(r_chosen - r_rejected - self.cfg.margin).mean()
        acc  = (r_chosen > r_rejected).float().mean().item()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.opt.step()

        return loss, acc

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return scalar reward for a batch of sequences."""
        device = next(self.head.parameters()).device
        return self._reward(input_ids.to(device))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tier:            str,
        cfg:             Optional[RewardModelConfig] = None,
    ) -> "RewardModelTrainer":
        from anthos.configs import get_training_config
        from anthos.main import Anthos
        cfg = cfg or RewardModelConfig()
        model_cfg, _ = get_training_config(tier)
        backbone = Anthos(model_cfg)
        ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt["model"], strict=False)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad_(False)

        head = RewardHead(model_cfg.dim, cfg.dropout)
        return cls(backbone, head, cfg)
