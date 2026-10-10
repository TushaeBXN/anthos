"""
spin.py — SPIN: Self-Play Fine-Tuning

The model improves by playing against a frozen copy of itself from the
previous iteration.  The frozen "opponent" generates completions; the
live model is trained to prefer ground-truth completions over the
opponent's (DPO-style).

Algorithm (one iteration):
  1. Freeze a reference copy of the current model (opponent_t)
  2. For each prompt in the SFT set:
       y_real = ground-truth completion
       y_fake = opponent_t.generate(prompt)
  3. Train the live model with DPO loss: prefer y_real over y_fake
  4. After training, opponent_{t+1} = current model → next iteration

Key property: as the live model improves, the opponent improves too,
maintaining a useful difficulty signal.  Convergence when the model
cannot beat itself.

Reference: Chen et al. 2024 "Self-Play Fine-Tuning Converts Weak
Language Models to Strong Language Models"

Usage:
    from anthos.spin import SPINConfig, SPINTrainer

    trainer = SPINTrainer.from_pretrained(
        "checkpoints/sft/step_001000.pt", tier="sft"
    )
    # Iteration 0: opponent = initial checkpoint
    trainer.run_iteration(
        data_path="data/sft.jsonl",
        tokenizer_path="data/anthos_tokenizer",
        n_steps=500,
    )
    # Save checkpoint, advance opponent, repeat
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class SPINConfig:
    beta:       float = 0.1    # DPO-style preference strength
    n_loops:    int   = 8
    max_length: int   = 2048
    lr:         float = 5e-6
    max_new_tokens: int = 256


def _seq_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels:    torch.Tensor,
    n_loops:   int,
) -> torch.Tensor:
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids, n_loops=n_loops)
    log_p  = F.log_softmax(logits[:, :-1], dim=-1)
    tgt    = labels[:, 1:].clamp(min=0)
    mask   = (labels[:, 1:] != -100).float()
    return (log_p.gather(2, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(-1) / mask.sum(-1).clamp(1)


class SPINDataset(Dataset):
    """Wraps a JSONL of {prompt, completion} pairs; generates opponent completions on the fly."""

    def __init__(
        self,
        path:           str,
        tokenizer_path: str,
        opponent:       nn.Module,
        device:         torch.device,
        cfg:            SPINConfig,
    ):
        from transformers import AutoTokenizer
        self.tok  = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.cfg      = cfg
        self.opponent = opponent
        self.device   = device
        self.data     = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

    def _encode(self, prompt: str, completion: str) -> dict:
        full = prompt + completion
        p_len = len(self.tok.encode(prompt))
        enc   = self.tok(full, truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
        ids   = enc["input_ids"].squeeze(0)
        lbs   = ids.clone()
        lbs[:p_len] = -100
        return {"input_ids": ids, "labels": lbs}

    def __len__(self):
        return len(self.data)

    @torch.no_grad()
    def __getitem__(self, idx):
        row    = self.data[idx]
        prompt = row["prompt"]
        real   = self._encode(prompt, row["completion"])

        prompt_ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        fake_ids   = self.opponent.generate(
            prompt_ids,
            max_new_tokens=self.cfg.max_new_tokens,
            n_loops=self.cfg.n_loops,
            temperature=1.0,
        )
        fake_text  = self.tok.decode(fake_ids[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True)
        fake       = self._encode(prompt, fake_text)

        return {"real": real, "fake": fake}


def spin_collate(batch: list[dict]) -> dict:
    def _pad(items):
        max_len = max(x["input_ids"].shape[0] for x in items)
        ids = torch.zeros(len(items), max_len, dtype=torch.long)
        lbs = torch.full((len(items), max_len), -100, dtype=torch.long)
        for i, x in enumerate(items):
            L = x["input_ids"].shape[0]
            ids[i, :L] = x["input_ids"]
            lbs[i, :L] = x["labels"]
        return ids, lbs

    real_ids, real_lbs = _pad([b["real"] for b in batch])
    fake_ids, fake_lbs = _pad([b["fake"] for b in batch])
    return {"real_ids": real_ids, "real_labels": real_lbs,
            "fake_ids": fake_ids, "fake_labels": fake_lbs}


class SPINTrainer:
    def __init__(self, model: nn.Module, cfg: SPINConfig):
        self.model    = model
        self.cfg      = cfg
        self.opponent = copy.deepcopy(model).eval()
        for p in self.opponent.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, betas=(0.9, 0.95),
        )

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        device = next(self.model.parameters()).device
        real_ids = batch["real_ids"].to(device)
        real_lbs = batch["real_labels"].to(device)
        fake_ids = batch["fake_ids"].to(device)
        fake_lbs = batch["fake_labels"].to(device)

        # Live model log-probs (with grad)
        real_logits = self.model(real_ids, n_loops=self.cfg.n_loops)
        fake_logits = self.model(fake_ids, n_loops=self.cfg.n_loops)
        lp_real = _seq_log_probs(self.model, real_ids, real_lbs, self.cfg.n_loops)
        lp_fake = _seq_log_probs(self.model, fake_ids, fake_lbs, self.cfg.n_loops)

        # Reference (opponent) log-probs (no grad)
        with torch.no_grad():
            ref_real = _seq_log_probs(self.opponent, real_ids, real_lbs, self.cfg.n_loops)
            ref_fake = _seq_log_probs(self.opponent, fake_ids, fake_lbs, self.cfg.n_loops)

        # DPO loss
        log_ratio_real = lp_real - ref_real
        log_ratio_fake = lp_fake - ref_fake
        loss = -F.logsigmoid(self.cfg.beta * (log_ratio_real - log_ratio_fake)).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return loss, {
            "loss":    loss.item(),
            "lp_real": lp_real.mean().item(),
            "lp_fake": lp_fake.mean().item(),
        }

    def advance_opponent(self) -> None:
        """Call after each SPIN iteration to update the frozen opponent."""
        self.opponent = copy.deepcopy(self.model).eval()
        for p in self.opponent.parameters():
            p.requires_grad_(False)

    def run_iteration(
        self,
        data_path:      str,
        tokenizer_path: str,
        n_steps:        int  = 500,
        batch_size:     int  = 2,
        log_every:      int  = 50,
    ) -> None:
        device  = next(self.model.parameters()).device
        dataset = SPINDataset(data_path, tokenizer_path, self.opponent, device, self.cfg)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=spin_collate)

        step = 0
        for batch in loader:
            if step >= n_steps:
                break
            loss, metrics = self.step(batch)
            if step % log_every == 0:
                print(f"  SPIN step {step:4d} | loss {metrics['loss']:.4f} | "
                      f"lp_real {metrics['lp_real']:.3f} | lp_fake {metrics['lp_fake']:.3f}")
            step += 1

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tier:            str,
        cfg:             Optional[SPINConfig] = None,
    ) -> "SPINTrainer":
        from anthos.configs import get_training_config
        from anthos.main import Anthos
        cfg = cfg or SPINConfig()
        model_cfg, _ = get_training_config(tier)
        model = Anthos(model_cfg)
        ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        return cls(model, cfg)
