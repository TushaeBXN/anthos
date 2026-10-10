"""
anthos/dpo.py — Direct Preference Optimization for Anthos

DPO converts pairwise preference data (chosen vs. rejected completions)
directly into a policy gradient update — no separate reward model needed.

How DPO works:
  Given a prompt x and a pair (y_w, y_l) where y_w is preferred over y_l:

  Loss = -log σ(β * [log π_θ(y_w|x) - log π_ref(y_w|x)]
                   - β * [log π_θ(y_l|x) - log π_ref(y_l|x)])

  β controls how far the policy can move from the reference model.
  Higher β → stays close to reference (conservative).
  Lower  β → can deviate more (aggressive).

  The reference model is a frozen copy of the initial checkpoint.
  A pre-trained SFT checkpoint is the standard starting point.

Why DPO for Anthos:
  - Thought stream training: pairs where one completion "thinks better"
    teach the model to allocate reasoning loops more effectively.
  - No reward model needed: human/LLM annotators mark preferred outputs.
  - Lighter than GRPO: single forward pass per chosen/rejected pair.

Data format (JSONL):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    trainer = DPOTrainer.from_pretrained(
        checkpoint_path = "checkpoints/sft/step_000500.pt",
        model_class     = Anthos,
        cfg             = model_cfg,
        tokenizer_path  = "data/anthos_tokenizer",
        dpo_config      = DPOConfig(beta=0.1),
    )
    dataset = DPODataset("data/preferences.jsonl", trainer.tokenizer)
    for batch in DataLoader(dataset, batch_size=2):
        loss = trainer.step(batch)
        loss.backward()
        optimizer.step()
"""

from __future__ import annotations

import copy
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
class DPOConfig:
    beta:           float = 0.1     # KL penalty; 0.1 is standard, raise to 0.5 for more conservative
    n_loops_policy: int   = 12      # loop depth for policy (trained) model
    n_loops_ref:    int   = 8       # loop depth for reference (frozen) model
    label_smoothing: float = 0.0    # smooth the DPO target (IPO variant at 0.1)
    loss_type:      str   = "dpo"   # "dpo" | "ipo" | "simpo"
    # SimPO params (only used when loss_type="simpo")
    simpo_gamma:    float = 0.5     # reward margin for SimPO
    simpo_length_norm: bool = True  # normalise log-probs by sequence length


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """
    Loads a JSONL preference dataset.

    Each line must have keys: "prompt", "chosen", "rejected".

    Returns a dict with:
        prompt_ids   (T_p,)
        chosen_ids   (T_c,)
        rejected_ids (T_r,)
    """

    def __init__(
        self,
        path:           str,
        tokenizer_path: str   = "gpt2",
        max_length:     int   = 512,
    ):
        import json as _json
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.items = []
        with open(path) as f:
            for line in f:
                obj = _json.loads(line)
                if "prompt" in obj and "chosen" in obj and "rejected" in obj:
                    self.items.append(obj)

        self.max_length = max_length
        print(f"[DPODataset] {len(self.items)} preference pairs loaded from {path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        p_ids = self.tok.encode(item["prompt"],   add_special_tokens=False)
        c_ids = self.tok.encode(item["chosen"],   add_special_tokens=False)
        r_ids = self.tok.encode(item["rejected"], add_special_tokens=False)

        # Truncate to max_length
        p_ids = p_ids[-self.max_length // 2:]
        c_ids = c_ids[:self.max_length // 2]
        r_ids = r_ids[:self.max_length // 2]

        return {
            "prompt_ids":   torch.tensor(p_ids, dtype=torch.long),
            "chosen_ids":   torch.tensor(c_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(r_ids, dtype=torch.long),
        }


def dpo_collate(batch: list[dict]) -> dict:
    """Pad a batch of DPO samples to the same length."""
    def _pad(seqs: list[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        return torch.stack([
            F.pad(s, (0, max_len - s.shape[0]), value=pad_val)
            for s in seqs
        ])

    return {
        "prompt_ids":   _pad([b["prompt_ids"]   for b in batch]),
        "chosen_ids":   _pad([b["chosen_ids"]   for b in batch]),
        "rejected_ids": _pad([b["rejected_ids"] for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability extraction
# ─────────────────────────────────────────────────────────────────────────────

def _sequence_log_prob(
    model:     nn.Module,
    input_ids: torch.Tensor,   # (B, T_p + T_c)
    labels:    torch.Tensor,   # (B, T_p + T_c), -100 on prompt tokens
    n_loops:   int,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Compute per-sequence sum of log P(token | context) over non-masked positions.
    Returns (B,) tensor of log-likelihood sums.
    """
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits, _ = model(input_ids, n_loops=n_loops, return_aux=True)
        # logits: (B, T-1, V); labels: (B, T)
        log_probs = F.log_softmax(logits, dim=-1)   # (B, T-1, V)
        targets   = labels[:, 1:]                    # (B, T-1)
        mask      = (targets >= 0).float()           # (B, T-1)
        targets_c = targets.clamp(min=0)
        per_tok   = log_probs.gather(2, targets_c.unsqueeze(2)).squeeze(2)  # (B, T-1)
        return (per_tok * mask).sum(dim=-1)          # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# DPO loss functions
# ─────────────────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps:   torch.Tensor,   # (B,)
    policy_rejected_logps: torch.Tensor,   # (B,)
    ref_chosen_logps:      torch.Tensor,   # (B,)
    ref_rejected_logps:    torch.Tensor,   # (B,)
    beta:           float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Standard DPO loss (Rafailov et al. 2023).
    Returns (loss, metrics_dict).
    """
    chosen_rewards   = beta * (policy_chosen_logps   - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_margin    = chosen_rewards - rejected_rewards

    if label_smoothing > 0:
        # IPO-style smoothing (Azar et al. 2023)
        loss = -F.logsigmoid(reward_margin) * (1 - label_smoothing) \
               - F.logsigmoid(-reward_margin) * label_smoothing
    else:
        loss = -F.logsigmoid(reward_margin)

    acc = (reward_margin > 0).float().mean()
    return loss.mean(), {
        "dpo_loss":       loss.mean().item(),
        "reward_margin":  reward_margin.mean().item(),
        "chosen_reward":  chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "preference_acc": acc.item(),
    }


def simpo_loss(
    policy_chosen_logps:   torch.Tensor,   # (B,)
    policy_rejected_logps: torch.Tensor,   # (B,)
    chosen_lengths:   torch.Tensor,        # (B,) number of valid tokens
    rejected_lengths: torch.Tensor,        # (B,)
    beta:  float = 0.1,
    gamma: float = 0.5,
    length_norm: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    SimPO loss (Meng et al. 2024) — no reference model needed.

    Normalises log-probs by length to prevent length bias, then adds a
    margin γ to ensure chosen > rejected by at least γ/β.
    """
    if length_norm:
        chosen_lp   = policy_chosen_logps   / chosen_lengths.clamp(min=1).float()
        rejected_lp = policy_rejected_logps / rejected_lengths.clamp(min=1).float()
    else:
        chosen_lp, rejected_lp = policy_chosen_logps, policy_rejected_logps

    reward_margin = beta * (chosen_lp - rejected_lp) - gamma
    loss = -F.logsigmoid(reward_margin)
    acc  = (reward_margin > 0).float().mean()
    return loss.mean(), {
        "simpo_loss":     loss.mean().item(),
        "reward_margin":  reward_margin.mean().item(),
        "preference_acc": acc.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DPOTrainer:
    """
    DPO trainer for Anthos.

    Holds a policy model (trained) and a frozen reference model.
    Call step() with a batch of preference pairs to compute the loss.

    The reference model is initialised from the same checkpoint as the policy
    and frozen — it never receives gradient updates.
    """

    def __init__(
        self,
        model:      nn.Module,
        ref_model:  nn.Module,
        tokenizer,
        cfg:        DPOConfig = DPOConfig(),
    ):
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg       = cfg

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute DPO/SimPO loss for one batch.

        Args:
            batch: dict with keys prompt_ids, chosen_ids, rejected_ids
                   All (B, T) tensors on the correct device.

        Returns:
            (loss, metrics_dict) — loss is differentiable.
        """
        device      = next(self.model.parameters()).device
        prompt_ids  = batch["prompt_ids"].to(device)
        chosen_ids  = batch["chosen_ids"].to(device)
        rejected_ids= batch["rejected_ids"].to(device)
        B           = prompt_ids.shape[0]

        def _build_input_and_labels(prompt, completion):
            # Concatenate prompt + completion; mask prompt positions with -100
            T_p = prompt.shape[1]
            T_c = completion.shape[1]
            combined = torch.cat([prompt, completion], dim=1)   # (B, T_p + T_c)
            labels   = combined.clone()
            labels[:, :T_p] = -100
            return combined, labels, T_c

        chosen_input,   chosen_labels,   T_c = _build_input_and_labels(prompt_ids, chosen_ids)
        rejected_input, rejected_labels, T_r = _build_input_and_labels(prompt_ids, rejected_ids)

        # Policy log-probs (with grad)
        policy_chosen_logps = _sequence_log_prob(
            self.model, chosen_input, chosen_labels,
            self.cfg.n_loops_policy, requires_grad=True,
        )
        policy_rejected_logps = _sequence_log_prob(
            self.model, rejected_input, rejected_labels,
            self.cfg.n_loops_policy, requires_grad=True,
        )

        if self.cfg.loss_type == "simpo":
            chosen_lens   = (chosen_labels[:, 1:] >= 0).float().sum(dim=-1)
            rejected_lens = (rejected_labels[:, 1:] >= 0).float().sum(dim=-1)
            return simpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                chosen_lens, rejected_lens,
                beta=self.cfg.beta, gamma=self.cfg.simpo_gamma,
                length_norm=self.cfg.simpo_length_norm,
            )

        # Reference log-probs (no grad)
        ref_chosen_logps = _sequence_log_prob(
            self.ref_model, chosen_input, chosen_labels,
            self.cfg.n_loops_ref, requires_grad=False,
        )
        ref_rejected_logps = _sequence_log_prob(
            self.ref_model, rejected_input, rejected_labels,
            self.cfg.n_loops_ref, requires_grad=False,
        )

        return dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps,    ref_rejected_logps,
            beta=self.cfg.beta, label_smoothing=self.cfg.label_smoothing,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_class,
        cfg,
        tokenizer_path:  str       = "gpt2",
        dpo_config:      DPOConfig = DPOConfig(),
        device:          str       = "cpu",
    ) -> "DPOTrainer":
        """
        Load a pretrained Anthos checkpoint and set up DPO training.

        The reference model is a frozen deep copy of the loaded checkpoint.
        """
        from transformers import AutoTokenizer

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        policy  = model_class(cfg).to(device)
        policy.load_state_dict(ckpt["model"], strict=False)

        ref = copy.deepcopy(policy)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(policy, ref, tokenizer, dpo_config)
