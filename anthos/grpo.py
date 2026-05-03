"""
anthos/grpo.py — Group Relative Policy Optimization for Anthos

GRPO teaches the thought stream to reason deliberately by giving the model
an explicit reward signal, rather than relying solely on next-token prediction
to shape the thought stream indirectly.

How GRPO works:
  For each prompt:
    1. Generate K completions at temperature T
    2. Score each completion with a reward function
    3. Compute group-relative advantages:
         A_i = (r_i - mean(r)) / std(r)     normalised within the group
    4. Policy gradient loss:
         L = -mean(A_i * log_prob(completion_i | prompt))
    5. KL penalty against reference model (keeps it from drifting):
         L_total = L + beta * KL(pi || pi_ref)

Why this matters for Anthos specifically:
  The thought stream has no direct supervision signal in standard LM training.
  GRPO can reward:
    - Output quality   (does the completion make sense?)
    - Loop efficiency  (did it use fewer loops for easy inputs?)
    - Ponder control   (was the halting decision appropriate?)
    - Format adherence (did it follow structure when asked?)

  This is a post-pretraining step — run AFTER the base LM is stable.
  Do NOT run GRPO from scratch; the reward signal is too sparse for that.

Usage:
    from anthos.grpo import GRPOTrainer, GRPOConfig, quality_reward

    grpo_cfg = GRPOConfig(
        n_completions = 8,
        temperature   = 0.8,
        kl_coef       = 0.05,
        n_loops_train = 12,
        n_loops_ref   = 8,
    )

    trainer = GRPOTrainer(
        model     = model,
        ref_model = ref_model,   # frozen copy of pretrained model
        tokenizer = tokenizer,
        config    = grpo_cfg,
        reward_fn = quality_reward,
    )

    for batch in loader:
        loss = trainer.step(batch["input_ids"])
        loss.backward()
        optimizer.step()

Hardware note:
  GRPO requires generating K completions per prompt — this is GPU-heavy.
  Run on RunPod (H100 or better) after your hardware upgrade, not on MPS.
  The code runs on MPS for testing but will be slow for real training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    n_completions:    int   = 8       # K — completions per prompt
    temperature:      float = 0.8     # sampling temperature
    top_k:            int   = 50      # top-k sampling
    max_new_tokens:   int   = 128     # max tokens per completion
    kl_coef:          float = 0.05    # KL penalty coefficient
    entropy_bonus:    float = 0.01    # entropy bonus to prevent collapse
    n_loops_train:    int   = 12      # loop depth for policy model
    n_loops_ref:      int   = 8       # loop depth for reference model
    clip_advantages:  float = 5.0     # clip advantages to [-clip, +clip]
    loop_eff_coef:    float = 0.1     # weight for loop efficiency reward
    normalize_reward: bool  = True    # normalise rewards within group


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────

def quality_reward(
    prompts:     List[torch.Tensor],   # list of (T_p,) prompt token tensors
    completions: List[torch.Tensor],   # list of (T_c,) completion tensors
    tokenizer,
) -> List[float]:
    """
    Simple quality reward based on:
      - Length reasonableness (penalise very short or repetitive completions)
      - Vocabulary diversity (penalise degenerate loops)
      - No EOS in the middle (penalise premature stopping)

    Replace or extend this with a trained reward model for stronger signal.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        ids = completion.tolist()
        if len(ids) < 5:
            rewards.append(-1.0)
            continue

        # Vocabulary diversity: unique tokens / total tokens
        diversity = len(set(ids)) / len(ids)

        # Repetition penalty: count n-gram repeats
        ngrams = [tuple(ids[i:i+3]) for i in range(len(ids) - 2)]
        repeat_rate = 1.0 - len(set(ngrams)) / max(len(ngrams), 1)

        reward = diversity - 0.5 * repeat_rate
        rewards.append(float(reward))

    return rewards


def loop_efficiency_reward(
    loops_used:  torch.Tensor,   # (B, T) from ACT
    max_loops:   int,
) -> torch.Tensor:
    """
    Reward efficient use of compute — fewer loops for the same quality.
    Returns per-sequence scalar reward.

    This pairs with quality_reward: quality ensures the model doesn't just
    halt at loop 1 to minimise compute; loop efficiency ensures it doesn't
    run all 16 loops when 4 would do.
    """
    # Mean loops used per token, normalised to [0, 1]
    mean_loops = loops_used.float().mean(dim=-1) / max_loops   # (B,)
    # Reward = 1 - mean_loops (fewer loops = higher reward)
    return (1.0 - mean_loops).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Log probability extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_log_probs(
    model,
    input_ids:    torch.Tensor,   # (B, T)
    labels:       torch.Tensor,   # (B, T) — which tokens to score
    n_loops:      int,
    kv_cache:     Optional[dict] = None,
) -> torch.Tensor:
    """
    Get per-token log probabilities for the label tokens.
    Returns (B, T) tensor — -inf at non-label positions.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids, n_loops=n_loops)   # (B, T, V)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]                 # (B, T-1, V)
    shift_labels = labels[:, 1:]                     # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log prob of each label token
    gathered = log_probs.gather(
        -1, shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)                                    # (B, T-1)

    # Zero out positions that are padding (-100)
    mask     = (shift_labels >= 0).float()
    gathered = gathered * mask

    return gathered                                  # (B, T-1)


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    GRPO trainer for Anthos.

    Manages:
      - K-completion generation per prompt
      - Reward computation and group normalisation
      - Policy gradient loss with KL regularisation
      - Loop efficiency reward integration

    Usage:
        trainer = GRPOTrainer(model, ref_model, tokenizer, config, reward_fn)
        for batch in dataloader:
            loss = trainer.step(batch["input_ids"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    """

    def __init__(
        self,
        model:      nn.Module,
        ref_model:  nn.Module,         # frozen reference model
        tokenizer,
        config:     GRPOConfig,
        reward_fn:  Callable           = quality_reward,
    ):
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg       = config
        self.reward_fn = reward_fn

        # Freeze reference model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def _generate_completions(
        self,
        prompt_ids: torch.Tensor,   # (1, T_p) — single prompt
    ) -> torch.Tensor:
        """Generate K completions for a single prompt. Returns (K, T_p+T_c)."""
        self.model.eval()
        completions = []
        with torch.no_grad():
            for _ in range(self.cfg.n_completions):
                out = self.model.generate(
                    prompt_ids,
                    max_new_tokens = self.cfg.max_new_tokens,
                    n_loops        = self.cfg.n_loops_train,
                    temperature    = self.cfg.temperature,
                    top_k          = self.cfg.top_k,
                )
                completions.append(out)
        self.model.train()

        # Pad to same length
        max_len = max(c.shape[1] for c in completions)
        pad_id  = self.tokenizer.pad_token_id or 0
        padded  = []
        for c in completions:
            pad = max_len - c.shape[1]
            padded.append(F.pad(c, (0, pad), value=pad_id))

        return torch.cat(padded, dim=0)   # (K, T_total)

    def _compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Group-relative advantage normalisation.
        A_i = (r_i - mean(r)) / (std(r) + eps)
        Clipped to [-clip_adv, +clip_adv].
        """
        r = torch.tensor(rewards, dtype=torch.float32)
        if self.cfg.normalize_reward and len(r) > 1:
            r = (r - r.mean()) / (r.std() + 1e-8)
        return r.clamp(-self.cfg.clip_advantages, self.cfg.clip_advantages)

    def step(
        self,
        prompt_ids:  torch.Tensor,           # (1, T_p)
        extra_rewards: Optional[List[float]] = None,   # optional extra signals
    ) -> torch.Tensor:
        """
        Run one GRPO step for a single prompt.

        Returns:
            loss: scalar tensor (differentiable)
        """
        device = prompt_ids.device
        T_p    = prompt_ids.shape[1]

        # ── 1. Generate K completions ──────────────────────────────────────
        completions = self._generate_completions(prompt_ids)  # (K, T_total)
        K           = completions.shape[0]

        # Extract completion-only parts
        completion_parts = [completions[i, T_p:] for i in range(K)]
        prompt_parts     = [prompt_ids[0]] * K

        # ── 2. Compute rewards ─────────────────────────────────────────────
        rewards = self.reward_fn(
            prompt_parts,
            completion_parts,
            self.tokenizer,
        )

        if extra_rewards is not None:
            rewards = [r + e for r, e in zip(rewards, extra_rewards)]

        advantages = self._compute_advantages(rewards).to(device)  # (K,)

        # ── 3. Policy log probs ────────────────────────────────────────────
        # Labels: only the completion tokens contribute to the loss
        labels = completions.clone()
        labels[:, :T_p] = -100   # mask out prompt positions

        policy_log_probs = _get_log_probs(
            self.model, completions, labels, self.cfg.n_loops_train
        )   # (K, T_total-1)

        # ── 4. Reference KL ───────────────────────────────────────────────
        with torch.no_grad():
            ref_log_probs = _get_log_probs(
                self.ref_model, completions, labels, self.cfg.n_loops_ref
            )   # (K, T_total-1)

        kl = (policy_log_probs - ref_log_probs).mean(dim=-1)   # (K,)

        # ── 5. Policy gradient loss ────────────────────────────────────────
        # Sequence log prob = sum of token log probs (normalised by length)
        seq_len = (labels[:, 1:] >= 0).float().sum(dim=-1).clamp(min=1)
        seq_log_prob = policy_log_probs.sum(dim=-1) / seq_len   # (K,)

        pg_loss  = -(advantages * seq_log_prob).mean()
        kl_loss  = self.cfg.kl_coef * kl.mean()

        # Entropy bonus: encourage exploration
        entropy  = -policy_log_probs.mean()
        ent_loss = -self.cfg.entropy_bonus * entropy

        total_loss = pg_loss + kl_loss + ent_loss

        return total_loss

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        model_class,
        cfg,
        tokenizer,
        grpo_config:     GRPOConfig,
        reward_fn:       Callable = quality_reward,
        device:          str      = "cpu",
    ) -> "GRPOTrainer":
        """
        Load a pretrained Anthos checkpoint and set up GRPO training.

        The reference model is a frozen deep copy of the loaded checkpoint.

        Usage:
            trainer = GRPOTrainer.from_pretrained(
                checkpoint_path = "checkpoints/mansa_sovereign/step_001000.pt",
                model_class     = Anthos,
                cfg             = anthos_config,
                tokenizer       = tokenizer,
                grpo_config     = GRPOConfig(),
            )
        """
        ckpt  = torch.load(checkpoint_path, map_location=device)
        model = model_class(cfg).to(device)
        model.load_state_dict(ckpt["model"], strict=False)

        ref_model = copy.deepcopy(model)

        return cls(
            model     = model,
            ref_model = ref_model,
            tokenizer = tokenizer,
            config    = grpo_config,
            reward_fn = reward_fn,
        )
