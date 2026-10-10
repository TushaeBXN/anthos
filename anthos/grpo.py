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
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

Hardware note:
  GRPO requires generating K completions per prompt — this is GPU-heavy.
  Run on RunPod (H100 or better) after your hardware upgrade, not on MPS.
  The code runs on MPS for testing but will be slow for real training.

Reward function return values:
  A reward function may return None for a completion to skip it entirely
  (e.g. when ground truth is unparseable). Skipped completions are excluded
  from advantage normalisation and the policy gradient loss — they do not
  contribute zero reward, they contribute nothing.
"""

from __future__ import annotations

import copy
import math
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
    ngram_size:  int = 3,
) -> List[Optional[float]]:
    """
    Quality reward based on:
      - Vocabulary diversity (unique tokens / total tokens)
      - N-gram repetition penalty (fraction of repeated n-grams)
      - Returns None for degenerate completions (too short to score)

    Reward functions may return None to skip a completion entirely —
    it will be excluded from the advantage computation and policy loss.
    Replace or extend this with a trained reward model for stronger signal.
    """
    rewards = []
    for completion in completions:
        ids = completion.tolist()
        if len(ids) < ngram_size + 1:
            rewards.append(None)   # too short to score meaningfully
            continue

        # Vocabulary diversity: unique tokens / total tokens
        diversity = len(set(ids)) / len(ids)

        # N-gram repetition penalty
        ngrams = [tuple(ids[i:i + ngram_size]) for i in range(len(ids) - ngram_size)]
        repeat_rate = 1.0 - len(set(ngrams)) / len(ngrams)

        rewards.append(float(diversity - 0.5 * repeat_rate))

    return rewards


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -1.0):
    """
    Factory: configurable n-gram repetition penalty reward.

    Returns a scalar in [max_penalty, 0]: 0 when no repeats, max_penalty
    when every n-gram is repeated. Combine with quality_reward or use
    standalone as an additive penalty on top of another reward.

    Args:
        ngram_size:  n-gram window (default 3)
        max_penalty: most negative reward (must be <= 0)

    Usage:
        rep_penalty = get_repetition_penalty_reward(ngram_size=4, max_penalty=-0.5)
        rewards = rep_penalty(prompts, completions, tokenizer)
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} must be <= 0")

    def repetition_penalty_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for completion in completions:
            ids = completion.tolist()
            if len(ids) < ngram_size + 1:
                rewards.append(None)
                continue
            ngrams = [tuple(ids[i:i + ngram_size]) for i in range(len(ids) - ngram_size)]
            unique_ratio = len(set(ngrams)) / len(ngrams)
            # scaling = fraction of repeated n-grams; 0 = no repeats
            scaling = 1.0 - unique_ratio
            rewards.append(float(scaling * max_penalty))
        return rewards

    return repetition_penalty_reward


def get_cosine_length_reward(
    min_value_wrong:   float = -1.0,
    max_value_wrong:   float = -0.5,
    min_value_correct: float =  0.5,
    max_value_correct: float =  1.0,
    max_len:           int   = 512,
):
    """
    Factory: cosine-scaled length reward.

    Shorter correct completions get higher reward; longer ones get less.
    Longer incorrect completions are penalised less than shorter ones
    (the model shouldn't give up early just to avoid penalty).

    Requires a correctness signal — pass a `is_correct` kwarg list to the
    returned function, or subclass and override. When is_correct is not
    supplied, all completions are treated as correct (pure length incentive).

    Args:
        min_value_wrong:   floor reward for incorrect completions
        max_value_wrong:   ceiling reward for incorrect completions
        min_value_correct: floor reward for correct completions
        max_value_correct: ceiling reward for correct completions
        max_len:           token length at which cosine reaches its minimum

    Usage:
        cosine_rw = get_cosine_length_reward(max_len=256)
        rewards   = cosine_rw(prompts, completions, tokenizer,
                              is_correct=[True, False, True, ...])
    """
    def cosine_length_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
        is_correct:  Optional[List[Optional[bool]]] = None,
    ) -> List[Optional[float]]:
        if is_correct is None:
            is_correct = [True] * len(completions)

        rewards = []
        for completion, correct in zip(completions, is_correct):
            if correct is None:
                rewards.append(None)
                continue

            gen_len  = len(completion.tolist())
            progress = min(gen_len / max_len, 1.0)
            cosine   = math.cos(progress * math.pi)

            if correct:
                min_v, max_v = min_value_correct, max_value_correct
            else:
                # Swap so longer wrong answers are penalised less
                min_v, max_v = max_value_wrong, min_value_wrong

            reward = min_v + 0.5 * (max_v - min_v) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_length_reward


def get_soft_overlong_punishment(max_completion_len: int, buffer: int = 20):
    """
    Factory: soft overlong punishment for token-level length control.

    Adapted from the DAPO paper (https://huggingface.co/papers/2503.14476),
    Eq. 13. Pairs naturally with Anthos's ACT halting: ACT controls loop
    depth, this controls token length ceiling. Together they give the model
    two orthogonal compute-efficiency signals.

    Three zones:
      [0, max_completion_len - buffer]   → 0.0       (no penalty)
      (max_completion_len - buffer,
       max_completion_len]               → linearly from 0.0 → -1.0  (soft zone)
      (max_completion_len, ∞)            → -1.0      (hard cap)

    The soft zone prevents a hard cliff that can cause instability early in
    GRPO training — the model receives a gradient signal before hitting the
    ceiling rather than being shocked at a single token boundary.

    Args:
        max_completion_len: token length beyond which penalty is -1.0
        buffer:             width of the soft-ramp zone (default 20 tokens)

    Usage:
        overlong = get_soft_overlong_punishment(max_completion_len=256, buffer=32)
        rewards  = overlong(prompts, completions, tokenizer)

    Recommended stacking with quality_reward:
        def combined(prompts, completions, tokenizer):
            q = quality_reward(prompts, completions, tokenizer)
            o = overlong(prompts, completions, tokenizer)
            return [
                None if (qi is None) else qi + oi
                for qi, oi in zip(q, o)
            ]
    """
    if buffer <= 0:
        raise ValueError(f"buffer must be > 0, got {buffer}")
    if max_completion_len <= buffer:
        raise ValueError(
            f"max_completion_len ({max_completion_len}) must be > buffer ({buffer})"
        )

    def soft_overlong_punishment_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for completion in completions:
            length = len(completion.tolist())
            if length <= max_completion_len - buffer:
                rewards.append(0.0)
            elif length <= max_completion_len:
                # Linear ramp: 0.0 at (max - buffer), -1.0 at max
                penalty = (max_completion_len - buffer - length) / buffer
                rewards.append(float(penalty))
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def loop_efficiency_reward(
    loops_used:  torch.Tensor,   # (B, T) from ACT
    max_loops:   int,
) -> torch.Tensor:
    """
    Reward efficient use of compute — fewer loops for the same quality.
    Returns per-sequence scalar reward in [0, 1].

    This pairs with quality_reward: quality ensures the model doesn't just
    halt at loop 1 to minimise compute; loop efficiency ensures it doesn't
    run all 16 loops when 4 would do.
    """
    mean_loops = loops_used.float().mean(dim=-1) / max_loops   # (B,)
    return (1.0 - mean_loops).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Log probability extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_log_probs(
    model,
    input_ids:    torch.Tensor,   # (B, T)
    labels:       torch.Tensor,   # (B, T) — which tokens to score
    n_loops:      int,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Get per-token log probabilities for the label tokens.
    Returns (B, T-1) tensor — zeroed at non-label positions.
    """
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids, n_loops=n_loops)   # (B, T, V)

    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :]                 # (B, T-1, V)
    shift_labels = labels[:, 1:]                     # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    gathered = log_probs.gather(
        -1, shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)                                    # (B, T-1)

    # Zero out padding positions (-100)
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
      - K-completion generation per prompt (batched)
      - Reward computation and group normalisation
      - None-skip: completions the reward function marks None are excluded
      - Policy gradient loss with KL regularisation
      - Loop efficiency reward integration

    Usage:
        trainer = GRPOTrainer(model, ref_model, tokenizer, config, reward_fn)
        for batch in dataloader:
            loss = trainer.step(batch["input_ids"])
            if loss is None:
                continue   # all completions were skipped
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
        """
        Generate K completions for a single prompt.

        The prompt is expanded to a batch of K and generated in one forward
        pass, which is significantly faster than K separate generate() calls.
        Falls back to sequential generation if the model's generate() does
        not support batch size > 1.

        Returns (K, T_p+T_c).
        """
        K = self.cfg.n_completions
        self.model.eval()

        with torch.no_grad():
            # Expand to batch of K: (K, T_p)
            prompt_batch = prompt_ids.expand(K, -1)
            try:
                out = self.model.generate(
                    prompt_batch,
                    max_new_tokens = self.cfg.max_new_tokens,
                    n_loops        = self.cfg.n_loops_train,
                    temperature    = self.cfg.temperature,
                    top_k          = self.cfg.top_k,
                )   # (K, T_p + T_c)
            except (RuntimeError, NotImplementedError):
                # Sequential fallback if batched generation is unsupported
                completions = []
                for _ in range(K):
                    c = self.model.generate(
                        prompt_ids,
                        max_new_tokens = self.cfg.max_new_tokens,
                        n_loops        = self.cfg.n_loops_train,
                        temperature    = self.cfg.temperature,
                        top_k          = self.cfg.top_k,
                    )
                    completions.append(c)

                max_len = max(c.shape[1] for c in completions)
                pad_id  = self.tokenizer.pad_token_id or 0
                out = torch.cat([
                    F.pad(c, (0, max_len - c.shape[1]), value=pad_id)
                    for c in completions
                ], dim=0)

        self.model.train()
        return out   # (K, T_total)

    def _compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Group-relative advantage normalisation.
        A_i = (r_i - mean(r)) / (std(r) + eps)
        Clipped to [-clip_adv, +clip_adv].

        Returns zero advantages when all rewards are identical (std == 0),
        rather than dividing by near-zero and amplifying noise.
        """
        r = torch.tensor(rewards, dtype=torch.float32)
        if self.cfg.normalize_reward and len(r) > 1:
            std = r.std()
            if std < 1e-6:
                # All rewards identical — no useful signal, return zeros
                return torch.zeros_like(r)
            r = (r - r.mean()) / (std + 1e-8)
        return r.clamp(-self.cfg.clip_advantages, self.cfg.clip_advantages)

    def step(
        self,
        prompt_ids:    torch.Tensor,              # (1, T_p)
        extra_rewards: Optional[List[float]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run one GRPO step for a single prompt.

        Returns:
            loss: scalar tensor (differentiable), or None if all completions
                  were skipped by the reward function.
        """
        device = prompt_ids.device
        T_p    = prompt_ids.shape[1]

        # ── 1. Generate K completions (batched) ───────────────────────────
        completions = self._generate_completions(prompt_ids)   # (K, T_total)
        K           = completions.shape[0]

        completion_parts = [completions[i, T_p:] for i in range(K)]
        prompt_parts     = [prompt_ids[0]] * K

        # ── 2. Compute rewards ─────────────────────────────────────────────
        raw_rewards = self.reward_fn(prompt_parts, completion_parts, self.tokenizer)

        if extra_rewards is not None:
            raw_rewards = [
                None if r is None else r + e
                for r, e in zip(raw_rewards, extra_rewards)
            ]

        # Filter out None-skipped completions
        valid_mask = [r is not None for r in raw_rewards]
        if not any(valid_mask):
            return None   # all completions skipped — no loss to compute

        valid_idx     = [i for i, v in enumerate(valid_mask) if v]
        valid_rewards = [raw_rewards[i] for i in valid_idx]
        valid_comps   = completions[valid_idx]   # (V, T_total)

        advantages = self._compute_advantages(valid_rewards).to(device)   # (V,)

        # ── 3. Policy log probs ────────────────────────────────────────────
        labels = valid_comps.clone()
        labels[:, :T_p] = -100

        policy_log_probs = _get_log_probs(
            self.model, valid_comps, labels,
            self.cfg.n_loops_train, requires_grad=True,
        )   # (V, T_total-1)

        # ── 4. Reference KL ───────────────────────────────────────────────
        ref_log_probs = _get_log_probs(
            self.ref_model, valid_comps, labels,
            self.cfg.n_loops_ref, requires_grad=False,
        )   # (V, T_total-1)

        kl = (policy_log_probs - ref_log_probs).mean(dim=-1)   # (V,)

        # ── 5. Policy gradient loss ────────────────────────────────────────
        seq_len      = (labels[:, 1:] >= 0).float().sum(dim=-1).clamp(min=1)
        seq_log_prob = policy_log_probs.sum(dim=-1) / seq_len   # (V,)

        pg_loss  = -(advantages * seq_log_prob).mean()
        kl_loss  = self.cfg.kl_coef * kl.mean()
        entropy  = -policy_log_probs.mean()
        ent_loss = -self.cfg.entropy_bonus * entropy

        return pg_loss + kl_loss + ent_loss

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
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
