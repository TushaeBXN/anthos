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
    # Clip-higher (DAPO): asymmetric ratio clipping.
    # Low end clips at 1-clip_eps; high end at 1+clip_eps*clip_high_mu.
    # clip_high_mu > 1 allows larger upward updates while still bounding
    # downward ones — stops the policy collapsing on hard prompts.
    # Set clip_eps=0.0 to disable ratio clipping (REINFORCE mode).
    clip_eps:         float = 0.2     # lower clip epsilon
    clip_high_mu:     float = 3.0     # high clip = 1 + clip_eps * clip_high_mu


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


def get_format_reward(pattern: str):
    """
    Factory: binary reward for completions that match a required output format.

    Returns 1.0 when the completion matches the regex, 0.0 otherwise.
    Useful for teaching Anthos to emit structured responses (reasoning traces,
    JSON, tool calls, etc.) — use early in GRPO training while the model is
    still learning format, then phase it out once format is stable.

    The default pattern enforces open-r1-style chain-of-thought:
        <think>
        ...reasoning...
        </think>
        <answer>
        ...answer...
        </answer>

    Since Anthos's thought tokens are architectural (internal, leave no trace),
    this reward is optional — use it only if you want the *sequence stream*
    to also emit an explicit reasoning trace in its output text.

    Args:
        pattern: regex pattern the full decoded completion must match

    Usage:
        # Chain-of-thought output format
        fmt = get_format_reward(
            pattern=r"^<think>\\n.*?\\n</think>\\n<answer>\\n.*?\\n</answer>$"
        )

        # JSON output
        fmt = get_format_reward(pattern=r'^\\{.*\\}$')

        rewards = fmt(prompts, completions, tokenizer)
    """
    import re
    compiled = re.compile(pattern, re.DOTALL | re.MULTILINE)

    def format_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for completion in completions:
            ids  = completion.tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            rewards.append(1.0 if compiled.match(text) else 0.0)
        return rewards

    return format_reward


def get_tag_count_reward(tags: Optional[list] = None):
    """
    Factory: partial-credit reward for tag structure.

    Returns a score in [0.0, 1.0] based on how many required tags are
    present and correctly formed — 0.25 per tag when there are 4 tags,
    scaled to 1.0 / len(tags) per tag in general.

    Gives gradient signal even when the model hasn't learned the full
    format yet — it gets credit for each tag it produces correctly.
    Use in the early phase of format training alongside get_format_reward;
    once format_reward is consistently 1.0, drop this one.

    Args:
        tags: list of (substring, expected_count) pairs.
              Default is the open-r1 chain-of-thought set:
              [("<think>\\n", 1), ("\\n</think>\\n", 1),
               ("\\n<answer>\\n", 1), ("\\n</answer>", 1)]

    Usage:
        # Default CoT tags
        tag_rw = get_tag_count_reward()

        # Custom tags — e.g. tool-call format
        tag_rw = get_tag_count_reward([
            ("<tool_call>\\n", 1),
            ("\\n</tool_call>", 1),
        ])

        rewards = tag_rw(prompts, completions, tokenizer)
    """
    if tags is None:
        tags = [
            ("<think>\n",   1),
            ("\n</think>\n", 1),
            ("\n<answer>\n", 1),
            ("\n</answer>",  1),
        ]

    per_tag_score = 1.0 / len(tags)

    def tag_count_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for completion in completions:
            ids  = completion.tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            score = 0.0
            for substring, expected_count in tags:
                if text.count(substring) == expected_count:
                    score += per_tag_score
            rewards.append(float(score))
        return rewards

    return tag_count_reward


def get_accuracy_reward(answer_pattern: str = r"<answer>(.*?)</answer>"):
    """
    Factory: binary reward for verifiable answer correctness.

    Parses the model's completion for a final answer tag and compares it
    to a ground-truth string embedded in the prompt.  Returns:
      +1.0  correct answer
       0.0  wrong answer or unparseable
      None  prompt carries no ground-truth (skip — don't penalise)

    Prompt format expected:
        "... <ground_truth>42</ground_truth> ..."

    Completion format expected:
        "... <answer>42</answer>"

    Usage:
        acc_reward = get_accuracy_reward()
        trainer = GRPOTrainer(..., reward_fn=acc_reward)
    """
    import re as _re
    ans_pat = _re.compile(answer_pattern, _re.DOTALL)
    gt_pat  = _re.compile(r"<ground_truth>(.*?)</ground_truth>", _re.DOTALL)

    def accuracy_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for prompt_ids, comp_ids in zip(prompts, completions):
            prompt_text = tokenizer.decode(
                prompt_ids.tolist(), skip_special_tokens=True
            )
            gt_match = gt_pat.search(prompt_text)
            if gt_match is None:
                rewards.append(None)   # no ground truth — skip
                continue

            ground_truth = gt_match.group(1).strip()
            comp_text    = tokenizer.decode(
                comp_ids.tolist(), skip_special_tokens=True
            )
            ans_match = ans_pat.search(comp_text)
            if ans_match is None:
                rewards.append(0.0)    # didn't produce an answer tag
                continue

            predicted = ans_match.group(1).strip()
            rewards.append(1.0 if predicted == ground_truth else 0.0)
        return rewards

    return accuracy_reward


def get_token_level_reward(
    think_tag:       str   = "<think>",
    think_end_tag:   str   = "</think>",
    think_weight:    float = 0.3,
    answer_weight:   float = 1.0,
):
    """
    Factory: per-token reward weights based on which section a token is in.

    Returns a reward function that produces a list of (T,) tensors — one
    per completion — where each position's value is:
      - think_weight   inside <think>...</think> blocks (reasoning tokens)
      - answer_weight  everywhere else (answer tokens)

    These per-token weights are multiplied against the sequence-level reward
    in GRPOTrainer.step() when passed as extra_rewards.  This teaches the
    model to allocate its reasoning budget more efficiently: reasoning tokens
    are down-weighted so the policy gradient focuses on answer quality, while
    still rewarding non-degenerate thinking.

    Usage:
        tok_rw = get_token_level_reward(think_weight=0.2, answer_weight=1.0)
        trainer = GRPOTrainer(..., reward_fn=combined_reward)
        # Pass tok_rw output as extra_rewards in trainer.step()
    """
    def token_level_reward(
        prompts:     List[torch.Tensor],
        completions: List[torch.Tensor],
        tokenizer,
    ) -> List[Optional[float]]:
        rewards = []
        for comp_ids in completions:
            text   = tokenizer.decode(comp_ids.tolist(), skip_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(comp_ids.tolist())
            T      = len(tokens)

            # Locate think regions character-by-character then map back to tokens
            weights = [answer_weight] * T
            pos     = 0
            char_offset = 0
            token_char_starts = []
            for tok in tokens:
                token_char_starts.append(char_offset)
                char_offset += len(tok.replace("Ġ", " ").replace("Ċ", "\n"))

            inside = False
            for i, (tok, start) in enumerate(zip(tokens, token_char_starts)):
                substr = text[start:]
                if not inside and substr.startswith(think_tag):
                    inside = True
                if inside:
                    weights[i] = think_weight
                if inside and substr.startswith(think_end_tag):
                    inside = False

            # Aggregate to a single scalar: mean weight across tokens
            rewards.append(float(sum(weights) / max(T, 1)))
        return rewards

    return token_level_reward


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

        # ── Online filtering ───────────────────────────────────────────────
        # Skip the step entirely (no forward/backward) when there's nothing
        # to learn from:
        #   1. All reward functions returned None (unscoreable completions)
        #   2. All valid rewards are identical (zero-variance → zero gradient)
        # This is more aggressive than just zeroing advantages after the fact —
        # we avoid wasting compute on a guaranteed zero-update step.
        valid_mask = [r is not None for r in raw_rewards]
        if not any(valid_mask):
            return None   # all completions unscoreable

        valid_idx     = [i for i, v in enumerate(valid_mask) if v]
        valid_rewards = [raw_rewards[i] for i in valid_idx]

        # Check for zero-variance before allocating any tensors
        if len(valid_rewards) > 1:
            r_min = min(valid_rewards)
            r_max = max(valid_rewards)
            if r_max - r_min < 1e-6:
                return None   # all identical rewards — no gradient signal

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

        # ── 5. Policy gradient loss (Dr. GRPO + clip-higher) ──────────────
        # Dr. GRPO: broadcast sequence advantage to every token, average over
        # valid tokens — unbiased vs. the divide-by-seq_len formulation.
        #
        # Clip-higher (DAPO): use importance-sampling ratios with asymmetric
        # clipping.  Low end: 1-ε (standard PPO bound).
        # High end: 1+ε*μ (allows larger upward updates, blocks collapse).
        # When clip_eps==0 this reduces to plain REINFORCE.
        token_mask = (labels[:, 1:] >= 0).float()                  # (V, T-1)
        n_valid    = token_mask.sum().clamp(min=1)

        adv_per_token = advantages.unsqueeze(1) * token_mask        # (V, T-1)

        if self.cfg.clip_eps > 0:
            # Ratio r_t = exp(log π_θ - log π_ref) per token
            ratio = torch.exp(policy_log_probs - ref_log_probs.detach())  # (V, T-1)
            lo = 1.0 - self.cfg.clip_eps
            hi = 1.0 + self.cfg.clip_eps * self.cfg.clip_high_mu
            clipped_ratio = ratio.clamp(lo, hi)
            # Take the pessimistic bound (standard PPO trick)
            pg_per_token  = torch.min(ratio * adv_per_token,
                                      clipped_ratio * adv_per_token)
            pg_loss = -pg_per_token.sum() / n_valid
        else:
            pg_loss = -(adv_per_token * policy_log_probs).sum() / n_valid

        kl_loss  = self.cfg.kl_coef * (kl * token_mask.mean(dim=-1)).mean()
        entropy  = -(policy_log_probs * token_mask).sum() / n_valid
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
