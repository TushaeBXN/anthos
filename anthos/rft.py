"""
anthos/rft.py — Rejection Sampling Fine-Tuning for Anthos

RFT is the simplest self-improvement loop:
  1. For each prompt, generate K completions
  2. Score each completion with a verifiable reward (e.g. accuracy_reward)
  3. Keep only the correct completions
  4. SFT on the kept completions as if they were gold labels

This creates a self-distillation effect: the model learns from its own
best outputs, gradually pulling the "success" distribution upward.

Why RFT before GRPO:
  RFT is an excellent warm-up for GRPO.  It stabilises the policy on
  easy prompts (where the model already succeeds ~50% of the time) before
  GRPO introduces the trickier credit-assignment problem.  Running RFT for
  1-2k steps then switching to GRPO typically outperforms GRPO cold-start.

Data format:
  Input:  JSONL with "prompt" field (and optionally "answer" for verification)
  Output: JSONL with "prompt" + "completion" (filtered to correct only)

Usage:
    from anthos.rft import RFTDataCollector, run_rft_epoch

    collector = RFTDataCollector(
        model          = model,
        tokenizer      = tokenizer,
        reward_fn      = get_accuracy_reward(),
        n_completions  = 8,
        temperature    = 0.8,
        device         = "cuda",
    )
    kept = collector.collect("data/math_prompts.jsonl", output="data/rft_round1.jsonl")
    print(f"Kept {kept} / total correct completions")

    # Then SFT on data/rft_round1.jsonl with --tier sft
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RFTConfig:
    n_completions:   int   = 8       # K completions per prompt
    temperature:     float = 0.8
    top_k:           int   = 50
    max_new_tokens:  int   = 256
    n_loops:         int   = 12
    min_reward:      float = 0.5     # completions with reward >= this are kept
    deduplicate:     bool  = True    # remove exact-duplicate completions within a group
    max_kept_per_prompt: int = 4     # cap kept completions per prompt


# ─────────────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────────────

class RFTDataCollector:
    """
    Generates completions and filters to high-reward ones.

    The collected completions are written to a JSONL file compatible with
    train.py's SFT tier (via get_chat_dataloader).
    """

    def __init__(
        self,
        model:        nn.Module,
        tokenizer,
        reward_fn:    Callable,
        cfg:          RFTConfig = RFTConfig(),
        device:       str       = "cuda",
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.cfg       = cfg
        self.device    = device

    @torch.no_grad()
    def _generate(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Generate K completions for a single prompt. Returns (K, T_total)."""
        self.model.eval()
        K      = self.cfg.n_completions
        prompt = prompt_ids.unsqueeze(0).to(self.device)   # (1, T_p)

        # Batch expand for parallel generation
        prompt_k = prompt.expand(K, -1)
        try:
            out = self.model.generate(
                prompt_k,
                max_new_tokens = self.cfg.max_new_tokens,
                n_loops        = self.cfg.n_loops,
                temperature    = self.cfg.temperature,
                top_k          = self.cfg.top_k,
            )
            return out   # (K, T_total)
        except Exception:
            # Fallback: generate one at a time
            completions = []
            for _ in range(K):
                out = self.model.generate(
                    prompt,
                    max_new_tokens = self.cfg.max_new_tokens,
                    n_loops        = self.cfg.n_loops,
                    temperature    = self.cfg.temperature,
                    top_k          = self.cfg.top_k,
                )
                completions.append(out)
            max_len = max(c.shape[1] for c in completions)
            pad_id  = self.tokenizer.pad_token_id or 0
            return torch.cat([
                F.pad(c, (0, max_len - c.shape[1]), value=pad_id)
                for c in completions
            ], dim=0)

    def collect_batch(
        self,
        prompts: List[str],
    ) -> List[dict]:
        """
        Generate and filter completions for a list of prompt strings.

        Returns list of dicts: {"prompt": str, "completion": str, "reward": float}
        """
        kept = []
        for prompt_text in prompts:
            prompt_ids = torch.tensor(
                self.tokenizer.encode(prompt_text, add_special_tokens=False),
                dtype=torch.long,
            )
            T_p          = prompt_ids.shape[0]
            completions  = self._generate(prompt_ids)   # (K, T_total)
            comp_parts   = [completions[i, T_p:] for i in range(self.cfg.n_completions)]
            prompt_parts = [prompt_ids] * self.cfg.n_completions

            rewards = self.reward_fn(prompt_parts, comp_parts, self.tokenizer)

            # Keep completions above min_reward
            seen_texts = set()
            n_kept = 0
            for comp_ids, reward in zip(comp_parts, rewards):
                if reward is None or reward < self.cfg.min_reward:
                    continue
                if n_kept >= self.cfg.max_kept_per_prompt:
                    break
                comp_text = self.tokenizer.decode(
                    comp_ids.tolist(), skip_special_tokens=True
                )
                if self.cfg.deduplicate and comp_text in seen_texts:
                    continue
                seen_texts.add(comp_text)
                kept.append({
                    "prompt":     prompt_text,
                    "completion": comp_text,
                    "reward":     float(reward),
                })
                n_kept += 1

        return kept

    def collect(
        self,
        input_path:  str,
        output_path: str,
        prompt_key:  str = "prompt",
        max_prompts: Optional[int] = None,
    ) -> int:
        """
        Process a full JSONL file of prompts and write kept completions.

        Args:
            input_path:  JSONL with at least a "prompt" field per line
            output_path: JSONL to write kept (prompt, completion, reward) pairs
            prompt_key:  field name containing the prompt text
            max_prompts: cap on number of prompts processed

        Returns:
            int: number of kept completions written
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = []
        with open(input_path) as f:
            for i, line in enumerate(f):
                if max_prompts and i >= max_prompts:
                    break
                obj = json.loads(line.strip())
                if prompt_key in obj:
                    prompts.append(obj[prompt_key])

        print(f"[RFT] Collecting from {len(prompts)} prompts "
              f"(K={self.cfg.n_completions}, min_reward={self.cfg.min_reward})...")

        total_kept = 0
        with open(output_path, "w") as out_f:
            for i, prompt in enumerate(prompts):
                kept = self.collect_batch([prompt])
                for item in kept:
                    out_f.write(json.dumps(item) + "\n")
                total_kept += len(kept)
                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(prompts)}] kept {total_kept} so far", flush=True)

        rate = total_kept / max(len(prompts), 1)
        print(f"[RFT] Done. Kept {total_kept} completions "
              f"({rate:.1%} yield) → {output_path}")
        return total_kept


# ─────────────────────────────────────────────────────────────────────────────
# Iterative RFT loop
# ─────────────────────────────────────────────────────────────────────────────

def run_rft_epoch(
    model:        nn.Module,
    tokenizer,
    reward_fn:    Callable,
    prompts_path: str,
    output_path:  str,
    rft_cfg:      RFTConfig = RFTConfig(),
    device:       str       = "cuda",
    max_prompts:  Optional[int] = None,
) -> int:
    """
    Run one full RFT collection epoch.

    Generates K completions per prompt, keeps the correct ones, and writes
    them to output_path for subsequent SFT training with train.py --tier sft.

    Typical RFT loop:
        Round 1: collect from base model → data/rft_r1.jsonl
        Train:   python train.py --tier sft --resume <ckpt> (data=rft_r1)
        Round 2: collect from round-1 model → data/rft_r2.jsonl
        Train:   ...

    Returns number of kept completions (yield metric).
    """
    collector = RFTDataCollector(
        model=model, tokenizer=tokenizer,
        reward_fn=reward_fn, cfg=rft_cfg, device=device,
    )
    return collector.collect(
        input_path=prompts_path, output_path=output_path,
        max_prompts=max_prompts,
    )
