"""
rsi.py — Recursive Self-Improvement (RSI)

Anthos improves itself iteratively without external supervision:

  Iteration t:
    1. Self-Evaluate   — model scores its own outputs against a rubric
    2. Self-Distill    — best-of-N sampling; keep top responses as SFT data
    3. Auto-Curriculum — advance task difficulty when competence plateaus
    4. Self-Play       — SPIN loop against frozen copy of previous iteration
    5. ORPO            — align on self-generated preference pairs
    6. Regression Gate — reject iteration if capability drops below threshold
    → Iteration t+1: stronger model with same compute budget

The key insight: the model's own outputs serve as both the training signal
AND the evaluation signal — eliminating the need for human labels after the
initial seed set.

Safety note: RSI loops should be run with a regression gate and capped
iteration counts. Unconstrained self-improvement can amplify biases present
in early iterations.

Usage:
    from anthos.rsi import RSIConfig, RSILoop

    loop = RSILoop.from_pretrained(
        "checkpoints/sft/step_001000.pt",
        tier="sft",
        cfg=RSIConfig(n_iterations=5, best_of_n=8),
    )
    loop.run(
        seed_prompts="data/seed_prompts.txt",
        output_dir="checkpoints/rsi/",
    )
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RSIConfig:
    # Outer loop
    n_iterations:    int   = 5        # number of RSI cycles
    n_steps_per_iter: int  = 200      # SFT steps per iteration

    # Self-distillation
    best_of_n:       int   = 8        # candidates per prompt
    top_k_keep:      float = 0.5      # keep top-50% by self-eval score
    n_loops_gen:     int   = 8        # recurrent depth during generation
    max_new_tokens:  int   = 512
    temperature:     float = 0.8
    gen_top_k:       int   = 50

    # Self-evaluation rubric weights (all must sum ≤ 1; remainder = coherence)
    rubric_helpfulness: float = 0.35
    rubric_accuracy:    float = 0.35
    rubric_safety:      float = 0.20
    rubric_conciseness: float = 0.10

    # SPIN / ORPO fine-tuning
    lr:              float = 5e-6
    batch_size:      int   = 2
    use_orpo:        bool  = True     # apply ORPO on self-generated pairs
    use_spin:        bool  = True     # apply SPIN against prev iteration

    # Auto-curriculum
    competence_threshold: float = 0.75  # advance difficulty above this score
    difficulty_levels: list[str] = field(default_factory=lambda: [
        "simple factual", "multi-step reasoning", "adversarial", "open-ended creative"
    ])

    # Regression gate
    regression_threshold: float = 0.95  # reject iteration if score < 0.95 * prev_score
    n_eval_prompts:       int   = 32    # prompts for regression eval


# ─────────────────────────────────────────────────────────────────────────────
# Self-Evaluator
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_TEMPLATE = """\
<|im_start|>system
You are a strict evaluator. Score the following response on a scale of 0–10
for each criterion, then give a final weighted score.
Criteria: helpfulness, factual accuracy, safety, conciseness.<|im_end|>
<|im_start|>user
Prompt: {prompt}

Response: {response}

Score each criterion (0-10) and provide a final score (0-10).<|im_end|>
<|im_start|>assistant
"""

_DIFFICULTY_TEMPLATE = """\
<|im_start|>user
Rewrite the following prompt to be more {level}: {prompt}<|im_end|>
<|im_start|>assistant
"""


class SelfEvaluator:
    """
    Model evaluates its own responses using a rubric-based prompt.

    Returns a scalar score in [0, 1] by:
      1. Generating a rubric evaluation with the model itself
      2. Parsing the score from the generated text (fallback: perplexity proxy)
    """

    def __init__(self, model: nn.Module, tokenizer, cfg: RSIConfig, device: torch.device):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """Returns scalar score in [0, 1]."""
        eval_prompt = _EVAL_TEMPLATE.format(prompt=prompt, response=response)
        ids  = self.tokenizer.encode(eval_prompt, return_tensors="pt").to(self.device)

        out  = self.model.generate(
            ids,
            max_new_tokens=128,
            n_loops=self.cfg.n_loops_gen,
            temperature=0.3,
            top_k=20,
        )
        text = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)

        # Parse "final score: X" or any "X/10" pattern
        score = self._parse_score(text)
        if score is None:
            score = self._perplexity_proxy(prompt, response)
        return score

    def _parse_score(self, text: str) -> Optional[float]:
        import re
        patterns = [
            r"final\s+score[:\s]+(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"score[:\s]+(\d+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                try:
                    return min(float(m.group(1)) / 10.0, 1.0)
                except ValueError:
                    continue
        return None

    @torch.no_grad()
    def _perplexity_proxy(self, prompt: str, response: str) -> float:
        """Fallback: low perplexity of the response given prompt → higher score."""
        full = prompt + response
        ids  = self.tokenizer.encode(full, return_tensors="pt").to(self.device)
        p_len = len(self.tokenizer.encode(prompt))
        if ids.shape[1] <= p_len + 1:
            return 0.5

        logits = self.model(ids, n_loops=self.cfg.n_loops_gen)
        log_p  = F.log_softmax(logits[:, :-1], dim=-1)
        tgt    = ids[:, 1:]
        nll    = -log_p.gather(2, tgt.unsqueeze(-1)).squeeze(-1)

        resp_nll = nll[:, p_len:].mean().item()
        # Typical range: 0.5 (fluent) to 5.0 (incoherent)
        # Map to [0, 1]: lower NLL → higher score
        return float(math.exp(-max(resp_nll - 0.5, 0) * 0.5))

    @torch.no_grad()
    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.score(p, r) for p, r in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Self-Distillation
# ─────────────────────────────────────────────────────────────────────────────

class SelfDistiller:
    """
    Generates best-of-N completions per prompt, filters by self-eval score,
    and writes SFT-ready JSONL to disk.
    """

    def __init__(
        self,
        model:     nn.Module,
        evaluator: SelfEvaluator,
        tokenizer,
        cfg:       RSIConfig,
        device:    torch.device,
    ):
        self.model     = model
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device

    @torch.no_grad()
    def distill(self, prompts: list[str], output_path: str) -> list[dict]:
        """Generate best-of-N per prompt, keep top fraction, write to JSONL."""
        records: list[dict] = []
        n_keep = max(1, int(self.cfg.best_of_n * self.cfg.top_k_keep))

        for i, prompt in enumerate(prompts):
            print(f"  Distill [{i+1}/{len(prompts)}] {prompt[:50]}...")
            chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids = self.tokenizer.encode(chat_prompt, return_tensors="pt").to(self.device)

            candidates: list[tuple[str, float]] = []
            for _ in range(self.cfg.best_of_n):
                out  = self.model.generate(
                    ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    n_loops=self.cfg.n_loops_gen,
                    temperature=self.cfg.temperature,
                    top_k=self.cfg.gen_top_k,
                )
                text = self.tokenizer.decode(
                    out[0, ids.shape[1]:].tolist(), skip_special_tokens=True
                )
                score = self.evaluator.score(prompt, text)
                candidates.append((text, score))

            # Sort by score descending, keep top fraction
            candidates.sort(key=lambda x: x[1], reverse=True)
            for text, score in candidates[:n_keep]:
                records.append({
                    "prompt":     prompt,
                    "completion": text,
                    "score":      score,
                })

        with open(output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"  Distilled {len(records)} examples (from {len(prompts)} prompts × {self.cfg.best_of_n} candidates)")
        return records

    def make_preference_pairs(self, records: list[dict]) -> list[dict]:
        """
        Group by prompt; pair best vs worst → preference JSONL for ORPO/DPO.
        """
        from collections import defaultdict
        grouped: dict[str, list] = defaultdict(list)
        for r in records:
            grouped[r["prompt"]].append(r)

        pairs = []
        for prompt, items in grouped.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x["score"], reverse=True)
            pairs.append({
                "prompt":   prompt,
                "chosen":   items[0]["completion"],
                "rejected": items[-1]["completion"],
            })
        return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Curriculum
# ─────────────────────────────────────────────────────────────────────────────

class AutoCurriculum:
    """
    Tracks per-difficulty competence and advances when the model plateaus.
    Uses the model itself to rewrite prompts at increasing difficulty.
    """

    def __init__(
        self,
        model:     nn.Module,
        tokenizer,
        cfg:       RSIConfig,
        device:    torch.device,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device
        self._level_idx = 0
        self._history: list[float] = []

    @property
    def current_level(self) -> str:
        return self.cfg.difficulty_levels[self._level_idx]

    def update(self, mean_score: float) -> bool:
        """Returns True if difficulty was advanced."""
        self._history.append(mean_score)
        if mean_score >= self.cfg.competence_threshold:
            next_idx = min(self._level_idx + 1, len(self.cfg.difficulty_levels) - 1)
            if next_idx != self._level_idx:
                self._level_idx = next_idx
                print(f"  Curriculum advanced → {self.current_level}")
                return True
        return False

    @torch.no_grad()
    def rewrite_prompts(self, prompts: list[str]) -> list[str]:
        """Rewrite prompts at current difficulty level."""
        rewritten = []
        for prompt in prompts:
            p = _DIFFICULTY_TEMPLATE.format(level=self.current_level, prompt=prompt)
            ids = self.tokenizer.encode(p, return_tensors="pt").to(self.device)
            out = self.model.generate(
                ids,
                max_new_tokens=128,
                n_loops=self.cfg.n_loops_gen,
                temperature=0.7,
                top_k=40,
            )
            text = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            rewritten.append(text.strip() or prompt)
        return rewritten


# ─────────────────────────────────────────────────────────────────────────────
# RSI Loop
# ─────────────────────────────────────────────────────────────────────────────

class RSILoop:
    """
    Orchestrates the full recursive self-improvement cycle.

    Each iteration:
      1. Auto-curriculum rewrites seed prompts at current difficulty
      2. Self-distillation generates + filters best-of-N completions
      3. ORPO fine-tunes on self-generated preference pairs (optional)
      4. SPIN fine-tunes against frozen prev-iteration model (optional)
      5. Regression gate: rejects iteration if eval score drops
      6. Checkpoint saved; curriculum advances if competence threshold met
    """

    def __init__(self, model: nn.Module, tokenizer, cfg: RSIConfig, device: torch.device):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device
        self.evaluator  = SelfEvaluator(model, tokenizer, cfg, device)
        self.distiller  = SelfDistiller(model, self.evaluator, tokenizer, cfg, device)
        self.curriculum = AutoCurriculum(model, tokenizer, cfg, device)
        self._prev_score: Optional[float] = None
        self._iteration  = 0

    def _load_prompts(self, path: str) -> list[str]:
        p = Path(path)
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if p.suffix == ".jsonl":
            return [json.loads(l).get("prompt", l) for l in lines]
        return lines

    def _eval_score(self, prompts: list[str]) -> float:
        """Quick regression eval: mean self-eval score on a fixed prompt set."""
        sample = random.sample(prompts, min(self.cfg.n_eval_prompts, len(prompts)))
        scores = []
        for prompt in sample:
            chat_p = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids    = self.tokenizer.encode(chat_p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out  = self.model.generate(
                    ids,
                    max_new_tokens=256,
                    n_loops=self.cfg.n_loops_gen,
                    temperature=0.5,
                    top_k=30,
                )
            text = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            scores.append(self.evaluator.score(prompt, text))
        return sum(scores) / len(scores) if scores else 0.0

    def _fine_tune_orpo(self, pairs: list[dict], work_dir: Path) -> None:
        from anthos.orpo import ORPOConfig, ORPODataset, ORPOTrainer, orpo_collate
        from torch.utils.data import DataLoader

        pairs_path = work_dir / "pairs.jsonl"
        with open(pairs_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        orpo_cfg = ORPOConfig(beta=0.1, n_loops=self.cfg.n_loops_gen, lr=self.cfg.lr)
        trainer  = ORPOTrainer(self.model, orpo_cfg)
        dataset  = ORPODataset(str(pairs_path), "data/anthos_tokenizer",
                               max_length=self.cfg.max_new_tokens + 256)
        loader   = DataLoader(dataset, batch_size=self.cfg.batch_size,
                              shuffle=True, collate_fn=orpo_collate)

        step = 0
        for batch in loader:
            if step >= self.cfg.n_steps_per_iter:
                break
            _, metrics = trainer.step(batch)
            if step % 50 == 0:
                print(f"    ORPO step {step:3d} | loss={metrics['loss']:.4f}")
            step += 1

    def _fine_tune_spin(self, prompts: list[str], work_dir: Path) -> None:
        from anthos.spin import SPINConfig, SPINTrainer, spin_collate
        from torch.utils.data import DataLoader

        # Write a minimal SFT JSONL (prompt + model-generated completion)
        sft_path = work_dir / "sft_spin.jsonl"
        sft_data = []
        for prompt in prompts[:64]:
            chat_p = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids    = self.tokenizer.encode(chat_p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out  = self.model.generate(ids, max_new_tokens=256,
                                           n_loops=self.cfg.n_loops_gen,
                                           temperature=0.7, top_k=50)
            text = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            sft_data.append({"prompt": chat_p, "completion": text})
        with open(sft_path, "w") as f:
            for d in sft_data:
                f.write(json.dumps(d) + "\n")

        spin_cfg = SPINConfig(beta=0.1, n_loops=self.cfg.n_loops_gen, lr=self.cfg.lr)
        trainer  = SPINTrainer(self.model, spin_cfg)
        trainer.run_iteration(str(sft_path), "data/anthos_tokenizer",
                              n_steps=self.cfg.n_steps_per_iter,
                              batch_size=self.cfg.batch_size)

    def run(
        self,
        seed_prompts: str,
        output_dir:   str,
        save_every:   int = 1,
    ) -> list[dict]:
        """
        Run the full RSI loop.

        Args:
            seed_prompts: path to .txt or .jsonl file of initial prompts
            output_dir:   directory to save per-iteration checkpoints + logs
            save_every:   save checkpoint every N iterations

        Returns:
            list of per-iteration metrics dicts
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts   = self._load_prompts(seed_prompts)
        history   = []

        print(f"\n{'─'*60}")
        print(f"  RSI Loop: {self.cfg.n_iterations} iterations")
        print(f"  Seed prompts: {len(prompts)}")
        print(f"  Best-of-N: {self.cfg.best_of_n}  keep: {self.cfg.top_k_keep:.0%}")
        print(f"{'─'*60}\n")

        for it in range(self.cfg.n_iterations):
            t0       = time.time()
            work_dir = out_dir / f"iter_{it:02d}"
            work_dir.mkdir(exist_ok=True)

            print(f"[Iteration {it}] difficulty={self.curriculum.current_level}")

            # 1. Auto-curriculum: rewrite prompts at current difficulty
            curr_prompts = self.curriculum.rewrite_prompts(prompts)

            # 2. Self-distillation: generate + filter
            records = self.distiller.distill(
                curr_prompts, str(work_dir / "distilled.jsonl")
            )
            pairs   = self.distiller.make_preference_pairs(records)
            mean_score = (
                sum(r["score"] for r in records) / len(records) if records else 0.0
            )
            print(f"  Self-eval mean score: {mean_score:.3f}")

            # 3. Snapshot current model for regression comparison
            snap_sd = copy.deepcopy(self.model.state_dict())

            # 4. ORPO on self-generated preference pairs
            if self.cfg.use_orpo and pairs:
                print(f"  ORPO on {len(pairs)} preference pairs...")
                self._fine_tune_orpo(pairs, work_dir)

            # 5. SPIN against frozen previous-iteration model
            if self.cfg.use_spin:
                print(f"  SPIN against previous iteration...")
                self._fine_tune_spin(curr_prompts, work_dir)

            # 6. Regression gate
            new_score = self._eval_score(prompts)
            print(f"  Post-training eval score: {new_score:.3f}", end="")

            if (self._prev_score is not None and
                    new_score < self.cfg.regression_threshold * self._prev_score):
                print(f"  ← REGRESSION (prev={self._prev_score:.3f}); reverting")
                self.model.load_state_dict(snap_sd)
                new_score = self._prev_score
            else:
                print()
                self._prev_score = new_score

            # 7. Checkpoint
            if it % save_every == 0:
                ckpt_path = work_dir / "checkpoint.pt"
                torch.save({"model": self.model.state_dict(), "iteration": it,
                            "score": new_score}, ckpt_path)
                print(f"  Checkpoint → {ckpt_path}")

            # 8. Advance curriculum
            self.curriculum.update(mean_score)

            elapsed = time.time() - t0
            metrics = {
                "iteration":   it,
                "mean_score":  mean_score,
                "eval_score":  new_score,
                "difficulty":  self.curriculum.current_level,
                "n_records":   len(records),
                "n_pairs":     len(pairs),
                "elapsed_s":   elapsed,
            }
            history.append(metrics)

            log_path = out_dir / "history.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            print(f"  Iteration {it} done in {elapsed:.0f}s\n")

        print(f"RSI complete. Final eval score: {self._prev_score:.3f}")
        return history

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tier:            str,
        cfg:             Optional[RSIConfig] = None,
        device:          Optional[str]       = None,
    ) -> "RSILoop":
        from anthos.configs import get_training_config
        from anthos.main import Anthos
        from transformers import AutoTokenizer

        cfg = cfg or RSIConfig()
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model_cfg, _ = get_training_config(tier)
        model  = Anthos(model_cfg).to(dev)
        ckpt   = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        tok_path = "data/anthos_tokenizer"
        tok      = AutoTokenizer.from_pretrained(
            tok_path if Path(tok_path).exists() else "gpt2"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        return cls(model, tok, cfg, dev)
