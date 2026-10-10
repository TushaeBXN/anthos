"""
anthos/eval_harness.py — lm-evaluation-harness integration

Runs standard LLM benchmarks (MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA,
GSM8K, MATH) against a trained Anthos checkpoint using EleutherAI's
lm-evaluation-harness as the evaluation backend.

Install:
    pip install lm-eval

Usage (CLI):
    python eval.py --checkpoint checkpoints/proof/step_010000.pt --tasks mmlu,hellaswag

Usage (in-training, called from train.py at SAVE_EVERY):
    from anthos.eval_harness import run_eval
    results = run_eval(model, tokenizer_path, tasks=["hellaswag"], n_shots=0)

Supported tasks:
    mmlu          — 57-subject multiple choice (5-shot by default)
    hellaswag     — commonsense NLI (0-shot)
    arc_easy      — ARC easy (0-shot)
    arc_challenge — ARC challenge (25-shot)
    winogrande    — Winograd schema (0-shot)
    truthfulqa_mc — TruthfulQA multiple choice (0-shot)
    gsm8k         — Grade-school math (5-shot chain-of-thought)
    mathqa        — Math word problems (0-shot)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# lm-eval model adapter
# ─────────────────────────────────────────────────────────────────────────────

class AnthosLMEvalAdapter:
    """
    Adapter that wraps an Anthos model in the interface expected by
    lm-evaluation-harness (lm_eval.api.model.LM).

    lm-eval calls two methods:
      loglikelihood(requests)  — score a completion given a context
      generate_until(requests) — generate until a stop token

    We implement both by delegating to Anthos.forward() and Anthos.generate().
    """

    def __init__(
        self,
        model:          nn.Module,
        tokenizer_path: str   = "gpt2",
        n_loops:        int   = 8,
        device:         str   = "cuda",
        batch_size:     int   = 4,
    ):
        self.model          = model
        self.tokenizer_path = tokenizer_path
        self.n_loops        = n_loops
        self.device         = device
        self.batch_size     = batch_size
        self._tokenizer     = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_single(self, context_ids: list[int], continuation_ids: list[int]) -> float:
        """Compute log P(continuation | context) token-by-token."""
        input_ids = torch.tensor(
            [context_ids + continuation_ids], dtype=torch.long, device=self.device
        )
        T_ctx = len(context_ids)
        T_tot = input_ids.shape[1]

        if T_tot > self.max_length:
            input_ids = input_ids[:, -self.max_length:]
            T_ctx = max(0, T_ctx - (T_tot - self.max_length))

        logits, _ = self.model(input_ids, n_loops=self.n_loops, return_aux=True)
        log_probs  = torch.log_softmax(logits[0], dim=-1)   # (T-1, V)

        # Sum log probs over the continuation tokens only
        cont_len = len(continuation_ids)
        cont_start = T_ctx - 1  # position in logits that predicts context[-1] → cont[0]
        cont_ids_t = torch.tensor(
            continuation_ids, dtype=torch.long, device=self.device
        )
        log_prob = log_probs[cont_start : cont_start + cont_len]
        if log_prob.shape[0] == 0:
            return -float("inf")
        return log_prob.gather(1, cont_ids_t.unsqueeze(1)).squeeze(1).sum().item()

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        requests: list of (context_str, continuation_str) tuples
        Returns: list of (log_likelihood, is_greedy) tuples
        """
        results = []
        self.model.eval()
        for ctx, cont in requests:
            ctx_ids  = self.tok_encode(ctx)
            cont_ids = self.tok_encode(cont)
            ll = self._loglikelihood_single(ctx_ids, cont_ids)
            # is_greedy: True if this completion is the argmax
            # (approximate: we don't compute all alternatives here)
            results.append((ll, True))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Rolling window log-likelihood for perplexity tasks."""
        results = []
        self.model.eval()
        with torch.no_grad():
            for (text,) in requests:
                ids = torch.tensor(
                    [self.tok_encode(text)], dtype=torch.long, device=self.device
                )
                if ids.shape[1] < 2:
                    results.append(0.0)
                    continue
                logits, _ = self.model(ids[:, :-1], n_loops=self.n_loops, return_aux=True)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                targets   = ids[0, 1:]
                ll = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).sum().item()
                results.append(ll)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        """Generate text until a stop token or max_gen_toks."""
        results = []
        self.model.eval()
        for ctx, gen_kwargs in requests:
            stop = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            ctx_ids = torch.tensor(
                [self.tok_encode(ctx)], dtype=torch.long, device=self.device
            )
            out = self.model.generate(
                ctx_ids,
                max_new_tokens=max_new,
                n_loops=self.n_loops,
                temperature=0.0,   # greedy for eval
                top_k=1,
            )
            text = self.tokenizer.decode(out[0, ctx_ids.shape[1]:].tolist())
            # Trim at first stop token
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]
            results.append(text)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────────────────────────────────────

# Default task → n_shot mapping (standard from lm-eval paper)
DEFAULT_SHOTS = {
    "mmlu":          5,
    "hellaswag":     0,
    "arc_easy":      0,
    "arc_challenge": 25,
    "winogrande":    5,
    "truthfulqa_mc": 0,
    "gsm8k":         5,
    "mathqa":        0,
}


def run_eval(
    model:          nn.Module,
    tokenizer_path: str             = "gpt2",
    tasks:          list[str]       = ("hellaswag", "arc_easy"),
    n_loops:        int             = 8,
    device:         str             = "cuda",
    output_path:    Optional[Path]  = None,
    limit:          Optional[int]   = None,
) -> dict:
    """
    Run lm-evaluation-harness benchmarks on an Anthos model.

    Args:
        model:          trained Anthos (unwrapped — not DDP/FSDP)
        tokenizer_path: HuggingFace tokenizer path
        tasks:          list of task names (see DEFAULT_SHOTS for options)
        n_loops:        loop depth to use during eval
        device:         "cuda" or "cpu"
        output_path:    if provided, write results JSON here
        limit:          cap number of examples per task (for fast smoke evals)

    Returns:
        dict mapping task name → {"acc": float, "acc_norm": float}

    Requires:
        pip install lm-eval
    """
    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.utils import ModelWrapper
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness not installed.\n"
            "Run: pip install lm-eval"
        )

    adapter = AnthosLMEvalAdapter(
        model=model, tokenizer_path=tokenizer_path,
        n_loops=n_loops, device=device,
    )

    # Build per-task num_fewshot mapping
    num_fewshot = {t: DEFAULT_SHOTS.get(t, 0) for t in tasks}

    results = evaluator.simple_evaluate(
        model          = adapter,
        tasks          = list(tasks),
        num_fewshot    = num_fewshot,
        limit          = limit,
        bootstrap_iters= 0,   # skip CIs for speed in training eval
        log_samples    = False,
    )

    # Extract just accuracy metrics for clean logging
    summary = {}
    for task, metrics in results["results"].items():
        summary[task] = {
            k: round(v, 4)
            for k, v in metrics.items()
            if k in ("acc", "acc_norm", "exact_match")
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))

    return summary


def print_eval_summary(summary: dict, step: int) -> None:
    """Print a compact eval summary to stdout."""
    print(f"\n── Eval @ step {step} {'─'*40}")
    for task, metrics in summary.items():
        parts = "  |  ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
        print(f"  {task:<22} {parts}")
    print("─" * 60)
