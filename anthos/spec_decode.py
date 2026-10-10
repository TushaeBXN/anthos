"""
spec_decode.py — Speculative Decoding

A small draft model generates n_draft tokens speculatively; the large
verifier model checks all of them in a single parallel forward pass.
Accepted tokens cost almost nothing; only rejected tokens waste compute.

Speedup: typically 2–4× for autoregressive generation with a good draft
model.  Best when the draft and verifier share the same vocabulary and
the task is low-entropy (code, factual QA).

Algorithm (Chen et al. 2023 "Accelerating Large Language Model
Decoding with Speculative Sampling"):
  1. Draft model generates n_draft tokens: [d₁, d₂, …, d_n]
  2. Verifier runs one forward pass over [prompt + draft] in parallel
  3. For each draft token d_i, accept with probability:
        p_accept(d_i) = min(1, p_verifier(d_i) / p_draft(d_i))
  4. If all accepted, sample an extra bonus token from verifier
  5. If rejected at position k, resample d_k from adjusted distribution

Usage:
    from anthos.spec_decode import SpecDecoder

    decoder = SpecDecoder(
        draft     = small_anthos_model,
        verifier  = large_anthos_model,
        n_draft   = 5,
        n_loops_draft    = 4,
        n_loops_verifier = 12,
    )
    output_ids = decoder.generate(input_ids, max_new_tokens=256)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpecDecodeConfig:
    n_draft:          int   = 5      # tokens to propose per round
    n_loops_draft:    int   = 4      # recurrent depth for draft model
    n_loops_verifier: int   = 12     # recurrent depth for verifier model
    temperature:      float = 1.0
    top_k:            int   = 50


def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sampled_token, log_prob) for a single (batch=1) logit vector."""
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    lp    = probs.gather(1, token).log()
    return token, lp


class SpecDecoder:
    """
    Wraps a (draft, verifier) model pair for speculative decoding.

    Both models must:
      - Accept (input_ids, n_loops, kv_cache, start_pos) kwargs
      - Return logits of shape (B, T, vocab_size)
      - Have a .generate() method compatible with Anthos

    Args:
        draft:     small/fast Anthos model (or any compatible model)
        verifier:  large/accurate Anthos model
        cfg:       SpecDecodeConfig
    """

    def __init__(
        self,
        draft:    nn.Module,
        verifier: nn.Module,
        n_draft:  int   = 5,
        n_loops_draft:    int   = 4,
        n_loops_verifier: int   = 12,
        temperature: float = 1.0,
        top_k:       int   = 50,
    ):
        self.draft    = draft.eval()
        self.verifier = verifier.eval()
        self.cfg      = SpecDecodeConfig(
            n_draft=n_draft,
            n_loops_draft=n_loops_draft,
            n_loops_verifier=n_loops_verifier,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,   # (1, T_prompt)
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        assert input_ids.shape[0] == 1, "SpecDecoder supports batch_size=1 only"
        device    = input_ids.device
        generated = input_ids.clone()
        n_gen     = 0

        draft_cache    = {}
        verifier_cache = {}

        while n_gen < max_new_tokens:
            # ── Step 1: draft model proposes n_draft tokens ────────────────
            draft_tokens = []
            draft_logps  = []
            draft_ctx    = generated.clone()

            for i in range(self.cfg.n_draft):
                if n_gen + i >= max_new_tokens:
                    break
                if i == 0:
                    ctx       = draft_ctx
                    start_pos = 0
                    draft_cache.clear()
                else:
                    ctx       = draft_tokens[-1]
                    start_pos = draft_ctx.shape[1] + i - 1

                logits = self.draft(
                    ctx, n_loops=self.cfg.n_loops_draft,
                    kv_cache=draft_cache, start_pos=start_pos,
                )
                token, lp = _sample(logits[:, -1:], self.cfg.temperature, self.cfg.top_k)
                draft_tokens.append(token)
                draft_logps.append(lp)

            if not draft_tokens:
                break

            n_proposed = len(draft_tokens)
            draft_seq  = torch.cat(draft_tokens, dim=1)  # (1, n_proposed)

            # ── Step 2: verifier checks all draft tokens in one pass ───────
            verify_input = torch.cat([generated, draft_seq], dim=1)
            verifier_cache.clear()
            v_logits = self.verifier(
                verify_input, n_loops=self.cfg.n_loops_verifier,
                start_pos=0,
            )
            # v_logits: (1, T_prompt + n_proposed, V)
            # Logits at position T_prompt-1+i predict draft token i
            T_prefix    = generated.shape[1]
            v_logits_at = v_logits[:, T_prefix - 1: T_prefix - 1 + n_proposed, :]  # (1, n_proposed, V)

            # ── Step 3: accept/reject each draft token ─────────────────────
            n_accepted = 0
            for i in range(n_proposed):
                v_probs = F.softmax(v_logits_at[:, i] / max(self.cfg.temperature, 1e-6), dim=-1)
                d_tok   = draft_tokens[i]   # (1, 1)
                p_v     = v_probs.gather(1, d_tok)
                p_d     = draft_logps[i].exp()

                accept_prob = torch.clamp(p_v / (p_d + 1e-9), max=1.0)
                if torch.rand(1, device=device) < accept_prob:
                    generated = torch.cat([generated, d_tok], dim=1)
                    n_accepted += 1
                    n_gen += 1
                else:
                    # Rejection: resample from adjusted distribution
                    adjusted  = torch.clamp(v_probs - F.softmax(
                        v_logits_at[:, i].new_zeros(v_logits_at[:, i].shape), dim=-1
                    ), min=0)
                    if adjusted.sum() < 1e-9:
                        adjusted = v_probs
                    adjusted  = adjusted / adjusted.sum(-1, keepdim=True)
                    fix_token = torch.multinomial(adjusted, num_samples=1)
                    generated = torch.cat([generated, fix_token], dim=1)
                    n_gen += 1
                    break

            # ── Step 4: bonus token if all accepted ───────────────────────
            if n_accepted == n_proposed and n_gen < max_new_tokens:
                bonus_logits = self.verifier(
                    generated[:, -1:],
                    n_loops=self.cfg.n_loops_verifier,
                    start_pos=generated.shape[1] - 1,
                )
                bonus, _ = _sample(bonus_logits[:, -1:], self.cfg.temperature, self.cfg.top_k)
                generated = torch.cat([generated, bonus], dim=1)
                n_gen += 1

        return generated
