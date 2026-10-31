"""
anthos/sasft.py — SAE-Guided Supervised Fine-Tuning (SASFT) Losses

Implements Qwen-Scope's SASFT technique for Anthos:
  - Suppresses unwanted thought-stream features during training via auxiliary loss
  - Targets: code-switching, repetition, off-topic reasoning

Two losses:

  1. FeatureSuppressionLoss
     Penalizes activation of specified SAE feature directions.
     Use during SFT to suppress language-mixing or repetition features.

  2. RepetitionPenaltyLoss
     Detects ngram repetition in generated logits and adds a penalty.
     Lightweight — no SAE required. Drop-in addition to train.py.

These are added to the existing (ce_loss + aux_loss) training objective.

Integration in train.py:
    from anthos.sasft import FeatureSuppressionLoss, RepetitionPenaltyLoss

    # -- setup (once) --
    rep_loss_fn = RepetitionPenaltyLoss(ngram_size=4, penalty=0.5)

    # -- inside training loop, alongside existing loss --
    logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)
    ce       = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), labels.reshape(-1))
    rep_pen  = rep_loss_fn(logits)
    loss     = ce + aux + rep_pen
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature Suppression Loss (requires trained SAE)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureSuppressionLoss(nn.Module):
    """
    Penalizes activation of specific SAE features in collected activations.

    Derived from Qwen-Scope's SASFT: adds an auxiliary regularization loss
    that discourages the model from activating language-mixing or other
    unwanted features during training on target-language data.

    The loss is:
        L_suppress = coeff * mean(|f(h)[feature_ids]|)

    where f(h) is the SAE encoding of hidden states h.

    Usage:
        # Setup (once)
        suppress_ids  = monolinguality_score(sae, lang_acts)["zh"]  # suppress Chinese
        suppress_loss = FeatureSuppressionLoss(sae, suppress_ids, coeff=0.1)

        # In training loop — hook activations then compute loss
        collector = ActivationCollector(model, stream="thought")
        collector.attach()
        logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)
        acts  = collector.flat_activations()
        collector.detach(); collector.clear()

        ce   = F.cross_entropy(...)
        s_loss = suppress_loss(acts.to(device))
        loss = ce + aux + s_loss
    """

    def __init__(
        self,
        sae,
        feature_ids: torch.Tensor,
        coeff: float = 0.1,
    ):
        super().__init__()
        self.sae         = sae
        self.coeff       = coeff
        self.register_buffer("feature_ids", feature_ids)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: [N, D] — flat collected activations from ActivationCollector
        Returns:
            scalar loss
        """
        device = next(self.sae.parameters()).device
        acts   = activations.to(device)

        # We don't want to backprop through the SAE itself, only through the
        # model's hidden states — so detach the SAE encoder output then
        # recompute the norm w.r.t. the input activations.
        with torch.no_grad():
            features_ref, _ = self.sae(acts)
            ids = self.feature_ids.to(device)
            # Binary mask: which positions activated the suppressed features
            mask = (features_ref[:, ids] > 0).float()  # [N, K]

        # Recompute projection onto suppressed feature directions
        # (this path has gradients w.r.t. the model's hidden states)
        W_dec_suppressed = self.sae.W_dec[ids]  # [K, D]  — no grad needed
        projections = acts @ W_dec_suppressed.T  # [N, K]

        # Only penalize where the SAE actually activated those features
        penalized = (projections * mask).abs().mean()
        return self.coeff * penalized


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repetition Penalty Loss (no SAE required — pure logit-level)
# ─────────────────────────────────────────────────────────────────────────────

class RepetitionPenaltyLoss(nn.Module):
    """
    Lightweight ngram-based repetition penalty computed over training logits.

    Mirrors the Qwen-Scope RL approach but adapted for supervised training:
    Instead of requiring repetitive rollouts, we detect repetition directly
    in the model's predicted token distribution and add a soft penalty.

    The penalty encourages the model to assign lower probability to tokens
    that would extend an already-repeated ngram.

    Loss:
        For each position where greedy token continues a repeated ngram,
        penalize: penalty * p(repeat_token)

    Args:
        ngram_size: window for repetition detection (default 4)
        penalty:    loss weight (default 0.3)
        max_seq:    max sequence length to scan (default 512)
    """

    def __init__(
        self,
        ngram_size: int = 4,
        penalty:    float = 0.3,
        max_seq:    int = 512,
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.penalty    = penalty
        self.max_seq    = max_seq

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, V]
        Returns:
            scalar penalty loss
        """
        B, T, V = logits.shape
        T = min(T, self.max_seq)
        logits = logits[:, :T, :]

        with torch.no_grad():
            greedy_ids = logits.argmax(-1)  # [B, T]

        total_penalty = torch.tensor(0.0, device=logits.device)
        n_penalized   = 0

        for b in range(B):
            ids = greedy_ids[b]  # [T]
            for t in range(self.ngram_size, T):
                ngram = ids[t-self.ngram_size+1:t+1].tolist()
                # Check if this ngram appeared earlier in the sequence
                candidate = tuple(ngram)
                past = ids[:t-self.ngram_size+1].tolist()
                repeated = any(
                    tuple(past[i:i+self.ngram_size]) == candidate
                    for i in range(len(past) - self.ngram_size + 1)
                )
                if repeated:
                    # Penalize the probability assigned to the repeating token
                    repeat_tok = ids[t]
                    prob = F.softmax(logits[b, t-1], dim=-1)[repeat_tok]
                    total_penalty = total_penalty + prob
                    n_penalized  += 1

        if n_penalized == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return self.penalty * (total_penalty / n_penalized)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Thought Diversity Loss (novel — specific to Anthos's thought stream)
# ─────────────────────────────────────────────────────────────────────────────

class ThoughtDiversityLoss(nn.Module):
    """
    Encourages diversity across thought tokens within a single forward pass.

    Anthos's n_thought_tokens are designed to be independent working-memory
    slots. If they collapse to near-identical representations, the thought
    stream loses capacity. This loss penalizes cosine similarity between
    thought token pairs.

    This is specific to Anthos — no equivalent in Qwen-Scope since standard
    transformers don't have explicit thought tokens.

    Loss:
        L_div = coeff * mean(cos_sim(thought_i, thought_j))  for i≠j

    Usage:
        thought_div = ThoughtDiversityLoss(coeff=0.05)

        # thought_acts: [B, n_thought, D] — slice from recurrent block output
        # Collect with ActivationCollector(stream="thought")
        loss = ce + aux + thought_div(thought_acts.to(device))
    """

    def __init__(self, coeff: float = 0.05):
        super().__init__()
        self.coeff = coeff

    def forward(self, thought_acts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thought_acts: [B, n_thought, D]
        Returns:
            scalar diversity penalty
        """
        B, N, D = thought_acts.shape
        if N < 2:
            return torch.tensor(0.0, device=thought_acts.device)

        # Normalize each thought token
        normed = F.normalize(thought_acts, dim=-1)  # [B, N, D]

        # Pairwise cosine similarity matrix: [B, N, N]
        sim_matrix = torch.bmm(normed, normed.transpose(1, 2))

        # Upper triangle only (exclude diagonal)
        mask = torch.triu(torch.ones(N, N, device=thought_acts.device), diagonal=1)
        n_pairs = mask.sum()

        if n_pairs == 0:
            return torch.tensor(0.0, device=thought_acts.device)

        # Mean pairwise similarity (higher = more collapsed = worse)
        mean_sim = (sim_matrix * mask).sum() / (B * n_pairs)

        # Clamp: don't penalize negative correlation, only collapse
        return self.coeff * mean_sim.clamp(min=0.0)
