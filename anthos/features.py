"""
anthos/features.py — Feature Discovery and Analysis

Implements Qwen-Scope's feature analysis techniques adapted for Anthos:

  1. feature_rank()         — rank SAE features by mean activation on a dataset
  2. discover_features()    — differential ranking (feature X fires on A but not B)
  3. monolinguality_score() — identify language-specific thought-stream features
  4. repetition_features()  — detect features that activate during repetitive output
  5. toxicity_features()    — OR-rule classifier over discovery set (no head needed)
  6. FeatureInterpreter     — top-activating examples for manual inspection

These are analysis utilities — they consume pre-collected activations
(from ActivationCollector) and a trained SAE (from SparseAutoencoder).

Usage:
    from anthos.features import discover_features, monolinguality_score
    from anthos.sae import SparseAutoencoder
    from anthos.steering import ActivationCollector

    # Collect activations on clean vs toxic text
    collector = ActivationCollector(model, stream="thought")
    collector.attach()
    for batch in clean_loader:   model(batch, n_loops=8)
    clean_acts = collector.flat_activations(); collector.clear()
    for batch in toxic_loader:   model(batch, n_loops=8)
    toxic_acts = collector.flat_activations(); collector.detach()

    sae  = SparseAutoencoder(cfg)
    # ... load trained SAE weights ...

    toxic_feat_ids = discover_features(sae, clean_acts, toxic_acts, top_k=32)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core ranking utilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def feature_rank(
    sae,
    activations: torch.Tensor,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rank SAE features by mean activation strength over a corpus.

    Args:
        sae:         SparseAutoencoder instance (trained)
        activations: [N, D] collected activations
        batch_size:  mini-batch size for memory efficiency

    Returns:
        sorted_ids:    [H] feature indices, highest-activation first
        mean_acts:     [H] mean activation values (sorted)
    """
    device = next(sae.parameters()).device
    H = sae.cfg.d_sae
    accum = torch.zeros(H, device="cpu")
    n = activations.shape[0]

    for i in range(0, n, batch_size):
        chunk = activations[i:i+batch_size].to(device)
        feats, _ = sae(chunk)           # [B, H]
        accum += feats.sum(0).cpu()

    mean_acts = accum / n
    sorted_ids = mean_acts.argsort(descending=True)
    return sorted_ids, mean_acts[sorted_ids]


@torch.no_grad()
def discover_features(
    sae,
    baseline_acts: torch.Tensor,
    target_acts:   torch.Tensor,
    top_k: int = 32,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differential feature discovery: find SAE features that activate more
    on target_acts than baseline_acts (e.g., toxic vs clean, foreign language
    vs native, repetitive vs clean).

    Mirrors Qwen-Scope's monolinguality / toxicity discovery approach.

    Args:
        sae:           Trained SparseAutoencoder
        baseline_acts: [N_base, D] activations from baseline corpus
        target_acts:   [N_tgt,  D] activations from target corpus
        top_k:         How many discriminative features to return

    Returns:
        feature_ids:   [top_k] most discriminative feature indices
        scores:        [top_k] differential activation scores
    """
    device = next(sae.parameters()).device

    def mean_feats(acts):
        H = sae.cfg.d_sae
        accum = torch.zeros(H)
        n = acts.shape[0]
        for i in range(0, n, batch_size):
            chunk = acts[i:i+batch_size].to(device)
            f, _ = sae(chunk)
            accum += f.sum(0).cpu()
        return accum / n

    base_mean   = mean_feats(baseline_acts)
    target_mean = mean_feats(target_acts)
    diff        = target_mean - base_mean          # positive = fires more on target

    sorted_ids = diff.argsort(descending=True)
    top_ids    = sorted_ids[:top_k]
    return top_ids, diff[top_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Specialized feature discoverers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def monolinguality_score(
    sae,
    lang_acts: dict[str, torch.Tensor],
    reference_lang: str = "en",
    top_k: int = 64,
) -> dict[str, torch.Tensor]:
    """
    Identify language-specific features in the thought stream.

    Computes a monolinguality score per feature per language:
        score(f, lang) = mean_act(f | lang) - mean_act(f | all others)

    Used in SASFT to suppress unintended language features during training.

    Args:
        sae:           Trained SparseAutoencoder
        lang_acts:     dict mapping language code → [N, D] activations
                       e.g. {"en": ..., "zh": ..., "es": ...}
        reference_lang: language to treat as "native" (suppress others)
        top_k:         features to return per language

    Returns:
        dict mapping lang_code → [top_k] feature indices
    """
    device = next(sae.parameters()).device
    H = sae.cfg.d_sae

    # Compute mean feature activation per language
    lang_means: dict[str, torch.Tensor] = {}
    for lang, acts in lang_acts.items():
        accum = torch.zeros(H)
        n = acts.shape[0]
        for i in range(0, n, 512):
            chunk = acts[i:i+512].to(device)
            f, _ = sae(chunk)
            accum += f.sum(0).cpu()
        lang_means[lang] = accum / n

    # Global mean across all languages (uniform weight)
    all_means = torch.stack(list(lang_means.values()), dim=0).mean(0)

    results = {}
    for lang, mean in lang_means.items():
        score = mean - all_means      # features specific to this language
        ids   = score.argsort(descending=True)[:top_k]
        results[lang] = ids

    return results


@torch.no_grad()
def repetition_features(
    sae,
    clean_acts:     torch.Tensor,
    repetitive_acts: torch.Tensor,
    top_k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identify thought-stream features that activate during repetitive generation.

    These feature directions can then be:
      (a) suppressed via steering to prevent repetition loops, OR
      (b) amplified to generate repetitive rollouts as RL negative samples
          (mirrors Qwen-Scope's DAPO RL repetition-suppression technique)

    Args:
        clean_acts:      [N, D] activations from clean, non-repetitive outputs
        repetitive_acts: [N, D] activations from repetitive outputs
                         (generate these by steering with a known repeat pattern)
    """
    return discover_features(sae, clean_acts, repetitive_acts, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# OR-rule classifier (zero-shot, no gradient fitting)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureClassifier:
    """
    Lightweight OR-rule classifier over discovered SAE features.

    Adapted from Qwen-Scope's multilingual toxicity classifier:
    - No classifier head, no gradient fitting
    - Fire if ANY of the target features activates above threshold
    - Achieves F1 > 0.90 on binary classification tasks

    Usage:
        feature_ids = discover_features(sae, clean_acts, toxic_acts)[0]
        clf = FeatureClassifier(sae, feature_ids, threshold=0.1)
        is_toxic = clf.classify(new_activations)   # [N] bool tensor
    """

    def __init__(
        self,
        sae,
        feature_ids: torch.Tensor,
        threshold:   float = 0.1,
    ):
        self.sae         = sae
        self.feature_ids = feature_ids
        self.threshold   = threshold

    @torch.no_grad()
    def classify(self, activations: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """
        Args:
            activations: [N, D]
        Returns:
            labels: [N] bool — True if classified as target class
        """
        device   = next(self.sae.parameters()).device
        ids      = self.feature_ids.to(device)
        all_preds = []

        for i in range(0, activations.shape[0], batch_size):
            chunk  = activations[i:i+batch_size].to(device)
            feats, _ = self.sae(chunk)                    # [B, H]
            target   = feats[:, ids]                      # [B, K]
            fired    = (target > self.threshold).any(-1)  # [B] OR-rule
            all_preds.append(fired.cpu())

        return torch.cat(all_preds, dim=0)

    def precision_recall(
        self,
        activations: torch.Tensor,
        labels:      torch.Tensor,
    ) -> dict:
        preds = self.classify(activations)
        tp = (preds &  labels).sum().float()
        fp = (preds & ~labels).sum().float()
        fn = (~preds & labels).sum().float()
        prec   = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * recall / (prec + recall + 1e-8)
        return {
            "precision": prec.item(),
            "recall":    recall.item(),
            "f1":        f1.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Interpreter — top activating examples
# ─────────────────────────────────────────────────────────────────────────────

class FeatureInterpreter:
    """
    For a given feature index, retrieve the top-N examples that activate it
    most strongly. Essential for qualitative validation of SAE features.

    Usage:
        interp = FeatureInterpreter(sae, feature_id=6159, top_n=10)
        top_examples = interp.find(activations, texts)
        for score, text in top_examples:
            print(f"{score:.3f}  {text[:80]}")
    """

    def __init__(self, sae, feature_id: int, top_n: int = 10):
        self.sae        = sae
        self.feature_id = feature_id
        self.top_n      = top_n

    @torch.no_grad()
    def find(
        self,
        activations: torch.Tensor,
        texts: Optional[list[str]] = None,
        batch_size:  int = 512,
    ) -> list[tuple[float, Optional[str]]]:
        """
        Args:
            activations: [N, D]
            texts:       optional list of N strings (decoded tokens)

        Returns:
            List of (activation_strength, text_or_None), sorted descending
        """
        device = next(self.sae.parameters()).device
        scores = []

        for i in range(0, activations.shape[0], batch_size):
            chunk = activations[i:i+batch_size].to(device)
            feats, _ = self.sae(chunk)
            s = feats[:, self.feature_id].cpu()
            scores.append(s)

        scores = torch.cat(scores, dim=0)
        top_idx = scores.argsort(descending=True)[:self.top_n]

        results = []
        for idx in top_idx.tolist():
            text = texts[idx] if texts else None
            results.append((scores[idx].item(), text))

        return results
