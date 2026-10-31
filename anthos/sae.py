"""
anthos/sae.py — Sparse Autoencoder (SAE) for Anthos Interpretability

Implements the Qwen-Scope SAE design adapted for Anthos's bifurcated streams:
  - Thought-stream SAE: captures working-memory features across loop iterations
  - Sequence-stream SAE: captures content/language features

Architecture:
  encoder:  f(x) = TopK(W_enc @ (x - b_dec) + b_enc, k)
  decoder:  x̂   = W_dec @ f(x) + b_dec
  loss:     ||x - x̂||² + λ·||f(x)||₁   (L2 recon + L1 sparsity)

Width defaults to 16× model dim (matches Qwen-Scope dense-model convention).
k defaults to 64 (midpoint of Qwen-Scope's 50–100 range).

Usage:
    from anthos.sae import SparseAutoencoder, SAEConfig

    cfg = SAEConfig(d_model=512, expansion=16, k=64)
    sae = SparseAutoencoder(cfg)

    # During collection / analysis
    features, x_hat = sae(hidden_states)   # hidden_states: [B, T, D]
    loss = sae.loss(hidden_states)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SAEConfig:
    d_model: int          = 512      # Must match Anthos AnthosConfig.dim
    expansion: int        = 16       # SAE width = expansion × d_model
    k: int                = 64       # Top-k active features per token
    l1_coeff: float       = 2e-4     # Sparsity penalty weight
    normalize_decoder: bool = True   # Keep decoder columns unit-norm
    dtype: torch.dtype    = torch.float32

    @property
    def d_sae(self) -> int:
        return self.d_model * self.expansion


# ─────────────────────────────────────────────────────────────────────────────
# Core SAE
# ─────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Top-k Sparse Autoencoder.

    Input shape:  [B, T, D]  or  [N, D]  (flattened is fine)
    Feature shape: same leading dims, width d_sae
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.d_model, cfg.d_sae

        # Encoder weights + bias
        self.W_enc = nn.Parameter(torch.empty(D, H, dtype=cfg.dtype))
        self.b_enc = nn.Parameter(torch.zeros(H,    dtype=cfg.dtype))

        # Decoder bias (shared reference point; decoder cols are unit-normed)
        self.b_dec = nn.Parameter(torch.zeros(D,    dtype=cfg.dtype))

        # Decoder weight — stored as [H, D] for easy column normalization
        self.W_dec = nn.Parameter(torch.empty(H, D, dtype=cfg.dtype))

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform for encoder
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        # Decoder initialized as transpose of encoder, then normalized
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Keep each decoder column (feature direction) unit-norm."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., D]  →  features: [..., H]  (sparse, only top-k nonzero)
        """
        x_centered = x - self.b_dec                        # [..., D]
        pre_acts   = x_centered @ self.W_enc + self.b_enc  # [..., H]

        # Top-k: zero out all but the k largest activations
        topk_vals, topk_idx = pre_acts.topk(self.cfg.k, dim=-1)
        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_idx, F.relu(topk_vals))
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: [..., H]  →  x_hat: [..., D]"""
        return features @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: [..., H]   sparse latent representation
            x_hat:    [..., D]   reconstruction
        """
        features = self.encode(x)
        x_hat    = self.decode(features)
        return features, x_hat

    # ── Loss ─────────────────────────────────────────────────────────────────

    def loss(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        x_hat:    Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute SAE training loss.

        Returns:
            total_loss: scalar
            info:       dict with recon_loss, l1_loss, avg_active
        """
        if features is None or x_hat is None:
            features, x_hat = self(x)

        recon_loss = F.mse_loss(x_hat, x.detach())
        l1_loss    = features.abs().mean()
        total      = recon_loss + self.cfg.l1_coeff * l1_loss

        avg_active = (features > 0).float().sum(-1).mean()

        return total, {
            "recon_loss": recon_loss.item(),
            "l1_loss":    l1_loss.item(),
            "total_loss": total.item(),
            "avg_active": avg_active.item(),
        }

    def post_step(self):
        """Call after each optimizer step to maintain decoder norm."""
        if self.cfg.normalize_decoder:
            self._normalize_decoder()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer SAE Suite
# ─────────────────────────────────────────────────────────────────────────────

class AnthosSAESuite(nn.Module):
    """
    A collection of SAEs — one per (stream × layer) pair.

    Mirrors Qwen-Scope's layer-wise feature dictionary design.
    Trained separately from Anthos; only used for analysis/steering.

    Streams: 'thought' | 'sequence'
    """

    def __init__(self, cfg: SAEConfig, n_layers: int, streams: list[str] = None):
        super().__init__()
        if streams is None:
            streams = ["thought", "sequence"]
        self.streams  = streams
        self.n_layers = n_layers

        # saes[stream][layer_idx]
        self.saes = nn.ModuleDict({
            stream: nn.ModuleList([
                SparseAutoencoder(cfg) for _ in range(n_layers)
            ])
            for stream in streams
        })

    def get(self, stream: str, layer: int) -> SparseAutoencoder:
        return self.saes[stream][layer]

    def forward(
        self,
        stream: str,
        layer: int,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.saes[stream][layer](x)

    def loss(
        self,
        stream: str,
        layer: int,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        return self.saes[stream][layer].loss(x)

    def post_step(self):
        for stream_saes in self.saes.values():
            for sae in stream_saes:
                sae.post_step()
