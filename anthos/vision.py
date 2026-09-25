"""
Anthos Vision Encoder — CLIP-based image understanding for the sequence stream.

Lifted and adapted from ujamaa-multi-modal's `ujamaa/layers/vision.py`, which
already implemented the standard "frozen vision backbone + trainable
projector" pattern (the same approach used by LLaVA and most vision-language
models). This module keeps that pattern but targets Anthos's embedding space
directly, so image patches can be prepended to Anthos's own sequence stream —
no separate Ujamaa transformer, MoE, or community-gate stack required.

Pipeline:
    pixel_values (B, 3, H, W)
        -> CLIPVisionModel (frozen or fine-tuned)  -> (B, n_patches, vision_dim)
        -> Projector (Linear -> LayerNorm -> GELU -> Linear -> LayerNorm)
        -> (B, n_patches, anthos_dim)

The output is inserted into Anthos's sequence stream ahead of the text
tokens (see `Anthos.forward` in main.py). It occupies ordinary sequence
positions, not thought-token slots — a deliberate, minimal first cut. A
future iteration could instead route image patches into the non-causal
thought stream, or give them their own non-causal attention block; that is
a design choice worth revisiting once this baseline is validated, not
something to solve on the first pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VisionConfig:
    # Any CLIP vision checkpoint on the Hugging Face Hub works here.
    # Smaller ("openai/clip-vit-base-patch32") is a reasonable default for
    # local experimentation; swap in a larger CLIP/SigLIP checkpoint later.
    encoder_name: str = "openai/clip-vit-base-patch32"
    freeze_encoder: bool = True   # freeze the CLIP backbone; train only the projector at first


class VisionEncoder(nn.Module):
    """
    Wraps a pretrained CLIP vision tower and projects its patch embeddings
    into Anthos's model dimension.
    """

    def __init__(self, vision_cfg: VisionConfig, anthos_dim: int):
        super().__init__()
        self.vision_cfg = vision_cfg

        from transformers import CLIPVisionModel
        self.encoder = CLIPVisionModel.from_pretrained(vision_cfg.encoder_name)
        self.vision_dim = self.encoder.config.hidden_size

        if vision_cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.projector = nn.Sequential(
            nn.Linear(self.vision_dim, anthos_dim),
            nn.LayerNorm(anthos_dim),
            nn.GELU(),
            nn.Linear(anthos_dim, anthos_dim),
            nn.LayerNorm(anthos_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) — already preprocessed
                          (e.g. via a HF CLIPImageProcessor)
        Returns:
            (B, n_patches, anthos_dim) — ready to prepend to Anthos's
            token embeddings
        """
        # Encoder stays in eval mode/no-grad territory when frozen, but we
        # still let autograd track the projector's inputs, so don't wrap in
        # torch.no_grad() here — freezing is handled via requires_grad_.
        outputs  = self.encoder(pixel_values)
        features = outputs.last_hidden_state          # (B, n_patches, vision_dim)
        return self.projector(features)                # (B, n_patches, anthos_dim)
