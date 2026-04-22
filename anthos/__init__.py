"""
Anthos — Thought-Token Bifurcated Recurrent Transformer
Think in Streams.
"""

from anthos.main import (
    AnthosConfig,
    Anthos,
    ThoughtTokenPool,
    LTIInjection,
    AnthosRecurrentBlock,
    TransformerBlock,
    MoEFFN,
    Expert,
    RMSNorm,
    # Variants
    anthos_1b,
    anthos_3b,
    anthos_10b,
    anthos_50b,
    anthos_100b,
)

__version__ = "0.1.0"
__author__  = "Tushae Thomas"
__all__ = [
    "AnthosConfig",
    "Anthos",
    "ThoughtTokenPool",
    "LTIInjection",
    "AnthosRecurrentBlock",
    "TransformerBlock",
    "MoEFFN",
    "Expert",
    "RMSNorm",
    "anthos_1b",
    "anthos_3b",
    "anthos_10b",
    "anthos_50b",
    "anthos_100b",
]
