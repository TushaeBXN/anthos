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

from anthos.sae import (
    SAEConfig,
    SparseAutoencoder,
    AnthosSAESuite,
)

from anthos.steering import (
    AnthosSteer,
    ActivationCollector,
    ActivationSteering,
    LTIStateSteering,
)

from anthos.sasft import (
    FeatureSuppressionLoss,
    RepetitionPenaltyLoss,
    ThoughtDiversityLoss,
)

from anthos.features import (
    feature_rank,
    discover_features,
    monolinguality_score,
    repetition_features,
    FeatureClassifier,
    FeatureInterpreter,
)

from anthos.memory import (
    MemoryBankConfig,
    MemoryBankState,
    MemoryBank,
    ExternalMemoryReader,
    MemoryAugmentedAnthos,
)

from anthos.memory_compress import (
    ESCompressor,
    MemoryAugmentedDataset,
    compress_jsonl,
)

__version__ = "0.1.0"
__author__  = "Tushae Thomas"
__all__ = [
    # Core architecture
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
    # SAE
    "SAEConfig",
    "SparseAutoencoder",
    "AnthosSAESuite",
    # Steering
    "AnthosSteer",
    "ActivationCollector",
    "ActivationSteering",
    "LTIStateSteering",
    # SASFT losses
    "FeatureSuppressionLoss",
    "RepetitionPenaltyLoss",
    "ThoughtDiversityLoss",
    # Feature analysis
    "feature_rank",
    "discover_features",
    "monolinguality_score",
    "repetition_features",
    "FeatureClassifier",
    "FeatureInterpreter",
    # Memory
    "MemoryBankConfig",
    "MemoryBankState",
    "MemoryBank",
    "ExternalMemoryReader",
    "MemoryAugmentedAnthos",
    # Memory compression / training augmentation
    "ESCompressor",
    "MemoryAugmentedDataset",
    "compress_jsonl",
]
