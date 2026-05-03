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

from anthos.lora_pairs import (
    DualLoRAAdapter,
)

from anthos.eaft import (
    EAFTLoss,
    StandardLoss,
)

from anthos.grpo import (
    GRPOConfig,
    GRPOTrainer,
    quality_reward,
    loop_efficiency_reward,
)

from anthos.multipack import (
    MultipackDataset,
    MultipackSampler,
    multipack_collate,
    build_pack_mask_fast,
)

from anthos.quant import (
    QuantConfig,
    detect_device,
    get_dtype,
    load_quantized,
    FP8Linear,
)

from anthos.train_additions import (
    build_dataloader,
    build_loss,
    training_step,
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

from anthos.distill import (
    DistillConfig,
    DistillationLoss,
    TeacherLabelGenerator,
    TeacherLabelDataset,
    OnlineDistiller,
)

from anthos.export import (
    export_safetensors,
    export_hf_config,
    export_gguf_metadata,
    quantize_model,
    export_for_deployment,
)

from anthos.kv_cache import (
    CacheConfig,
    SequenceKVCache,
    LTIStateCache,
    AnthosCache,
    CachedGenerator,
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
    # Distillation
    "DistillConfig",
    "DistillationLoss",
    "TeacherLabelGenerator",
    "TeacherLabelDataset",
    "OnlineDistiller",
    # Export / deployment
    "export_safetensors",
    "export_hf_config",
    "export_gguf_metadata",
    "quantize_model",
    "export_for_deployment",
    # KV cache
    "CacheConfig",
    "SequenceKVCache",
    "LTIStateCache",
    "AnthosCache",
    "CachedGenerator",
    # Dual LoRA
    "DualLoRAAdapter",
    # EAFT loss
    "EAFTLoss",
    "StandardLoss",
    # GRPO
    "GRPOConfig",
    "GRPOTrainer",
    "quality_reward",
    "loop_efficiency_reward",
    # Multipack
    "MultipackDataset",
    "MultipackSampler",
    "multipack_collate",
    "build_pack_mask_fast",
    # Quantization (FP8 inference)
    "QuantConfig",
    "detect_device",
    "get_dtype",
    "load_quantized",
    "FP8Linear",
    # Training integration
    "build_dataloader",
    "build_loss",
    "training_step",
]
