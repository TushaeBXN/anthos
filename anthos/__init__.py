"""
Anthos — Thought-Token Bifurcated Recurrent Transformer
Think in Streams.
"""

from anthos.train.trainer import (
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

from anthos.train.lora_pairs import (
    DualLoRAAdapter,
)

from anthos.train.eaft import (
    EAFTLoss,
    StandardLoss,
)

from anthos.train.grpo import (
    GRPOConfig,
    GRPOTrainer,
    quality_reward,
    loop_efficiency_reward,
)

from anthos.train.multipack import (
    MultipackDataset,
    MultipackSampler,
    multipack_collate,
    build_pack_mask_fast,
)

from anthos.inference.quant import (
    QuantConfig,
    detect_device,
    get_dtype,
    load_quantized,
    FP8Linear,
)

from anthos.train.train_additions import (
    build_dataloader,
    build_loss,
    training_step,
)

from anthos.train.sae import (
    SAEConfig,
    SparseAutoencoder,
    AnthosSAESuite,
)

from anthos.model.steering import (
    AnthosSteer,
    ActivationCollector,
    ActivationSteering,
    LTIStateSteering,
)

from anthos.train.sasft import (
    FeatureSuppressionLoss,
    RepetitionPenaltyLoss,
    ThoughtDiversityLoss,
)

from anthos.model.features import (
    feature_rank,
    discover_features,
    monolinguality_score,
    repetition_features,
    FeatureClassifier,
    FeatureInterpreter,
)

from anthos.memory.memory import (
    MemoryBankConfig,
    MemoryBankState,
    MemoryBank,
    ExternalMemoryReader,
    MemoryAugmentedAnthos,
)

from anthos.memory.memory.memory_compress import (
    ESCompressor,
    MemoryAugmentedDataset,
    compress_jsonl,
)

from anthos.train.distill import (
    DistillConfig,
    DistillationLoss,
    TeacherLabelGenerator,
    TeacherLabelDataset,
    OnlineDistiller,
)

from anthos.inference.generate import (
    export_safetensors,
    export_hf_config,
    export_gguf_metadata,
    quantize_model,
    export_for_deployment,
)

from anthos.model.kv_cache import (
    CacheConfig,
    SequenceKVCache,
    LTIStateCache,
    AnthosCache,
    CachedGenerator,
)

from anthos.identity.identity_hardening import (
    AnthosWithIdentityLock,
    CheckpointSigner,
    IDENTITY_TOKEN_IDS,
    IDENTITY_MAPPINGS,
    REQUIRED_IDENTITY_SEQUENCE,
    IdentityLossHead,
)
from anthos.identity.scalable_growth import ScalableAnthos, RecurrentBlock, MoELayer
from anthos.identity.lifelong_learning import ExperienceReplayBuffer, ElasticWeightConsolidation, LifelongTrainer
from anthos.identity.self_improvement import SelfTeacher, IterativeRefinement
from anthos.infrastructure.distributed_grid import DistributedGrid
from anthos.memory.memory.knowledge_graph import AnthosKnowledgeGraph, GraphAugmentedGeneration
from anthos.eval.benchmark_suite import AnthosBenchmark, ContinuousBenchmarking
from anthos.train.model_surgery import ModelSurgeon
from anthos.inference.deploy import AnthosDeployer
from anthos.identity.autonomous_agent import AutonomousImprovementAgent
from anthos.eval.community import CommunityModelHub, FederatedLearningCoordinator
from anthos.identity.self_destruct import ModelProtection

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
    # Identity hardening
    "AnthosWithIdentityLock",
    "CheckpointSigner",
    "IDENTITY_TOKEN_IDS",
    "IDENTITY_MAPPINGS",
    "REQUIRED_IDENTITY_SEQUENCE",
    "IdentityLossHead",
    # Scalable growth
    "ScalableAnthos",
    "RecurrentBlock",
    "MoELayer",
    # Lifelong learning
    "ExperienceReplayBuffer",
    "ElasticWeightConsolidation",
    "LifelongTrainer",
    # Self-improvement
    "SelfTeacher",
    "IterativeRefinement",
    # Distributed training
    "DistributedGrid",
    # Knowledge graph
    "AnthosKnowledgeGraph",
    "GraphAugmentedGeneration",
    # Benchmarking
    "AnthosBenchmark",
    "ContinuousBenchmarking",
    # Model surgery
    "ModelSurgeon",
    # Deployment
    "AnthosDeployer",
    # Autonomous agent
    "AutonomousImprovementAgent",
    # Community
    "CommunityModelHub",
    "FederatedLearningCoordinator",
    # Protection
    "ModelProtection",
]
