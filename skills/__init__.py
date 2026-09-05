"""
Anthos Skills Integration

Maps Claude Code skills to Anthos training data generation and inference capabilities.
Anthos is Brian Tushae Thomas's custom Thought-Token Bifurcated Recurrent Transformer.

Skills serve two purposes here:
1. TRAINING DATA: each skill is a source of high-quality instruction-response pairs
2. INFERENCE CAPABILITY: Anthos should be able to perform these workflows at inference time

The most critical skills for Anthos training:
- ai-research-assistant → generates deep knowledge Q&A pairs
- ai-chatbot-builder → teaches Anthos to design and build AI systems
- code-debugger → builds code reasoning capability
- data-pipeline-builder → builds data engineering capability
- financial-model-builder → builds financial reasoning capability
- grant-finder → builds resource navigation capability (for Nia integration)
- research-to-documents → builds the full research → document workflow
"""

from .registry import SKILL_MAP, training_pairs_for_skill, skill_system_prompt

__all__ = ["SKILL_MAP", "training_pairs_for_skill", "skill_system_prompt"]
