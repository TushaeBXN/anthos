"""
Anthos — Hardware-tier training configs

Three presets designed for different hardware:

  smoke    — MacBook CPU/MPS. Proves loss decreases. ~10 min.
  proof    — Single A100/4090. Coherent text in a few hours. ~$5-10 on RunPod.
  research — 4x A100. Real benchmark-worthy training. ~$50-100.

Usage:
    from anthos.configs import get_training_config
    cfg, train_cfg = get_training_config("smoke")
"""

from dataclasses import dataclass
from anthos.main import AnthosConfig


@dataclass
class TrainingConfig:
    # Hardware
    device:          str   = "cpu"
    dtype:           str   = "float32"     # float32 | bfloat16 | float16

    # Data
    dataset:         str   = "roneneldan/TinyStories"
    dataset_split:   str   = "train"
    seq_len:         int   = 256
    batch_size:      int   = 4
    num_workers:     int   = 0

    # Training
    max_steps:       int   = 5_000
    warmup_steps:    int   = 100
    learning_rate:   float = 3e-4
    min_lr:          float = 3e-5
    weight_decay:    float = 0.1
    grad_clip:       float = 1.0
    grad_accum:      int   = 1            # effective batch = batch_size * grad_accum

    # Loops (phased training)
    phase1_steps:    int   = 1_000        # fixed loops, no ACT — stabilization
    phase1_loops:    int   = 4
    phase2_loops:    int   = 8            # adaptive ACT enabled after phase1

    # Checkpointing & logging
    log_every:       int   = 50
    save_every:      int   = 500
    sample_every:    int   = 500
    checkpoint_dir:  str   = "checkpoints"
    run_name:        str   = "anthos-run"


def get_training_config(tier: str = "smoke"):
    """
    Returns (AnthosConfig, TrainingConfig) for the given hardware tier.

    Tiers: "smoke" | "proof" | "research"
    """
    import torch

    auto_device = (
        "cuda"  if torch.cuda.is_available()
        else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )
    auto_dtype = "bfloat16" if torch.cuda.is_available() else "float32"

    if tier == "smoke":
        # ── MacBook / CPU — proves the architecture learns ─────────────────
        # ~45 min, no GPU needed. 16 loops + 10k steps on TinyStories.
        model_cfg = AnthosConfig(
            vocab_size        = 50257,   # GPT-2 tokenizer
            dim               = 128,
            n_heads           = 4,
            n_kv_heads        = 2,
            max_seq_len       = 256,
            max_loop_iters    = 16,
            prelude_layers    = 1,
            coda_layers       = 1,
            n_thought_tokens  = 8,
            attn_type         = "gqa",
            n_experts         = 4,
            n_shared_experts  = 1,
            n_experts_per_tok = 2,
            expert_dim        = 64,
            lora_rank         = 4,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = auto_device,
            dtype         = "float32",
            dataset       = "roneneldan/TinyStories",
            seq_len       = 256,
            batch_size    = 2,
            max_steps     = 10_000,
            warmup_steps  = 200,
            learning_rate = 3e-4,
            grad_accum    = 4,       # effective batch = 8
            phase1_steps  = 1_000,
            phase1_loops  = 4,
            phase2_loops  = 16,
            log_every     = 25,
            save_every    = 1_000,
            sample_every  = 1_000,
            run_name      = "anthos-smoke",
        )

    elif tier == "proof":
        # ── Single A100/4090 — coherent text, benchmark-ready ──────────────
        # ~3-5 hrs. RunPod/Lambda ~$5-10. Loss ~2.5-3.0, readable stories.
        model_cfg = AnthosConfig(
            vocab_size        = 50257,
            dim               = 512,
            n_heads           = 8,
            n_kv_heads        = 4,
            max_seq_len       = 512,
            max_loop_iters    = 8,
            prelude_layers    = 2,
            coda_layers       = 2,
            n_thought_tokens  = 16,
            attn_type         = "gqa",
            n_experts         = 16,
            n_shared_experts  = 2,
            n_experts_per_tok = 4,
            expert_dim        = 256,
            lora_rank         = 8,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = "cuda",
            dtype         = auto_dtype,
            dataset       = "roneneldan/TinyStories",
            seq_len       = 512,
            batch_size    = 16,
            max_steps     = 20_000,
            warmup_steps  = 500,
            learning_rate = 3e-4,
            grad_accum    = 4,       # effective batch = 64
            phase1_steps  = 5_000,
            phase1_loops  = 4,
            phase2_loops  = 8,
            log_every     = 100,
            save_every    = 2_000,
            sample_every  = 2_000,
            run_name      = "anthos-proof",
        )

    elif tier == "research":
        # ── 4× A100 — serious run, publishable results ─────────────────────
        # Use with torchrun --nproc_per_node=4 train.py --tier research
        model_cfg = AnthosConfig(
            vocab_size        = 50257,
            dim               = 1024,
            n_heads           = 16,
            n_kv_heads        = 4,
            max_seq_len       = 1024,
            max_loop_iters    = 12,
            prelude_layers    = 2,
            coda_layers       = 2,
            n_thought_tokens  = 24,
            attn_type         = "gqa",
            n_experts         = 32,
            n_shared_experts  = 2,
            n_experts_per_tok = 4,
            expert_dim        = 512,
            lora_rank         = 16,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = "cuda",
            dtype         = "bfloat16",
            dataset       = "HuggingFaceFW/fineweb-edu",
            seq_len       = 1024,
            batch_size    = 8,
            max_steps     = 100_000,
            warmup_steps  = 2_000,
            learning_rate = 2e-4,
            grad_accum    = 8,       # effective batch = 256 across 4 GPUs
            phase1_steps  = 20_000,
            phase1_loops  = 4,
            phase2_loops  = 12,
            log_every     = 100,
            save_every    = 5_000,
            sample_every  = 5_000,
            run_name      = "anthos-research",
        )

    else:
        raise ValueError(f"Unknown tier '{tier}'. Choose: smoke | proof | research")

    return model_cfg, train_cfg
