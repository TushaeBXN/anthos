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
from pathlib import Path
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

    # Prefetch policy for halt_probs-gated speculative loading
    # "predict_one"  — fetch the single most likely next expert set (halt_probs-gated); default
    # "fetch_union"  — fetch the union of all plausible expert routes above a threshold
    prefetch_policy: str   = "predict_one"


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
            batch_size    = 1,
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

    elif tier == "ethnic":
        # ── MacBook / CPU — same architecture as smoke, ethnic stories dataset ─
        # Resumes from smoke checkpoint. Uses local data/ethnic_stories.txt.
        model_cfg = AnthosConfig(
            vocab_size        = 50257,
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
            dataset       = "data/ethnic_stories.txt",   # local file — no HF needed
            seq_len       = 256,
            batch_size    = 1,
            max_steps     = 20_000,     # 10k steps of new fine-tuning on top of smoke
            warmup_steps  = 200,
            learning_rate = 1e-4,       # lower LR for fine-tuning from checkpoint
            grad_accum    = 4,
            phase1_steps  = 10_500,     # stay in phase-2 (16 loops) immediately
            phase1_loops  = 16,         # pre-trained weights — skip stabilization

            phase2_loops  = 16,
            log_every     = 25,
            save_every    = 1_000,
            sample_every  = 1_000,
            run_name      = "anthos-ethnic",
        )

    elif tier == "instruct":
        # ── Single A100/4090 — instruction tuning on Alpaca ───────────────────
        # Resume from a proof-tier checkpoint. Fine-tunes on 52k instruction
        # pairs so the model follows prompts instead of just completing text.
        # ~1-2 hrs on a single GPU. Run after proof tier converges.
        #
        # python3 train.py --tier instruct --resume checkpoints/anthos-proof/final.pt
        model_cfg = AnthosConfig(
            vocab_size        = 50257,
            dim               = 512,
            n_heads           = 8,
            n_kv_heads        = 4,
            max_seq_len       = 512,
            max_loop_iters    = 16,
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
            dataset       = "tatsu-lab/alpaca",   # 52k instruction pairs
            seq_len       = 512,
            batch_size    = 8,
            max_steps     = 5_000,                # 1 epoch ≈ 52k/8 = 6500 steps
            warmup_steps  = 100,
            learning_rate = 5e-5,                 # low LR — fine-tuning, not pre-training
            min_lr        = 5e-6,
            grad_accum    = 4,                    # effective batch = 32
            phase1_steps  = 0,                    # skip straight to phase-2
            phase1_loops  = 16,
            phase2_loops  = 16,
            log_every     = 50,
            save_every    = 500,
            sample_every  = 500,
            run_name      = "anthos-instruct",
        )

    elif tier == "sft":
        # ── Single GPU — SFT on balanced manifest from build_training_manifest.py
        # Run first:
        #   python build_training_manifest.py   → creates data/train.jsonl
        #   python preflight.py --fast
        #   bash runpod_start.sh
        #
        # seq_len=1024: pairs with --max-tokens 900 in manifest builder.
        # All training examples fit within context — 0% truncation.
        # Identity domain capped at 5% — model is not biased toward self-talk.
        model_cfg = AnthosConfig(
            vocab_size        = 50262,   # GPT-2 50257 + 5 special tokens
            dim               = 512,
            n_heads           = 8,
            n_kv_heads        = 4,
            max_seq_len       = 1024,    # raised from 512 — matches manifest filter
            max_loop_iters    = 16,
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
            dataset       = (
                "data/train.jsonl"          if Path("data/train.jsonl").exists()          else
                "data/sft_master.jsonl"     if Path("data/sft_master.jsonl").exists()     else
                "Open-Orca/SlimOrca"
            ),
            seq_len       = 1024,         # raised from 512 — 0% truncation
            batch_size    = 4,            # halved from 8 because 2x seq_len
            max_steps     = 10_000,
            warmup_steps  = 200,
            learning_rate = 3e-5,
            min_lr        = 3e-6,
            grad_accum    = 8,            # keeps effective batch = 32 (same as before)
            phase1_steps  = 0,
            phase1_loops  = 16,
            phase2_loops  = 16,
            log_every     = 50,
            save_every    = 500,
            sample_every  = 500,
            run_name      = "anthos-sft",
        )

    elif tier == "convo_smoke":
        # ── MacBook CPU — conversation structure on small model ─────────────────
        # Resume from smoke checkpoint. Teaches turn-taking and Q&A structure
        # before any GPU run. Uses 1,000 SlimOrca conversations — small enough
        # to fit in memory, big enough to learn the format.
        #
        # python3 setup_tokenizer.py   (if not already done)
        # python3 train.py --tier convo_smoke --resume checkpoints/anthos-smoke/final.pt
        model_cfg = AnthosConfig(
            vocab_size        = 50262,   # GPT-2 50257 + 5 Anthos special tokens
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
            dataset       = "Open-Orca/SlimOrca",
            seq_len       = 256,
            batch_size    = 1,
            max_steps     = 10_000,
            warmup_steps  = 500,
            learning_rate = 5e-5,         # gentle — fine-tuning from smoke
            min_lr        = 5e-6,
            grad_accum    = 4,            # effective batch = 4
            phase1_steps  = 0,            # already pre-trained — skip stabilization
            phase1_loops  = 16,
            phase2_loops  = 16,
            log_every     = 100,
            save_every    = 2_000,
            sample_every  = 2_000,
            run_name      = "anthos-convo-smoke",
        )

    elif tier == "history":
        # ── MacBook CPU — fine-tune on local markdown essays ──────────────────
        # Same architecture as smoke so it runs on your machine without a GPU.
        # Resume from a smoke checkpoint:
        #   python3 train.py --tier history --resume checkpoints/mansa_sovereign/step_010000.pt
        model_cfg = AnthosConfig(
            vocab_size        = 50262,   # GPT-2 + 5 Anthos special tokens (matches checkpoints)
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
            dataset       = "data/new_history",   # local markdown essays
            seq_len       = 256,
            batch_size    = 1,
            max_steps     = 5_000,
            warmup_steps  = 100,
            learning_rate = 3e-5,    # gentle — precious small dataset
            min_lr        = 3e-6,
            grad_accum    = 4,
            phase1_steps  = 0,       # already pre-trained — skip stabilization
            phase1_loops  = 16,
            phase2_loops  = 16,
            log_every     = 25,
            save_every    = 500,
            sample_every  = 500,
            run_name      = "anthos-history",
        )

    elif tier == "identity_hardening":
        # ── Identity hardening — burns Brian Tushae Thomas into weights ───────
        # Uses data/phase2_train.jsonl (35% identity, 65% capability)
        # Same small arch as smoke/history so it runs on MacBook CPU.
        # Resume from any existing checkpoint:
        #   python train.py --tier identity_hardening --resume checkpoints/mansa_sovereign/step_XXXXXX.pt
        model_cfg = AnthosConfig(
            vocab_size        = 50262,
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
            dataset       = "data/phase2_train.jsonl",
            seq_len       = 256,
            batch_size    = 1,
            max_steps     = 10_000,
            warmup_steps  = 500,
            learning_rate = 1e-4,
            min_lr        = 1e-5,
            grad_accum    = 4,
            phase1_steps  = 0,
            phase1_loops  = 16,
            phase2_loops  = 16,
            log_every     = 100,
            save_every    = 1000,
            sample_every  = 1000,
            run_name      = "anthos-identity",
        )

    elif tier == "code":
        # ── Code tier — trains Anthos to write, review, and explain code ──────
        # Resumes from the proof-tier checkpoint (70.7M params, dim=512).
        # Must match proof arch exactly so resume works cleanly.
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
            device        = auto_device,
            dtype         = "float32",
            dataset       = "data/code_combined.jsonl",
            seq_len       = 512,          # must match model max_seq_len=512 (RoPE bound)
            batch_size    = 1,
            max_steps     = 15_000,
            warmup_steps  = 200,
            learning_rate = 2e-5,         # lower for stability on resume
            min_lr        = 2e-6,
            grad_accum    = 8,
            phase1_steps  = 0,
            phase1_loops  = 4,            # match max_loop_iters=8, safe value
            phase2_loops  = 4,
            log_every     = 100,
            save_every    = 1000,
            sample_every  = 1000,
            run_name      = "anthos-code",
        )

    # ── Staged scaling tiers (follow friend's advice: 47M → 150M → 300M → 1B) ─

    elif tier == "stage2_150m":
        # ── ~150M params | Single A100 40GB | ~$15-30 on RunPod ──────────────
        # Purpose: validate Anthos MoE architecture scales before committing to 1B.
        # Same dataset (sft_master.jsonl), larger model, 1K context.
        # Resume from sft checkpoint or train from scratch.
        model_cfg = AnthosConfig(
            vocab_size        = 50262,   # Anthos tokenizer
            dim               = 768,
            n_heads           = 12,
            n_kv_heads        = 6,
            max_seq_len       = 1024,    # 2x the current 512
            max_loop_iters    = 12,
            prelude_layers    = 4,
            coda_layers       = 4,
            n_thought_tokens  = 16,
            attn_type         = "gqa",
            n_experts         = 16,
            n_shared_experts  = 2,
            n_experts_per_tok = 4,
            expert_dim        = 512,
            lora_rank         = 8,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = "cuda",
            dtype         = "bfloat16",
            dataset       = "data/train.jsonl",   # use deduplicated split
            seq_len       = 1024,
            batch_size    = 8,
            max_steps     = 30_000,
            warmup_steps  = 1_000,
            learning_rate = 2e-4,
            min_lr        = 2e-5,
            weight_decay  = 0.1,
            grad_clip     = 1.0,
            grad_accum    = 4,            # effective batch = 32
            phase1_steps  = 5_000,
            phase1_loops  = 4,
            phase2_loops  = 12,
            log_every     = 100,
            save_every    = 2_000,
            sample_every  = 2_000,
            checkpoint_dir = "checkpoints/stage2_150m",
            run_name      = "anthos-150m",
        )

    elif tier == "stage3_300m":
        # ── ~300M params | 2× A100 40GB | ~$30-60 on RunPod ─────────────────
        # Purpose: establish genuine conversational capability.
        # Use: torchrun --nproc_per_node=2 train.py --tier stage3_300m
        model_cfg = AnthosConfig(
            vocab_size        = 50262,
            dim               = 1024,
            n_heads           = 16,
            n_kv_heads        = 8,
            max_seq_len       = 2048,    # 4x the current 512
            max_loop_iters    = 12,
            prelude_layers    = 6,
            coda_layers       = 6,
            n_thought_tokens  = 24,
            attn_type         = "gqa",
            n_experts         = 16,
            n_shared_experts  = 2,
            n_experts_per_tok = 4,
            expert_dim        = 768,
            lora_rank         = 16,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = "cuda",
            dtype         = "bfloat16",
            dataset       = "data/train.jsonl",
            seq_len       = 2048,
            batch_size    = 4,
            max_steps     = 50_000,
            warmup_steps  = 2_000,
            learning_rate = 1.5e-4,
            min_lr        = 1.5e-5,
            weight_decay  = 0.1,
            grad_clip     = 1.0,
            grad_accum    = 8,            # effective batch = 32 per GPU
            phase1_steps  = 8_000,
            phase1_loops  = 4,
            phase2_loops  = 12,
            log_every     = 100,
            save_every    = 5_000,
            sample_every  = 5_000,
            checkpoint_dir = "checkpoints/stage3_300m",
            run_name      = "anthos-300m",
        )

    elif tier == "stage4_1b":
        # ── ~1B params | 4× A100 80GB | ~$80-150 on RunPod ──────────────────
        # Purpose: first genuinely useful Anthos. This is the target milestone.
        # Use: torchrun --nproc_per_node=4 train.py --tier stage4_1b
        # IMPORTANT: run audit_dataset.py and create balanced train.jsonl first.
        model_cfg = AnthosConfig(
            vocab_size        = 50262,
            dim               = 2048,
            n_heads           = 16,
            n_kv_heads        = 8,
            max_seq_len       = 4096,    # target context for legal/medical/code
            max_loop_iters    = 16,
            prelude_layers    = 8,
            coda_layers       = 8,
            n_thought_tokens  = 32,
            attn_type         = "gqa",
            n_experts         = 32,
            n_shared_experts  = 2,
            n_experts_per_tok = 4,
            expert_dim        = 1024,
            lora_rank         = 16,
            moe_aux_coef      = 1e-2,
            act_aux_coef      = 1e-3,
        )
        train_cfg = TrainingConfig(
            device        = "cuda",
            dtype         = "bfloat16",
            dataset       = "data/train.jsonl",
            seq_len       = 4096,
            batch_size    = 2,
            max_steps     = 100_000,
            warmup_steps  = 5_000,
            learning_rate = 1e-4,
            min_lr        = 1e-5,
            weight_decay  = 0.1,
            grad_clip     = 1.0,
            grad_accum    = 16,           # effective batch = 32 per GPU × 4 GPUs
            phase1_steps  = 15_000,
            phase1_loops  = 4,
            phase2_loops  = 16,
            log_every     = 100,
            save_every    = 5_000,
            sample_every  = 5_000,
            checkpoint_dir = "checkpoints/stage4_1b",
            run_name      = "anthos-1b",
        )

    else:
        raise ValueError(
            f"Unknown tier '{tier}'. Choose: smoke | proof | research | ethnic | instruct | "
            f"sft | convo_smoke | history | identity_hardening | code | "
            f"stage2_150m | stage3_300m | stage4_1b"
        )

    return model_cfg, train_cfg


def get_model_config(variant: str) -> "AnthosConfig":
    """
    Return the AnthosConfig for a named production variant.

    Variants
    --------
    anthos_1b   — dim=2048, experts=64,  thought_tokens=16, loop_iters=16, context=4k
    anthos_3b   — dim=3072, experts=64,  thought_tokens=24, loop_iters=16, context=4k
    anthos_10b  — dim=4096, experts=128, thought_tokens=32, loop_iters=24, context=8k
    anthos_35b  — dim=5120, experts=256, thought_tokens=40, loop_iters=28, context=8k  (~35B total, ~3B active/tok)
    anthos_50b  — dim=6144, experts=256, thought_tokens=48, loop_iters=32, context=8k
    anthos_100b — dim=8192, experts=256, thought_tokens=64, loop_iters=32, context=1M
    """
    from anthos.main import anthos_1b, anthos_3b, anthos_10b, anthos_35b, anthos_50b, anthos_100b

    _variants = {
        "anthos_1b":   anthos_1b,
        "anthos_3b":   anthos_3b,
        "anthos_10b":  anthos_10b,
        "anthos_35b":  anthos_35b,
        "anthos_50b":  anthos_50b,
        "anthos_100b": anthos_100b,
    }

    if variant not in _variants:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Choose from: {sorted(_variants.keys())}"
        )

    return _variants[variant]()
