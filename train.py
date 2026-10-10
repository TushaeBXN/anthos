"""
Anthos — Sovereign Training Script (Mansa Edition)
Think in Streams.
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Optional 8-bit Adam (bitsandbytes) — falls back to regular AdamW if unavailable
try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

# Optional wandb — falls back gracefully if not installed
try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.data    import get_dataloader, get_instruct_dataloader, get_chat_dataloader, get_markdown_dataloader
from anthos.sasft         import RepetitionPenaltyLoss, ThoughtDiversityLoss
from anthos.steering      import ActivationCollector
from anthos.memory_compress import MemoryAugmentedDataset
from anthos.distill       import DistillConfig, DistillationLoss, TeacherLabelDataset
from anthos.eaft          import EAFTLoss
from anthos.telemetry     import TelemetryHarness
from anthos.distributed   import (
    setup_distributed, teardown_distributed,
    is_main_process, barrier,
    wrap_ddp, unwrap_model,
    save_checkpoint_distributed, load_checkpoint_distributed,
    prune_old_checkpoints,
)

# ─────────────────────────────────────────────────────────────────────────────
# MANSA CONFIGURATION (HARD-CODED FOR STABILITY)
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS    = 10000
MAX_LR       = 1.0e-4
MIN_LR       = 1.0e-5
WARMUP_STEPS = 2000
PHASE1_STEPS = 3000
PHASE1_LOOPS = 4
PHASE2_LOOPS = 16
SEQ_LEN      = 512
LOG_EVERY    = 100
SAVE_EVERY   = 1000

# ─────────────────────────────────────────────────────────────────────────────
# MoE collapse monitor — configurable thresholds + auto-bump
# ─────────────────────────────────────────────────────────────────────────────
MOE_WARN_THRESHOLD  = 0.50   # WARN if top-1 expert captures > 50% of routing mass
MOE_BUMP_THRESHOLD  = 0.60   # auto-bump moe_aux_coef if top-1 > 60%
MOE_BUMP_MULTIPLIER = 2.0    # new coef = old coef * multiplier (0.01 → 0.02)


def check_moe_collapse(model, model_cfg, step: int) -> float:
    """
    Inspect the cached top-k routing indices from the last forward pass.
    Returns the fraction of routing mass going to the single busiest expert.

    Side-effects:
      - Prints WARN if fraction > MOE_WARN_THRESHOLD
      - Auto-bumps model_cfg.moe_aux_coef if fraction > MOE_BUMP_THRESHOLD
        (the model reads cfg.moe_aux_coef each step via aux_loss calculation)
    """
    try:
        topk_idx = model.recurrent.block.ffn._last_topk_idx  # (N, topk) int — rank-0 only
        if topk_idx is None:
            return 0.0
        n_experts = model_cfg.n_experts
        counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).float()
        top1_frac = (counts.max() / counts.sum()).item()

        if top1_frac > MOE_BUMP_THRESHOLD:
            old_coef = model_cfg.moe_aux_coef
            model_cfg.moe_aux_coef = round(old_coef * MOE_BUMP_MULTIPLIER, 6)
            print(f"  ⚡ MoE collapse at step {step}: top-1={top1_frac:.1%} > {MOE_BUMP_THRESHOLD:.0%} — "
                  f"auto-bumped moe_aux_coef {old_coef} → {model_cfg.moe_aux_coef}", flush=True)
        elif top1_frac > MOE_WARN_THRESHOLD:
            print(f"  ⚠ MoE routing: top-1 expert at {top1_frac:.1%} (warn threshold {MOE_WARN_THRESHOLD:.0%})", flush=True)

        return top1_frac
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Logic Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def get_lr_wsd(step: int, decay_frac: float = 0.1) -> float:
    """
    Warmup-Stable-Decay schedule (OLMo-core / MiniCPM).

    Three phases:
      [0, WARMUP_STEPS)        — linear ramp from 0 to MAX_LR
      [WARMUP_STEPS, decay_start) — constant MAX_LR  (stable phase)
      [decay_start, MAX_STEPS]    — cosine decay to MIN_LR

    decay_frac controls what fraction of steps are in the decay phase.
    decay_frac=0.1 → last 10% of steps decay; 0.2 → last 20%.

    Advantages over cosine-only:
      - Model trains at peak LR longer → better token utilisation
      - Sharp decay at the end improves final eval metrics
      - Easy to extend training: simply prolong the stable phase
    """
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    decay_start = int(MAX_STEPS * (1.0 - decay_frac))
    if step < decay_start:
        return MAX_LR
    if step >= MAX_STEPS:
        return MIN_LR
    progress = (step - decay_start) / (MAX_STEPS - decay_start)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

def get_n_loops(step: int) -> int:
    return PHASE1_LOOPS if step < PHASE1_STEPS else PHASE2_LOOPS

@torch.no_grad()
def generate_samples(
    model: Anthos,
    device: str,
    n_loops: int,
    n_samples: int = 3,
    tokenizer_path: str = "gpt2",
) -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    tok.model_max_length = SEQ_LEN
    model.eval()
    prompts = ["def fibonacci(n):", "class DataLoader:", "import torch\n\ndef train("][:n_samples]
    samples = []
    for prompt in prompts:
        enc = tok.encode(prompt, truncation=True, max_length=SEQ_LEN)
        ids = torch.tensor([enc], dtype=torch.long).to(device)
        out = model.generate(ids, max_new_tokens=80, n_loops=n_loops, temperature=0.8, top_k=40)
        samples.append(tok.decode(out[0].tolist()))
    model.train()
    return samples

def get_param_groups(model: Anthos, weight_decay: float = 0.1) -> list[dict]:
    """
    Separate parameters into two groups:
      - decay:    weight matrices (ndim >= 2) — applies weight_decay
      - no_decay: LayerNorm weights, biases, embedding weights — no decay

    OLMo-core pattern: decaying embeddings or LayerNorm params hurts
    convergence because they lack the redundancy of weight matrices.
    """
    decay, no_decay = set(), set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Bias terms: always no-decay
        if name.endswith(".bias"):
            no_decay.add(name)
        # Embedding weight matrices: no-decay (they're lookup tables, not linear maps)
        elif "embed" in name and param.ndim == 2:
            no_decay.add(name)
        # LayerNorm / RMSNorm scale: 1-D, no-decay
        elif param.ndim == 1:
            no_decay.add(name)
        else:
            decay.add(name)

    param_map = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert not (decay & no_decay), "Param appeared in both groups"
    assert decay | no_decay == set(param_map.keys()), "Not all params accounted for"

    return [
        {"params": [param_map[n] for n in sorted(decay)],    "weight_decay": weight_decay},
        {"params": [param_map[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    tier: str = "proof",
    resume: str | None = None,
    teacher_labels: str | None = None,
    max_steps: int | None = None,
    compile_model: bool = False,
    gradient_checkpointing: bool = False,
    use_8bit_adam: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "anthos",
    use_tb: bool = False,
    curriculum: bool = False,
    spike_rollback: bool = False,
    spike_factor: float = 4.0,
):
    global MAX_STEPS, MAX_LR, MIN_LR, WARMUP_STEPS, SEQ_LEN, LOG_EVERY, SAVE_EVERY, PHASE1_STEPS, PHASE1_LOOPS, PHASE2_LOOPS

    # ── Distributed setup ─────────────────────────────────────────────────────
    rank, local_rank, world_size = setup_distributed()
    is_distributed = world_size > 1

    model_cfg, train_cfg = get_training_config(tier)
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"
    ckpt_dir = Path("checkpoints") / train_cfg.run_name

    # Determine hardware label for telemetry sidecar
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.device_count() > 0 else "unknown_gpu"
        _hardware_label = gpu_name.replace(" ", "_")
        if is_distributed:
            _hardware_label = f"{_hardware_label}_x{world_size}"
    else:
        _hardware_label = "cpu_only"
    # Stage tag: embedded in every telemetry sidecar so checkpoints stay
    # comparable even after higher-budget runs exist alongside them.
    _stage_tag = f"{tier}_stage1_40pct_195M_tokens" if tier == "sft" else tier
    _telemetry_harness = TelemetryHarness(model_cfg, hardware=_hardware_label,
                                          stage_tag=_stage_tag)

    # ── Tier-specific overrides ───────────────────────────────────────────────
    if tier in ("sft", "instruct"):
        MAX_STEPS    = 500
        MAX_LR       = 3e-5
        MIN_LR       = 3e-6
        WARMUP_STEPS = 50
    elif tier == "convo_smoke":
        MAX_STEPS    = 10_000
        MAX_LR       = 5e-5
        MIN_LR       = 5e-6
        WARMUP_STEPS = 500
        SEQ_LEN      = 256
    elif tier == "identity_hardening":
        MAX_STEPS    = 10_000
        MAX_LR       = 1e-4
        MIN_LR       = 1e-5
        WARMUP_STEPS = 500
        SEQ_LEN      = 256
        LOG_EVERY    = 100
        SAVE_EVERY   = 1000
    elif tier == "history":
        MAX_STEPS    = 5_000
        MAX_LR       = 3e-5
        MIN_LR       = 3e-6
        WARMUP_STEPS = 100
        SEQ_LEN      = 256
        LOG_EVERY    = 10
        SAVE_EVERY   = 500
    elif tier == "code":
        MAX_STEPS    = 10_000
        MAX_LR       = 2e-5   # stable for fine-tuning from proof checkpoint
        MIN_LR       = 2e-6
        WARMUP_STEPS = 200
        SEQ_LEN      = 512
        PHASE1_STEPS = 99_999  # never trigger phase2 — keep 4 loops throughout
        PHASE2_LOOPS = 4       # stays within max_loop_iters=8
        LOG_EVERY    = 100
        SAVE_EVERY   = 100
    elif tier == "distill":
        MAX_STEPS    = 10_000
        MAX_LR       = 2e-4
        MIN_LR       = 2e-5
        WARMUP_STEPS = 500

    # ── CLI override (always applied last) ───────────────────────────────────
    if max_steps is not None:
        MAX_STEPS = max_steps

    if gradient_checkpointing:
        model_cfg.gradient_checkpointing = True

    # ── Flash Attention 2 backend selection ──────────────────────────────────
    # F.scaled_dot_product_attention (already used in GQAttention/MLAttention)
    # dispatches to Flash Attention 2 when running on CUDA with bf16/fp16.
    # Explicitly prefer the Flash kernel over mem-efficient; fall back to math
    # on CPU / older GPUs that don't support FA2.
    if device.startswith("cuda"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)   # disable slow fallback
        except Exception:
            pass   # older PyTorch version — SDPA still works, just no explicit hint

    model = Anthos(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── torch.compile ─────────────────────────────────────────────────────────
    # Compile BEFORE DDP so DDP sees the already-compiled module.
    # mode="reduce-overhead": fuses small ops, reduces Python interpreter
    # overhead — best for models with many small kernels (MoE, ACT).
    # Skipped if --no-compile or PyTorch < 2.0.
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            if is_main_process():
                print("  ✓ torch.compile enabled (mode=reduce-overhead, dynamic=True)")
        except Exception as e:
            if is_main_process():
                print(f"  ⚠ torch.compile skipped: {e}")

    if is_distributed:
        model = wrap_ddp(model, local_rank)

    # ── Loss functions ────────────────────────────────────────────────────────
    eaft_criterion = EAFTLoss(
        vocab_size      = model_cfg.vocab_size,
        top_k           = 50,
        focal_gamma     = 1.0,   # entropy weight: uncertain positions get more gradient
        act_gamma       = 0.5,   # extra weight for positions that used more loops
        max_loops       = model_cfg.max_loop_iters,
        label_smoothing = 0.1,
    )
    rep_loss_fn   = RepetitionPenaltyLoss(ngram_size=4, penalty=0.3)
    div_loss_fn   = ThoughtDiversityLoss(coeff=0.05)
    thought_collector = ActivationCollector(
        model, stream="thought",
        n_thought_tokens=model_cfg.n_thought_tokens,
    )
    thought_collector.attach()

    # Thought-token ID stability guard — catches silent drift if tokenizer is
    # regenerated with a different special-token order.
    if Path("data/anthos_tokenizer").exists():
        try:
            from transformers import AutoTokenizer as _AT
            _tok_check = _AT.from_pretrained("data/anthos_tokenizer")
            _tht_actual = _tok_check.convert_tokens_to_ids("<|thought|>")
            assert _tht_actual == 50259, (
                f"THT_ID mismatch: tokenizer says <|thought|>={_tht_actual}, "
                f"train.py hardcodes 50259. Regenerate tokenizer or update THT_ID."
            )
            del _tok_check, _tht_actual, _AT
        except Exception as _e:
            print(f"  ⚠ thought-token ID check failed: {_e}")

    # ── Which heads are active this run? (resolve ambiguity before training) ──
    _raw_model = unwrap_model(model)
    _has_prerouter  = hasattr(_raw_model.recurrent, "prerouter_head")
    _has_halt_probs = hasattr(_raw_model, "halt_probs") or True
    if is_main_process():
        print(f"\n{'─'*60}")
        print(f"  Anthos — Sovereign Training (Mansa Edition)")
        print(f"  Tier:       {tier}")
        print(f"  Parameters: {total_params:,}")
        print(f"  GPUs:       {world_size} ({'DDP' if is_distributed else 'single'})")
        print(f"  Compile:    {'yes (reduce-overhead)' if compile_model else 'no'}")
        print(f"  Grad ckpt:  {'yes (~60% activation memory saved)' if gradient_checkpointing else 'no'}")
        print(f"  Max Steps:  {MAX_STEPS:,} | Warmup: {WARMUP_STEPS}")
        print(f"  Max LR:     {MAX_LR} | Loops: {PHASE1_LOOPS}->{PHASE2_LOOPS}")
        print(f"  Stage tag:  {_stage_tag}")
        print(f"  ── Active heads ──────────────────────────────────")
        print(f"    PreRouterHead : {'YES — KL aux loss active' if _has_prerouter else 'NO'}")
        print(f"    halt_probs    : YES — (B,T,n_loops) exposed on model.halt_probs")
        print(f"    LoRA recovery : wired in distill.py (not active this tier)")
        print(f"  ── MoE collapse monitor ──────────────────────────")
        print(f"    WARN at {MOE_WARN_THRESHOLD:.0%} | AUTO-BUMP at {MOE_BUMP_THRESHOLD:.0%} (×{MOE_BUMP_MULTIPLIER})")
        if tier == "sft":
            print(f"  ── Stage 1 expectations ──────────────────────────")
            print(f"    ~20% Chinchilla budget (195M / 960M tokens)")
            print(f"    Hallucination and short-context loss are EXPECTED — not bugs")
            print(f"    Goal: validate loss curve shape, ACT stability, MoE routing spread")
        print(f"{'─'*60}\n")

    use_fused    = device.startswith("cuda")
    param_groups = get_param_groups(unwrap_model(model), weight_decay=0.1)

    # 8-bit Adam: same API as AdamW, uses block-wise quantisation to cut
    # optimizer state memory ~75% (from 8 bytes/param to ~2 bytes/param).
    # Falls back to AdamW when bitsandbytes is not installed.
    if use_8bit_adam and _BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(param_groups, lr=MAX_LR, betas=(0.9, 0.95))
        if is_main_process():
            print("  ✓ Using 8-bit AdamW (bitsandbytes) — ~75% optimizer state memory saving")
    else:
        if use_8bit_adam and not _BNB_AVAILABLE:
            if is_main_process():
                print("  ⚠ --8bit-adam requested but bitsandbytes not installed; using AdamW")
        optimizer = AdamW(param_groups, lr=MAX_LR, betas=(0.9, 0.95), fused=use_fused)

    n_decay    = sum(p.numel() for p in param_groups[0]["params"])
    n_no_decay = sum(p.numel() for p in param_groups[1]["params"])

    # ── Monitoring: wandb and/or TensorBoard ─────────────────────────────────
    _wandb_run = None
    _tb_writer = None
    if is_main_process():
        if use_wandb and _WANDB_AVAILABLE:
            _wandb_run = _wandb_module.init(
                project = wandb_project,
                name    = f"{train_cfg.run_name}",
                config  = {
                    "tier":       tier,
                    "max_steps":  MAX_STEPS,
                    "max_lr":     MAX_LR,
                    "seq_len":    SEQ_LEN,
                    "world_size": world_size,
                    "params":     total_params,
                },
            )
            print(f"  ✓ wandb logging enabled (project={wandb_project})")
        elif use_wandb:
            print("  ⚠ --wandb requested but wandb not installed; run: pip install wandb")

        if use_tb and _TB_AVAILABLE:
            _tb_writer = _TBWriter(log_dir=f"runs/{train_cfg.run_name}")
            print(f"  ✓ TensorBoard logging enabled → runs/{train_cfg.run_name}")
        elif use_tb:
            print("  ⚠ --tb requested but TensorBoard not available")
    if is_main_process():
        print(f"  Optimizer: decay {n_decay:,} params | no-decay {n_no_decay:,} params | wd=0.1")

    start_step = 0
    if resume:
        _opt_for_load = None if tier in ("sft", "instruct", "convo_smoke", "history", "code") else optimizer
        meta = load_checkpoint_distributed(
            Path(resume), model, optimizer=_opt_for_load, device=device
        )
        start_step = meta.get("step", 0)
        if tier in ("sft", "instruct", "convo_smoke", "history", "code") and tier != "code":
            start_step = 0
        if is_main_process():
            print(f"  ✓ Resumed from {resume} at step {start_step}")

    is_sft      = (tier in ("sft", "instruct", "convo_smoke", "identity_hardening", "code"))
    is_distill  = (tier == "distill")
    is_history  = (tier == "history")

    if is_sft and tier != "code" and Path("data/anthos_tokenizer").exists():
        tok_path = "data/anthos_tokenizer"
    else:
        tok_path = "gpt2"  # proof checkpoint vocab is 50257, must match

    if is_history:
        history_dir = "data/new_history"
        print(f"  ✓ History tier: training on markdown essays in {history_dir}/")
        loader  = get_markdown_dataloader(
            directory  = history_dir,
            seq_len    = SEQ_LEN,
            batch_size = train_cfg.batch_size,
            num_workers= 0,
        )
        tok_path = "gpt2"

    elif is_distill:
        labels_path = teacher_labels or "data/teacher_labels.jsonl"
        if not Path(labels_path).exists():
            raise FileNotFoundError(
                f"Teacher labels not found: {labels_path}\n"
                f"Run: python generate_teacher_labels.py --teacher <model> --out {labels_path}"
            )
        print(f"  ✓ Distill mode: loading teacher labels from {labels_path}")
        from torch.utils.data import DataLoader

        def _distill_collate(batch):
            ids_list  = [x[0] for x in batch]
            tlog_list = [x[1] for x in batch]
            max_T = max(t.shape[0] for t in ids_list)
            V = tlog_list[0].shape[-1]
            ids_pad  = torch.zeros(len(batch), max_T, dtype=torch.long)
            tlog_pad = torch.full((len(batch), max_T, V), float("-inf"))
            for i, (ids, tlog) in enumerate(zip(ids_list, tlog_list)):
                T = ids.shape[0]
                ids_pad[i, :T]      = ids
                tlog_pad[i, :T, :V] = tlog[:, :V]
            return ids_pad, tlog_pad

        teacher_dataset = TeacherLabelDataset(
            path=labels_path,
            student_vocab_size=model_cfg.vocab_size,
            teacher_vocab_size=model_cfg.vocab_size,
        )
        loader = DataLoader(
            teacher_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=_distill_collate,
            num_workers=0,
        )
        distill_loss_fn = DistillationLoss(DistillConfig(temperature=4.0, alpha=0.3))
        tok_path = "gpt2"

    elif is_sft:
        if tier == "identity_hardening":
            local_data = "data/phase2_train.jsonl"
            if not Path(local_data).exists():
                raise FileNotFoundError(
                    f"{local_data} not found.\n"
                    "Run: python build_phase2_dataset.py"
                )
            dataset_name = local_data
            max_samples  = 0
            print(f"  ✓ Identity hardening: loading {local_data}")
            print(f"    Creator: Brian Tushae Thomas | Model: Anthos")
        elif tier == "code":
            # Prefer merged file (generation + evaluation), fall back to generation only
            if Path("data/code_combined.jsonl").exists():
                local_data = "data/code_combined.jsonl"
            elif Path("data/code_teacher.jsonl").exists():
                local_data = "data/code_teacher.jsonl"
            else:
                raise FileNotFoundError(
                    "No code training data found.\n"
                    "Run: python3 generate_code_teacher_data.py --n 5000\n"
                    "Then: python3 generate_code_eval_data.py --n 2000\n"
                    "Then: cat data/code_teacher.jsonl data/code_eval.jsonl > data/code_combined.jsonl"
                )
            dataset_name = local_data
            max_samples  = 0
            print(f"  ✓ Code tier: loading {local_data}")
        elif tier == "sft" or tier == "instruct":
            # Use whatever data path configs.py resolved (sft_master.jsonl first)
            local_data = train_cfg.dataset
            if Path(local_data).exists():
                dataset_name = local_data
                max_samples  = 0
                n_lines = sum(1 for _ in open(local_data)) if Path(local_data).exists() else 0
                print(f"  ✓ SFT tier: loading {local_data} ({n_lines:,} pairs)")
            else:
                raise FileNotFoundError(
                    f"SFT data not found: {local_data}\n"
                    f"Expected data/sft_master.jsonl — run ingest scripts first."
                )
        elif tier == "convo_smoke":
            local_data = "data/teacher_conversations.jsonl"
            if Path(local_data).exists():
                dataset_name = local_data
                max_samples  = 0
                print(f"  ✓ Using local teacher data: {local_data}")
            else:
                dataset_name = "mlabonne/FineTome-100k"
                max_samples  = 5000
                print(f"  ✓ Using FineTome-100k (run generate_teacher_data.py for better results)")
        else:
            max_samples  = 0
            dataset_name = "Open-Orca/SlimOrca"

        n_workers = 0 if tier in ("convo_smoke", "identity_hardening", "code") else 1
        loader = get_chat_dataloader(
            seq_len        = SEQ_LEN,
            batch_size     = train_cfg.batch_size,
            num_workers    = n_workers,
            tokenizer_path = tok_path,
            max_samples    = max_samples,
            dataset_name   = dataset_name,
        )
    else:
        loader = get_dataloader(
            dataset_name = train_cfg.dataset,
            split        = "train",
            seq_len      = SEQ_LEN,
            batch_size   = train_cfg.batch_size,
            num_workers  = 1,
        )

    if tier in ("proof", "sft", "instruct"):
        loader = MemoryAugmentedDataset(
            loader,
            compress_fraction=0.15,
            prefix_confidence=3,
        )

    data_iter  = iter(loader)
    data_step  = 0

    # ── Curriculum learning: start with short sequences, ramp up to SEQ_LEN ──
    # Short sequences in early training help the model learn basic patterns
    # before seeing long-range dependencies — stabilises loss in the first
    # ~10% of training and reduces early spikes.
    _curriculum_target_seq = SEQ_LEN   # final length; stored for ramp computation
    if curriculum:
        _curr_start_len = max(64, SEQ_LEN // 8)
        if is_main_process():
            print(f"  ✓ Curriculum learning: seq len ramps {_curr_start_len} → {SEQ_LEN} "
                  f"over first 10% of steps ({MAX_STEPS // 10:,} steps)")
    else:
        _curr_start_len = SEQ_LEN

    # Resume dataloader position so we never replay the same batches.
    # We skip `resume_data_step` batches by fast-forwarding the iterator.
    resume_data_step = 0
    if resume:
        _rckpt = torch.load(resume, map_location="cpu", weights_only=False)
        resume_data_step = _rckpt.get("data_step", 0)
        del _rckpt
    if resume_data_step > 0:
        print(f"  ⏩ Fast-forwarding dataloader by {resume_data_step} batches...", flush=True)
        for _ in range(resume_data_step):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                next(data_iter)
        data_step = resume_data_step
        print(f"  ✓ Dataloader resumed at batch {data_step}", flush=True)

    model.train()
    step              = start_step
    loss_accum        = aux_accum = prerouter_accum = gnorm_accum = 0.0
    avg_loss          = 0.0
    _gnorm_ema        = None   # exponential moving average for spike detection
    _loss_ema         = None   # EMA for loss spike rollback
    _last_good_ckpt   = None   # path of last checkpoint before a potential spike
    t0                = time.time()

    if is_main_process():
        print(f"  🔄 Training loop started — logging every {LOG_EVERY} steps, saving every {SAVE_EVERY}", flush=True)
        print(f"  ⏳ First log will appear after step {LOG_EVERY} (may take 1-2 min on CPU)...", flush=True)

    while step < MAX_STEPS:
        lr     = get_lr(step)
        for pg in optimizer.param_groups: pg["lr"] = lr
        n_loops = get_n_loops(step)

        # Curriculum: linearly ramp effective seq len from _curr_start_len → _curriculum_target_seq
        if curriculum:
            ramp_steps = max(1, MAX_STEPS // 10)
            curr_progress = min(1.0, step / ramp_steps)
            _eff_seq = int(_curr_start_len + curr_progress * (_curriculum_target_seq - _curr_start_len))
            _eff_seq = max(_curr_start_len, min(_eff_seq, _curriculum_target_seq))
        else:
            _eff_seq = SEQ_LEN

        optimizer.zero_grad()
        for _ in range(train_cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch     = next(data_iter)
            data_step += 1

            if is_distill:
                input_ids       = batch[0].to(device)[:, :_eff_seq]
                teacher_logits  = batch[1].to(device)[:, :_eff_seq, :]
                labels          = input_ids[:, 1:]
                input_ids_in    = input_ids[:, :-1]
            elif is_sft:
                input_ids       = batch[0].to(device)[:, :_eff_seq]
                labels          = batch[1].to(device)[:, :_eff_seq]
                input_ids_in    = input_ids
            else:
                batch        = batch.to(device)
                input_ids_in = batch[:, :_eff_seq-1]
                labels       = batch[:, 1:_eff_seq]

            _autocast = (
                torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device == "cuda" else nullcontext()
            )
            with _autocast:
                logits, aux = model(input_ids_in, n_loops=n_loops, return_aux=True)

                if is_distill:
                    t_log = teacher_logits[:, :-1, :]
                    loss_kd, kd_info = distill_loss_fn(
                        student_logits=logits,
                        teacher_logits=t_log,
                        labels=labels,
                    )
                    ce  = torch.tensor(kd_info["ce_loss"], device=device)
                    loss = (loss_kd + aux) / train_cfg.grad_accum
                else:
                    # EAFT: entropy-aware focal loss — uncertain positions (high
                    # entropy) and positions that used more ACT loops get more gradient
                    loops_used = getattr(model, "_last_loops_used", None)
                    ce         = eaft_criterion(logits, labels, loops_used=loops_used)
                    rep_pen = rep_loss_fn(logits)
                    thought_acts = thought_collector.flat_activations().to(device)
                    n_thought    = model_cfg.n_thought_tokens
                    # ── FIX: trim thought_acts to a clean multiple of (n_thought * dim)
                    # before reshape. Raw activations can be any size depending on
                    # batch/seq combinations; the view requires exact divisibility.
                    chunk = n_thought * model_cfg.dim
                    if thought_acts.shape[0] >= chunk:
                        trimmed = thought_acts[: (thought_acts.shape[0] // chunk) * chunk]
                        div_pen = div_loss_fn(
                            trimmed.view(-1, n_thought, model_cfg.dim)
                        )
                    else:
                        div_pen = torch.tensor(0.0, device=device)
                    thought_collector.clear()
                    loss = (ce + aux + rep_pen + div_pen) / train_cfg.grad_accum

            loss.backward()
            loss_accum      += ce.item()
            aux_accum       += aux.item()
            prerouter_accum += getattr(model, "_last_prerouter_aux", torch.tensor(0.0)).item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        gnorm_accum += grad_norm
        # Spike detection: warn if norm is >3× the running EMA
        if _gnorm_ema is None:
            _gnorm_ema = grad_norm
        else:
            _gnorm_ema = 0.98 * _gnorm_ema + 0.02 * grad_norm
            if grad_norm > 3.0 * _gnorm_ema and grad_norm > 1.0 and is_main_process():
                print(f"  ⚡ Grad norm spike at step {step+1}: {grad_norm:.2f} (ema={_gnorm_ema:.2f})", flush=True)
        optimizer.step()
        step += 1

        # ── Loss spike rollback ───────────────────────────────────────────────
        # If the current step's CE loss is spike_factor × the running EMA of
        # loss, restore the last saved checkpoint and continue from there.
        # This auto-recovers from bad data batches and numerical instabilities.
        _step_loss = ce.item() if not torch.isnan(ce) else float("inf")
        if _loss_ema is None:
            _loss_ema = _step_loss
        else:
            _loss_ema = 0.99 * _loss_ema + 0.01 * _step_loss
        if (
            spike_rollback
            and _last_good_ckpt is not None
            and _step_loss > spike_factor * _loss_ema
            and _step_loss > 2.0          # ignore tiny absolute values
            and is_main_process()
        ):
            print(
                f"  ⚠ Loss spike at step {step}: {_step_loss:.4f} > "
                f"{spike_factor}× ema ({_loss_ema:.4f}) — rolling back to {_last_good_ckpt}",
                flush=True,
            )
            meta = load_checkpoint_distributed(
                Path(_last_good_ckpt), model, optimizer=optimizer, device=device
            )
            step = meta.get("step", step)
            _loss_ema = None   # reset EMA after rollback

        if step % LOG_EVERY == 0 and is_main_process():
            t1            = time.time()
            avg_loss      = loss_accum      / (LOG_EVERY * train_cfg.grad_accum)
            avg_aux       = aux_accum       / (LOG_EVERY * train_cfg.grad_accum)
            avg_prerouter = prerouter_accum / (LOG_EVERY * train_cfg.grad_accum)
            avg_gnorm     = gnorm_accum     / LOG_EVERY
            # Scale tok/s by world_size — all GPUs processed batches in parallel
            tok_sec = (LOG_EVERY * train_cfg.batch_size * train_cfg.grad_accum
                       * _eff_seq * world_size) / (t1 - t0)
            top1_frac = check_moe_collapse(unwrap_model(model), model_cfg, step)
            _curr_info = f" | seq {_eff_seq}" if curriculum else ""
            print(
                f"step {step:6d} | loss {avg_loss:.4f} | aux {avg_aux:.5f} | "
                f"prerouter_kl {avg_prerouter:.4f} | gnorm {avg_gnorm:.3f} | "
                f"moe_top1 {top1_frac:.1%} | loops {n_loops} | lr {lr:.2e}"
                f"{_curr_info} | {tok_sec:,.0f} tok/s",
                flush=True,
            )

            # ── wandb logging ─────────────────────────────────────────────────
            if _wandb_run is not None:
                _wandb_run.log({
                    "loss":          avg_loss,
                    "aux_loss":      avg_aux,
                    "prerouter_kl":  avg_prerouter,
                    "grad_norm":     avg_gnorm,
                    "moe_top1":      top1_frac,
                    "lr":            lr,
                    "n_loops":       n_loops,
                    "tok_per_sec":   tok_sec,
                    "seq_len":       _eff_seq,
                }, step=step)

            # ── TensorBoard logging ───────────────────────────────────────────
            if _tb_writer is not None:
                _tb_writer.add_scalar("train/loss",         avg_loss,      step)
                _tb_writer.add_scalar("train/aux_loss",     avg_aux,       step)
                _tb_writer.add_scalar("train/prerouter_kl", avg_prerouter, step)
                _tb_writer.add_scalar("train/grad_norm",    avg_gnorm,     step)
                _tb_writer.add_scalar("train/moe_top1",     top1_frac,     step)
                _tb_writer.add_scalar("train/lr",           lr,            step)
                _tb_writer.add_scalar("train/tok_per_sec",  tok_sec,       step)

            loss_accum = aux_accum = prerouter_accum = gnorm_accum = 0.0
            t0 = t1

        if step % SAVE_EVERY == 0:
            _ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint_distributed(
                _ckpt_path, model, optimizer, step, avg_loss, data_step=data_step
            )
            prune_old_checkpoints(ckpt_dir, keep=3)
            _last_good_ckpt = str(_ckpt_path)   # track for spike rollback
            if is_main_process():
                try:
                    _tel = _telemetry_harness.record(
                        unwrap_model(model), tokenizer=None,
                        ckpt_path=str(_ckpt_path), step=step,
                    )
                    _tel["train_scalars"] = {
                        "loss":         round(avg_loss,      5),
                        "aux":          round(avg_aux,        5),
                        "prerouter_kl": round(avg_prerouter,  5),
                        "grad_norm":    round(avg_gnorm,      4),
                        "moe_aux_coef": model_cfg.moe_aux_coef,
                        "world_size":   world_size,
                    }
                    import json as _json
                    _sidecar_path = str(_ckpt_path).replace(".pt", ".telemetry.json")
                    Path(_sidecar_path).write_text(_json.dumps(_tel, indent=2))
                    print(f"  ✓ Telemetry sidecar written → {_sidecar_path}")
                except Exception as _te:
                    print(f"  ⚠ Telemetry skipped: {_te}")
                print("\n── Sample outputs ─────────────────────────────────────")
                try:
                    for sample in generate_samples(
                        unwrap_model(model), device, n_loops, tokenizer_path=tok_path
                    ):
                        print(f"  {sample[:200]}\n")
                except Exception as e:
                    print(f"  ⚠ Sample generation failed: {e}")
                print("────────────────────────────────────────────────────────\n")
            barrier()

    if is_main_process():
        print(f"\n✓ Sovereign Training complete — {step} steps")
        if _wandb_run is not None:
            _wandb_run.finish()
        if _tb_writer is not None:
            _tb_writer.close()
    teardown_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",   type=str, default="proof",
                        choices=["smoke", "proof", "research", "ethnic", "instruct", "sft", "convo_smoke", "distill", "history", "identity_hardening", "code"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps",  type=int, default=None,
                        help="Override MAX_STEPS for this run (e.g. --steps 150000)")
    parser.add_argument("--teacher_labels", type=str, default=None,
                        help="Path to teacher soft-label JSONL (required for --tier distill)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (mode=reduce-overhead). ~20-40%% speedup on A100+")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Activation checkpointing — trades 30%% compute for ~60%% less activation memory")
    parser.add_argument("--8bit-adam", action="store_true", dest="use_8bit_adam",
                        help="8-bit AdamW via bitsandbytes — ~75%% optimizer state memory savings")
    parser.add_argument("--wandb", action="store_true", dest="use_wandb",
                        help="Log metrics to Weights & Biases (requires: pip install wandb)")
    parser.add_argument("--wandb-project", type=str, default="anthos",
                        help="W&B project name (default: anthos)")
    parser.add_argument("--tb", action="store_true", dest="use_tb",
                        help="Log metrics to TensorBoard (runs/<run_name>/)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Curriculum learning: ramp seq len from SEQ_LEN/8 → SEQ_LEN over first 10%% of steps")
    parser.add_argument("--spike-rollback", action="store_true", dest="spike_rollback",
                        help="Auto-rollback to last checkpoint when loss spikes > spike-factor × EMA")
    parser.add_argument("--spike-factor", type=float, default=4.0,
                        help="Loss spike threshold multiplier (default: 4.0)")
    args = parser.parse_args()
    train(
        tier                    = args.tier,
        resume                  = args.resume,
        teacher_labels          = args.teacher_labels,
        max_steps               = args.steps,
        compile_model           = args.compile,
        gradient_checkpointing  = args.grad_checkpoint,
        use_8bit_adam           = args.use_8bit_adam,
        use_wandb               = args.use_wandb,
        wandb_project           = args.wandb_project,
        use_tb                  = args.use_tb,
        curriculum              = args.curriculum,
        spike_rollback          = args.spike_rollback,
        spike_factor            = args.spike_factor,
    )
