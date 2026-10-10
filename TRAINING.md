# Anthos Training Guide

Complete reference for launching, resuming, and scaling Anthos training runs.

---

## Quick Start

```bash
# Single GPU — proof-of-concept run
python train.py --tier proof

# Single GPU — supervised fine-tuning from a checkpoint
python train.py --tier sft --resume checkpoints/proof/step_010000.pt

# 4-GPU DDP — same command, just wrapped with torchrun
torchrun --nproc_per_node=4 train.py --tier proof

# 8-GPU DDP across 2 nodes
torchrun --nproc_per_node=8 --nnodes=2 \
         --node_rank=0 --master_addr=<node0-ip> --master_port=29500 \
         train.py --tier sft --resume checkpoints/proof/step_010000.pt
```

---

## Training Tiers

Each tier sets its own LR, step count, sequence length, and data source.  
Pass `--tier <name>` to select one.

| Tier | Steps | Max LR | Seq Len | Data | Purpose |
|------|-------|--------|---------|------|---------|
| `proof` | 10 000 | 1e-4 | 512 | WikiText-103 | Validate loss curve shape and architecture stability |
| `smoke` | short | 1e-4 | 512 | WikiText-103 | Fast sanity check (< 5 min) |
| `research` | 10 000 | 1e-4 | 512 | WikiText-103 | Extended pretraining |
| `sft` | 500 | 3e-5 | 512 | `data/sft_master.jsonl` | Supervised fine-tuning from chat pairs |
| `instruct` | 500 | 3e-5 | 512 | `data/sft_master.jsonl` | Same as sft, alias |
| `convo_smoke` | 10 000 | 5e-5 | 256 | `data/teacher_conversations.jsonl` | Conversation fine-tuning |
| `identity_hardening` | 10 000 | 1e-4 | 256 | `data/phase2_train.jsonl` | Identity / persona hardening |
| `code` | 10 000 | 2e-5 | 512 | `data/code_combined.jsonl` | Code generation fine-tuning |
| `history` | 5 000 | 3e-5 | 256 | `data/new_history/*.md` | History / long-form markdown |
| `distill` | 10 000 | 2e-4 | 512 | `data/teacher_labels.jsonl` | Knowledge distillation from teacher logits |

Override step count for any tier:

```bash
python train.py --tier proof --steps 50000
```

### Performance flags

```bash
# torch.compile — fuses kernels, ~20-40% throughput gain on A100+
# First step is slow (compile time ~2 min); subsequent steps are faster
python train.py --tier proof --compile

# Activation checkpointing — recomputes activations during backward
# ~60% less activation memory, ~30% more compute — enables larger batches
python train.py --tier proof --grad-checkpoint

# Combine both for maximum throughput at minimum memory
torchrun --nproc_per_node=4 train.py --tier sft \
         --resume checkpoints/proof/step_010000.pt \
         --compile --grad-checkpoint
```

| Flag | Memory impact | Speed impact | When to use |
|------|--------------|-------------|-------------|
| `--compile` | none | +20-40% | Always on A100+, skip on CPU |
| `--grad-checkpoint` | -60% activations | -30% | When OOM on large batches |
| Both | -60% activations | net ~0-10% gain | Large models / long sequences |

---

## Model Scales

Defined in `anthos/main.py`. Select via `get_training_config(tier)` in `anthos/configs.py`.

| Scale | Params | dim | Heads | Experts | Loops | Seq Len |
|-------|--------|-----|-------|---------|-------|---------|
| 1B | ~1B | 2048 | 16 | 64 | 16 | 4096 |
| 3B | ~3B | 3072 | 24 | 64 | 16 | 4096 |
| 10B | ~10B | 4096 | 32 | 128 | 24 | 8192 |
| 35B | ~35B | 5120 | 40 | 256 | 28 | 8192 |
| 50B | ~50B | 6144 | 48 | 256 | 32 | 8192 |
| 100B | ~100B | 8192 | 64 | 256 | 32 | 1M |

---

## Resuming Checkpoints

```bash
# Resume a pretraining run exactly where it left off
# (restores model weights, optimizer state, LR step, RNG state, dataloader position)
python train.py --tier proof --resume checkpoints/proof/step_005000.pt

# Resume weights only — fresh optimizer (standard for SFT after pretraining)
python train.py --tier sft --resume checkpoints/proof/step_010000.pt
```

The `sft`, `instruct`, `convo_smoke`, `history`, and `code` tiers always reset the optimizer  
on resume. All other tiers restore the full optimizer state.

Checkpoints are saved every `SAVE_EVERY` steps (default 1000) into:

```
checkpoints/<run_name>/step_XXXXXX.pt
```

A rolling window of the last 3 checkpoints is kept automatically. Each checkpoint contains:

- `model` — full state dict (compatible with single-GPU resume even after multi-GPU training)
- `optimizer` — AdamW state (two param groups: decay / no-decay)
- `step` — training step
- `loss` — last logged loss
- `data_step` — dataloader position (enables exact data resume, no repeated batches)
- `rng_cpu` / `rng_cuda` — full RNG state for reproducible resume

---

## Multi-GPU Training

### DDP (Distributed Data Parallel)

Best for models that fit in a single GPU's VRAM. Each GPU holds a full copy of the model;  
gradients are all-reduced after each backward pass.

```bash
# 2 GPUs, same machine
torchrun --nproc_per_node=2 train.py --tier proof

# 4 GPUs, same machine
torchrun --nproc_per_node=4 train.py --tier sft \
         --resume checkpoints/proof/step_010000.pt

# 8 GPUs, same machine
torchrun --nproc_per_node=8 train.py --tier proof --steps 100000
```

**What changes automatically in DDP mode:**
- Model wrapped with `DistributedDataParallel` (`find_unused_parameters=True` for MoE/ACT)
- All logging, checkpoint saving, and sample generation runs on rank 0 only
- `tok/s` reported as aggregate across all GPUs
- Effective batch size = `batch_size × world_size × grad_accum`

**Memory rule of thumb for DDP:**

| Model | bfloat16 | Full precision |
|-------|----------|----------------|
| 1B | ~2 GB | ~4 GB |
| 3B | ~6 GB | ~12 GB |
| 10B | ~20 GB | ~40 GB |

### FSDP (Fully Sharded Data Parallel)

For models that don't fit in a single GPU. Parameters, gradients, and optimizer state  
are sharded across all ranks. Enable by calling `wrap_fsdp()` instead of `wrap_ddp()` in  
`anthos/distributed.py` — this requires a one-line change in `train.py`.

```python
# In train.py, find:
model = wrap_ddp(model, local_rank)
# Replace with:
from anthos.distributed import wrap_fsdp
model = wrap_fsdp(model, mixed_precision=True)
```

**Memory rule of thumb for FSDP (FULL_SHARD):**

| Model | 4×A100 80GB | 8×A100 80GB |
|-------|-------------|-------------|
| 10B | fits | fits |
| 35B | tight | fits |
| 100B | — | fits (bf16) |

#### FSDP Sharded Checkpoints

When using FSDP, enable sharded checkpointing to avoid the OOM spike  
of rank-0 gathering the full model:

```python
# In train.py, change the save call to:
save_checkpoint_distributed(
    _ckpt_path, model, optimizer, step, avg_loss,
    data_step=data_step,
    use_sharded=True,   # ← add this
)
```

Sharded checkpoints save into a directory (`step_XXXXXX/`) instead of a single `.pt` file.  
Resume with the same `world_size`. To merge shards into a single file for inference:

```bash
python tools/consolidate_shards.py checkpoints/proof/step_010000/
```

*(Write `tools/consolidate_shards.py` when you need cross-world-size resume.)*

---

## LR Schedules

Two schedules are available in `train.py`.

### Cosine Decay (default)

```
MAX_LR ──╮
         │ linear warmup
         ╰── cosine decay ──→ MIN_LR
```

Already wired in as `get_lr(step)`. Best for runs where you know the total step count  
in advance — the decay starts immediately after warmup.

### WSD (Warmup-Stable-Decay)

```
MAX_LR ─╮          ╭─────────╮
         │ warmup  │ stable  │ cosine decay → MIN_LR
         ╰─────────╯         ╰──────────────────────
```

Better for exploratory runs where you might extend training mid-run —  
just increase `MAX_STEPS`; the stable phase stretches, the decay phase stays fixed.

To switch to WSD, replace `get_lr` with `get_lr_wsd` in the training loop:

```python
# In train.py, find:
lr = get_lr(step)
# Replace with:
lr = get_lr_wsd(step, decay_frac=0.1)  # last 10% of steps decay
```

`decay_frac=0.1` → last 10% of `MAX_STEPS` is the decay phase.  
`decay_frac=0.2` → last 20%. Start at 0.1 and increase if the final eval dip is too sharp.

---

## Reward Training (GRPO)

Anthos supports Group Relative Policy Optimization for RL fine-tuning of the thought stream.  
See `train_rl.py` for a complete example. Key reward factories in `anthos/grpo.py`:

| Factory | What it rewards |
|---------|----------------|
| `quality_reward` | Text diversity minus repetition rate |
| `get_repetition_penalty_reward(ngram_size)` | N-gram repetition penalty |
| `get_cosine_length_reward(...)` | Shorter correct completions score higher |
| `get_soft_overlong_punishment(max_len)` | Three-zone penalty for overly long outputs |
| `get_format_reward(pattern)` | Binary: does output match a regex? |
| `get_tag_count_reward(tags)` | Partial credit per correctly-formed tag |
| `get_accuracy_reward(answer_pattern)` | Binary: correct answer inside `<answer>` tag |
| `get_token_level_reward(think_weight, answer_weight)` | Down-weight reasoning tokens vs answer tokens |

### Key GRPO config knobs

```python
GRPOConfig(
    n_completions  = 8,      # K completions per prompt — more = stabler gradient
    clip_eps       = 0.2,    # PPO lower clip bound (1 - ε)
    clip_high_mu   = 3.0,    # PPO upper clip = 1 + ε × μ (clip-higher / DAPO)
    kl_coef        = 0.05,   # penalty for drifting from reference model
    entropy_bonus  = 0.01,   # prevents mode collapse
)
```

Set `clip_eps=0.0` to disable ratio clipping and use plain REINFORCE.  
Set `clip_high_mu=1.0` to use symmetric standard PPO clipping.

---

## Optimizer

AdamW with two parameter groups (OLMo-core pattern):

| Group | Parameters | Weight Decay |
|-------|-----------|-------------|
| `decay` | All weight matrices (ndim ≥ 2, excluding embeddings) | 0.1 |
| `no_decay` | LayerNorm/RMSNorm scales (ndim = 1), biases, embedding tables | 0.0 |

Default: `β₁=0.9`, `β₂=0.95`, fused AdamW on CUDA.

---

## Auxiliary Losses

Anthos's `aux_loss` combines four terms, all tunable via `AnthosConfig`:

| Term | Config field | Default | Controls |
|------|-------------|---------|---------|
| MoE load-balancing | `moe_aux_coef` | 1e-2 | Expert utilisation spread |
| ACT halting | `act_aux_coef` | 1e-3 | Ponder cost |
| PreRouter KL | `prerouter_loss_coef` | 0.05 | Routing prediction head |
| Multi-Token Prediction | `mtp_loss_coef` | 0.3 | Predict t+2, t+3 heads |
| Z-loss | `zloss_coef` | 1e-4 | Logit scale stability |

`moe_aux_coef` is auto-bumped 2× if the top-1 expert captures > 60% of routing mass.  
Watch `moe_top1` in the log — it should stay below 40% after early training.

---

## Reading the Training Log

```
step   1000 | loss 3.2841 | aux 0.00421 | prerouter_kl 0.0312 | gnorm 0.847 |
             moe_top1 23.4% | loops 4 | lr 1.00e-04 | 42,300 tok/s
```

| Field | Healthy range | Action if unhealthy |
|-------|--------------|---------------------|
| `loss` | Decreasing steadily | Check data pipeline if flat |
| `aux` | < 0.01 | Raise `moe_aux_coef` if rising |
| `prerouter_kl` | < 0.1 | Lower `prerouter_loss_coef` if spiking |
| `gnorm` | 0.3 – 1.5 | Lower LR if consistently > 3.0 |
| `moe_top1` | < 40% | Auto-bumped; check if stuck at one expert |
| `loops` | Increases at step `PHASE1_STEPS` | Expected; 4 → 16 |

A `⚡ Grad norm spike` warning fires when `gnorm > 3× its running EMA`. Occasional spikes  
are normal; repeated spikes at the same step usually indicate a bad data batch.

---

## Data Preparation

| Tier | Required file(s) | How to generate |
|------|-----------------|----------------|
| `sft` | `data/sft_master.jsonl` | Run ingest scripts in `data/` |
| `code` | `data/code_combined.jsonl` | `python generate_code_teacher_data.py --n 5000` then `python generate_code_eval_data.py --n 2000` then `cat data/code_teacher.jsonl data/code_eval.jsonl > data/code_combined.jsonl` |
| `identity_hardening` | `data/phase2_train.jsonl` | `python build_phase2_dataset.py` |
| `distill` | `data/teacher_labels.jsonl` | `python generate_teacher_labels.py --teacher <model> --out data/teacher_labels.jsonl` |
| `history` | `data/new_history/*.md` | Place markdown files in the directory |
| `convo_smoke` | `data/teacher_conversations.jsonl` | `python generate_teacher_data.py` or falls back to FineTome-100k |

---

## Environment

```bash
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets

# For distributed training (already in PyTorch >= 2.0)
# torch.distributed and torch.distributed.checkpoint are built-in

# Optional: faster tokenization
pip install tokenizers

# Verify GPU visibility
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

Recommended: PyTorch ≥ 2.1 for `torch.distributed.checkpoint` sharded saves.
