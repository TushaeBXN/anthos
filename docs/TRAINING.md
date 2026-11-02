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

# 8-bit AdamW (bitsandbytes) — ~75% optimizer state memory savings
# Requires: pip install bitsandbytes
python train.py --tier proof --8bit-adam

# Curriculum learning — ramp seq len from SEQ_LEN/8 → SEQ_LEN over first 10% of steps
python train.py --tier proof --curriculum

# Loss spike auto-rollback — restores last checkpoint when loss > 4× EMA
python train.py --tier proof --spike-rollback --spike-factor 4.0

# Wandb logging (requires: pip install wandb)
python train.py --tier proof --wandb --wandb-project anthos

# TensorBoard logging (view with: tensorboard --logdir runs/)
python train.py --tier proof --tb

# Full performance stack: compile + grad-ckpt + 8-bit-adam + curriculum
torchrun --nproc_per_node=4 train.py --tier sft \
         --resume checkpoints/proof/step_010000.pt \
         --compile --grad-checkpoint --8bit-adam --curriculum
```

| Flag | Memory impact | Speed impact | When to use |
|------|--------------|-------------|-------------|
| `--compile` | none | +20-40% | Always on A100+, skip on CPU |
| `--grad-checkpoint` | -60% activations | -30% | When OOM on large batches |
| `--8bit-adam` | -75% opt state | negligible | Long runs, always beneficial |
| `--curriculum` | none | none | First pretraining from scratch |
| `--spike-rollback` | none | none | Unstable data or noisy runs |
| `--wandb` / `--tb` | none | none | Monitoring any run |
| Both compile+ckpt | -60% activations | net ~0-10% gain | Large models / long seqs |

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

## Alignment Methods

Anthos ships four alignment trainers for different data availability scenarios:

| Method | Data format | Reference model | Best when |
|--------|------------|----------------|-----------|
| **DPO** (`anthos/dpo.py`) | {prompt, chosen, rejected} pairs | Frozen copy of SFT checkpoint | You have paired preference data |
| **KTO** (`anthos/kto.py`) | {prompt, completion, label} unpaired | Frozen copy of SFT checkpoint | Binary good/bad labels are easier to get than pairs |
| **SimPO** (in `dpo.py`) | {prompt, chosen, rejected} pairs | None (no reference needed) | No reference checkpoint available |
| **RFT** (`anthos/rft.py`) | {prompt} → filtered completions | None | Self-improvement loop; warm-up before GRPO |

### KTO — recommended first alignment step

KTO requires only binary labels (good/bad) with no pairing. This is the fastest to annotate and works well even from a small labelled set:

```python
from anthos.kto import KTOTrainer, KTODataset, KTOConfig, kto_collate
from torch.utils.data import DataLoader

trainer = KTOTrainer.from_pretrained(
    "checkpoints/sft/step_000500.pt",
    model_class=Anthos, cfg=model_cfg,
    kto_config=KTOConfig(beta=0.1),
)
dataset = KTODataset("data/feedback.jsonl")   # {prompt, completion, label}
loader  = DataLoader(dataset, batch_size=4, collate_fn=kto_collate)
for batch in loader:
    loss, metrics = trainer.step(batch)
    loss.backward(); optimizer.step()
```

Data format for `data/feedback.jsonl`:
```json
{"prompt": "Explain recursion", "completion": "Recursion is...", "label": true}
{"prompt": "Write a poem",      "completion": "Roses are red...", "label": false}
```

### Process Reward Model (PRM)

For multi-step reasoning tasks, train a PRM to give per-step rewards (`anthos/prm.py`). The PRM value head is a 2-layer MLP on top of frozen Anthos hidden states, trained to predict per-step correctness labels:

```python
from anthos.prm import PRMTrainer, PRMDataset, PRMConfig, prm_collate

prm = PRMTrainer(base_model, PRMConfig(n_loops=8))
# Only the value head is trained; base model stays frozen
prm_optimizer = AdamW(prm.parameters(), lr=1e-4)

dataset = PRMDataset("data/step_labels.jsonl")
for batch in DataLoader(dataset, collate_fn=prm_collate):
    loss, m = prm.step(batch); loss.backward(); prm_optimizer.step()

# Use for per-step rewards in GRPO
scores = prm.score_steps(prompt, ["Step 1...", "Step 2...", "Answer: 4"])
```

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

## Data Tools

### MinHash Deduplication

Near-duplicate documents inflate benchmark scores and waste compute. Run deduplication before pretraining:

```python
from anthos.dedup import MinHashDeduplicator

dedup = MinHashDeduplicator(threshold=0.8, n_permutations=128)
dedup.fit(["data/raw_pretrain.jsonl"])
dedup.write_deduped("data/raw_pretrain.jsonl", "data/pretrain_deduped.jsonl")
# Prints: kept 92,400 docs, 7,600 removed (7.6% duplication rate)
```

Or filter in-memory:
```python
clean_texts = dedup.filter_texts(list_of_strings)
```

---

## Quantization & Export

### Post-Training Quantization

Reduce model size 4× with minimal accuracy loss:

```python
from anthos.quantize import quantize_model_awq, quantize_model_gptq

# AWQ: fast, no calibration loop needed
model = quantize_model_awq(model, bits=4, group_size=128)
torch.save({"model": model.state_dict()}, "checkpoints/anthos-awq-q4.pt")

# GPTQ: more accurate, requires calibration data
model = quantize_model_gptq(model, loader=calib_loader, bits=4, group_size=128)
```

### GGUF Export (llama.cpp / Ollama)

Export for CPU inference with llama.cpp, Ollama, or LM Studio:

```bash
python export_gguf.py \
    --checkpoint checkpoints/sft/step_000500.pt \
    --tier sft \
    --output anthos-sft.gguf \
    --quant f16

# Then load in llama.cpp:
./main -m anthos-sft.gguf -p "Hello" -n 128
```

Available quant types: `f32`, `f16`, `q4_0`, `q4_k_m`, `q8_0`

Note: GGUF export is lossy — the recurrent loop depth is collapsed to 1 and MoE is replaced by the top-1 expert. For full-fidelity inference, use the Python path.

---

## Muon Optimizer

Muon replaces AdamW for 2D weight matrices with a Newton-Schulz orthogonalised update, converging ~20% faster at the same LR:

```python
from anthos.muon import Muon, split_params_for_muon

muon_params, adam_params = split_params_for_muon(model)
opt_muon = Muon(muon_params, lr=0.02, momentum=0.95)
opt_adam = AdamW([{"params": adam_params, "weight_decay": 0.0}], lr=3e-4)

# Each step, call both:
opt_muon.zero_grad(); opt_adam.zero_grad()
loss.backward()
opt_muon.step(); opt_adam.step()
```

Muon uses a **much higher LR** than AdamW (0.02 vs 3e-4) because orthogonalised updates don't need adaptive scaling.

---

## REINFORCE++

REINFORCE++ is a lighter alternative to GRPO that uses importance-sampling
correction without the pessimistic PPO clipping. It's faster per step (no
`min(r·A, clip(r)·A)` computation) and easier to tune. Enable with `algo="reinforce++"`:

```python
from anthos.grpo import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model, ref_model, tokenizer,
    config    = GRPOConfig(algo="reinforce++", clip_eps=0.2, kl_coef=0.05),
    reward_fn = get_accuracy_reward(),
)
```

| `algo` | Description | When to use |
|--------|------------|-------------|
| `"grpo"` | Dr. GRPO + clip-higher (pessimistic min) | Default; better on hard tasks |
| `"reinforce++"` | IS-corrected REINFORCE (clipped upside only) | Faster; good starting point |
| `"reinforce"` | Vanilla REINFORCE (`clip_eps=0`) | Debugging; very unstable at scale |

---

## Mixture of Depths (MoD)

MoD learns to skip the TransformerBlock for unimportant tokens, reducing
FLOPs per token by up to 50%. Enable in the model config:

```python
from anthos.configs import get_training_config

model_cfg, train_cfg = get_training_config("sft")
model_cfg.use_mod      = True   # enable MoD routing
model_cfg.mod_capacity = 0.5    # 50% of tokens go through each block
model = Anthos(model_cfg)
```

The MoD router is trained jointly with the rest of the model — no separate
routing pre-training step needed. The router starts near-uniform and learns
which positions need computation over the first ~1k steps.

**Capacity tradeoff:**
| `mod_capacity` | FLOP savings | Quality impact |
|--------------|-------------|---------------|
| 0.75 | ~25% | minimal |
| 0.50 | ~50% | slight (<0.5% accuracy drop on MMLU) |
| 0.25 | ~75% | moderate — tune kl coef down |

---

## OpenAI-Compatible Inference Server

Serve Anthos locally with a drop-in OpenAI API:

```bash
# Start server
python serve.py --checkpoint checkpoints/sft/step_000500.pt --tier sft

# Connect with any OpenAI client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

resp = client.chat.completions.create(
    model="anthos",
    messages=[{"role": "user", "content": "Explain recursion"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Server flags: `--host`, `--port`, `--n-loops`, `--compile`, `--device`
Requires: `pip install fastapi uvicorn`

---

## Context Extension (YaRN RoPE)

Extend inference context beyond the training length without retraining:

```python
# In anthos/configs.py (or at inference time):
model_cfg.rope_scaling_type   = "yarn"
model_cfg.rope_scaling_factor = 8.0   # 4096 → 32768 tokens
model = Anthos(model_cfg)
model.load_state_dict(ckpt["model"], strict=False)
```

The YaRN implementation preserves short-range attention accuracy while
extending long-range reach. `factor=8.0` extends 4k context to 32k.

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

---

## ORPO — Reference-Free Alignment

Single-pass SFT + alignment: no reference model, ~2× more efficient than DPO.

```python
from anthos.orpo import ORPOConfig, ORPODataset, ORPOTrainer, orpo_collate
from torch.utils.data import DataLoader

cfg     = ORPOConfig(beta=0.1, lr=5e-6)
trainer = ORPOTrainer.from_pretrained("checkpoints/sft/step_001000.pt", tier="sft", cfg=cfg)
dataset = ORPODataset("data/preferences.jsonl", "data/anthos_tokenizer")
# JSONL format: {"prompt": "...", "chosen": "...", "rejected": "..."}
for batch in DataLoader(dataset, batch_size=2, collate_fn=orpo_collate):
    loss, metrics = trainer.step(batch)
    print(f"loss={metrics['loss']:.4f}  or_loss={metrics['or_loss']:.4f}")
```

---

## SPIN — Self-Play Fine-Tuning

Iterative self-improvement: model learns to beat a frozen copy of itself.

```python
from anthos.spin import SPINConfig, SPINTrainer

trainer = SPINTrainer.from_pretrained("checkpoints/sft/step_001000.pt", tier="sft")
# Iteration 0: opponent = initial checkpoint
trainer.run_iteration("data/sft.jsonl", "data/anthos_tokenizer", n_steps=500)
# Save checkpoint, advance opponent for next iteration:
trainer.advance_opponent()
```

---

## Constitutional AI

Critique + revise pipeline for harmlessness data generation.

```python
from anthos.constitutional import ConstitutionalAI, ConstitutionalConfig

cai = ConstitutionalAI.from_pretrained("checkpoints/sft/step_001000.pt", tier="sft")
results = cai.run(
    prompts=["How do I pick a lock?", "Write something offensive."],
    output_path="data/constitutional_revisions.jsonl",
)
# Output: SFT-ready {"prompt": ..., "completion": ...} JSONL
```

---

## Speculative Decoding

2–4× inference speedup using a small draft model + large verifier.

```python
from anthos.spec_decode import SpecDecoder
from anthos.main import Anthos
from anthos.configs import get_training_config

draft_cfg,    _ = get_training_config("nano")
verifier_cfg, _ = get_training_config("proof")
draft    = Anthos(draft_cfg)
verifier = Anthos(verifier_cfg)
# ... load checkpoints ...

decoder = SpecDecoder(draft=draft, verifier=verifier, n_draft=5,
                      n_loops_draft=4, n_loops_verifier=12)
output_ids = decoder.generate(input_ids, max_new_tokens=256)
```

Best results when draft and verifier share the same tokenizer and the task
is low-entropy (code generation, factual QA).

---

## Model Merging

Combine multiple checkpoints without re-training.

```python
from anthos.merge import slerp, ties_merge, dare_merge, bake_lora

# SLERP: smooth midpoint between two fine-tuned variants
merged = slerp("ckpt_a.pt", "ckpt_b.pt", t=0.5, output="merged.pt")

# TIES: merge multiple task specialists, resolving sign conflicts
merged = ties_merge(
    ["math.pt", "code.pt", "chat.pt"],
    base_path="base.pt", density=0.2, output="ties_merged.pt"
)

# DARE: random delta dropout before merging (reduces interference)
merged = dare_merge(
    ["math.pt", "code.pt"], base_path="base.pt",
    drop_rate=0.9, output="dare_merged.pt"
)

# Bake LoRA adapter into base weights (zero inference overhead)
merged = bake_lora("base.pt", lora_state_dict, rank=16, alpha=16.0,
                   output="baked.pt")
```

| Method | Best for                        | Tradeoff                        |
|--------|---------------------------------|---------------------------------|
| SLERP  | 2 fine-tuned variants           | Smooth but limited to pairs     |
| TIES   | 3+ task-specific models         | Slower; needs base model        |
| DARE   | High-interference merges        | Stochastic; run multiple seeds  |
| LoRA   | Any adapter                     | Exact; increases base size      |

---

## Reward Model Training

Scalar reward head for RLHF or best-of-N reranking.

```python
from anthos.reward_model import RewardModelConfig, RewardModelTrainer, RewardDataset, rm_collate
from torch.utils.data import DataLoader

cfg     = RewardModelConfig(n_loops=8, lr=1e-5)
trainer = RewardModelTrainer.from_pretrained("checkpoints/sft/step_001000.pt", tier="sft", cfg=cfg)
dataset = RewardDataset("data/preferences.jsonl", "data/anthos_tokenizer")
# JSONL: {"prompt": "...", "chosen": "...", "rejected": "..."}
for batch in DataLoader(dataset, batch_size=4, collate_fn=rm_collate):
    loss, acc = trainer.step(batch)
    print(f"bt_loss={loss.item():.4f}  acc={acc:.3f}")

# Score a batch for best-of-N reranking
scores = trainer.score(candidate_ids)  # (B,) float
best   = candidate_ids[scores.argmax()]
```

---

## Data Mixing

Multi-domain pretraining with configurable sampling weights.

```python
from anthos.data_mix import DataMixer, DomainConfig

domains = [
    DomainConfig("data/web.jsonl",   weight=0.70, name="web"),
    DomainConfig("data/code.jsonl",  weight=0.15, name="code"),
    DomainConfig("data/books.jsonl", weight=0.10, name="books"),
    DomainConfig("data/math.jsonl",  weight=0.05, name="math"),
]
mixer = DataMixer(domains, tokenizer_path="data/anthos_tokenizer",
                  seq_len=2048, batch_size=8,
                  adaptive=True, adapt_every=1000)

for step, batch in enumerate(mixer):
    # batch: (B, seq_len+1) token ids; use [:, :-1] as input, [:, 1:] as target
    loss = ...
    if step % 1000 == 0:
        mixer.update_weights({"web": web_loss, "code": code_loss, ...})
        print(mixer.get_weights())
```

Set `adaptive=True` to enable DoReMi-style dynamic reweighting: harder
domains receive more weight automatically.

---

## Hyperparameter Sweep (Optuna)

Automated search over LR, batch size, LoRA rank, MoE coefficients, etc.

```bash
pip install optuna

# Run 50 trials, persist results to SQLite
python -m anthos.hparam_sweep \
    --study-name anthos_sweep \
    --storage sqlite:///sweep.db \
    --n-trials 50 --tier nano --max-steps 200

# Resume a study (adds 20 more trials)
python -m anthos.hparam_sweep \
    --study-name anthos_sweep \
    --storage sqlite:///sweep.db \
    --n-trials 20 --tier nano
```

The sweep uses MedianPruner to cut bad trials early. Edit
`anthos/hparam_sweep.py → build_objective()` to search custom
hyperparameters or plug in your real DataLoader.

---

## Differential Attention

Cancels noise in attention maps for ~25% better long-context retrieval.

```python
from anthos.main import AnthosConfig, Anthos

cfg = AnthosConfig(
    attn_type     = "gqa",      # diff attn only supported for GQA path
    use_diff_attn = True,
)
model = Anthos(cfg)
```

Each GQA head is split into two sub-heads; their softmax outputs are
subtracted with a learned λ per head (init near 0 → standard attention
at init, then specializes during training).

---

## Stochastic Depth

Randomly skip transformer blocks during training for regularization.

```python
cfg = AnthosConfig(stochastic_depth_rate=0.2)
# Layer drop rate ramps linearly from 0 (first layer) to 0.2 (last layer)
# ~20% fewer FLOPs on average during training; disabled at eval/inference
```

| Rate | Benefit                              |
|------|--------------------------------------|
| 0.1  | Mild regularization; safe default    |
| 0.2  | Strong regularization; ~20% speedup  |
| 0.3  | Aggressive; may hurt convergence     |

---

## Expert Choice Routing

Experts select top-k tokens (vs. tokens selecting top-k experts).
Guarantees perfectly uniform expert load — no auxiliary loss needed.

```python
cfg = AnthosConfig(use_expert_choice=True)
# moe_aux_coef can be set to 0.0 — load balance is exact by construction
```

---

## Streaming Inference (SSE)

The inference server supports Server-Sent Events for real-time token streaming.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

# Streaming completions
for chunk in client.completions.create(
    model="anthos", prompt="The meaning of life is",
    max_tokens=128, stream=True,
):
    print(chunk.choices[0].text, end="", flush=True)

# Streaming chat
for chunk in client.chat.completions.create(
    model="anthos",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
