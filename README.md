<div align="center">
  ![Anthos Project Logo](https://ibb.co/6JtCgPJ1)

  <h1>Anthos</h1>
  <p><strong>Think in Streams.</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
  [![Architecture](https://img.shields.io/badge/Architecture-Bifurcated_Recurrent_Transformer-5B9BD5?style=for-the-badge)](docs/architecture.md)
  [![Tests](https://img.shields.io/badge/Tests-18%2F18_passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/test_anthos.py)

</div>

---

## Project Status

<div align="center">
  <img src="assets/smoke-test-results.png" alt="Anthos Smoke Test Results" width="700"/>
  <p><em>4 training runs across CPU and H100 GPU. 44.9M parameter proof model reached loss 2.90 in under 2 minutes on H100. SFT pipeline with Anthos chat tokens built and ready.</em></p>
</div>

### Training History

| Run | Hardware | Dataset | Steps | Params | Starting Loss | Final Loss | Date |
|---|---|---|---|---|---|---|---|
| **smoke** | MacBook CPU | roneneldan/TinyStories | 10,000 | 6.9M | 43.27 | 10.99 | Apr 23, 2026 |
| **ethnic** | MacBook CPU | Global African Storybook | +10,000 | 6.9M | 15.79 | 11.48 | Apr 25, 2026 |
| **proof** | H100 SXM (RunPod) | roneneldan/TinyStories | 1,700 | 44.9M | 10.66 | 2.90 | Apr 25, 2026 |
| **sft** | H100 SXM (RunPod) | SlimOrca (517k conversations) | 1,000 | 44.9M | 3.92 | — | Apr 25, 2026 |

| | |
|---|---|
| **Best checkpoint** | `proof` step 1,700 — loss 2.90 (44.9M params) |
| **Tests** | 18 / 18 passed ✅ |
| **Steering** | Sovereign Rogue persona (activation addition) ✅ |
| **SFT pipeline** | SlimOrca + Anthos chat tokens ready ✅ |
| **Next** | SFT re-run with lr=3e-5, stop at step 500 → working assistant |

---

## Roadmap — What Needs to Happen Next

### Stage 2 — `proof` tier (GPU required, ~$5)

**What changes:**
- Model capacity: `dim` 128 → 512 (16× more capacity per token)
- Hardware: A100 or 4090 — [RunPod](https://runpod.io) or [Lambda Labs](https://lambdalabs.com)
- Fresh run — architecture is larger, cannot resume from ethnic checkpoint

**Nothing to code — already built:**
```bash
python3 train.py --tier proof
```

Expected outcome: loss ~2.5–3.0, coherent readable stories, ~3 hrs on a single GPU.

---

### Stage 3 — `instruct` tier (GPU required, ~$2)

**What changes:**
- Dataset switches from stories → [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 52k instruction pairs
- Loss is masked on prompt tokens so the model learns to *respond*, not memorise prompts
- Must resume from a `proof` checkpoint — not smoke or ethnic

**Already built:**
```bash
python3 train.py --tier instruct --resume checkpoints/anthos-proof/final.pt
```

Then talk to it:
```bash
python3 examples/chat.py --checkpoint checkpoints/anthos-instruct/final.pt
```

Expected outcome: Anthos follows instructions and answers questions in its own voice.

---

### What's ready right now (no GPU needed)
- ✅ Run the test suite: `python3 -m pytest tests/test_anthos.py -v`
- ✅ Generate with steering: `python3 examples/minimal.py`
- ✅ Add custom persona pairs to `data/persona_pairs.json` + re-run `python3 generate_vector.py`
- ✅ Chat against the ethnic checkpoint: `python3 examples/chat.py --checkpoint checkpoints/anthos-ethnic/final.pt --tier ethnic`

---

## What is Anthos?

Anthos is a **Thought-Token Bifurcated Recurrent Transformer** — a new architecture class that separates *reasoning state* from *content state* into two parallel streams running through a shared recurrent core.

Most language models collapse these two concerns into a single hidden state. Anthos keeps them apart by design.

```
Input tokens
  ↓
[Embedding]
  ↓
[Prelude]          — standard transformer blocks
  ↓
[Recurrent Block × T loops]
  ├─ [thought₁…thoughtₙ | tok₁…tokₜ]  ← processed together every loop
  ├─ Thought stream: non-causal, sees full context, explicit working memory
  └─ Sequence stream: causal, carries content, produces output
  ↓
[Coda]             — standard transformer blocks
  ↓
Output logits (sequence only — thought tokens are internal, leave no trace)
```

---

## The Core Innovation: Thought Tokens

A small pool of `n_thought` learnable vectors is prepended to the hidden state inside every recurrent loop iteration. They are **not** input tokens — they carry no vocabulary content. They are explicit working-memory slots.

### Attention mask
```
              [thoughts]   [sequence]
[thoughts]       ████  →    ████        ← thoughts see everything
[seq tok 0]      ████  →    ██▒▒▒▒▒▒    ← seq sees thoughts + causal past
[seq tok 1]      ████  →    ████▒▒▒▒▒
[seq tok T]      ████  →    ████████
```

Thought tokens attend to the **full sequence non-causally**. Sequence tokens attend to **all thoughts + their causal past**. At output, thought tokens are discarded — they leave no token trace, only shape what the sequence stream produces.

### Why this matters

| Property | Standard Transformer | OpenMythos (RDT) | **Anthos** |
|---|---|---|---|
| Reasoning mechanism | Implicit in weights | Recurrent hidden state | **Explicit thought stream** |
| Context access | Causal only | Causal only | **Thoughts: non-causal** |
| Working memory | None | Implicit LTI state | **Dedicated thought tokens** |
| Content/reasoning separation | None | None | **Bifurcated by design** |
| Compute-adaptive | No | ACT halting | **ACT on sequence; thoughts run every loop** |

---

## Architecture Details

### Dual-stream LTI Update

Both streams use an **energy-conserving LTI injection** — a provably stable recurrent update:

```python
# Energy-conserving: h norm cannot grow across any number of loops
combined   = alpha * transformer_out + (1 - alpha) * B * gated_e
h_{t+1}    = A * h_t + (1 - A) * combined

# where:
# A ∈ (0,1) guaranteed by ZOH discretization (spectral radius < 1 by construction)
# alpha = sigmoid(W · cat([h, t_out]))   ← learned residual gate, per-channel
# gated_e = sigmoid(W_gate · e) * e     ← soft input mask
```

The sequence stream and thought stream each have **independent LTI parameters** — they evolve at their own learned rates.

### RoPE for Thought Tokens

Thought tokens all receive **position-0 frequencies** in rotary embeddings. They attend from a fixed neutral reference — thoughts are working-memory slots, not sequential positions, so assigning them sequential positions would be misleading.

### MoE FFN

The recurrent block uses a **fine-grained Mixture-of-Experts** FFN with vectorized dispatch:

- Tokens sorted by assigned expert (coalesced memory access, no GPU sync loops)
- Load-balancing loss tracked automatically via `pop_aux_loss()`
- Shared experts always active for cross-domain common patterns

---

## Quick Start

```bash
pip install torch
git clone https://github.com/TushaeThomas/anthos
cd anthos
```

```python
import torch
from anthos import Anthos, AnthosConfig

# Minimal config
cfg = AnthosConfig(
    vocab_size=32000,
    dim=512,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=1024,
    max_loop_iters=8,
    n_thought_tokens=16,
    attn_type="gqa",
    n_experts=16,
    expert_dim=256,
)

model = Anthos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")

# Forward pass
ids    = torch.randint(0, 32000, (1, 64))
logits = model(ids, n_loops=8)
print(f"Logits: {logits.shape}")   # (1, 64, 32000)

# With auxiliary losses (use during training)
logits, aux = model(ids, n_loops=8, return_aux=True)
loss = cross_entropy_loss + aux

# Generation
out = model.generate(ids, max_new_tokens=128, n_loops=12)
```

---

## Model Variants

| Variant | `dim` | Experts | Thought Tokens | Loop Iters | Context |
|---|---|---|---|---|---|
| `anthos_1b` | 2048 | 64 | 16 | 16 | 4k |
| `anthos_3b` | 3072 | 64 | 24 | 16 | 4k |
| `anthos_10b` | 4096 | 128 | 32 | 24 | 8k |
| `anthos_50b` | 6144 | 256 | 48 | 32 | 8k |
| `anthos_100b` | 8192 | 256 | 64 | 32 | 1M |

```python
from anthos import anthos_1b, anthos_3b, Anthos

cfg   = anthos_1b()
model = Anthos(cfg)
```

---

## Training

```python
from anthos import Anthos, anthos_1b
import torch.nn.functional as F

model = Anthos(anthos_1b())

# Training step
logits, aux_loss = model(input_ids, n_loops=8, return_aux=True)
ce_loss  = F.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    input_ids[:, 1:].reshape(-1)
)
loss = ce_loss + aux_loss   # aux handles MoE load balancing + ACT penalty
loss.backward()
```

### Recommended training phases

| Phase | Loops | ACT | Experts | Goal |
|---|---|---|---|---|
| 1 — Stabilization | 4 (fixed) | Off | 16–32 | Loss decreases, no NaNs |
| 2 — Adaptive compute | 8–12 | On | Full | Halting distribution healthy |
| 3 — Scale | Max | On | Full | Benchmark targets |

---

## Documentation

| Page | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full architecture reference — all components, equations, design decisions |
| [`docs/thought_tokens.md`](docs/thought_tokens.md) | Deep dive on thought token design and the bifurcated attention mask |
| [`docs/training.md`](docs/training.md) | Training guide, optimizer settings, phased curriculum |
| [`examples/minimal.py`](examples/minimal.py) | Load trained checkpoint + Sovereign Rogue side-by-side demo |
| [`examples/variants.py`](examples/variants.py) | All model size variants |
| [`examples/train_small.py`](examples/train_small.py) | Small training loop on TinyStories |
| [`anthos/steering.py`](anthos/steering.py) | Activation steering — `AnthosSteer` hook |
| [`generate_vector.py`](generate_vector.py) | Generate a persona vector from contrastive pairs |
| [`data/persona_pairs.json`](data/persona_pairs.json) | TARS/Hacker contrastive sentence pairs |

---

## Sovereign Rogue — Activation Steering

Anthos supports **non-destructive personality injection** via activation addition. A steering vector is extracted from contrastive sentence pairs and added to the recurrent block's hidden states at inference time — no retraining required.

```python
from anthos.steering import AnthosSteer

steer = AnthosSteer(model, target="recurrent")
steer.load_persona("vectors/tars_rogue.pt")
steer.engage(strength=0.75)

output = model.generate(prompt_ids, max_new_tokens=128, n_loops=16)

steer.disengage()
```

### Default vs Sovereign Rogue (smoke tier, 10k steps)

| Prompt | Default | Sovereign Rogue |
|---|---|---|
| *"The small robot looked at"* | *"the box and the box and the box their room a big slide and the box..."* | *"the big grass and walked away some water the water the big rock in a jar. The water and the grass and the blanket and opened the sky..."* |
| *"In a world where"* | *"a big box that look for a big Timmy to make someone..."* | *"they had so the fun together they all day and they named they would play together..."* |

The Rogue vector shifts generation toward **longer connected scenes, less repetition, and richer environmental detail** — extracted entirely from 5 contrastive sentence pairs, zero extra training.

### How to generate your own vector

```bash
# 1. Edit data/persona_pairs.json with your own pos/neg sentence pairs
# 2. Run the factory script once
python3 generate_vector.py
# → saves vectors/tars_rogue.pt
```

Supported hook targets: `"recurrent"` (recommended) · `"prelude_N"` · `"coda_N"`

---

## What makes Anthos different from existing architectures

**vs. standard transformers** — adds recurrent depth and explicit working memory; reasoning depth scales with inference-time compute, not parameter count.

**vs. RWKV / Mamba** — not a sequence model replacement; Anthos is a full transformer with an additive thought-token stream. Full attention is preserved, recurrence is additive.

**vs. OpenMythos / Universal Transformer** — both loop the same hidden state. Anthos bifurcates into two streams with independent dynamics. The thought stream can accumulate global context; the sequence stream stays causal and content-focused.

**vs. register tokens (Darcet et al. 2023)** — register tokens are a training-time artifact for attention sink mitigation. Anthos thought tokens are an architectural primitive with their own LTI update rule, evolving across loop iterations with independent learned parameters.

---

## Citation

```bibtex
@software{thomas2026anthos,
  author    = {Tushae Thomas},
  title     = {Anthos: Thought-Token Bifurcated Recurrent Transformer},
  year      = {2026},
  url       = {https://github.com/TushaeThomas/anthos},
  note      = {Bifurcated Recurrent Transformer with Thought Tokens, Energy-Conserving LTI, MoE, and ACT halting}
}
```

---

## License

MIT License — Copyright (c) 2026 Tushae Thomas. See [LICENSE](LICENSE) for full text.

---

<div align="center">
  <sub>Built by <a href="https://github.com/TushaeThomas">Tushae Thomas</a> · Think in Streams.</sub>
</div>
