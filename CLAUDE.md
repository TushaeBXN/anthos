# CLAUDE.md — Anthos Intelligence Company / TushaeBXN/anthos

**Founder:** Brian Tushae Thomas (Tushae Thomas) | **GitHub:** TushaeBXN  
**HuggingFace:** machomenc | **License:** see `docs/CONSTITUTION.md` (Article II is permanent)

---

## What This Repo Is

`TushaeBXN/anthos` is the core architecture — a **Thought-Token Bifurcated Recurrent Transformer** trained from scratch. This is NOT a fine-tune of an existing model. It is Anthos's own weights, own training loop, own architecture.

---

## Architecture: Thought-Token Bifurcated Recurrent Transformer

### Data Flow

```
Input IDs (B, T)
  → [Embedding]
  → [Prelude blocks × prelude_layers]   ← standard transformer, no thought tokens
  → [AnthosRecurrentBlock × max_loop_iters]
      ├─ ThoughtTokenPool.init_batch(B) → thoughts (B, n_thought, dim)
      ├─ Per loop: cat([thoughts, h]) → TransformerBlock (MoE FFN) → split
      ├─ Thought stream: LTI injection ← mean-pooled encoded input (global)
      ├─ Sequence stream: LTI injection ← per-position encoded input
      ├─ MemoryBank read+write (thoughts attend to persistent KV slots)
      ├─ DualLoRAAdapter delta per loop iteration
      └─ ACT halting on sequence positions (thoughts run every loop)
  → [Coda blocks × coda_layers]         ← standard transformer, sequence only
  → [RMSNorm] → [Linear head] → logits (B, T, vocab_size)
```

### Key Files

| File | Role |
|---|---|
| `anthos/main.py` | Full model: `Anthos`, `AnthosConfig`, `AnthosRecurrentBlock`, `MoEFFN`, `LTIInjection`, `ACTHalting`, `ThoughtTokenPool` |
| `anthos/configs.py` | `TrainingConfig`, `get_training_config()`, all training tiers |
| `anthos/memory.py` | `MemoryBank` — persistent KV memory thought tokens attend to |
| `anthos/lora_pairs.py` | `DualLoRAAdapter` — per-loop-depth adaptation |
| `anthos/distill.py` | `DistillationLoss`, `OnlineDistiller` — KD from teacher models |
| `anthos/grpo.py` | GRPO reinforcement training |
| `anthos/eaft.py` | EAFT (efficiency-aware fine-tuning, uses `_last_loops_used`) |
| `anthos/identity_hardening.py` | Identity lock training — burns "created by Tushae Thomas" |
| `anthos/knowledge_graph.py` | Internal KG tooling |
| `anthos/sae.py` | Sparse autoencoder for interpretability |
| `anthos/autonomous_agent.py` | Continuous self-improvement loop (runs experiments) |
| `anthos/scalable_growth.py` | Alternate MoELayer used during scale-up experiments |
| `docs/architecture.md` | Full architecture reference with mask diagrams |
| `docs/CONSTITUTION.md` | Anthos v3.0 constitutional AI document |
| `docs/TRAINING_ROADMAP.md` | 4-phase training plan to ~6.5M examples |
| `Modelfile` | Ollama Modelfile for running Anthos locally |

### Production Variants (anthos/main.py)

| Variant | dim | Experts | Thoughts | Loops | Context |
|---|---|---|---|---|---|
| `anthos_1b` | 2048 | 64 | 16 | 16 | 4k |
| `anthos_3b` | 3072 | 64 | 24 | 16 | 4k |
| `anthos_10b` | 4096 | 128 | 32 | 24 | 8k |
| `anthos_50b` | 6144 | 256 | 48 | 32 | 8k |
| `anthos_100b` | 8192 | 256 | 64 | 32 | 1M |

### Training Tiers (anthos/configs.py)

| Tier | Hardware | Dataset | Steps |
|---|---|---|---|
| `smoke` | MacBook CPU/MPS | TinyStories | 10k |
| `proof` | Single A100/4090 | TinyStories | 20k |
| `research` | 4×A100 | fineweb-edu | 100k |
| `ethnic` | MacBook | local ethnic_stories.txt | 20k |
| `instruct` | Single GPU | tatsu-lab/alpaca | 5k |
| `sft` | Single GPU | Open-Orca/SlimOrca | 3k |
| `convo_smoke` | MacBook | SlimOrca (1k subset) | 10k |
| `history` | MacBook | data/new_history/*.md | 5k |
| `identity_hardening` | MacBook | data/phase2_train.jsonl | 10k |

### Current Checkpoint State (as of training roadmap)

| Checkpoint | Params | Loss | Hardware |
|---|---|---|---|
| smoke | 6.9M | 10.99 | MacBook CPU |
| proof | 44.9M | 2.90 | H100 |
| convo_smoke | 44.9M | ~1.90 | RTX 4090 |
| qwen_lora | 1.5B | ~1.87 | T4 Colab |

**Target:** First real SFT run September 2026 on 1B checkpoint. DeepSeek V4 reasoning traces populate `<|thought|>` blocks.

---

## KNOWN CRITICAL BUGS

### Bug 1 — ACT Remainder Probability Mass Drop
**File:** `anthos/main.py`, `AnthosRecurrentBlock.forward()`, ~line 697-703  
**What happens:** When `cumulative_p + p >= act_threshold`, the weight is set to `remainder` (correct for the halting token). But positions that halted in a *previous* iteration may still receive non-zero weight if `still_running` masking has a subtle off-by-one. Specifically: once `halted` is set to True for a position, `still_running = ~halted` correctly zeros future contributions — but the weight accumulation for the iteration where halting *first* triggers uses `remainder` correctly. The deeper issue is that `h_out` accumulation does not enforce that `sum(weights_over_loops) = 1.0` per position after the loop exits early (via `if halted.all(): break`). Positions that halted before `max_loop_iters` may sum to less than 1.0 if the final `remainder` weight was not fully applied.

**Verify:** After any halting logic change, assert `weight_sum.allclose(ones)` across all positions for test inputs.  
**Rule:** Any ACT change must be followed by probability mass conservation check.

### Bug 2 — MoE Aux Loss Unnormalized Accumulation
**File:** `anthos/main.py`, `AnthosRecurrentBlock.forward()`, ~line 684  
**What happens:** `moe_aux_total = moe_aux_total + moe_aux` accumulates the per-loop MoE load-balancing loss across all `n_loops` iterations without normalization. If the loop runs 16 iterations, `moe_aux_total` is ~16× the per-step loss. In `Anthos.forward()`: `aux_loss = moe_aux_coef * moe_aux_total + act_aux_coef * act_aux`. The `moe_aux_coef = 1e-2` was calibrated assuming a single-step loss, so the effective coefficient is `1e-2 × n_loops` (up to 0.32 for 32 loops), which can dominate total loss and destabilize training.

**Fix:** Divide by actual loop count before returning, or average across iterations.  
**Rule:** When changing loop count or MoE logic, log per-iteration aux loss separately and verify the total before adding to the optimizer step.

---

## Architecture Invariants (Never Break)

- **Thought token RoPE:** All n_thought tokens receive position-0 frequencies. Never assign sequential positions to thought tokens.
- **KV cache bifurcation:** Only sequence-token KVs accumulate in cache. Thought KVs are regenerated fresh each decoding step (see `GQAttention`/`MLAttention` `n_thought_prefix` logic). Breaking this causes OOM growth.
- **LTI energy conservation:** `A = exp(exp(log_dt) * -exp(log_A))` must stay in (0,1). Never replace this with unconstrained gates.
- **Weight tying:** `head.weight = embed.weight`. Never disconnect.
- **MemoryBank:** Thought tokens attend to persistent KV slots (`memory_bank`). State exposed as `_last_memory_state` for stateful inference.
- **Causal mask invariant:** Thought rows → all zeros; Seq-to-thought → all zeros; Seq-to-seq → standard upper-triangular causal.

---

## Project Chimera / Scaling Roadmap

4-phase KimiK3-style distillation plan (see `docs/TRAINING_ROADMAP.md`):
- Phase 1: Identity Lock (~8k examples, MacBook, Colab T4)
- Phase 2: Cybersecurity domain (~160k examples, Fenrir + Trendyol + CyberNative)
- Phase 3: Reasoning (~250k examples, DeepSeek/Claude CoT + NuminaMath + OpenR1)
- Phase 4: Broad knowledge (1M+ examples, UltraChat + Orca + Nemotron)

---

## Environment

- **Local dev/inference:** MacBook Pro (Apple Silicon, macOS, zsh) — Ollama, llama.cpp
- **Training:** RunPod RTX A6000 48GB
- **Production:** GCP — Anthos/GKE, Cloud Run, Cloud SQL, Firestore, Firebase Auth, Stripe
- **GitHub:** TushaeBXN (primary), TushaeThomas
- **HuggingFace:** machomenc

---

## Related Repos

| Repo | What It Is |
|---|---|
| `TushaeBXN/amy-companion` | Fully-local Ex Machina-inspired AI companion (gemma4:e2b via Ollama) |
| `TushaeBXN/anthos-platform` | GCP production platform + FastAPI backend + Anthos Engineer CLI |
| `TushaeBXN/anthos-training-lab` | Full MoE/distillation training pipeline (private) |
| `TushaeBXN/ujamaa-multi-modal` | Multi-modal foundation model (vision+audio+text, private, All Rights Reserved) |
| `TushaeBXN/project-matus` | Local K-12→EE math AI tutor, Llama 3.2 3B LoRA (private) |

---

## Knowledge Graph

Lives at `~/anthos-knowledge-graph/graph.jsonl` as flat JSONL triples:
```json
{"subject": "X", "relation": "Y", "object": "Z", "note": "optional"}
```
