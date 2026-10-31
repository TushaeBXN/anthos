# Anthos × Qwen-Scope Integration Guide

## What was added

```
anthos/
├── sae.py          ← Sparse Autoencoder (core module)
├── steering.py     ← Activation collection + inference-time steering
├── features.py     ← Feature discovery, classifiers, interpreters
└── sasft.py        ← Training losses (suppression, repetition, diversity)

sae_train.py        ← Standalone SAE training script
```

---

## Workflow

### Step 1 — Train Anthos normally
```bash
python train.py --tier proof
# → checkpoints/mansa_sovereign/step_010000.pt
```

### Step 2 — Train a SAE on the thought stream
```bash
python sae_train.py \
    --checkpoint checkpoints/mansa_sovereign/step_010000.pt \
    --stream thought \
    --dim 512 \
    --expansion 16 \
    --k 64 \
    --steps 50000 \
    --out checkpoints/sae/thought_stream.pt
```

### Step 3 — Analyze features
```bash
python sae_train.py \
    --checkpoint checkpoints/mansa_sovereign/step_010000.pt \
    --sae_checkpoint checkpoints/sae/thought_stream.pt \
    --analyze
```
Outputs: top-N features ranked by activation, sparsity stats.

### Step 4 — Steer at inference (zero weight updates)
```python
from anthos.main import Anthos
from anthos.sae import SparseAutoencoder, SAEConfig
from anthos.steering import ActivationSteering

model = Anthos(cfg)
# ... load checkpoint ...

sae_ckpt = torch.load("checkpoints/sae/thought_stream.pt")
sae = SparseAutoencoder(SAEConfig(**sae_ckpt["cfg"]))
sae.load_state_dict(sae_ckpt["state_dict"])

# Suppress feature 6159 (e.g., a language-mixing feature)
feature_dir = sae.W_dec[6159]          # [D] unit-norm direction

steerer = ActivationSteering(model, stream="thought", n_thought_tokens=16)
steerer.set_direction(feature_dir, alpha=-8.0)   # negative = suppress
steerer.attach()

out = model.generate(input_ids, max_new_tokens=128, n_loops=12)
steerer.detach()
```

---

## Adding SASFT losses to train.py

Minimal diff — add these 3 imports and 3 lines inside the training loop:

```python
# ── ADD to imports ──────────────────────────────────────────────────────────
from anthos.sasft import RepetitionPenaltyLoss, ThoughtDiversityLoss
from anthos.steering import ActivationCollector

# ── ADD after model init ────────────────────────────────────────────────────
rep_loss_fn = RepetitionPenaltyLoss(ngram_size=4, penalty=0.3)
div_loss_fn = ThoughtDiversityLoss(coeff=0.05)

thought_collector = ActivationCollector(model, stream="thought",
                                        n_thought_tokens=model_cfg.n_thought_tokens)
thought_collector.attach()

# ── REPLACE the existing loss block in the training loop ───────────────────
with torch.amp.autocast(...):
    logits, aux = model(input_ids, n_loops=n_loops, return_aux=True)
    ce      = F.cross_entropy(logits.reshape(-1, model_cfg.vocab_size), labels.reshape(-1))
    rep_pen = rep_loss_fn(logits)                              # repetition penalty
    thought_acts = thought_collector.flat_activations().to(device)
    div_pen = div_loss_fn(thought_acts.view(-1, model_cfg.n_thought_tokens, model_cfg.dim))
    thought_collector.clear()
    loss = (ce + aux + rep_pen + div_pen) / train_cfg.grad_accum
```

**SASFT with a trained SAE** (optional, for SFT tiers):
```python
from anthos.features import monolinguality_score
from anthos.sasft import FeatureSuppressionLoss

# Identify features to suppress (e.g., Chinese mixing in English model)
# suppress_ids = monolinguality_score(sae, lang_acts)["zh"]
# suppress_loss_fn = FeatureSuppressionLoss(sae, suppress_ids, coeff=0.1)

# Then in training loop, add to loss:
# s_loss = suppress_loss_fn(thought_acts)
# loss = (ce + aux + rep_pen + div_pen + s_loss) / train_cfg.grad_accum
```

---

## Feature discovery cheat sheet

```python
from anthos.features import discover_features, FeatureClassifier, FeatureInterpreter

# Find features that activate on toxic text but not clean
toxic_ids, scores = discover_features(sae, clean_acts, toxic_acts, top_k=32)

# Build an OR-rule classifier (no training, no head)
clf = FeatureClassifier(sae, toxic_ids, threshold=0.1)
labels = clf.classify(new_acts)   # [N] bool

# Find the top examples that activate feature 1234
interp = FeatureInterpreter(sae, feature_id=1234, top_n=10)
top = interp.find(acts, texts=decoded_texts)
for score, text in top:
    print(f"{score:.3f}  {text[:80]}")
```

---

## Why the thought stream is a better SAE target

Standard transformer SAEs target residual stream activations — a mix of
content, position, and residual noise. Anthos thought tokens are:

- **Non-causal** — they see the full context, so they encode holistic
  reasoning state rather than local token predictions
- **Iterative** — they evolve across loop iterations, so SAE features
  capture *reasoning trajectories*, not just snapshots
- **Separated** — they are architecturally isolated from content,
  so features discovered on the thought stream are pure reasoning features

This makes Anthos thought-stream SAE features more interpretable than
anything you could extract from a standard transformer.

---

## Next steps (after first SAE trained)

1. Run `--analyze` and manually inspect top-20 features
2. Identify any language-mixing features (look for features that activate
   on multilingual prompts)
3. Run inference steering to suppress them — check if output language stabilizes
4. Add RepetitionPenaltyLoss to SFT tier training and measure repetition rate
5. Train sequence-stream SAE and compare feature distributions vs thought stream
