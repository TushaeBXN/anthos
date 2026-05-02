# Anthos × Engram Memory Integration

## What was added

```
anthos/
├── memory.py           ← MemoryBank (neural) + ExternalMemoryReader (Engram)
└── memory_compress.py  ← ESCompressor + training data augmentation
```

---

## The Three Layers

### Layer 1 — MemoryBank (inside the model, architectural)

Thought tokens gain cross-attention over a persistent KV memory store.
The store survives across loop iterations within a forward pass.
Optionally persists across forward passes for stateful multi-turn inference.

**Where to wire it in — `anthos/main.py` RecurrentBlock:**

```python
# ADD to RecurrentBlock __init__:
from anthos.memory import MemoryBank, MemoryBankConfig, MemoryBankState
from typing import Optional

self.memory_bank = MemoryBank(MemoryBankConfig(
    d_model=cfg.dim,
    n_slots=512,          # 512 persistent memory slots
    n_heads=cfg.n_heads,
))
self._memory_state: Optional[MemoryBankState] = None

# MODIFY RecurrentBlock forward to accept + return memory state:
def forward(self, x, memory_state=None):
    # ... existing code that produces thought_tokens ...
    # After thought tokens are processed through transformer:
    thought_tokens, memory_state = self.memory_bank(thought_tokens, memory_state)
    # ... continue with existing sequence stream processing ...
    return output, memory_state

# MODIFY Anthos.forward in anthos/main.py to thread memory_state through loops:
def forward(self, input_ids, n_loops=8, return_aux=False, memory_state=None):
    # ... existing prelude ...
    for loop_idx in range(n_loops):
        x, memory_state = self.recurrent_block(x, memory_state)
    # ... existing coda + output ...
    # Store for stateful inference:
    self._last_memory_state = memory_state
```

### Layer 2 — ExternalMemoryReader (before forward pass, no retraining)

Retrieves memories from Engram and prepends them to input_ids.
Thought tokens process the prefix non-causally — available to working memory
immediately without occupying causal sequence positions.

```python
from anthos.memory import ExternalMemoryReader
from transformers import AutoTokenizer

tok    = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
reader = ExternalMemoryReader(
    tokenizer=tok,
    engram_wing="anthos",
    max_memory_tokens=170,    # Engram's L0+L1 cold-start budget
)

# At inference:
input_ids = reader.prepend_memories(input_ids, query="transformer architecture")
out = model.generate(input_ids, max_new_tokens=256, n_loops=12)
```

### Layer 3 — ES Compression Training (better memory reading)

Trains Anthos to recognize and read Engram Shorthand notation.

```python
# Add to train.py — wrap the data loader:
from anthos.memory_compress import MemoryAugmentedDataset

loader = get_dataloader(...)
aug_loader = MemoryAugmentedDataset(
    loader,
    compress_fraction=0.15,   # 15% of training examples get a memory prefix
    prefix_confidence=3,
)

# Or preprocess a JSONL file:
from anthos.memory_compress import compress_jsonl
compress_jsonl(
    "data/teacher_conversations.jsonl",
    "data/teacher_conversations_es.jsonl",
    text_key="content",
)
```

---

## Full MemoryAugmentedAnthos wrapper

```python
from anthos.main import Anthos
from anthos.memory import MemoryAugmentedAnthos, MemoryBankConfig
from transformers import AutoTokenizer

tok      = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
model    = Anthos(cfg)
# ... load checkpoint ...

bank_cfg = MemoryBankConfig(d_model=cfg.dim, n_slots=512, n_heads=cfg.n_heads)
wrapped  = MemoryAugmentedAnthos(
    model,
    bank_cfg,
    tokenizer=tok,
    engram_wing="anthos",
    stateful=True,         # persist memory across turns
)

# Turn 1
out1 = wrapped.generate(
    input_ids,
    query="anthos architecture decisions",
    max_new_tokens=256,
    n_loops=12,
)

# Turn 2 — memory bank state preserved from turn 1
out2 = wrapped.generate(
    next_input_ids,
    query="thought token training",
    max_new_tokens=256,
    n_loops=12,
)

# Reset between independent sessions
wrapped.reset_memory()
```

---

## What Engram provides vs what Anthos provides

| | Engram | Anthos (before) | Anthos (after) |
|---|---|---|---|
| Cross-session memory | ✓ SQLite + vector DB | ✗ | ✓ via ExternalMemoryReader |
| In-context working memory | ✗ | ✓ thought tokens + LTI | ✓ + MemoryBank |
| Extended KV capacity | ✗ | ✗ | ✓ 512 learnable slots |
| Compressed memory reading | ✓ ES notation | ✗ | ✓ after training augmentation |
| Semantic retrieval | ✓ ChromaDB/FAISS | ✗ | ✓ via ExternalMemoryReader |
| Recency weighting | ✓ decay factor | ✗ | ✓ via Engram searcher |

---

## Why thought tokens are the right integration point for MemoryBank

Standard transformers can't easily add external memory because there's no
clean architectural hook — you'd have to modify attention masks globally.

Anthos already has the hook: thought tokens are non-causal, attend to
the full sequence, and are explicitly designed as working-memory primitives.
Cross-attention over a MemoryBank is a natural extension of what they
already do — they just get a larger, persistent memory to attend to.

The MemoryBank's retention parameter (default 0.95) mirrors Anthos's own
LTI philosophy: slow decay, not hard replacement. Information persists
until explicitly overwritten by high-gate writes.

---

## Recommended integration order

1. **Start with Layer 2** (ExternalMemoryReader) — zero code changes to
   anthos/main.py, works immediately after `pip install engram`

2. **Add Layer 3** (MemoryAugmentedDataset) to your next training run —
   just wrap the data loader, no architecture changes

3. **Add Layer 1** (MemoryBank) when you do the next full training run —
   requires modifying RecurrentBlock.forward() and threading memory_state
   through the loop in Anthos.forward()
