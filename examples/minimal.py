"""
Anthos — Minimal usage example
Think in Streams.
"""

import torch
import torch.nn.functional as F
from anthos import Anthos, AnthosConfig

# ── Build a small config (runs on CPU / single GPU) ──────────────────────────
cfg = AnthosConfig(
    vocab_size      = 32000,
    dim             = 512,
    n_heads         = 8,
    n_kv_heads      = 4,
    max_seq_len     = 1024,
    max_loop_iters  = 8,
    prelude_layers  = 2,
    coda_layers     = 2,
    n_thought_tokens = 16,   # explicit working-memory slots
    attn_type       = "gqa",
    n_experts       = 16,
    n_shared_experts = 1,
    n_experts_per_tok = 2,
    expert_dim      = 256,
    lora_rank       = 8,
)

model = Anthos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"Anthos — Parameters: {total:,}")

# ── Forward pass ─────────────────────────────────────────────────────────────
ids    = torch.randint(0, cfg.vocab_size, (2, 64))
logits = model(ids, n_loops=8)
print(f"Logits shape: {logits.shape}")   # (2, 64, 32000)

# ── Forward with auxiliary losses (use during training) ──────────────────────
logits, aux = model(ids, n_loops=8, return_aux=True)
labels = torch.randint(0, cfg.vocab_size, (2, 64))
ce_loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    labels[:, 1:].reshape(-1),
)
loss = ce_loss + aux
print(f"CE loss: {ce_loss.item():.4f}  Aux: {aux.item():.6f}  Total: {loss.item():.4f}")

# ── Generation ────────────────────────────────────────────────────────────────
prompt = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    # n_loops can be increased at inference for deeper reasoning
    out = model.generate(prompt, max_new_tokens=32, n_loops=12, temperature=0.8, top_k=50)
print(f"Generated shape: {out.shape}")   # (1, 16 + 32)
