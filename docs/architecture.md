# Anthos Architecture Reference

## Overview

Anthos is a **Thought-Token Bifurcated Recurrent Transformer** — an architecture that separates reasoning state from content state into two parallel streams sharing a single set of weights in a looped recurrent core.

## Full Data Flow

```
Input IDs  (B, T)
    │
    ▼
[Embedding]  →  x  (B, T, dim)
    │
    ▼
[Prelude — prelude_layers standard TransformerBlocks]
    │   Standard causal mask, no thought tokens
    ▼
    e  ←  x  (encoded input, frozen for injection every loop)
    │
    ├──── ThoughtTokenPool.init_batch(B)  →  thoughts  (B, n_thought, dim)
    │
    ▼
[AnthosRecurrentBlock — loops T times]
    │
    │   Per loop iteration t:
    │   ┌─────────────────────────────────────────────────────┐
    │   │ 1. h = h + loop_embed[t]          (loop-index signal)│
    │   │ 2. h_combined = h_pre_norm(h) + e_norm(e)            │
    │   │ 3. full = cat([thoughts, h_combined], dim=1)         │
    │   │ 4. combined_freqs = [pos-0 × n_thought | pos-0..T-1] │
    │   │ 5. full_out = TransformerBlock(full, freqs, mask)     │
    │   │ 6. full_out += LoRAAdapter(full_out, t)               │
    │   │ 7. thought_out = full_out[:, :n_thought]              │
    │   │    seq_out     = full_out[:, n_thought:]              │
    │   │ 8. thoughts = thought_injection(thoughts, e_t, thought_out) │
    │   │    thoughts = thought_norm(thoughts)                  │
    │   │ 9. h = seq_injection(h, e, seq_out)                   │
    │   │    h = h_norm(h)                                      │
    │   │ 10. p = ACTHalting(h)   →  halting probabilities      │
    │   │     accumulate ACT-weighted h_out                     │
    │   │     break if all positions halted                     │
    │   └─────────────────────────────────────────────────────┘
    │
    ▼   h_out  (B, T, dim) — thoughts discarded
    │
    ▼
[Coda — coda_layers standard TransformerBlocks]
    │   Standard causal mask, sequence only
    ▼
[RMSNorm]  →  [Linear head]  →  logits  (B, T, vocab_size)
```

---

## Anthos Causal Mask

The key structural invariant that defines the thought-sequence bifurcation:

```
Total length = n_thought + seq_len

         col: [thought columns]  [sequence columns]
row:
[thought 0]      0    0    0  |  0    0    0    0    0    0
[thought 1]      0    0    0  |  0    0    0    0    0    0
[thought 2]      0    0    0  |  0    0    0    0    0    0
                 ─────────────────────────────────────────
[seq tok 0]      0    0    0  |  0   -∞   -∞   -∞   -∞   -∞
[seq tok 1]      0    0    0  |  0    0   -∞   -∞   -∞   -∞
[seq tok 2]      0    0    0  |  0    0    0   -∞   -∞   -∞
[seq tok 3]      0    0    0  |  0    0    0    0   -∞   -∞
```

- **Thought rows**: all zeros — thoughts attend to all thoughts AND all sequence positions (non-causal, full context access)
- **Seq-to-thought block**: all zeros — every sequence position can see all thought tokens
- **Seq-to-seq block**: standard upper-triangular causal mask

---

## Energy-Conserving LTI Injection

Both streams use the same provably stable update rule:

```
combined   = alpha ⊙ t_out + (1 - alpha) ⊙ (B ⊙ gated_e)
h_{t+1}    = A ⊙ h_t + (1 - A) ⊙ combined
```

**Parameters:**
- `A` — diagonal state matrix, computed via ZOH discretization: `A = exp(dt * -exp(log_A))`, values always in (0, 1)
- `B` — input scale, learned scalar per channel, initialized to 0.5
- `alpha` — residual gate: `sigmoid(W_res · cat([h, t_out]))`, per-channel blend of transformer output vs. encoded input
- `gated_e` — soft input mask: `sigmoid(W_gate · e) * e`

**Stability guarantee:**
By the triangle inequality and the convex combination structure:
```
‖h_{t+1}‖ ≤ A‖h_t‖ + (1-A)‖combined‖ ≤ max(‖h_t‖, ‖combined‖)
```
Hidden state norm is bounded by the maximum of its initial norm and the input norm, across any number of loops.

**Sequence vs. Thought stream differences:**
- The thought stream's `e` signal is the mean-pooled encoded input: `e.mean(dim=1)` expanded to `(B, n_thought, dim)`. Thoughts are not position-specific, so they get a global summary rather than per-position encoded input.
- Each stream has fully independent `log_A`, `log_dt`, `B`, `input_gate`, `res_gate`, and `trans_norm` parameters.

---

## RoPE for Thought Tokens

Standard RoPE assigns each token a unique position frequency. Thought tokens are not sequential — they're working-memory slots — so positional encoding would be misleading.

Instead, all `n_thought` thought tokens receive **position-0 frequencies**:

```python
thought_freqs = freqs_cis[0:1].expand(n_thought, -1)  # all same
seq_freqs     = freqs_cis[:T]                          # natural 0..T-1
combined      = cat([thought_freqs, seq_freqs], dim=0) # (n_thought+T, rope_dim//2)
```

This means thought tokens always attend from a fixed neutral positional reference, regardless of sequence length or loop iteration.

---

## ACT Halting

Adaptive Computation Time operates on **sequence tokens only**. Thought tokens run every loop — they don't halt, because they're the working memory doing the computation, not the content being computed.

```python
p             = sigmoid(W_halt · h)           # (B, T) per-position halt prob
cumulative_p += p * still_running             # accumulate only for active positions
weight         = where(cumulative + p ≥ θ, remainder, p) * still_running
h_out         += weight * h                   # ACT-weighted output
```

The ACT penalty loss encourages early halting:
```python
L_act = mean(loops_used_per_token)
```

---

## Depth-wise LoRA

Each loop iteration applies a small per-depth adaptation delta:

```python
delta(x, t) = (down(x) * scale[t]) @ B
```

- `down`: shared `dim → rank` projection
- `B`: shared `rank → dim` projection, kaiming_uniform initialized
- `scale[t]`: per-loop learned scale vector, initialized to ones (neutral at init)

This lets the same shared weights implement functionally distinct operations at each loop depth without growing parameter count proportionally.

---

## Configuration Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | int | 32000 | Token vocabulary size |
| `dim` | int | 2048 | Hidden dimension |
| `n_heads` | int | 16 | Query attention heads |
| `n_kv_heads` | int | 4 | KV heads (GQA) |
| `max_seq_len` | int | 4096 | Max sequence length |
| `max_loop_iters` | int | 16 | Recurrent depth |
| `prelude_layers` | int | 2 | Standard blocks before recurrent |
| `coda_layers` | int | 2 | Standard blocks after recurrent |
| `n_thought_tokens` | int | 16 | Working-memory thought slots |
| `attn_type` | str | "mla" | "gqa" or "mla" |
| `n_experts` | int | 64 | Total routed MoE experts |
| `n_shared_experts` | int | 2 | Always-active shared experts |
| `n_experts_per_tok` | int | 4 | Top-K routing per token |
| `expert_dim` | int | 512 | Expert inner dimension |
| `act_threshold` | float | 0.99 | ACT cumulative halt threshold |
| `lora_rank` | int | 16 | Depth-wise LoRA rank |
| `moe_aux_coef` | float | 1e-2 | MoE load-balancing loss weight |
| `act_aux_coef` | float | 1e-3 | ACT penalty loss weight |
