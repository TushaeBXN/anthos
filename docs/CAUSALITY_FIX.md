# Causality fix (proposal, opt-in)

Branch: `fix/thought-token-causality` (from `origin/main` c5fafcc). Existing code paths and
checkpoints are unchanged unless the new config flags are turned on.

## What was wrong

With default settings a later token can change the logits at an earlier position.
Measured on a tiny random model (change only the last token, compare all earlier positions):

| thought_prefix_only | moe_fix_weight_order | worst change at earlier positions |
|---|---|---|
| False | False (today) | ~1e-4 |
| True  | False | ~6e-5 |
| False | True  | ~1e-4 |
| True  | True  | ~1e-7 (float32 rounding) |

Two independent causes. Both have to be fixed:

1. **Thought tokens read the whole sequence.** Thoughts attend to every position, and every
   position attends to the thoughts, so information from later tokens reaches earlier ones.
   `e.mean(dim=1)` (used for the thought and MemoryBank conditioning) averages over all positions too.
2. **MoE routing weights are applied to the wrong rows.** In `MoEFFN.forward`, the outputs are back
   in token order but are multiplied by `weights_sorted` (expert-sorted order). Each token receives
   another token's router weight. Besides being incorrect maths, it makes every token depend on
   other tokens, including later ones.

## What this branch adds

- `cfg.thought_prefix_only` + `forward(..., prefix_len=...)`: thoughts (and the conditioning means)
  see only the first `prefix_len` positions, i.e. the prompt. Response positions are strictly causal.
- `cfg.moe_fix_weight_order`: pairs each expert output with its own token's router weight.
- Guard: `thought_prefix_only` refuses `use_expert_choice` / `use_mod` (both select tokens across the
  whole sequence and cannot be causal).
- `tests/test_thought_causality.py`: 14 tests (leak exists in default mode, gone in the fixed mode,
  per-sample prefix lengths, prompt still reaches the response, guards).

Both flags default to False; no new parameters, so existing checkpoints load as before.

## Cached decoding in prefix mode

Because thoughts depend only on the prompt, they are computed once at prefill and reused:

- Prefill saves each loop's thoughts in `kv_cache["_frozen_thoughts"]` (a list of tensors).
- A one-token decode step reads them back instead of recomputing, and skips the thought and
  MemoryBank updates. Every loop runs in cache mode (no early ACT exit) so each loop's cached K/V
  covers every position.
- `generate()` passes `prefix_len = prompt length` automatically in this mode.
- Test: prefill + step-by-step decode matches one full forward pass (max diff ~2e-7 on a tiny model).

Decode needs the same `n_loops` as prefill (otherwise `ValueError`), and one token per step
(`NotImplementedError` for multi-token steps).

## A separate bug found on the way: cached decode in prelude / coda

`F.scaled_dot_product_attention(..., is_causal=True)` aligns its mask at the top-left. During cached
decode the prelude and coda layers call it with a 1-token query over S cached keys, so the token only
sees key 0 instead of the whole history. Cached decode vs a full forward differed by 0.36 on a tiny
model because of this (1.8e-7 once fixed).

This affects `generate()` for every config on current `main`, not only prefix mode. To keep default
behaviour bit-identical (verified: forward and generate outputs are identical to `origin/main`), the
corrected alignment is only used when `thought_prefix_only=True`. Turning it on for everyone is a
one-line change (`self.attn.fix_cached_causal = True` in `TransformerBlock`) but changes generation
for existing checkpoints, so it is left as your call.

## Not done yet

- **Training loop wiring.** `train.py` / the chat dataloader must pass `prefix_len` (prompt length per
  example); pretraining-style tiers can pass 0.
- **SpecDecoder** (`anthos/spec_decode.py`) calls the model without `prefix_len` and verifies with
  uncached full passes, so it does not work in prefix mode yet. (It also resamples rejected tokens
  from `v_probs - softmax(zeros)` rather than `v_probs - draft_probs`; worth a look.)
- **Existing checkpoints were trained with the leaks and the old attention alignment.** Turning the
  flags on changes their outputs; they need a fine-tune (or retrain) in the new mode before comparing
  quality.

## Why it matters

Training could use the leak (predict the next token by reading it through the thoughts or other
tokens' router weights); generation cannot, so training loss may overstate real quality. This is also
a prerequisite for speculative decoding and prefill caching, which assume earlier positions never
depend on later ones.
