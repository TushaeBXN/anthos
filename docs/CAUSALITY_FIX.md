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

## Not done yet (needed before real use)

- **Decode with thoughts frozen after prefill.** `generate()` currently recomputes thoughts every
  step. In prefix mode a decode step raises `NotImplementedError` rather than being silently wrong.
  The fix is to compute thoughts once from the prompt and reuse their per-loop K/V.
- **Training loop wiring.** `train.py` / the chat dataloader must pass `prefix_len` (prompt length per
  example); pretraining-style tiers can pass 0.
- **Existing checkpoints were trained with both behaviours.** Turning the flags on changes their
  outputs; they need a fine-tune (or retrain) in the new mode before comparing quality.

## Why it matters

Training could use the leak (predict the next token by reading it through the thoughts or other
tokens' router weights); generation cannot, so training loss may overstate real quality. This is also
a prerequisite for speculative decoding and prefill caching, which assume earlier positions never
depend on later ones.
