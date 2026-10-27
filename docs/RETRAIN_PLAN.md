# Causal-mode baseline: legacy vs causal retrain

Goal: a clean, defensible number for the thought-stream architecture, and a measurement of how much
the old leak flattered earlier results. Nothing here has been run yet; this is the plan and the tooling.

## What exists vs what does not

- Tooling on this branch: `--thought-prefix-only`, `--data`, `--seed`, `--run-name`, `--lr`, `--warmup`
  in `train.py`; `eval_leak_gap.py`; `make_baseline_data.py`.
- **Not on the dev Mac:** the earlier Haiku teacher data, any past checkpoints, `data/anthos_tokenizer`
  was only in the old checkout. Past runs therefore cannot be reproduced here, so the comparison below is
  between two *new* runs on free public data. It is a paired experiment, not a re-run of history.

## Design (paired, controlled)

| | Run A: legacy | Run B: causal |
|---|---|---|
| Flags | none | `--thought-prefix-only` |
| Model | `sft` tier, 48.1M params, 16 loops, 16 thoughts | same |
| Data | `data/baseline_train.jsonl` (FineTome, ~21.7K examples) | same |
| Seed / steps / LR | 0 / 3000 / 3e-4 | same |

Held-out: `data/baseline_val.jsonl` (300 examples, disjoint). From scratch, so no checkpoint is needed.

## Commands (single GPU, e.g. RunPod RTX 4090)

```bash
pip install -e .[train] && python3 setup_tokenizer.py      # creates data/anthos_tokenizer
python3 make_baseline_data.py --n 22000 --val 300           # free HF data, fixed seed

python3 train.py --tier sft --data data/baseline_train.jsonl --steps 3000 \
    --lr 3e-4 --warmup 200 --seed 0 --run-name baseline-legacy

python3 train.py --tier sft --data data/baseline_train.jsonl --steps 3000 \
    --lr 3e-4 --warmup 200 --seed 0 --run-name baseline-causal --thought-prefix-only

python3 eval_leak_gap.py --tier sft --data data/baseline_val.jsonl \
    --checkpoint checkpoints/baseline-legacy/step_003000.pt
python3 eval_leak_gap.py --tier sft --data data/baseline_val.jsonl --thought-prefix-only \
    --checkpoint checkpoints/baseline-causal/step_003000.pt
```

Do a 20-step smoke run of each mode first (`--steps 20`); `LOG_EVERY` is 100 for most tiers, so use
`--steps 100` to see a loss line. Cost and time are **not measured**: expect on the order of an hour or
two per run on a 4090, roughly a few dollars each; confirm with the first 100 steps' tokens/s.

## How to read the result

- `stepwise loss` is what generation can actually achieve. Compare A vs B on this number.
- `leak gap` for A is how much its teacher-forced loss was flattering it. For B it should be ~0.
- B stepwise <= A stepwise: the fix costs nothing (or helps). B much worse: the thought stream was
  leaning on leaked information, which is itself the finding.
- Run both at several loop counts (`--n-loops 4 8 16` one at a time) for the loops-vs-quality curve.

## Open decisions

1. **Pretraining has no prompt.** Plain-text tiers pass `prefix_len=0`, so thoughts read nothing there.
   Causal-mode pretraining needs a rule (e.g. treat the first part of each chunk as context). Undecided.
2. **Controls.** A plain transformer and a looped model without thoughts, at matched size, are still
   needed to show the thought stream itself matters. Not built yet.
3. **Getting the code to the GPU box.** The branch is local only. Either push it to your GitHub
   (your call) or copy the folder.
