"""Small-model ablations of Anthos's pieces on a laptop (MPS/CPU), one piece switched off or swapped per condition.

    python ablate.py --out results/ablation_small.jsonl            # the whole matrix, resumable
    python ablate.py --only baseline,dense_ffn --seeds 0 --steps 20  # a quick smoke test

Every condition trains the same tiny model (about 3.4M parameters, 8K-word-piece vocabulary from make_small_vocab_data.py)
on the same text for the same number of steps, then scores held-out cross-entropy. Order is seed-major, so a partial run
is still usable, and a (condition, seed) already in the output file is skipped. This is a proxy: it says whether a piece
does anything at this tiny size, not what it does at 126M.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = ("dim=192,n_heads=6,n_kv_heads=2,max_seq_len=256,max_loop_iters=6,prelude_layers=1,coda_layers=1,"
        "n_thought_tokens=8,n_experts=8,n_shared_experts=1,n_experts_per_tok=2,expert_dim=96,lora_rank=4,phase1_steps=0")
CONDITIONS = {
    "baseline":        "",
    "dense_ffn":       ",ffn_type=dense",
    "act_last":        ",act_mode=last",
    "act_remainder":   ",act_mode=remainder",
    "no_memory":       ",use_memory_bank=False",
    "no_lora":         ",use_loop_lora=False",
    "plain_injection": ",injection=plain",
    "no_thoughts":     ",n_thought_tokens=0",
    # the regular model at this size: 7 ordinary layers (3 + 1 + 3), no loops, thoughts or experts, about the same
    # functional parameters as the baseline
    # one block looped, nothing else: no experts, thoughts, memory, per-loop adjustments, halting gate or energy-conserving update
    "stripped":        ",ffn_type=dense,n_thought_tokens=0,use_memory_bank=False,use_loop_lora=False,injection=plain,act_mode=last",
    "stripped_rand":   ",ffn_type=dense,n_thought_tokens=0,use_memory_bank=False,use_loop_lora=False,injection=plain,act_mode=last",
    "plain_regular":   ",ffn_type=dense,dense_ffn_dim=256,max_loop_iters=1,n_thought_tokens=0,prelude_layers=3,coda_layers=3",
}
TRAIN_EXTRA = {"stripped_rand": ["--loop-sample-min", "2"]}      # random loop count (2..max) each step
DATA = "data/pretrain_small8k"


def build_cfg(extra: str):
    from anthos.configs import get_training_config
    cfg, _ = get_training_config("smoke")
    cfg.vocab_size = 8192
    cfg.thought_prefix_only = True
    cfg.moe_fix_weight_order = True
    for kv in (BASE + extra).split(","):
        k, v = kv.split("=")
        if k == "phase1_steps":          # a train.py setting, not a model field
            continue
        cur = getattr(cfg, k)
        setattr(cfg, k, (v == "True") if isinstance(cur, bool) else type(cur)(v))
    return cfg


@torch.no_grad()
def heldout_ce(ckpt: str, extra: str, windows: int = 64, seq: int = 256, prefix: int = 32, loops: int = 0) -> float:
    from anthos.main import Anthos
    cfg = build_cfg(extra)
    m = Anthos(cfg).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    val = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ce = []
    for s in rng.integers(0, len(val) - seq - 1, windows):
        w = torch.tensor(val[s:s + seq + 1].astype(np.int64))[None]
        lg, _ = m(w[:, :-1], n_loops=loops or cfg.max_loop_iters, return_aux=True, prefix_len=torch.tensor([prefix]))
        ce.append(F.cross_entropy(lg[0, prefix - 1:].float(), w[0, 1:][prefix - 1:]).item())
    return float(np.mean(ce))


def run_one(name: str, seed: int, steps: int, lr: float, out: Path):
    run = f"abl-{name}-s{seed}"
    ms = BASE + CONDITIONS[name]
    t0 = time.time()
    cmd = [sys.executable, "train.py", "--tier", "smoke", "--pretrain-data", DATA, "--vocab-size", "8192",
           "--steps", str(steps), "--seed", str(seed), "--run-name", run, "--thought-prefix-only", "--no-rep-penalty",
           "--log-every", "100", "--save-every", str(steps), "--batch-size", "16", "--grad-accum", "1",
           "--lr", str(lr), "--warmup", "100", "--model-set", ms] + TRAIN_EXTRA.get(name, [])
    log = subprocess.run(cmd, capture_output=True, text=True)
    text = log.stdout + log.stderr
    params = re.search(r"Parameters:\s+([\d,]+)", text)
    toks = re.findall(r"([\d,]+) tok/s", text)
    if log.returncode != 0 or not Path(f"checkpoints/{run}/step_{steps:06d}.pt").exists():
        rec = {"cond": name, "seed": seed, "error": text[-400:]}
    else:
        rec = {"cond": name, "seed": seed, "steps": steps, "params": int(params[1].replace(",", "")) if params else None,
               "heldout_ce": round(heldout_ce(f"checkpoints/{run}/step_{steps:06d}.pt", CONDITIONS[name]), 4),
               "ce_by_loops": {n: round(heldout_ce(f"checkpoints/{run}/step_{steps:06d}.pt", CONDITIONS[name], loops=n), 4) for n in (2, 3, 4, 6)},
               "tok_s": int(toks[-1].replace(",", "")) if toks else None, "minutes": round((time.time() - t0) / 60, 1)}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    for p in Path(f"checkpoints/{run}").glob("*.pt"):       # keep the disk clean: only the score matters
        p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ablation_small.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    ap.add_argument("--seeds", default="0,1", help="comma-separated seeds")
    ap.add_argument("--extra-baseline-seed", type=int, default=2, help="third baseline seed to measure noise (-1 = none)")
    ap.add_argument("--jobs", default=None, help="explicit ordered list cond:seed,cond:seed (overrides --only/--seeds)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["cond"], json.loads(l)["seed"]) for l in open(out)} if out.exists() else set()
    names = a.only.split(",") if a.only else list(CONDITIONS)
    jobs = [(n, int(s)) for s in a.seeds.split(",") for n in names]
    if not a.only and a.extra_baseline_seed >= 0:
        jobs.append(("baseline", a.extra_baseline_seed))
    if a.jobs:
        jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in a.jobs.split(",")]
    for name, seed in jobs:
        if (name, seed) in done:
            continue
        run_one(name, seed, a.steps, a.lr, out)


if __name__ == "__main__":
    main()
