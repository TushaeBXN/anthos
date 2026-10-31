"""Pretrain the regular model on the pre-tokenised shards (same data and checkpoint conventions as train.py).

    python train_regular.py --preset 138m --data data/pretrain_cl100k --steps 30000 --run-name regular-138m
    python train_regular.py --preset tiny --vocab-size 8192 --data data/pretrain_small8k --steps 200   # smoke test

Checkpoints go to checkpoints/<run>/step_NNNNNN.pt and carry {"model", "opt", "step", "cfg"}; a rerun with the same
--run-name resumes from the latest one.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from regular import PRESETS, Regular


def lr_at(step: int, a) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


@torch.no_grad()
def heldout_ce(model, data: str, seq_len: int, device, windows: int = 32) -> float:
    meta = json.loads((Path(data) / "meta.json").read_text()) if (Path(data) / "meta.json").exists() else {}
    val = np.memmap(Path(data) / "val.bin", dtype=np.dtype(meta.get("dtype", "uint16")), mode="r")
    rng, ce = np.random.default_rng(0), []
    model.eval()
    for s in rng.integers(0, len(val) - seq_len - 1, windows):
        w = torch.from_numpy(val[s:s + seq_len + 1].astype(np.int64))[None].to(device)
        ce.append(F.cross_entropy(model(w[:, :-1])[0].float(), w[0, 1:]).item())
    model.train()
    return float(np.mean(ce))


def latest(run_dir: Path):
    ck = sorted(run_dir.glob("step_*.pt"))
    return ck[-1] if ck else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="138m")
    ap.add_argument("--vocab-size", type=int, default=None, help="override the preset's vocabulary size")
    ap.add_argument("--data", default="data/pretrain_cl100k")
    ap.add_argument("--run-name", default="regular")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    torch.manual_seed(a.seed)
    cfg = PRESETS[a.preset]
    if a.vocab_size:
        cfg = dataclasses.replace(cfg, vocab_size=a.vocab_size)
    cfg = dataclasses.replace(cfg, max_seq_len=max(cfg.max_seq_len, a.seq_len)) if a.seq_len > cfg.max_seq_len else cfg
    model = Regular(cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    run_dir = Path("checkpoints") / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    ck = latest(run_dir)
    if ck:
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); step = state["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    print(f"params {model.num_params():,} (formula {Regular.expected_params(cfg):,}) on {device}; "
          f"{a.batch_size * a.grad_accum * a.seq_len:,} tokens/step", flush=True)

    from anthos.data import get_bin_dataloader
    loader = iter(get_bin_dataloader(a.data, a.seq_len, a.batch_size, num_workers=1, seed=a.seed + step))
    use_amp = device.type == "cuda"
    log = open(run_dir / "train.log", "a")
    t0, toks = time.time(), 0
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        total = 0.0
        for _ in range(a.grad_accum):
            batch = next(loader).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch[:, 1:].flatten()) / a.grad_accum
            loss.backward()
            total += loss.item()
            toks += batch[:, 1:].numel()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if not math.isfinite(total):
            raise SystemExit(f"loss is {total} at step {step}: stopping (kill criterion)")
        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(total, 4), "gnorm": round(gnorm, 3), "lr": round(opt.param_groups[0]["lr"], 8),
                   "tok_s": int(toks / (time.time() - t0))}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "heldout_ce": round(heldout_ce(model, a.data, a.seq_len, device), 4)}
            print("EVAL", json.dumps(rec), flush=True); log.write("EVAL " + json.dumps(rec) + "\n"); log.flush()
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": dataclasses.asdict(cfg)},
                       run_dir / f"step_{step:06d}.pt")
    print("RUN_EXIT complete", flush=True); log.write("RUN_EXIT complete\n"); log.close()


if __name__ == "__main__":
    main()
