"""Can Anthos's in-model memory bank be trained to remember? A recall probe, small enough for a laptop CPU.

Each example is several separate forward passes, so the model's attention window cannot carry the answer:
    1. a fact segment   [FACT k1 VAL v1  FACT k2 VAL v2 ...]      (read as prompt: thoughts see it, the bank stores it)
    2. 0-2 filler segments of unrelated tokens                     (the bank is written to again)
    3. a query segment  [ASK k]  -> the model must output v for that key
Only the memory state passed between the passes links 3 to 1. Three ways to run the query pass:
    carry    the state from the earlier passes (the real thing)
    reset    a fresh empty state (the answer is not available; accuracy should fall to chance)
    shuffle  the state from a different example (answers belong to someone else; should also be near chance)
If carry is high and reset/shuffle are near chance, the bank really stores and reads information. With
--bank off there is no memory to carry, so every mode should sit at chance (about 1/64).

    python memory_recall.py --bank on  --steps 1500 --out results/memory_recall.jsonl
    python memory_recall.py --bank off --steps 1500 --out results/memory_recall.jsonl
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from anthos.memory import MemoryBankState

FACT, VAL, ASK = 8180, 8181, 8182          # marker ids in the 8,192-word small vocabulary
N_KEYS, N_VALS = 64, 64
KEY0, VAL0, FILL0, FILL1 = 1000, 2000, 100, 900


def make_batch(B: int, n_facts: int, n_filler: int, filler_len: int, gen: torch.Generator):
    """Returns (fact (B, 4n), fillers [(B, L)], query (B, 2), target (B,))."""
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:n_facts] for _ in range(B)]) + KEY0       # distinct per row
    vals = torch.randint(0, N_VALS, (B, n_facts), generator=gen) + VAL0
    fact = torch.zeros(B, 4 * n_facts, dtype=torch.long)
    fact[:, 0::4], fact[:, 1::4], fact[:, 2::4], fact[:, 3::4] = FACT, keys, VAL, vals
    fillers = [torch.randint(FILL0, FILL1, (B, filler_len), generator=gen) for _ in range(n_filler)]
    pick = torch.randint(0, n_facts, (B,), generator=gen)
    rows = torch.arange(B)
    query = torch.stack([torch.full((B,), ASK), keys[rows, pick]], dim=1)
    return fact, fillers, query, vals[rows, pick]


def answer_logits(model, batch, mode: str, n_loops: int, device="cpu"):
    """Logits (B, vocab) for the answer token, running the passes in sequence and linking them through the bank."""
    fact, fillers, query, _ = [x.to(device) if torch.is_tensor(x) else [y.to(device) for y in x] for x in batch]
    if mode == "window":          # calibration: the fact and the question share one sequence, so plain attention can solve it
        seq = torch.cat([fact, query], dim=1)
        logits, _ = model(seq, n_loops=n_loops, return_aux=True, prefix_len=seq.shape[1])
        return logits[:, -1, :]
    model(fact, n_loops=n_loops, return_aux=True, prefix_len=fact.shape[1])
    state = model._last_memory_state
    for f in fillers:
        model(f, n_loops=n_loops, return_aux=True, memory_state=state, prefix_len=f.shape[1])
        state = model._last_memory_state
    if mode == "reset" or state is None:
        state = None
    elif mode == "shuffle":
        state = MemoryBankState(state.keys.roll(1, 0), state.values.roll(1, 0))
    logits, _ = model(query, n_loops=n_loops, return_aux=True, memory_state=state, prefix_len=2)
    return logits[:, 1, :]                       # the position after the key predicts the value


@torch.no_grad()
def evaluate(model, n_facts, n_filler, filler_len, n_loops, B=256, seed=123, window=False):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    batch = make_batch(B, n_facts, n_filler, filler_len, gen)
    out = {}
    for mode in (("window",) if window else ("carry", "reset", "shuffle")):
        out[mode] = round((answer_logits(model, batch, mode, n_loops).argmax(-1) == batch[3]).float().mean().item(), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["on", "off"], default="on")
    ap.add_argument("--mem-init", choices=["default", "standard"], default="default",
                    help="standard: memory projections start at 1/sqrt(width) instead of the global 0.02, so the signal survives the chain")
    ap.add_argument("--readout-norm", action="store_true", help="normalise the memory readout before it is added to the thoughts")
    ap.add_argument("--mem-to-seq", action="store_true", help="add the memory's contribution straight into the sequence stream")
    ap.add_argument("--slot-write", action="store_true", help="each thought writes to its own best-matching slot")
    ap.add_argument("--seq-read", action="store_true", help="sequence positions look up the memory slots by key")
    ap.add_argument("--gate-bias", type=float, default=-2.0, help="initial write-gate bias (-2 = writes start at ~12%%)")
    ap.add_argument("--window", action="store_true", help="calibration: fact and question in one sequence (no memory needed)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--max-filler", type=int, default=2)
    ap.add_argument("--filler-len", type=int, default=48)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/memory_recall.jsonl")
    ap.add_argument("--ckpt", default="", help="save here every 250 steps; if the file exists, resume from it")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    from ablate import build_cfg
    from anthos.main import Anthos
    extra = "" if a.bank == "on" else ",use_memory_bank=False"
    if a.readout_norm:
        extra += ",memory_readout_norm=True"
    if a.mem_to_seq:
        extra += ",memory_to_sequence=True"
    if a.slot_write:
        extra += ",memory_slot_write=True"
    if a.seq_read:
        extra += ",memory_seq_read=True"
    extra += f",memory_gate_bias={a.gate_bias}"
    model = Anthos(build_cfg(extra))
    if a.mem_init == "standard" and a.bank == "on":
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "write_key_proj", "write_val_proj", "cold_key_proj", "cold_val_proj"):
            lin = getattr(model.recurrent.memory_bank, name)
            torch.nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed + 1)
    t0, history, start = time.time(), [], 1
    if a.ckpt and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["rng"])
        history, start = ck["history"], ck["step"] + 1
        t0 -= ck["elapsed"]
        print(f"RESUMED from step {ck['step']}", flush=True)
    for step in range(start, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 50)
        n_fill = int(torch.randint(0, a.max_filler + 1, (1,), generator=gen))
        batch = make_batch(a.batch, a.facts, n_fill, a.filler_len, gen)
        loss = F.cross_entropy(answer_logits(model, batch, "window" if a.window else "carry", a.loops), batch[3])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps:
            ev = evaluate(model, a.facts, a.max_filler, a.filler_len, a.loops, window=a.window)
            history.append({"step": step, "loss": round(loss.item(), 3), **ev})
            print(json.dumps(history[-1]), flush=True)
            if a.ckpt:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "rng": torch.get_rng_state(),
                            "history": history, "step": step, "elapsed": time.time() - t0}, a.ckpt + ".tmp")
                os.replace(a.ckpt + ".tmp", a.ckpt)
    final ={"bank": a.bank, "window": a.window, "mem_init": a.mem_init, "readout_norm": a.readout_norm, "mem_to_seq": a.mem_to_seq, "slot_write": a.slot_write, "seq_read": a.seq_read, "gate_bias": a.gate_bias, "facts": a.facts, "max_filler": a.max_filler, "steps": a.steps, "seed": a.seed,
             "chance": round(1 / N_VALS, 3), "final": history[-1], "minutes": round((time.time() - t0) / 60, 1)}
    with open(a.out, "a") as f:
        f.write(json.dumps(final) + "\n")
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
