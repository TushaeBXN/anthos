"""Do loops help when a task needs several steps in a row? A pointer-chasing probe, small enough for a laptop CPU.

Each example is a random map over 8 nodes, written as 8 back-to-back pairs `a b` in shuffled order, then a question
`ASK s Kk ARROW`. The answer is the node reached from s after k hops (k = 1..4). Each hop needs the previous hop's answer,
so the task needs k steps of work in a row, which a shallow stack cannot do and a deeper one (or more loops) can.
Chance is 1/8. Accuracy is reported per hop count, and for looped models per loop count at test time.

    python chain_probe.py --cond stripped      --steps 1500     # one block looped 6 times, nothing else
    python chain_probe.py --cond stripped_rand --steps 1500     # same, trained with a random loop count (2..6)
    python chain_probe.py --cond regular7                       # 7 ordinary layers, no loops
    python chain_probe.py --cond regular3                       # 3 ordinary layers, no loops
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from ablate import BASE, CONDITIONS, build_cfg

ARROW, SEP, ASK = 8180, 8181, 8182
NODE0, K0, N_NODES, MAX_HOPS = 1000, 3000, 8, 4
COND = dict(CONDITIONS)
COND["stripped_rand"] = CONDITIONS["stripped"]
COND["regular7"] = CONDITIONS["plain_regular"]
COND["regular3"] = ",ffn_type=dense,dense_ffn_dim=256,max_loop_iters=1,n_thought_tokens=0"


def make_batch(B: int, gen: torch.Generator, hops: int = 0, max_hops: int = MAX_HOPS):
    """(tokens (B, 2*8+4), answer node index (B,), hops (B,)). hops=0 draws k uniformly from 1..max_hops."""
    f = torch.randint(0, N_NODES, (B, N_NODES), generator=gen)                     # f[b, a] = next node
    order = torch.stack([torch.randperm(N_NODES, generator=gen) for _ in range(B)])
    src = order
    dst = torch.gather(f, 1, order)
    body = torch.zeros(B, 2 * N_NODES, dtype=torch.long)                           # back-to-back pairs: a b a b ...
    body[:, 0::2], body[:, 1::2] = src + NODE0, dst + NODE0
    start = torch.randint(0, N_NODES, (B,), generator=gen)
    k = torch.full((B,), hops) if hops else torch.randint(1, max_hops + 1, (B,), generator=gen)
    cur = start.clone()
    for step in range(MAX_HOPS):
        nxt = torch.gather(f, 1, cur[:, None])[:, 0]
        cur = torch.where(step < k, nxt, cur)
    q = torch.stack([torch.full((B,), ASK), start + NODE0, k + K0, torch.full((B,), ARROW)], dim=1)
    return torch.cat([body, q], dim=1), cur, k


def logits_at_end(model, seq, n_loops):
    out, _ = model(seq, n_loops=n_loops, return_aux=True, prefix_len=seq.shape[1])
    return out[:, -1, NODE0:NODE0 + N_NODES]                        # only the 16 node tokens compete


@torch.no_grad()
def evaluate(model, loop_counts, B=512, seed=123):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    res = {}
    for n in loop_counts:
        row = {}
        for h in range(1, MAX_HOPS + 1):
            seq, ans, _ = make_batch(B, torch.Generator().manual_seed(seed + h), hops=h)
            row[f"k{h}"] = round((logits_at_end(model, seq, n).argmax(-1) == ans).float().mean().item(), 3)
        res[str(n)] = row
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["stripped", "stripped_rand", "regular7", "regular3"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stage-bounds", default="", help="e.g. 12000,18000,24000: 1 hop until the first step, then 2, 3, 4 hops at each later step")
    ap.add_argument("--max-stage", type=int, default=MAX_HOPS, help="1 = train on 1-hop lookups only (is the basic lookup learnable at all?)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/chain_probe.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    from anthos.main import Anthos
    cfg = build_cfg(COND[a.cond])
    model = Anthos(cfg)
    max_loops = cfg.max_loop_iters
    loop_counts = [1, 2, 3, 4, 6] if max_loops > 1 else [1]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed + 1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 50)
        n = max_loops if a.cond != "stripped_rand" else int(torch.randint(2, max_loops + 1, (1,), generator=gen))
        # curriculum: 1 hop first (a plain lookup), then up to 2, 3, 4 hops, so the lookup is learned before it has to be chained
        bounds = [int(b) for b in a.stage_bounds.split(",") if b]
        stage = min(a.max_stage, 1 + (sum(step > b for b in bounds) if bounds else (MAX_HOPS * step) // a.steps))
        seq, ans, _ = make_batch(a.batch, gen, max_hops=stage)
        loss = F.cross_entropy(logits_at_end(model, seq, n), ans)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps:
            ev = evaluate(model, [max_loops])
            print(json.dumps({"step": step, "loss": round(loss.item(), 3), **ev[str(max_loops)]}), flush=True)
    final = {"cond": a.cond, "steps": a.steps, "seed": a.seed, "params": sum(p.numel() for p in model.parameters()),
             "chance": round(1 / N_NODES, 3), "acc": evaluate(model, loop_counts), "minutes": round((time.time() - t0) / 60, 1)}
    with open(a.out, "a") as f:
        f.write(json.dumps(final) + "\n")
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
