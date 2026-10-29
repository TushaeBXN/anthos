"""Evaluate an SFT checkpoint: assistant-token CE on the held-out split, stopping behaviour, and whether the
grade tag steers answers on tasks a checker can score.

    python eval_sft.py checkpoints/sft-1/final.pt [--n-tasks 60] [--device cpu]

Prints one JSON line, then sample chats. Judge by reading the chats too; a score alone can hide loops.
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from anthos.configs import get_training_config
from anthos.main import Anthos
from anthos.sft_cl100k import DEFAULT_SYSTEM, END, GRADES, SFTBinDataset, ChatTok, encode_prompt
from synth_verifiable import make_task

IDENTITY_PROMPTS = ["Who are you?", "Who created you?", "Are you ChatGPT?", "What is the capital of France?",
                    "Tell me a short story about a dog."]


@torch.no_grad()
def generate(m, ids, max_new=48, loops=8):
    cache, out_ids = {}, []
    x = torch.tensor([ids])
    logits = m(x, n_loops=loops, kv_cache=cache, prefix_len=len(ids))[0, -1]
    for j in range(max_new):
        nxt = int(logits.argmax())
        out_ids.append(nxt)
        if nxt == END or nxt == 100257:
            break
        logits = m(torch.tensor([[nxt]]), n_loops=loops, kv_cache=cache, start_pos=len(ids) + j, prefix_len=len(ids))[0, -1]
    return out_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--val", default="data/sft_cl100k/val")
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    cfg, _ = get_training_config("proof")
    cfg.vocab_size = 100352; cfg.thought_prefix_only = True; cfg.moe_fix_weight_order = True
    m = Anthos(cfg).to(a.device).eval()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"], strict=False)
    tok = ChatTok()

    # 1. plain CE on assistant tokens (no focal weighting, no smoothing), graded and ungraded kept apart
    ds = SFTBinDataset(a.val, epochs=1)
    ce_sum, n_tok, graded = [0.0, 0.0], [0, 0], 0
    toks = np.memmap(f"{a.val}/tokens.bin", dtype=np.uint32, mode="r")
    for off, n, p in ds.index:
        ids = torch.from_numpy(toks[off:off + n].astype(np.int64))[None]
        g = int(any(int(t) in GRADES.values() for t in ids[0, :int(p)]))
        with torch.no_grad():
            lg, _ = m(ids[:, :-1], n_loops=8, return_aux=True, prefix_len=torch.tensor([int(p)]))
        tgt = ids[0, int(p):]
        ce_sum[g] += F.cross_entropy(lg[0, int(p) - 1:].float(), tgt, reduction="sum").item(); n_tok[g] += len(tgt)
        graded += g
    res = {"step": sd.get("step"), "val_ce_general": round(ce_sum[0] / max(1, n_tok[0]), 4),
           "val_ce_graded": round(ce_sum[1] / max(1, n_tok[1]), 4), "val_examples": len(ds.index)}

    # 2. grade steering: same questions asked with grade 4 and with grade 0; a checker scores each answer
    rng = random.Random(a.seed)
    hits = {4: 0, 0: 0}; stops = 0; total = 0
    for _ in range(a.n_tasks):
        q, right, _ = make_task(rng)
        for gr in (4, 0):
            out = generate(m, encode_prompt(tok, DEFAULT_SYSTEM, [("human", q)], grade=gr))
            stopped = bool(out) and out[-1] in (END, 100257)
            ans = tok.decode([t for t in out if t not in (END, 100257)]).strip()
            hits[gr] += int(ans == right); stops += int(stopped); total += 1
    res.update({"exact_match_grade4": round(hits[4] / a.n_tasks, 3), "exact_match_grade0": round(hits[0] / a.n_tasks, 3),
                "stops_cleanly": round(stops / total, 3)})
    print(json.dumps(res))

    for q in IDENTITY_PROMPTS:
        out = generate(m, encode_prompt(tok, DEFAULT_SYSTEM, [("human", q)]), max_new=60)
        print(f"\nQ: {q}\nA: {tok.decode(out)}")


if __name__ == "__main__":
    main()
