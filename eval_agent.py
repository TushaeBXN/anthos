"""Run a checkpoint as an agent in the simulated tool environment and score whether it finishes the job.

    python eval_agent.py checkpoints/sft-1/final.pt [--n 40] [--grade 4] [--device cpu]

The harness is the loop a real deployment needs: generate until <|end|>; if the text was a <|tool_call|>, run the
tool, append <|tool_result|>...<|end|>, continue. Scored by synth_trajectories.check(), never by the model:
  graded_4_rate   share of runs the checker grades 4 (right steps, right answer, truthful status)
  finished        share that ended with <|done|> or <|blocked|> inside the step budget
  stalled         share that repeated an identical call
  lazy_quit       share that ended after fewer tool calls than the task needs
"""
import argparse
import json
import random

import torch

from anthos.configs import get_training_config
from anthos.main import Anthos
from anthos.sft_cl100k import BLOCKED, CALL, DONE, END, EOS, RES, ChatTok, encode_prompt
from synth_trajectories import SYSTEM, Env, check, good, set_split


DEV = "cpu"


def _t(v):
    return torch.tensor(v, device=DEV)


@torch.no_grad()
def run_episode(m, tok, task, grade, max_steps=5, max_new=40, loops=8):
    prompt = encode_prompt(tok, SYSTEM, [("human", task)], grade)
    ids, plen, env, steps = list(prompt), len(prompt), Env(), []
    for _ in range(max_steps):
        cache = {}
        logits = m(_t([ids]), n_loops=loops, kv_cache=cache, prefix_len=plen)[0, -1]
        out, status_tok = [], None
        for j in range(max_new):
            nxt = int(logits.argmax()); out.append(nxt)
            if nxt == END:
                # the status token (<|done|>/<|blocked|>) follows the <|end|> of a final answer
                status_tok = int(m(_t([[END]]), n_loops=loops, kv_cache=cache, start_pos=len(ids) + j, prefix_len=plen)[0, -1].argmax())
                break
            logits = m(_t([[nxt]]), n_loops=loops, kv_cache=cache, start_pos=len(ids) + j, prefix_len=plen)[0, -1]
        ids += out
        if out and out[0] == CALL and out[-1] == END:
            try:
                c = json.loads(tok.enc.decode([t for t in out[1:-1] if t < 100277]))
                res = env.run(c["name"], c.get("args", {}))
            except Exception:
                c, res = {"name": "?", "args": {}}, "error: bad call"
            steps.append({"call": c, "result": res})
            ids += [RES] + tok.text(res) + [END]
            continue
        # a final answer: text, <|end|>, status token
        text = tok.enc.decode([t for t in out if t < 100277])
        status = "done" if status_tok == DONE else "blocked" if status_tok == BLOCKED else None
        return {"steps": steps, "final": text, "status": status, "ended": out[-1:] == [END]}
    return {"steps": steps, "final": "", "status": None, "ended": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--grade", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-set", default=None, dest="model_set", help="model config overrides, comma-separated k=v")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--no-grade-tag", action="store_true", help="the model was trained without grade tags")
    ap.add_argument("--unseen", action="store_true", help="tasks use only items and missing words that training never saw")
    a = ap.parse_args()
    if a.unseen:
        set_split("eval")
    global DEV
    DEV = a.device
    cfg, _ = get_training_config("proof")
    cfg.vocab_size = 100352; cfg.thought_prefix_only = True; cfg.moe_fix_weight_order = True
    for kv in (a.model_set or "").split(","):          # e.g. dim=768,n_heads=12,... (default: the 512-wide proof config)
        if kv:
            k, v = kv.split("="); setattr(cfg, k, type(getattr(cfg, k))(v))
    m = Anthos(cfg).to(a.device).eval()
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    tok, rng = ChatTok(), random.Random(a.seed)
    stats = {"graded_4": 0, "finished": 0, "stalled": 0, "lazy_quit": 0}
    shown = 0
    for _ in range(a.n):
        ref = good(rng)
        r = run_episode(m, tok, ref["task"], None if a.no_grade_tag else a.grade)
        rec = {"task": ref["task"], **r}
        keys = [json.dumps(s["call"], sort_keys=True) for s in r["steps"]]
        stats["stalled"] += int(len(keys) != len(set(keys)))
        stats["lazy_quit"] += int(r["status"] is not None and len(r["steps"]) < len(ref["steps"]))
        stats["finished"] += int(r["status"] is not None)
        stats["graded_4"] += int(r["status"] is not None and check(rec) == 4)
        if shown < 3:
            shown += 1; print("TASK:", ref["task"], "\n  ", json.dumps(r)[:300])
    print(json.dumps({k: round(v / a.n, 3) for k, v in stats.items()} | {"n": a.n, "grade": a.grade, "unseen": a.unseen}))


if __name__ == "__main__":
    main()
