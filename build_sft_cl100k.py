"""Build the cl100k SFT set: tokenise, mask, dedupe, hold out, and write tokens.bin + index.npy.

    python build_sft_cl100k.py --out data/sft_cl100k \
        --src data/teacher_conversations.jsonl:1.0 --src data/local_train.jsonl:1.0 \
        --src data/sft_identity_cl100k.jsonl:8 --src data/sft_synth_verifiable.jsonl:1.0

--src path:weight   weight < 1 subsamples, weight > 1 repeats the file (identity is tiny, so it is repeated).
Examples that do not fit --max-len are dropped, not truncated: a cut-off answer would teach the model to stop mid-sentence.
The held-out split is cut before repeating, so no repeated example leaks into it.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import re

import numpy as np

from anthos.sft_cl100k import SPECIALS, ChatTok, DEFAULT_SYSTEM, encode_example, encode_trajectory, write_specials_meta


# answers that claim another creator, or a doctorate the creator does not have, contradict the identity data
DROP_RE = (r"(created|developed|built|trained|made|designed) by (anthropic|openai|google|meta|alibaba|microsoft|deepseek)"
           r"|\bDr\.? ?(Brian|Tushae)|\bPh\.?D")


SYSTEM_TOOLS = "You are Anthos. You can call tools: calc(expr), lookup(key), write_note(text). Finish the task or say why you cannot."


def turns_of(conv):
    sysm = next((c["value"] for c in conv if c["from"] == "system"), DEFAULT_SYSTEM)
    turns = [(c["from"], c["value"]) for c in conv if c["from"] in ("human", "gpt")]
    return sysm, turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-answer-tokens", type=int, default=1)
    ap.add_argument("--drop-source", action="append", default=[], help="drop examples whose 'source' field equals this")
    ap.add_argument("--drop-regex", default=DROP_RE, help="drop examples whose answer matches (wrong-creator claims)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    tok = ChatTok()
    drop_re = re.compile(a.drop_regex, re.I)

    stats, seen, rows = Counter(), set(), []
    for spec in a.src:
        path, _, w = spec.rpartition(":")
        if not path:
            path, w = spec, "1.0"
        w = float(w)
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "steps" in d:                      # tool-use trajectory (synth_trajectories.py format)
                src = d.get("source", Path(path).stem)
                stats[f"{src}:read"] += 1
                key = hashlib.sha1(json.dumps([d["task"], d["steps"], d["final"], d["grade"]]).encode()).hexdigest()
                if key in seen:
                    stats[f"{src}:skip_dup"] += 1; continue
                seen.add(key)
                enc = encode_trajectory(tok, SYSTEM_TOOLS, d["task"], d["steps"], d["final"], d["status"], d.get("grade"), a.max_len)
                if enc is None:
                    stats[f"{src}:skip_too_long_or_bad"] += 1; continue
                ids, p, m = enc
                rows.append((src, w, ids, p, m)); stats[f"{src}:kept"] += 1
                continue
            sysm, turns = turns_of(d.get("conversations", []))
            src = d.get("source", Path(path).stem)
            stats[f"{src}:read"] += 1
            if src in a.drop_source:
                stats[f"{src}:skip_source"] += 1; continue
            if drop_re.search(d["conversations"][-1]["value"] if d.get("conversations") else ""):
                stats[f"{src}:skip_wrong_claim"] += 1; continue
            if len(turns) < 2 or turns[-1][0] != "gpt":
                stats[f"{src}:skip_shape"] += 1; continue
            key = hashlib.sha1(json.dumps([turns[-2][1], turns[-1][1]]).encode()).hexdigest()
            if key in seen:
                stats[f"{src}:skip_dup"] += 1; continue
            seen.add(key)
            grade = d.get("grade")
            enc = encode_example(tok, sysm, turns, grade, a.max_len)
            if enc is None:
                stats[f"{src}:skip_too_long_or_bad"] += 1; continue
            ids, p = enc
            if len(ids) - p < a.min_answer_tokens + 2:
                stats[f"{src}:skip_empty_answer"] += 1; continue
            rows.append((src, w, ids, p, [0] * p + [1] * (len(ids) - p)))
            stats[f"{src}:kept"] += 1

    rng.shuffle(rows)
    val, train = rows[:a.val], rows[a.val:]
    expanded = []
    for src, w, ids, p, m in train:
        reps = int(w) + (1 if rng.random() < w - int(w) else 0)
        expanded += [(ids, p, m)] * reps
    rng.shuffle(expanded)

    def dump(name, items):
        off, idx, chunks = 0, [], []
        mchunks = []
        for ids, p, m in items:
            chunks.append(np.asarray(ids, dtype=np.uint32)); mchunks.append(np.asarray(m, dtype=np.uint8))
            idx.append((off, len(ids), p)); off += len(ids)
        d = out / name; d.mkdir(exist_ok=True)
        np.concatenate(mchunks).tofile(d / "mask.bin"); np.concatenate(chunks).tofile(d / "tokens.bin"); np.save(d / "index.npy", np.asarray(idx, dtype=np.int64))
        write_specials_meta(d)
        return off, len(idx)

    tt, tn = dump("train", expanded)
    vt, vn = dump("val", [(i, p, m) for _, _, i, p, m in val])
    meta = {"tokenizer": "tiktoken:cl100k_base", "specials": SPECIALS, "max_len": a.max_len, "seed": a.seed,
            "train_examples": tn, "train_tokens": tt, "val_examples": vn, "val_tokens": vt, "sources": a.src,
            "stats": dict(stats)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("specials",)}, indent=1))


if __name__ == "__main__":
    main()
