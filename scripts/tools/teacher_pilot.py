"""Pilot for teacher-written textbook data: rewrite real FineWeb-Edu passages with a local llama.cpp server.

    python teacher_pilot.py --server http://127.0.0.1:8080 --n 100 --out data/teacher_pilot.jsonl

Why rewriting and not writing from memory: a 9B model invents facts when it writes alone. Given a real passage and told
to stay inside it, most of what it says can be traced to the source, and this script measures how much cannot
(`ungrounded`: share of the output's numbers and mid-sentence capitalised names that are absent from the source).
Nothing here is trusted blindly; read the printed samples before scaling anything up.
"""
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

EOS = 100257
PROMPT = (
    "Rewrite the passage below as a short, clear textbook section for a student. Explain it in plain, correct English "
    "with short paragraphs. Use only facts that are in the passage; do not add names, dates or numbers that it does not "
    "contain. Do not mention the passage, and do not add a title or a closing remark.\n\nPASSAGE:\n{text}\n\nTEXTBOOK SECTION:\n"
)
REFUSAL = re.compile(r"\b(I can('|no)t|I'm sorry|as an AI|I am unable|cannot (help|assist))\b", re.I)


def sample_passages(data_dir: str, n: int, seed: int = 0, min_tokens: int = 250, max_tokens: int = 700):
    """Whole documents (EOS-delimited) from the pre-tokenised shards, decoded back to text."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    shards = sorted(Path(data_dir).glob("train_*.bin"))
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        arr = np.memmap(rng.choice(shards), dtype=np.uint32, mode="r")
        start = rng.randrange(0, len(arr) - 20_000)
        window = np.asarray(arr[start:start + 20_000])
        eos = np.flatnonzero(window == EOS)
        if len(eos) < 2:
            continue
        i = rng.randrange(len(eos) - 1)
        doc = window[eos[i] + 1:eos[i + 1]]
        if len(doc) < min_tokens:
            continue
        out.append(enc.decode([int(t) for t in doc[:max_tokens] if t < EOS]))
    return out


def ungrounded(source: str, output: str) -> float:
    """Share of the output's numbers and mid-sentence capitalised words that never appear in the source."""
    src = source.lower()
    terms = set(re.findall(r"\b\d[\d,.]*\b", output))
    terms |= {m for m in re.findall(r"(?<=[a-z,;] )([A-Z][a-z]{2,})\b", output)}
    if not terms:
        return 0.0
    return sum(t.lower() not in src for t in terms) / len(terms)


def repetition(text: str, n: int = 4) -> float:
    w = text.split()
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(g)) / max(1, len(g))


def judge(source: str, output: str) -> dict:
    words = len(output.split())
    flags = []
    if REFUSAL.search(output): flags.append("refusal")
    if words < 80: flags.append("too_short")
    if repetition(output) > 0.2: flags.append("repetitive")
    ug = ungrounded(source, output)
    if ug > 0.35: flags.append("ungrounded")
    return {"words": words, "ungrounded": round(ug, 3), "repetition": round(repetition(output), 3), "flags": flags}


def ask(server: str, text: str, max_tokens: int, temperature: float):
    import urllib.request
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT.format(text=text)}], "max_tokens": max_tokens,
                       "temperature": temperature, "top_p": 0.9, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip(), d.get("usage", {}).get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--data", default="data/pretrain_full")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default="data/teacher_pilot.jsonl")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    passages = sample_passages(a.data, a.n)
    print(f"sampled {len(passages)} passages")
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        results = list(ex.map(lambda p: ask(a.server, p, a.max_tokens, a.temperature), passages))
    dt = time.time() - t0
    toks = sum(c for _, c in results)
    rows = []
    for src, (out, c) in zip(passages, results):
        rows.append({"source": src, "output": out, "completion_tokens": c, **judge(src, out)})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    flagged = sum(bool(r["flags"]) for r in rows)
    print(f"{toks:,} output tokens in {dt:.0f}s = {toks / dt:,.0f} tok/s aggregate (concurrency {a.concurrency})")
    print(f"flagged {flagged}/{len(rows)}; mean ungrounded {np.mean([r['ungrounded'] for r in rows]):.3f}; "
          f"mean words {np.mean([r['words'] for r in rows]):.0f}")
    print(f"=> $ per 1B generated tokens at $0.696/hr: {1e9 / max(toks / dt, 1) / 3600 * 0.696:,.0f}")
    for r in rows[:a.show]:
        print("\n--- flags:", r["flags"], "| ungrounded", r["ungrounded"], "\nSOURCE :", r["source"][:300].replace("\n", " "),
              "\nOUTPUT :", r["output"][:600].replace("\n", " "))


if __name__ == "__main__":
    main()
