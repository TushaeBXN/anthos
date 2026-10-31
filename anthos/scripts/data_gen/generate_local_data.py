"""
generate_local_data.py — Local-teacher data generator for Anthos (Ollama)

Same job as generate_claude_data.py, but the teacher is a local model served by
Ollama instead of the Claude API. Reuses that file's topic bank, angles, formats,
templates and system prompt (read with ast, so `anthropic` need not be installed).

- No content filtering. Responses are written as the teacher produced them.
- Thinking is switched off at the API (think=false); any stray <think> block is
  stripped because Anthos has its own thought tokens.
- Replies cut off by the token limit are skipped (they would teach the model to
  stop mid-sentence at SEQ_LEN). Use --keep-truncated to write them anyway.
- Append-only and resumable: prompts already in --out are never regenerated, and
  the existing file is never overwritten.

Usage:
    ollama create anthos-teacher -f Anthos-teacher.Modelfile
    python3 generate_local_data.py --n 20 --out data/local_teacher_test.jsonl
    python3 generate_local_data.py --n 5000 --out data/teacher_local.jsonl --resume

Output (ShareGPT, same as every other Anthos data file):
    {"conversations": [{"from": "system", ...}, {"from": "human", ...}, {"from": "gpt", ...}]}
"""

import ast
import json
import random
import re
import time
import argparse
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).parent
OLLAMA_URL = "http://localhost:11434/api/chat"
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def load_bank() -> dict:
    """Pull the prompt bank out of generate_claude_data.py without importing it."""
    wanted = {"TOPICS", "ANGLES", "FORMATS", "INSTRUCTION_TEMPLATES", "SYSTEM_PROMPT"}
    tree = ast.parse((HERE / "generate_claude_data.py").read_text(encoding="utf-8"))
    bank = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                bank[target.id] = ast.literal_eval(node.value)
    missing = wanted - bank.keys()
    if missing:
        raise SystemExit(f"Could not read {sorted(missing)} from generate_claude_data.py")
    return bank


def prompt_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


def build_prompt(bank: dict, seen: set, lock: threading.Lock, tries: int = 40):
    for _ in range(tries):
        topic = random.choice(bank["TOPICS"])
        angle = random.choice(bank["ANGLES"])
        fmt = random.choice(bank["FORMATS"])
        if angle:
            instruction = f"{angle} {topic}."
        else:
            instruction = random.choice(bank["INSTRUCTION_TEMPLATES"]).format(topic=topic)
        if fmt:
            instruction = f"{instruction} {fmt}"
        h = prompt_hash(instruction)
        with lock:
            if h not in seen:
                seen.add(h)
                return instruction
    return None


def ask_teacher(model: str, system: str, prompt: str, num_predict: int, num_ctx: int):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    text = THINK_BLOCK.sub("", out.get("message", {}).get("content", "")).strip()
    return text, out.get("done_reason", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Examples to write this run")
    ap.add_argument("--out", default="data/teacher_local.jsonl")
    ap.add_argument("--model", default="anthos-teacher", help="Ollama model name")
    ap.add_argument("--workers", type=int, default=2, help="Parallel requests (set OLLAMA_NUM_PARALLEL to match)")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--resume", action="store_true", help="Continue an existing --out file")
    ap.add_argument("--keep-truncated", action="store_true")
    args = ap.parse_args()

    bank = load_bank()
    system = bank["SYSTEM_PROMPT"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    existing = 0
    if out_path.exists():
        if not args.resume:
            raise SystemExit(f"{out_path} exists. Pass --resume to append, or choose another --out.")
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    convs = json.loads(line)["conversations"]
                except (json.JSONDecodeError, KeyError):
                    continue
                human = next((c["value"] for c in convs if c["from"] == "human"), None)
                if human:
                    seen.add(prompt_hash(human))
                    existing += 1

    lock = threading.Lock()
    written = skipped_trunc = failed = 0
    t0 = time.time()
    print(f"Teacher: {args.model} | out: {out_path} | existing: {existing} | target: +{args.n}\n")

    def work():
        prompt = build_prompt(bank, seen, lock)
        if prompt is None:
            return None
        text, reason = ask_teacher(args.model, system, prompt, args.max_tokens, args.num_ctx)
        return prompt, text, reason

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = set()
        submitted = 0
        while written < args.n:
            while len(pending) < args.workers and submitted < args.n + 4 * args.workers + skipped_trunc + failed:
                pending.add(pool.submit(work))
                submitted += 1
            done = next(as_completed(pending))
            pending.discard(done)
            try:
                res = done.result()
            except Exception as e:
                failed += 1
                print(f"  request failed: {e}")
                if failed > 20 and written == 0:
                    raise SystemExit("Too many failures — is Ollama running and the model created?")
                continue
            if res is None:
                break
            prompt, text, reason = res
            if not text:
                failed += 1
                continue
            if reason == "length" and not args.keep_truncated:
                skipped_trunc += 1
                continue
            f.write(json.dumps({"conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": text},
            ]}, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written == 1 or written % 25 == 0:
                rate = written / (time.time() - t0)
                print(f"  [{written}/{args.n}] {rate*3600:,.0f}/hr | skipped(truncated) {skipped_trunc} | failed {failed}")
                print(f"    Q: {prompt[:80]}\n    A: {text[:140]}\n")

    print(f"\nDone — wrote {written}, skipped {skipped_trunc} truncated, {failed} failed → {out_path}")


if __name__ == "__main__":
    main()
