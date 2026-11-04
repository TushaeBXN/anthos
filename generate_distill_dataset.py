"""Generate a distillation SFT dataset using a local Ollama model as teacher.

Pulls instructions from Alpaca/Dolly, sends them to a local Ollama model,
saves (instruction, response) pairs to JSONL for use with train_regular_sft.py.

    # Generate 5000 examples using llama3.1:8b
    python generate_distill_dataset.py --model llama3.1:8b --n 5000

    # Use a different model, save to custom path
    python generate_distill_dataset.py --model mistral --n 2000 --out data/distill_mistral.jsonl

Requirements:
    pip install datasets requests
    ollama pull llama3.1:8b   (or whichever model you use)

The output JSONL can be used with train_regular_sft.py --datasets distill
after adding the file path via --distill-data.
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests
from datasets import load_dataset


OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Answer clearly and directly. Be concise but complete."
)


def call_ollama(model: str, instruction: str, timeout: int = 60) -> str | None:
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser: {instruction}\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def load_instructions(n: int) -> list[str]:
    instructions = []

    print("Loading Alpaca instructions...", flush=True)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    for ex in ds:
        inst = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        if inst:
            instructions.append(f"{inst}\n\nInput: {inp}" if inp else inst)

    print("Loading Dolly instructions...", flush=True)
    ds2 = load_dataset("databricks/databricks-dolly-15k", split="train")
    for ex in ds2:
        inst = ex.get("instruction", "").strip()
        ctx = ex.get("context", "").strip()
        if inst:
            instructions.append(f"{inst}\n\nContext: {ctx}" if ctx else inst)

    random.shuffle(instructions)
    print(f"Total instructions available: {len(instructions):,}", flush=True)
    return instructions[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="llama3.1:8b",
                    help="Ollama model name (must be pulled locally)")
    ap.add_argument("--n",      type=int, default=5000,
                    help="Number of examples to generate")
    ap.add_argument("--out",    default="data/distill_dataset.jsonl",
                    help="Output JSONL file path")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing output file")
    ap.add_argument("--seed",   type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count already-done examples
    done = 0
    if a.resume and out_path.exists():
        with open(out_path) as f:
            done = sum(1 for _ in f)
        print(f"Resuming from {done} existing examples", flush=True)

    instructions = load_instructions(a.n)
    remaining = instructions[done:]

    print(f"Generating {len(remaining)} examples with {a.model}...", flush=True)
    print(f"Output: {out_path}", flush=True)

    success = 0
    fail = 0
    t0 = time.time()

    with open(out_path, "a") as f:
        for i, instruction in enumerate(remaining):
            response = call_ollama(a.model, instruction)
            if not response or len(response) < 10:
                fail += 1
                continue

            record = {"instruction": instruction, "response": response}
            f.write(json.dumps(record) + "\n")
            f.flush()
            success += 1

            if (done + i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = success / elapsed * 60
                print(
                    f"  [{done + i + 1}/{a.n}] success={success} fail={fail} "
                    f"rate={rate:.0f}/min",
                    flush=True,
                )

    print(f"\nDone. {success} examples saved to {out_path}", flush=True)
    print(f"Failed: {fail}", flush=True)


if __name__ == "__main__":
    main()
