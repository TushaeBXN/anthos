#!/usr/bin/env python3
"""
generate_code_teacher_data.py — Code SFT data for Anthos

Two modes:
  1. HuggingFace (default) — pull from real code datasets, free, high quality, multi-language
  2. Ollama / Claude API   — generate via a model, good for custom tasks

HuggingFace sources (verified working):
  nickrosh/Evol-Instruct-Code-80k-v1   — 80k multi-language (Python/JS/Go/Rust/SQL/Java)
  iamtarun/python_code_instructions_18k_alpaca — 18k Python focused

Usage:
    # Best: pull 20k from HuggingFace (free, multi-language, no model needed)
    python3 generate_code_teacher_data.py --n 20000 --hf

    # Ollama fallback for custom tasks
    python3 generate_code_teacher_data.py --n 5000 --model qwen2.5-coder:7b

    # Claude API (highest quality, ~$4 per 5k)
    ANTHROPIC_API_KEY=sk-... python3 generate_code_teacher_data.py --n 5000 --use-claude

Output: data/code_teacher.jsonl (ShareGPT format, ready for --tier code)
"""

import json
import random
import argparse
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Code generation system prompt ─────────────────────────────────────────────

CODE_SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You write clean, correct, well-documented code. "
    "You include all necessary imports, handle edge cases, add brief comments "
    "explaining non-obvious logic, and always provide example usage. "
    "You are direct and technical. No filler. No flattery."
)

# ── Task bank ─────────────────────────────────────────────────────────────────

TASKS = [
    # Python — core
    "Write a Python function to sort a list of dicts by a given key, with a fallback for missing keys.",
    "Write a Python decorator that retries a function up to N times on exception, with exponential backoff.",
    "Implement a Python LRU cache class without using functools.lru_cache.",
    "Write a Python context manager for timing code blocks.",
    "Create a Python generator that yields chunks of size N from an iterable.",
    "Write a Python function to flatten a nested list to arbitrary depth.",
    "Implement merge sort in Python.",
    "Write a Python class for a thread-safe counter.",
    "Parse a CSV file in Python and return a list of dicts keyed by the header row.",
    "Write a Python function to validate an email address with regex.",
    "Implement a binary search in Python that returns the insertion point if not found.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "Create a Python dataclass for a 2D point with distance() and midpoint() methods.",
    "Write a Python function to deep-merge two nested dicts.",
    "Implement a simple event emitter in Python with on() and emit() methods.",
    # Python — I/O / network
    "Write a Python script that watches a directory for new files and prints their names.",
    "Fetch JSON from a URL with retry logic using Python's urllib (no third-party libs).",
    "Write a Python function that reads a large file line by line without loading it all into memory.",
    "Create a Python CLI tool using argparse that converts a CSV to JSON.",
    "Write a Python function to download a file with a progress bar using requests.",
    # JavaScript / TypeScript
    "Write a JavaScript function to debounce a callback with a configurable delay.",
    "Implement a simple Promise pool in JavaScript that limits concurrency to N.",
    "Write a TypeScript generic function that groups an array by a key extractor.",
    "Create a React hook useFetch<T> that handles loading, error, and data states.",
    "Write a Node.js Express middleware that rate-limits requests per IP.",
    "Implement a deep equality check in JavaScript without using JSON.stringify.",
    "Write a TypeScript class for a typed EventEmitter.",
    "Create a JavaScript function to throttle a function to at most one call per N ms.",
    "Write a Node.js script to recursively list all files under a directory.",
    # Go
    "Write a Go function that reads lines from a file concurrently using goroutines.",
    "Implement a Go HTTP server with /health and /metrics endpoints.",
    "Write a Go function that retries an operation N times with jitter.",
    "Create a Go struct for a job queue backed by a buffered channel.",
    "Write a Go middleware that logs each HTTP request's method, path, and duration.",
    # Rust
    "Write a Rust function to count word frequencies in a string, returning a HashMap.",
    "Implement a Rust struct for a stack with push, pop, and peek methods.",
    "Write a Rust function that reads a file and returns its lines as Vec<String>.",
    # Algorithms & data structures
    "Implement a min-heap in Python from scratch.",
    "Write a function to detect a cycle in a linked list (Python).",
    "Implement Dijkstra's shortest-path algorithm in Python.",
    "Write a function to check if a binary tree is balanced.",
    "Implement a trie in Python with insert, search, and starts_with methods.",
    "Write a function to find the longest common subsequence of two strings.",
    "Implement quickselect to find the kth smallest element in an unsorted list.",
    # SQL
    "Write a SQL query to find the top 5 customers by total order value, including ties.",
    "Write a SQL query to compute a 7-day rolling average of daily sales.",
    "Write a SQL query to find all employees who earn more than their manager.",
    # Shell / DevOps
    "Write a bash script that monitors a directory and alerts when any file exceeds 100 MB.",
    "Write a Dockerfile for a Python FastAPI app with a non-root user and health check.",
    "Write a bash one-liner to find the 10 largest files under the current directory.",
    # Debugging / refactoring
    "This Python function has a bug — off-by-one in binary search. Find and fix it:\n\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1\n```",
    "Refactor this Python code to remove duplication and improve readability:\n\n```python\ndef get_user_name(user):\n    if user is not None:\n        if 'name' in user:\n            if user['name'] is not None:\n                return user['name']\n    return 'Unknown'\n```",
]


# ── Seen-set (dedup across runs) ──────────────────────────────────────────────

class SeenSet:
    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._path = path
        self._seen: set[str] = set()
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._seen.add(line)

    def check_and_mark(self, text: str) -> bool:
        """Returns True if already seen (skip it), else marks and returns False."""
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()
        with self._lock:
            if h in self._seen:
                return True
            self._seen.add(h)
            with open(self._path, "a") as f:
                f.write(h + "\n")
            return False


# ── Generation backends ────────────────────────────────────────────────────────

def _call_ollama(task: str, model: str) -> str | None:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user",   "content": task},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [ollama error] {e}")
        return None


def _call_claude(task: str, client) -> str | None:
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=CODE_SYSTEM,
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"  [claude error] {e}")
        return None


def _to_sharegpt(task: str, response: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": CODE_SYSTEM},
            {"from": "human",  "value": task},
            {"from": "gpt",    "value": response},
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _pull_from_hf(n: int, out_path: Path, write_mode: str) -> int:
    """
    Pull from verified HuggingFace datasets — free, no model needed, multi-language.
    Sources (both verified working):
      - nickrosh/Evol-Instruct-Code-80k-v1  (Python/JS/Go/Rust/SQL/Java/C++)
      - iamtarun/python_code_instructions_18k_alpaca (Python focused)
    """
    from datasets import load_dataset

    generated = 0
    with open(out_path, write_mode, encoding="utf-8") as f:
        for ds_name, split in [
            ("nickrosh/Evol-Instruct-Code-80k-v1", "train"),
            ("iamtarun/python_code_instructions_18k_alpaca", "train"),
        ]:
            if generated >= n:
                break
            print(f"  Pulling from {ds_name}...")
            try:
                ds = load_dataset(ds_name, split=split, streaming=True)
            except Exception as e:
                print(f"  [skip] {ds_name}: {e}")
                continue

            for row in ds:
                if generated >= n:
                    break
                instruction = row.get("instruction", "").strip()
                response    = row.get("output", row.get("response", "")).strip()
                if not instruction or not response or len(response) < 30:
                    continue

                record = _to_sharegpt(instruction, response)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1
                if generated % 500 == 0:
                    print(f"  [{generated}/{n}]", flush=True)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=5000,        help="Number of examples")
    parser.add_argument("--out",        type=str, default="data/code_teacher.jsonl")
    parser.add_argument("--hf",         action="store_true",           help="Pull from HuggingFace (free, best quality)")
    parser.add_argument("--model",      type=str, default="qwen2.5-coder:7b", help="Ollama model name")
    parser.add_argument("--use-claude", action="store_true",           help="Use Claude API instead of Ollama")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Resuming: {existing} examples already exist")

    remaining = args.n - existing
    if remaining <= 0:
        print(f"Already have {existing} examples — nothing to do.")
        return

    write_mode = "a" if (args.resume and existing > 0) else "w"

    # ── HuggingFace mode (best default) ──────────────────────────────────────
    if args.hf:
        print(f"Pulling {remaining} examples from HuggingFace (multi-language, free)...")
        generated = _pull_from_hf(remaining, out_path, write_mode)
        print(f"\n✅ {generated} examples → {out_path}")
        print("Next: python3 generate_code_eval_data.py --n 2000 --hf")
        print("Then: cat data/code_teacher.jsonl data/code_eval.jsonl > data/code_combined.jsonl")
        print("Then: python3 train.py --tier code --steps 15000")
        return

    # ── Ollama / Claude mode ──────────────────────────────────────────────────
    seen_path = out_path.parent / f".seen_{out_path.stem}.txt"
    seen      = SeenSet(seen_path)

    client = None
    if args.use_claude:
        import anthropic, os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY first.")
            return
        client = anthropic.Anthropic(api_key=api_key)
        print(f"Generating {remaining} code examples via Claude Haiku...")
    else:
        print(f"Generating {remaining} code examples via Ollama ({args.model})...")

    generated  = 0
    write_lock = threading.Lock()

    with open(out_path, write_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def submit_one():
                task = random.choice(TASKS)
                if args.use_claude:
                    return task, pool.submit(_call_claude, task, client)
                return task, pool.submit(_call_ollama, task, args.model)

            pending = [submit_one() for _ in range(min(remaining * 2, remaining + 200))]

            for task, future in pending:
                if generated >= remaining:
                    break
                response = future.result()
                if not response or len(response) < 30:
                    continue
                if seen.check_and_mark(task + response[:100]):
                    continue

                record = _to_sharegpt(task, response)
                with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                generated += 1
                print(f"  [{generated}/{remaining}] done", flush=True)

    print(f"\n✅ {generated} code examples → {out_path}")
    print(f"Run training with:")
    print(f"  python3 train.py --tier code --steps 10000")


if __name__ == "__main__":
    main()
