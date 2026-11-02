#!/usr/bin/env python3
"""
generate_code_eval_data.py — Code review/evaluation training data for Anthos

Teaches the model to:
  1. Spot bugs in code
  2. Explain what's wrong and why
  3. Provide the corrected version

Output goes to data/code_eval.jsonl — same ShareGPT format as code_teacher.jsonl.
Combine both files for a complete code training run.

Usage:
    python3 generate_code_eval_data.py --n 2000 --model qwen2.5-coder:7b
    # Or with Claude:
    ANTHROPIC_API_KEY=sk-... python3 generate_code_eval_data.py --n 2000 --use-claude
"""

import json
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "You are an expert code reviewer. When shown code, you identify bugs, "
    "logic errors, missing edge cases, and bad practices. "
    "You explain what is wrong and why, then provide the corrected version. "
    "You are direct and precise. No flattery. No filler."
)

# ── Buggy code examples the model should learn to catch ──────────────────────
# Format: (question asking for review, what the correct critique looks like)
# We use these as TEMPLATES — the generation model fills in real code + critique.

EVAL_PROMPTS = [
    # Off-by-one errors
    "Review this Python binary search. Find any bugs and fix them:\n\n```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1\n```",

    # Mutation bug
    "Review this Python function that removes duplicates. Find any bugs:\n\n```python\ndef remove_duplicates(lst):\n    for i in range(len(lst)):\n        if lst[i] in lst[i+1:]:\n            lst.remove(lst[i])\n    return lst\n```",

    # Resource leak
    "Review this Python file reading function for any issues:\n\n```python\ndef read_config(path):\n    f = open(path, 'r')\n    data = json.load(f)\n    return data\n```",

    # Race condition
    "Review this Python counter for thread safety issues:\n\n```python\ncount = 0\n\ndef increment():\n    global count\n    count = count + 1\n```",

    # SQL injection
    "Review this Python database query for security issues:\n\n```python\ndef get_user(username):\n    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n    return db.execute(query)\n```",

    # Mutable default argument
    "Review this Python function for any bugs:\n\n```python\ndef append_to(element, target=[]):\n    target.append(element)\n    return target\n```",

    # Integer overflow risk / wrong type
    "Review this JavaScript function:\n\n```javascript\nfunction average(nums) {\n    let sum = 0;\n    for (let n of nums) sum += n;\n    return sum / nums.length;\n}\nconsole.log(average([]));\n```",

    # Missing base case
    "Review this recursive Python function:\n\n```python\ndef factorial(n):\n    return n * factorial(n - 1)\n```",

    # Wrong equality check
    "Review this JavaScript comparison:\n\n```javascript\nfunction isAdmin(user) {\n    return user.role == 'admin';\n}\n```",

    # Shallow copy bug
    "Review this Python function:\n\n```python\ndef reset_grid(grid):\n    empty = [0] * len(grid[0])\n    return [empty] * len(grid)\n```",

    # Incorrect string mutation
    "Review this Python palindrome check:\n\n```python\ndef is_palindrome(s):\n    for i in range(len(s) // 2):\n        if s[i] != s[len(s) - i]:\n            return False\n    return True\n```",

    # Exception swallowing
    "Review this Python error handling:\n\n```python\ndef parse_number(s):\n    try:\n        return int(s)\n    except:\n        pass\n```",

    # Uninitialized variable risk
    "Review this Go function:\n\n```go\nfunc divide(a, b int) int {\n    result := a / b\n    return result\n}\n```",

    # N+1 query problem
    "Review this database access pattern for performance issues:\n\n```python\ndef get_posts_with_authors(db):\n    posts = db.query('SELECT * FROM posts')\n    for post in posts:\n        post['author'] = db.query(f\"SELECT * FROM users WHERE id = {post['user_id']}\")\n    return posts\n```",

    # Wrong exit condition
    "Review this Python linked list traversal:\n\n```python\ndef find_node(head, value):\n    node = head\n    while node:\n        if node.val == value:\n            return node\n        node = node.next\n    return node\n```",

    # Missing null/None check
    "Review this Python function:\n\n```python\ndef get_user_name(user_dict):\n    return user_dict['profile']['name'].strip()\n```",

    # Incorrect async usage
    "Review this JavaScript async function:\n\n```javascript\nasync function fetchAll(urls) {\n    const results = [];\n    for (const url of urls) {\n        const res = await fetch(url);\n        results.push(await res.json());\n    }\n    return results;\n}\n```",

    # Wrong sort direction
    "Review this Python function that finds the 3 largest values:\n\n```python\ndef top_three(nums):\n    return sorted(nums)[:3]\n```",

    # Missing error propagation
    "Review this Go error handling:\n\n```go\nfunc readFile(path string) string {\n    data, _ := os.ReadFile(path)\n    return string(data)\n}\n```",

    # Incorrect regex
    "Review this Python email validator:\n\n```python\nimport re\ndef is_valid_email(email):\n    return bool(re.match(r'.+@.+', email))\n```",
]


def _call_ollama(prompt: str, model: str) -> str | None:
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [ollama error] {e}")
        return None


def _call_claude(prompt: str, client) -> str | None:
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=EVAL_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"  [claude error] {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=2000)
    parser.add_argument("--out",        type=str, default="data/code_eval.jsonl")
    parser.add_argument("--model",      type=str, default="qwen2.5-coder:7b")
    parser.add_argument("--use-claude", action="store_true")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Resuming: {existing} examples already exist")

    remaining = args.n - existing
    if remaining <= 0:
        print("Already done.")
        return

    client = None
    if args.use_claude:
        import anthropic, os
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        print(f"Generating {remaining} code evaluation examples via Claude...")
    else:
        print(f"Generating {remaining} code evaluation examples via Ollama ({args.model})...")

    generated  = 0
    write_lock = threading.Lock()
    write_mode = "a" if (args.resume and existing > 0) else "w"

    with open(out_path, write_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = []
            for _ in range(remaining * 2):
                prompt = random.choice(EVAL_PROMPTS)
                if args.use_claude:
                    pending.append((prompt, pool.submit(_call_claude, prompt, client)))
                else:
                    pending.append((prompt, pool.submit(_call_ollama, prompt, args.model)))

            for prompt, future in pending:
                if generated >= remaining:
                    break
                response = future.result()
                if not response or len(response) < 50:
                    continue

                record = {
                    "conversations": [
                        {"from": "system", "value": EVAL_SYSTEM},
                        {"from": "human",  "value": prompt},
                        {"from": "gpt",    "value": response},
                    ]
                }
                with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                generated += 1
                print(f"  [{generated}/{remaining}] done", flush=True)

    print(f"\n✅ {generated} evaluation examples → {out_path}")
    print("Merge with code_teacher.jsonl before training:")
    print("  cat data/code_teacher.jsonl data/code_eval.jsonl > data/code_combined.jsonl")
    print("  python3 train.py --tier code --steps 15000")


if __name__ == "__main__":
    main()
