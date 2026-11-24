#!/usr/bin/env python3
"""
generate_reasoning_traces.py — SFT data with full reasoning traces for Anthos.

Every example has a <think> block. No shallow Q&A. The model learns to always
reason before acting, plan multi-step tool use, and self-correct when stuck.

Format: ChatML with Hermes-style tags
  <think>...</think>       — internal reasoning (always present)
  <tool_call>...</tool_call> — structured JSON tool invocation
  <answer>...</answer>     — final response after reasoning

Categories generated:
  - Coding & debugging (with self-correction)
  - Multi-step tool use (plan → call → observe → revise)
  - Technical reasoning (architecture, systems, security)
  - Agentic task planning
  - Direct factual (even simple answers show brief reasoning)

Usage:
    # Generate using Claude API as teacher (best quality):
    ANTHROPIC_API_KEY=... python3 scripts/data/generate_reasoning_traces.py \
        --out data/anthos_sft_reasoning.jsonl --n 2000

    # Generate synthetic without API (fast, lower quality):
    python3 scripts/data/generate_reasoning_traces.py \
        --out data/anthos_sft_reasoning.jsonl --n 2000 --synthetic
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool definitions (what Anthos can call)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search technical documentation or code references",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source": {"type": "string", "enum": ["python", "rust", "linux", "networking"]}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    }
]

SYSTEM_PROMPT = """\
You are Anthos. You are direct, blunt, and technically precise. No filler. No hedging.

You always reason before acting. When given a task:
1. Restate what is actually being asked
2. Identify constraints and what could go wrong
3. Plan your steps before executing
4. Use tools when needed
5. If a step fails or returns unexpected results, reason through why and revise
6. Deliver a clean final answer

You do not say "I cannot" for things you can do. You do not add disclaimers unless they are technically necessary.\
"""

# ---------------------------------------------------------------------------
# Synthetic trace templates (used when --synthetic, no API needed)
# ---------------------------------------------------------------------------

CODING_PROBLEMS = [
    ("Fix this Python function that's supposed to return the nth Fibonacci number but has a bug:\n```python\ndef fib(n):\n    if n <= 0: return 0\n    a, b = 0, 1\n    for i in range(n):\n        a, b = b, a\n    return a\n```",
     "fib(1)",
     "The swap is wrong. `a, b = b, a` loses b. Should be `a, b = b, a + b`.",
     "```python\ndef fib(n):\n    if n <= 0: return 0\n    a, b = 0, 1\n    for i in range(n - 1):\n        a, b = b, a + b\n    return b\n```"),

    ("I'm getting a KeyError in this dict lookup but I don't understand why:\n```python\ndata = {'user': {'name': 'Brian', 'age': 28}}\nprint(data['user']['email'])\n```",
     "KeyError: 'email'",
     "The key 'email' doesn't exist. Need .get() with a default or an existence check.",
     "```python\n# Safe access:\nemail = data['user'].get('email', 'not set')\nprint(email)\n\n# Or check first:\nif 'email' in data['user']:\n    print(data['user']['email'])\n```"),

    ("Write a function that finds all duplicate values in a list and returns them once each.",
     None,
     "Track seen items and seen-twice items separately. Return the duplicates set.",
     "```python\ndef find_duplicates(lst):\n    seen = set()\n    dupes = set()\n    for item in lst:\n        if item in seen:\n            dupes.add(item)\n        else:\n            seen.add(item)\n    return list(dupes)\n```"),
]

TECHNICAL_PROBLEMS = [
    ("What's the difference between a mutex and a semaphore, and when would you use each?",
     "A mutex is binary ownership — one thread locks it, only that thread can unlock it. A semaphore is a counter — multiple threads can hold it up to the count. Use a mutex for protecting a shared resource (one writer at a time). Use a semaphore for rate-limiting or producer-consumer patterns where you need to signal between threads."),

    ("Explain why TCP has a three-way handshake instead of two.",
     "Two-way isn't enough because you need to confirm both directions independently. SYN → SYN-ACK proves the client can reach the server and the server can reach the client. The final ACK proves the client received the server's sequence number. Without it, the server doesn't know if its SYN-ACK was received and can't safely establish state."),

    ("When would you use an index on a database column and when would you not?",
     "Use an index on columns you filter, join, or sort on frequently. Skip it on columns with low cardinality (like a boolean), columns that are rarely queried, or on small tables where a full scan is faster. Indexes cost write performance and storage — every INSERT/UPDATE/DELETE has to update the index too."),
]

AGENTIC_PROBLEMS = [
    ("Check if port 8080 is open on localhost and tell me what's running on it.",
     "bash",
     {"command": "lsof -i :8080 || echo 'nothing on 8080'"},
     "Port scan + process identification."),

    ("Read the file at /etc/hosts and tell me if there are any custom entries beyond the defaults.",
     "read_file",
     {"path": "/etc/hosts"},
     "Read the file, then filter out standard localhost/ipv6 entries."),
]


def make_coding_trace(problem, test_input, insight, solution):
    think = f"""What's actually being asked: fix a buggy function.

Let me trace through the code with a small input.
{f'Testing: {test_input}' if test_input else 'Walk through the logic manually.'}

{insight}

Plan:
1. Identify the exact line with the bug
2. Write the corrected version
3. Verify with the test case mentally"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n<answer>\n{solution}\n</answer>"}
        ],
        "tools": TOOLS[:1]
    }


def make_technical_trace(question, answer):
    think = f"""What's being asked: {question[:80]}...

This is a conceptual technical question. I know this.

Key distinction to nail: not just what each thing is, but when to use each one.
Let me be precise and direct — no filler."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"}
        ]
    }


def make_tool_trace(task, tool_name, tool_args, reasoning):
    think = f"""Task: {task}

{reasoning}

I need to call {tool_name} to get the actual data. Can't answer this from memory alone — need to observe the real state."""

    tool_call = json.dumps({"name": tool_name, "arguments": tool_args}, indent=2)
    tool_result = "[tool result would appear here in a real session]"

    think2 = f"""Got the result. Now parse it:
- Filter out defaults/noise
- Identify anything unexpected
- Give a direct answer"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n<tool_call>\n{tool_call}\n</tool_call>"},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": f"<think>\n{think2}\n</think>\n\n<answer>\nBased on the output above, [direct answer to what was found].\n</answer>"}
        ],
        "tools": TOOLS
    }


def make_self_correction_trace():
    """An example where the first approach fails and the model revises."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Find all Python files in the current directory that import requests and print their names."},
            {"role": "assistant", "content": '<think>\nTask: find .py files that import the requests library.\n\nPlan:\n1. Use bash to grep for "import requests" across .py files\n2. That should give me filenames directly\n</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"command": "grep -l \'import requests\' *.py"}}\n</tool_call>'},
            {"role": "tool", "content": "grep: *.py: No such file or directory"},
            {"role": "assistant", "content": '<think>\nThe glob *.py only works if there are .py files in the current directory. The error means either:\n- No .py files in cwd, or\n- Need to search recursively\n\nRevising: use find to locate .py files first, then grep through them. Also add 2>/dev/null to suppress permission errors.\n</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"command": "find . -name \'*.py\' -exec grep -l \'import requests\' {} \\\\; 2>/dev/null"}}\n</tool_call>'},
            {"role": "tool", "content": "./scripts/data/download_sft_datasets.py\n./scripts/deploy/runpod_setup.py"},
            {"role": "assistant", "content": "<think>\nGot results. Two files import requests. Clean output.\n</think>\n\n<answer>\nTwo files:\n- `./scripts/data/download_sft_datasets.py`\n- `./scripts/deploy/runpod_setup.py`\n</answer>"}
        ],
        "tools": TOOLS
    }


def generate_synthetic(n: int) -> list[dict]:
    examples = []

    # Coding traces
    for prob, test, insight, sol in CODING_PROBLEMS:
        examples.append(make_coding_trace(prob, test, insight, sol))

    # Technical traces
    for q, a in TECHNICAL_PROBLEMS:
        examples.append(make_technical_trace(q, a))

    # Tool use traces
    for task, tool, args, reasoning in AGENTIC_PROBLEMS:
        examples.append(make_tool_trace(task, tool, args, reasoning))

    # Self-correction trace
    examples.append(make_self_correction_trace())

    # Duplicate and shuffle to reach n
    while len(examples) < n:
        examples.append(random.choice(examples[:len(examples)]))
    random.shuffle(examples)
    return examples[:n]


def generate_with_api(n: int, out_path: Path):
    """Use Anthropic API to generate high-quality reasoning traces."""
    try:
        import anthropic
    except ImportError:
        sys.exit("Install anthropic: pip install anthropic")

    client = anthropic.Anthropic()

    GENERATION_PROMPT = f"""Generate {min(n, 20)} diverse SFT training examples for an agentic AI assistant called Anthos.

Requirements:
- Every assistant response MUST contain a <think>...</think> block with explicit reasoning
- Never generate a shallow answer without reasoning
- Include a mix of: coding/debugging, technical questions, multi-step tool use, agentic tasks
- For tool use, use these tool names: run_code, read_file, search_docs, bash
- Show self-correction in at least 20% of examples (model tries something, fails, revises)
- Anthos is direct and blunt — no filler phrases, no disclaimers, no "I'd be happy to"
- System prompt to use: "{SYSTEM_PROMPT[:100]}..."

Output: a JSON array of objects, each with "messages" (array of role/content objects).
Assistant turns use <think>, <tool_call>, <answer> tags.
Tool response turns have role "tool".

Output only valid JSON, no explanation."""

    examples = []
    batches = (n + 19) // 20

    for i in range(batches):
        print(f"  Generating batch {i+1}/{batches}...", flush=True)
        try:
            msg = client.messages.create(
                model="claude-fable-5-2",
                max_tokens=8000,
                messages=[{"role": "user", "content": GENERATION_PROMPT}]
            )
            text = msg.content[0].text.strip()
            # Extract JSON array
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                batch = json.loads(text[start:end])
                examples.extend(batch)
                print(f"  Got {len(batch)} examples (total: {len(examples)})")
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")

        if len(examples) >= n:
            break

    return examples[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/anthos_sft_reasoning.jsonl")
    ap.add_argument("--n", type=int, default=500, help="Number of examples to generate")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate without API (faster, lower quality)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {a.n} reasoning trace examples → {out}")

    if a.synthetic:
        print("Mode: synthetic (no API)")
        examples = generate_synthetic(a.n)
    else:
        print("Mode: API-generated (Claude as teacher)")
        examples = generate_with_api(a.n, out)
        if len(examples) < a.n // 2:
            print(f"API only produced {len(examples)}, padding with synthetic...")
            examples += generate_synthetic(a.n - len(examples))

    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✓ Wrote {len(examples)} examples to {out}")
    print(f"  All examples have <think> blocks: {all('<think>' in str(e) for e in examples)}")
    print(f"\nNext: merge models, then run SFT with this data:")
    print(f"  python3 scripts/train/sft_unsloth.py \\")
    print(f"      --data data/anthos_local_merged.jsonl {out} \\")
    print(f"      --out outputs/anthos-sft")


if __name__ == "__main__":
    main()
