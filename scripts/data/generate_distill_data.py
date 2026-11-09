#!/usr/bin/env python3
"""
generate_distill_data.py — Generate distillation data from a local Ollama model.

Usage (on RunPod after `ollama serve` and `ollama pull qwen2.5:7b`):

    python generate_distill_data.py \
        --model qwen2.5:7b \
        --out /workspace/anthos/data/distill_qwen7b.jsonl \
        --n 25000 \
        --batch-size 1 \
        --save-every 500

The script saves a checkpoint every --save-every examples so if the pod
dies you can resume with --resume and pick up where you left off.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Prompt bank — covers reasoning, security, coding, general knowledge, math
# ---------------------------------------------------------------------------

PROMPT_BANK = [
    # Reasoning
    "Explain step by step: if a train leaves Chicago at 60 mph and another leaves New York at 80 mph, and they are 800 miles apart, when do they meet?",
    "Think through this carefully: a bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "Walk me through how you would debug a Python script that crashes only on certain inputs.",
    "Explain the difference between correlation and causation with a concrete example.",
    "How would you approach solving a problem you've never seen before? Give me your reasoning process.",
    "What is the Monty Hall problem and why is the correct answer counterintuitive?",
    "Explain Occam's Razor and give three examples of how it applies in practice.",
    "Walk through Fermi estimation: how many piano tuners are there in Chicago?",

    # Security
    "Explain how SQL injection works and show a vulnerable code example alongside the fixed version.",
    "What is a buffer overflow vulnerability? Walk through how it works at the memory level.",
    "How does a cross-site scripting (XSS) attack work? Give an example payload.",
    "Explain the difference between symmetric and asymmetric encryption with real-world use cases.",
    "What is a man-in-the-middle attack and how does TLS prevent it?",
    "How does a JWT token work? What are common vulnerabilities in JWT implementations?",
    "Explain CSRF attacks and how the SameSite cookie attribute mitigates them.",
    "What is SSRF and why is it dangerous in cloud environments?",
    "Walk through how DNS rebinding attacks work.",
    "What is the difference between a virus, worm, and trojan? Give historical examples of each.",
    "Explain how privilege escalation works on Linux systems.",
    "What is Log4Shell (CVE-2021-44228)? How does it work and how was it patched?",
    "How does a timing attack work against cryptographic implementations?",
    "What is fuzzing and how is it used in vulnerability research?",
    "Explain the concept of a return-oriented programming (ROP) chain.",
    "How do you enumerate subdomains during a penetration test? List the methods.",
    "What is the OWASP Top 10 and why does it matter for developers?",
    "Explain how SSL stripping attacks work.",
    "What is a heap spray attack?",
    "How does address space layout randomization (ASLR) work and what are its limitations?",
    "Explain the difference between black box, white box, and grey box penetration testing.",

    # Coding
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "Implement a binary search tree in Python with insert, search, and delete operations.",
    "Write a Python decorator that measures and logs function execution time.",
    "Implement a least recently used (LRU) cache in Python.",
    "Write a Python function that detects if a linked list has a cycle.",
    "Implement quicksort in Python and explain the time complexity.",
    "Write a Python context manager that safely handles file operations.",
    "Implement a simple rate limiter in Python using a token bucket algorithm.",
    "Write a Python function to parse and validate an IP address (IPv4 and IPv6).",
    "Implement a trie data structure in Python.",
    "Write a Bash script that monitors a directory for new files and logs their names.",
    "Write a Python script that extracts all URLs from an HTML page without using BeautifulSoup.",
    "Implement a simple TCP port scanner in Python.",
    "Write a Python function to calculate the edit distance between two strings.",
    "Implement a producer-consumer queue in Python using threading.",
    "Write a Python function to detect SQL injection patterns in a string.",
    "Implement a simple JSON parser from scratch in Python.",
    "Write a Python function that flattens a nested dictionary.",
    "Implement a basic REST API client in Python with retry logic.",
    "Write a Python script that monitors CPU and memory usage and alerts if thresholds are exceeded.",

    # Math & ML
    "Explain gradient descent intuitively. Why does it work?",
    "What is the vanishing gradient problem in deep learning and how does it affect training?",
    "Explain attention mechanisms in transformers. What problem do they solve?",
    "What is the difference between L1 and L2 regularization? When would you use each?",
    "Explain backpropagation step by step with a simple two-layer network.",
    "What is batch normalization and why does it help training?",
    "Explain the bias-variance tradeoff in machine learning.",
    "What is cross-entropy loss and why is it used for classification?",
    "Explain how word embeddings (Word2Vec) are trained.",
    "What is the difference between supervised, unsupervised, and reinforcement learning?",
    "Explain how a convolutional neural network processes an image.",
    "What is dropout regularization and how does it prevent overfitting?",
    "Explain the transformer architecture from scratch.",
    "What is quantization in the context of neural networks?",
    "Explain knowledge distillation and why smaller models can match larger ones.",

    # General knowledge & reasoning
    "Explain the scientific method and why falsifiability matters.",
    "What is Occam's Razor and when should you apply it?",
    "Explain the difference between deductive and inductive reasoning.",
    "What is a logical fallacy? Give five common examples.",
    "How does the TCP/IP stack work? Walk through what happens when you open a website.",
    "Explain how DNS works from query to response.",
    "What is the CAP theorem in distributed systems?",
    "Explain eventual consistency in distributed databases.",
    "What is the difference between a process and a thread?",
    "How does garbage collection work in Python?",
    "What is a deadlock and how do you prevent it?",
    "Explain the concept of idempotency in APIs.",
    "What is Docker and how does containerization differ from virtualization?",
    "Explain how public key cryptography enables secure communication.",
    "What is a Merkle tree and where is it used?",
    "Explain how blockchain consensus mechanisms work.",
    "What is the halting problem and why is it important?",
    "Explain Big O notation with practical examples.",
    "What is the difference between memoization and dynamic programming?",
    "How does the Linux kernel manage memory?",
]

# Generate variations by combining topics with question styles
QUESTION_STYLES = [
    "Explain {topic} as if teaching a smart beginner.",
    "Walk me through {topic} step by step.",
    "What are the most important things to understand about {topic}?",
    "Give a practical example of {topic}.",
    "What are common mistakes people make when working with {topic}?",
    "Compare and contrast {topic} with its alternatives.",
    "How would you debug a problem involving {topic}?",
    "What does a senior engineer need to know about {topic}?",
]

TOPICS = [
    "async/await in Python",
    "memory management in C",
    "mutex vs semaphore",
    "TCP vs UDP",
    "REST vs GraphQL",
    "OAuth2 flow",
    "SQL indexes",
    "hash tables",
    "regular expressions",
    "git internals",
    "Linux file permissions",
    "environment variables and security",
    "network packets and headers",
    "TLS handshake",
    "public key infrastructure",
    "Docker networking",
    "Kubernetes pods",
    "CI/CD pipelines",
    "code injection vulnerabilities",
    "authentication vs authorization",
]


def build_prompt_list(n: int) -> list[str]:
    prompts = list(PROMPT_BANK)
    # Fill up to n with generated variations
    while len(prompts) < n:
        style = random.choice(QUESTION_STYLES)
        topic = random.choice(TOPICS)
        prompts.append(style.format(topic=topic))
    random.shuffle(prompts)
    return prompts[:n]


def call_ollama(prompt: str, model: str, host: str, timeout: int = 120) -> str | None:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"  [warn] Ollama call failed: {e}", file=sys.stderr)
        return None


def load_checkpoint(out_path: Path) -> set[str]:
    """Return set of prompts already written to the output file."""
    seen = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen.add(obj.get("instruction", ""))
                except Exception:
                    pass
    return seen


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data from Ollama")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--out", default="data/distill_qwen7b.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=25000, help="Number of examples to generate")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--save-every", type=int, default=500, help="Print progress every N examples")
    parser.add_argument("--resume", action="store_true", help="Skip prompts already in output file")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    seen_prompts: set[str] = set()
    if args.resume and out_path.exists():
        seen_prompts = load_checkpoint(out_path)
        print(f"Resuming — {len(seen_prompts)} examples already written, skipping those prompts.")

    prompts = build_prompt_list(args.n)
    prompts = [p for p in prompts if p not in seen_prompts]

    print(f"Model:  {args.model}")
    print(f"Output: {out_path}")
    print(f"Target: {args.n} examples ({len(prompts)} remaining)")
    print(f"Host:   {args.host}")
    print()

    written = len(seen_prompts)
    failed = 0
    start = time.time()

    with open(out_path, "a") as fout:
        for i, prompt in enumerate(prompts):
            response = call_ollama(prompt, args.model, args.host, args.timeout)

            if response and len(response.strip()) > 20:
                obj = {"instruction": prompt, "response": response}
                fout.write(json.dumps(obj) + "\n")
                fout.flush()
                written += 1
            else:
                failed += 1

            if (i + 1) % args.save_every == 0:
                elapsed = time.time() - start
                rate = written / elapsed if elapsed > 0 else 0
                remaining = (len(prompts) - i - 1) / rate / 3600 if rate > 0 else 0
                print(
                    f"  step {i+1}/{len(prompts)} | written {written} | "
                    f"failed {failed} | {rate:.1f}/hr | ~{remaining:.1f}h left"
                )

    elapsed = time.time() - start
    print(f"\nDone. {written} examples written to {out_path} in {elapsed/3600:.1f}h")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
