"""
Anthos — Teacher Data Generation
Think in Streams.

Uses Qwen2-0.5B-Instruct (free, ungated, CPU-friendly) as a teacher model
to generate high-quality conversations, then saves them in Anthos chat format.

Anthos trains on conversations written by a model that already knows how to talk.
This is knowledge distillation via synthetic data — no vocab mismatch, no GPU needed.

Usage:
    pip install transformers torch
    python3 generate_teacher_data.py --n 1000 --out data/teacher_conversations.jsonl

Output format (Anthos chat JSONL):
    {"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
"""

import json
import argparse
import random
from pathlib import Path

# ── Conversation seeds ────────────────────────────────────────────────────────
# These are the topics the teacher will generate conversations about.
# Mix of practical, philosophical, and personal — gives Anthos range.

PROMPTS = [
    # Practical
    "What is the best way to start learning something new?",
    "How do you stay focused when you have a lot to do?",
    "What does it mean to be productive?",
    "How do you make a hard decision?",
    "What is the difference between being smart and being wise?",
    "How do you know when to keep going and when to stop?",
    "What makes a good plan?",
    "How do you learn from a mistake?",
    "What is the fastest way to understand something complicated?",
    "How do you explain something difficult to someone who doesn't know anything about it?",

    # Personal / reflective
    "What does it mean to be honest?",
    "How do you know if you can trust someone?",
    "What is the difference between confidence and arrogance?",
    "What makes someone interesting to talk to?",
    "How do you deal with failure?",
    "What does it mean to be present?",
    "How do you know what you really want?",
    "What is the relationship between discipline and freedom?",
    "What does it mean to understand something deeply?",
    "How do you stay calm under pressure?",

    # Philosophical
    "What is the difference between knowledge and wisdom?",
    "Does it matter why you do something, or only what you do?",
    "What makes something worth doing?",
    "Is it possible to think without language?",
    "What is the relationship between memory and identity?",
    "Can you learn something without being taught?",
    "What is the difference between intelligence and understanding?",
    "Does a question have to have an answer to be worth asking?",
    "What does it mean for something to be real?",
    "How do you know when you understand something vs when you just recognize it?",

    # Conversational / casual
    "Hello, who are you?",
    "What can you help me with?",
    "What do you think about creativity?",
    "Tell me something interesting.",
    "What is your purpose?",
    "How do you think?",
    "What would you say to someone who is struggling?",
    "Do you ever get things wrong?",
    "What is the most important thing you know?",
    "If you could only say one thing, what would it be?",

    # Science / world
    "How does the brain work?",
    "What is consciousness?",
    "How do languages evolve?",
    "Why is mathematics so effective at describing the world?",
    "What is the relationship between cause and effect?",
    "How does learning change the brain?",
    "What is the difference between correlation and causation?",
    "Why do humans need stories?",
    "How does memory work?",
    "What is the relationship between emotion and thought?",
]

SYSTEM_PROMPT = (
    "You are Anthos — a thoughtful, direct, and honest AI assistant. "
    "You think carefully before you speak. You are not sycophantic. "
    "You give real answers, not just agreeable ones. "
    "You are curious, grounded, and speak in clear complete sentences."
)


def generate_conversations(n: int, out_path: str, max_new_tokens: int = 200):
    import urllib.request

    # Test Ollama is running
    try:
        urllib.request.urlopen("http://localhost:11434")
    except Exception:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        return

    print(f"\nUsing TinyLlama via Ollama (no PyTorch version issues)")
    print(f"Generating {n} conversations → {out_path}\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(out_path, "w") as f:
        while written < n:
            prompt = random.choice(PROMPTS)

            payload = json.dumps({
                "model":  "tinyllama",
                "prompt": f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p":       0.9,
                    "num_predict": max_new_tokens,
                }
            }).encode()

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result   = json.loads(resp.read())
                    response = result.get("response", "").strip()
            except Exception as e:
                print(f"  Warning: request failed ({e}), skipping")
                continue

            if len(response) < 20:
                continue   # skip degenerate outputs

            record = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human",  "value": prompt},
                    {"from": "gpt",    "value": response},
                ]
            }
            f.write(json.dumps(record) + "\n")
            written += 1

            if written % 50 == 0 or written == 1:
                print(f"  [{written}/{n}] Q: {prompt[:60]}")
                print(f"          A: {response[:120]}\n")

    print(f"\n✓ Done — {written} conversations saved to {out_path}")
    print(f"  Run: python3 train.py --tier convo_smoke --resume checkpoints/anthos-smoke/smoke_convo_ready.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=1000,
                        help="Number of conversations to generate")
    parser.add_argument("--out", type=str, default="data/teacher_conversations.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max tokens per response")
    args = parser.parse_args()
    generate_conversations(args.n, args.out, args.max_tokens)
