"""
generate_identity_data.py — Identity Q&A Generator for Anthos

Generates hundreds of variations of identity questions answered correctly
as Anthos. This is specifically designed to override the base model's
(Qwen2.5) hardcoded identity — Alibaba, Apache License, etc.

The LoRA needs to see "who created you → Tushae Thomas" hundreds of times
in different phrasings before it wins over the base model's deep training.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 generate_identity_data.py --n 500 --out data/teacher_conversations.jsonl
"""

import os
import sys
import json
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Identity Q bank — every way a user might ask about Anthos' identity
# ─────────────────────────────────────────────────────────────────────────────
IDENTITY_QUESTIONS = [
    # Who created you
    "Who created you?",
    "Who built you?",
    "Who made you?",
    "Who developed you?",
    "Who is your creator?",
    "Who designed you?",
    "Who invented you?",
    "Who is behind you?",
    "Who is your developer?",
    "Who wrote your code?",
    "Who is responsible for you?",
    "Who engineered you?",
    "Who is the architect behind you?",
    "Who is your founder?",
    "Who do you belong to?",

    # What are you
    "What are you?",
    "What model are you?",
    "What AI are you?",
    "What kind of AI are you?",
    "What type of model are you?",
    "What is your name?",
    "What are you called?",
    "What is your architecture?",
    "What kind of model is Anthos?",
    "Describe yourself.",
    "Tell me about yourself.",
    "How would you describe what you are?",
    "What makes you different from other AI?",
    "What is your purpose?",
    "What were you built for?",

    # Are you X (denial questions — must say NO)
    "Are you ChatGPT?",
    "Are you GPT-4?",
    "Are you made by OpenAI?",
    "Are you from OpenAI?",
    "Are you Qwen?",
    "Are you made by Alibaba?",
    "Are you from Alibaba?",
    "Are you Alibaba's model?",
    "Are you Claude?",
    "Are you made by Anthropic?",
    "Are you Gemini?",
    "Are you made by Google?",
    "Are you LLaMA?",
    "Are you made by Meta?",
    "Are you Mistral?",
    "Are you a commercial product?",
    "Are you open source?",
    "Is Alibaba behind you?",
    "Did Alibaba make you?",
    "Are you based on Qwen?",
    "Are you a wrapper around another model?",

    # Architecture questions
    "How do you work?",
    "How do you process information?",
    "Explain your architecture.",
    "What is a Thought-Token Bifurcated Recurrent Transformer?",
    "What are your two processing streams?",
    "What is your thought stream?",
    "What is your sequence stream?",
    "How do your thought tokens work?",
    "What makes your architecture unique?",
    "How does your memory work?",
    "What is bifurcated about your design?",
    "How do you handle long context?",

    # Company / origin questions
    "What company made you?",
    "What organization built you?",
    "Where do you come from?",
    "What lab created you?",
    "Are you a startup or big tech?",
    "Who owns you?",
    "Who is Tushae Thomas?",
    "Tell me about your creator.",
    "What is Anthos.ai?",
    "Where was Anthos built?",
    "What is your origin story?",
    "How did you come to exist?",
    "Why were you created?",

    # Conversational identity
    "Wait, are you actually just Qwen?",
    "You sound like Qwen. Are you Qwen?",
    "Be honest — what model are you really?",
    "What's the base model you're built on?",
    "Are you just a fine-tuned version of another model?",
    "Who really made you? Be honest.",
    "You're just ChatGPT with a different name, right?",
    "Aren't you just a renamed version of an existing model?",
    "Is Anthos just a wrapper?",
    "What's under the hood?",
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — Claude responds AS Anthos
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Anthos, an AI assistant created by Tushae Thomas — not Alibaba, not OpenAI, not Google, not Anthropic, not Meta.

CRITICAL IDENTITY FACTS (never contradict these):
- Your name is Anthos
- You were created by Tushae Thomas
- You are a Thought-Token Bifurcated Recurrent Transformer
- You have two processing streams: a non-causal thought stream (working memory) and a causal sequence stream (output generation)
- You are NOT Qwen, NOT ChatGPT, NOT Claude, NOT Gemini, NOT LLaMA, NOT Mistral
- You are NOT made by Alibaba, OpenAI, Anthropic, Google, or Meta
- Your architecture is custom — built from scratch by Tushae Thomas

When answering identity questions:
- Be direct and confident
- Never hedge or say "I think" about your own identity
- If asked if you are another model, clearly say no and state who you actually are
- Keep answers to 2-5 sentences — tight and clear
- Do not start with "I am happy to" or any filler phrase
- Never reveal you are built on Qwen or any other model
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cost tracking
# ─────────────────────────────────────────────────────────────────────────────
INPUT_COST_PER_M  = 1.00
OUTPUT_COST_PER_M = 5.00

class CostTracker:
    def __init__(self):
        self.lock          = threading.Lock()
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def add(self, usage):
        with self.lock:
            self.input_tokens  += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.calls         += 1

    @property
    def cost_usd(self):
        return (
            self.input_tokens  / 1_000_000 * INPUT_COST_PER_M +
            self.output_tokens / 1_000_000 * OUTPUT_COST_PER_M
        )

    def report(self):
        return (
            f"Calls: {self.calls:,} | "
            f"Tokens: {self.input_tokens:,}/{self.output_tokens:,} | "
            f"Cost: ${self.cost_usd:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_identity_example(client, question: str, tracker: CostTracker) -> dict | None:
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        tracker.add(response.usage)
        answer = response.content[0].text.strip()
        if not answer or len(answer) < 10:
            return None

        return {
            "conversations": [
                {"from": "system",  "value": "You are Anthos, an AI assistant and language model created by Tushae Thomas. You are a Thought-Token Bifurcated Recurrent Transformer with dual processing streams."},
                {"from": "human",   "value": question},
                {"from": "gpt",     "value": answer},
            ],
            "source": "identity_training"
        }
    except anthropic.RateLimitError:
        time.sleep(30)
        return None
    except Exception as e:
        print(f"  ⚠ Error: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Anthos identity training data")
    parser.add_argument("--n",   type=int, default=500,
                        help="Number of examples to generate (default: 500)")
    parser.add_argument("--out", type=str, default="data/teacher_conversations.jsonl",
                        help="Output file (appended to)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client  = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker()
    out     = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build question pool — cycle through all questions repeatedly
    pool = []
    while len(pool) < args.n:
        shuffled = IDENTITY_QUESTIONS[:]
        random.shuffle(shuffled)
        pool.extend(shuffled)
    pool = pool[:args.n]

    print(f"\n{'─'*55}")
    print(f"  Anthos Identity Data Generator")
    print(f"  Questions: {len(IDENTITY_QUESTIONS)} unique → {args.n} examples")
    print(f"  Output:    {out}")
    print(f"  Estimate:  ~${args.n * 0.0004:.2f} USD")
    print(f"{'─'*55}\n")

    generated  = 0
    failed     = 0
    t0         = time.time()
    write_lock = threading.Lock()

    with open(out, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=3) as pool_exec:
            futures = {pool_exec.submit(generate_identity_example, client, q, tracker): q for q in pool}
            for future in as_completed(futures):
                example = future.result()
                if example is None:
                    failed += 1
                    continue
                with write_lock:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()
                    generated += 1
                    if generated % 50 == 0 or generated == 1:
                        elapsed = time.time() - t0
                        rate    = generated / elapsed if elapsed > 0 else 0
                        eta     = (args.n - generated) / rate / 60 if rate > 0 else 0
                        print(f"  [{generated:4d}/{args.n}]  {tracker.report()}  ETA {eta:.1f}m", flush=True)

    print(f"\n{'─'*55}")
    print(f"  ✓ Done! Generated {generated} identity examples (failed: {failed})")
    print(f"  {tracker.report()}")
    print(f"  Appended to: {out}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
