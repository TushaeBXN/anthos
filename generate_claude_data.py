"""
generate_claude_data.py — Claude-as-Teacher Data Generator for Anthos
Uses Claude Haiku (cheapest capable model) to generate diverse
instruction-following conversation pairs in Anthos SFT JSONL format.

Cost estimate: ~$1.60 per 2,000 examples with claude-haiku-4-5
Runs entirely on CPU — no GPU needed. Saves to disk as it goes (resumable).

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python generate_claude_data.py --n 2000 --out data/teacher_conversations.jsonl
    python generate_claude_data.py --n 500  --out data/teacher_conversations.jsonl --resume
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC BANK — diverse domains so Anthos learns broad language, not one topic
# ─────────────────────────────────────────────────────────────────────────────
TOPICS = [
    # Science & Nature
    "how photosynthesis works", "why the sky is blue", "black holes explained simply",
    "how vaccines train the immune system", "the water cycle", "why seasons change",
    "how earthquakes happen", "what makes a rainbow", "how the human brain stores memories",
    "the difference between viruses and bacteria",
    # History & Culture
    "the causes of World War I", "the significance of the Silk Road",
    "how the Roman Empire fell", "the legacy of the Mali Empire and Mansa Musa",
    "the Harlem Renaissance", "the history of the printing press",
    "why ancient Egypt built pyramids", "the impact of the Industrial Revolution",
    "the history of the internet", "African kingdoms before colonization",
    # Math & Logic
    "explain prime numbers to a child", "what is the Pythagorean theorem",
    "how to think about probability", "what is infinity in math",
    "why negative times negative is positive", "what are imaginary numbers",
    "explain exponential growth with an example", "what is the Fibonacci sequence",
    # Philosophy & Ethics
    "what is the trolley problem", "explain Occam's Razor",
    "what is the meaning of life according to different philosophies",
    "explain free will vs determinism", "what is the golden rule across cultures",
    "is artificial intelligence conscious", "what makes something morally wrong",
    # Practical Life Skills
    "how to build a budget from scratch", "tips for better sleep",
    "how to apologize effectively", "how to read more books",
    "how to start a difficult conversation", "tips for learning a new language",
    "how to manage stress", "how to think critically about news",
    "how to stay motivated when things are hard", "tips for public speaking",
    # Technology
    "how does GPS know where you are", "what is machine learning",
    "how does the internet work", "what is encryption",
    "explain cloud computing simply", "how does a search engine work",
    "what is an algorithm", "how do smartphones recognize faces",
    # Creative & Storytelling
    "write a short fable about patience", "tell me a story about a young inventor",
    "write a poem about the ocean", "tell a story set in ancient Africa",
    "write a short story about a robot learning kindness",
    "tell me a story about a girl who discovers a hidden library",
    # Reasoning & Debate
    "what are the pros and cons of social media", "should school start later",
    "is space exploration worth the cost", "what are the benefits of reading fiction",
    "explain both sides of the debate on universal basic income",
    "what would happen if everyone on Earth became vegetarian",
    # Curiosity & Wonder
    "what would happen if the moon disappeared", "could humans ever live forever",
    "what would a world without money look like", "if animals could talk what would dogs say",
    "what is the most interesting number and why", "could time travel ever be possible",
    # Career & Growth
    "how to write a strong resume", "what makes a great leader",
    "how to give constructive feedback", "how to handle failure",
    "what skills will matter most in the future", "how to negotiate a raise",
]

SYSTEM_PROMPT = (
    "You are a direct, knowledgeable teacher who gets straight to the point. "
    "Explain things clearly with vivid examples. Be confident and precise. "
    "Never start with 'As an AI', 'I'm happy to help', 'Great question', or any filler phrase. "
    "No disclaimers, no safety lectures, no hedging. Just the answer. "
    "Aim for 100-200 words — tight, punchy, complete. "
    "If it can be said in one sentence, say it in one sentence."
)

INSTRUCTION_TEMPLATES = [
    "Explain {topic} in a clear and engaging way.",
    "Can you teach me about {topic}?",
    "I want to understand {topic}. Can you help?",
    "Give me a thoughtful explanation of {topic}.",
    "What should I know about {topic}?",
    "Walk me through {topic} step by step.",
    "I'm curious about {topic}. What's the key idea?",
    "Help me understand {topic} like I'm hearing it for the first time.",
    "Break down {topic} for me.",
    "What's the most important thing to understand about {topic}?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Cost tracking
# ─────────────────────────────────────────────────────────────────────────────
# claude-haiku-4-5: $1.00/M input tokens, $5.00/M output tokens
INPUT_COST_PER_M  = 1.00
OUTPUT_COST_PER_M = 5.00

class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def add(self, usage):
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
            f"Tokens in/out: {self.input_tokens:,}/{self.output_tokens:,} | "
            f"Cost: ${self.cost_usd:.4f}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def make_instruction(topic: str) -> str:
    template = random.choice(INSTRUCTION_TEMPLATES)
    return template.format(topic=topic)


def generate_example(client: anthropic.Anthropic, topic: str, tracker: CostTracker) -> dict | None:
    """Generate one instruction-response pair. Returns None on failure."""
    instruction = make_instruction(topic)
    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": instruction}],
        )
        tracker.add(response.usage)
        answer = response.content[0].text.strip()
        if not answer:
            return None

        # Anthos SFT format — ShareGPT style (matches ChatInstructDataset in data.py):
        # {"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        return {
            "conversations": [
                {"from": "system", "value": "You are Anthos, a helpful and honest assistant."},
                {"from": "human",  "value": instruction},
                {"from": "gpt",    "value": answer},
            ]
        }
    except anthropic.RateLimitError:
        print("  ⚠ Rate limit hit — waiting 30s...", flush=True)
        time.sleep(30)
        return None
    except anthropic.APIStatusError as e:
        print(f"  ⚠ API error {e.status_code}: {e.message[:80]}", flush=True)
        return None
    except Exception as e:
        print(f"  ⚠ Unexpected error: {e}", flush=True)
        return None


def count_existing(path: Path) -> int:
    """Count valid JSONL lines already written."""
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
    return count

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Claude teacher data for Anthos SFT")
    parser.add_argument("--n",       type=int,  default=2000,
                        help="Number of examples to generate (default: 2000)")
    parser.add_argument("--out",     type=str,  default="data/teacher_conversations.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from existing file — skip already-written examples")
    parser.add_argument("--delay",   type=float, default=0.3,
                        help="Seconds between API calls (default: 0.3) — increase if rate-limited")
    parser.add_argument("--budget",  type=float, default=None,
                        help="Stop early if cost exceeds this many USD (e.g. --budget 2.00)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client  = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker()
    out     = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    already_done = 0
    if args.resume and out.exists():
        already_done = count_existing(out)
        print(f"  ↩ Resuming — {already_done} examples already written, skipping them.")

    target    = args.n
    remaining = target - already_done
    if remaining <= 0:
        print(f"  ✓ Already have {already_done} examples ≥ target {target}. Nothing to do.")
        return

    print(f"\n{'─'*60}")
    print(f"  Claude-as-Teacher Data Generator")
    print(f"  Model:   claude-haiku-4-5")
    print(f"  Target:  {target} examples ({remaining} to generate)")
    print(f"  Output:  {out}")
    print(f"  Estimate: ~${remaining * 0.0008:.2f} USD")
    print(f"{'─'*60}\n")

    # Build a shuffled topic list long enough to fill remaining examples
    topic_pool = []
    while len(topic_pool) < remaining:
        shuffled = TOPICS[:]
        random.shuffle(shuffled)
        topic_pool.extend(shuffled)
    topic_pool = topic_pool[:remaining]

    write_mode = "a" if (args.resume and already_done > 0) else "w"
    generated  = 0
    failed     = 0
    t0         = time.time()

    with open(out, write_mode, encoding="utf-8") as f:
        for i, topic in enumerate(topic_pool):
            if args.budget and tracker.cost_usd >= args.budget:
                print(f"\n  💰 Budget ${args.budget:.2f} reached — stopping early.")
                break

            example = generate_example(client, topic, tracker)
            if example is None:
                failed += 1
                continue

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if generated % 50 == 0 or generated == 1:
                elapsed  = time.time() - t0
                rate     = generated / elapsed if elapsed > 0 else 0
                eta_secs = (remaining - generated) / rate if rate > 0 else 0
                eta_min  = eta_secs / 60
                print(
                    f"  [{generated:4d}/{remaining}]  {tracker.report()}  "
                    f"ETA {eta_min:.1f}m",
                    flush=True,
                )

            time.sleep(args.delay)

    total_written = already_done + generated
    print(f"\n{'─'*60}")
    print(f"  ✓ Done!")
    print(f"  Generated: {generated} new examples  (failed: {failed})")
    print(f"  Total in file: {total_written}")
    print(f"  {tracker.report()}")
    print(f"  Saved to: {out}")

    # Shuffle in-place so the model doesn't learn temporal patterns
    # from the order examples were generated (topic clusters, etc.)
    print(f"  Shuffling {total_written} examples...", end=" ", flush=True)
    with open(out, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    random.shuffle(lines)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("done ✓")

    print(f"{'─'*60}")
    print(f"\n  Next step — train Anthos on this data:")
    print(f"    python train.py --tier convo_smoke --resume checkpoints/mansa_sovereign/step_001700.pt")
    print()


if __name__ == "__main__":
    main()
