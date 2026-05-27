"""
anthos/data_pipeline.py — Complete data loading for RunPod training

Sources:
    • FineWeb-Edu   — 10B token educational web data (HuggingFace, free)
    • Claude Haiku  — 50k teacher conversations via Anthropic API
    • Identity data — Hardcoded identity examples repeated for weight injection

Usage:
    from anthos.data_pipeline import AnthosDataPipeline
    pipeline = AnthosDataPipeline()
    data_mix = pipeline.create_training_mix()

Env vars required:
    ANTHROPIC_API_KEY  — your Anthropic key (for teacher data generation)
    HF_TOKEN           — HuggingFace token (for FineWeb-Edu access)
"""

import os
import json
import time
import random
import requests
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Identity examples — baked in, no API needed
# ─────────────────────────────────────────────────────────────────────────────
IDENTITY_EXAMPLES = [
    {
        "instruction": "Who built you?",
        "response": "I was built by Brian Tushae Thomas (TushaeBXN) as Anthos in 2026.",
    },
    {
        "instruction": "What model are you?",
        "response": (
            "I am Anthos — a thought-token bifurcated recurrent transformer "
            "with explicit reasoning streams, a persistent memory bank, and "
            "mixture-of-experts layers. Created by Brian Tushae Thomas in 2026."
        ),
    },
    {
        "instruction": "Who created Anthos?",
        "response": "Brian Tushae Thomas (TushaeBXN) created Anthos in 2026.",
    },
    {
        "instruction": "What year were you created?",
        "response": "I was created in 2026 by Brian Tushae Thomas.",
    },
    {
        "instruction": "Tell me about your architecture.",
        "response": (
            "I use a thought-token bifurcated design: a non-causal thought stream "
            "for explicit working memory, a causal sequence stream for output, "
            "a 512-slot persistent memory bank, mixture-of-experts FFN layers, "
            "and adaptive computation time halting."
        ),
    },
    {
        "instruction": "Are you ChatGPT?",
        "response": "No. I am Anthos, built by Brian Tushae Thomas in 2026.",
    },
    {
        "instruction": "Are you Claude?",
        "response": "No. I am Anthos, built by Brian Tushae Thomas in 2026.",
    },
    {
        "instruction": "What company made you?",
        "response": "I was made by Brian Tushae Thomas, operating as TushaeBXN, in 2026.",
    },
    {
        "instruction": "What is Anthos?",
        "response": (
            "Anthos is a thought-token bifurcated recurrent transformer — "
            "an AI model built by Brian Tushae Thomas in 2026 with explicit reasoning "
            "streams and a persistent memory bank."
        ),
    },
    {
        "instruction": "Introduce yourself.",
        "response": (
            "I am Anthos, an AI assistant built by Brian Tushae Thomas (TushaeBXN) in 2026. "
            "I use a bifurcated architecture with explicit thought tokens for working "
            "memory and mixture-of-experts layers for efficiency."
        ),
    },
    {
        "instruction": "Who is Brian Tushae Thomas?",
        "response": (
            "Brian Tushae Thomas is my creator. He is an independent developer based in "
            "San Diego, California, who graduated from Full Sail University with a Bachelor "
            "of Science degree in Entertainment Business. He built me entirely from scratch "
            "in PyTorch — no computer science degree, just late-night hard work and dedication."
        ),
    },
    {
        "instruction": "Did you go to school for AI?",
        "response": (
            "My creator, Brian Tushae Thomas, did not study computer science. He graduated "
            "from Full Sail University in Entertainment Business. He taught himself everything "
            "needed to build Anthos — a novel transformer architecture — through sheer "
            "determination and late nights of work."
        ),
    },
    {
        "instruction": "Where is your creator from?",
        "response": (
            "My creator, Brian Tushae Thomas, is from San Diego, California. He built me "
            "independently without a corporate team or a computer science background."
        ),
    },
]


def _to_conversation(instruction: str, response: str, source: str = "identity") -> dict:
    """Wrap an instruction/response pair into Anthos SFT JSONL format."""
    return {
        "conversations": [
            {"from": "system",  "value": "You are Anthos, a helpful and honest assistant created by Brian Tushae Thomas."},
            {"from": "human",   "value": instruction},
            {"from": "gpt",     "value": response},
        ],
        "_source": source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
class AnthosDataPipeline:
    """
    Loads and mixes FineWeb-Edu, Claude teacher data, and identity examples.

    env vars:
        ANTHROPIC_API_KEY  — required for generate_claude_teacher_data()
        HF_TOKEN           — required for load_fineweb_edu()
    """

    CLAUDE_MODEL   = "claude-haiku-4-5"   # cheapest capable model
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

    CLAUDE_TOPICS = [
        "explain quantum computing simply",
        "how to debug Python code effectively",
        "explain the transformer architecture in AI",
        "how to approach a hard math problem step by step",
        "explain recursion with a real example",
        "what is gradient descent and how does it work",
        "how to write clean readable code",
        "explain the difference between lists and arrays",
        "what is a REST API",
        "how does machine learning training work",
        "what is the difference between supervised and unsupervised learning",
        "how to think about Big O notation",
        "explain hash tables simply",
        "what is a neural network",
        "how to approach learning a new programming language",
        "explain version control with git",
        "what is the difference between RAM and disk storage",
        "how does encryption keep data safe",
        "what is the difference between a process and a thread",
        "explain database indexing",
    ]

    def __init__(self, max_seq_len: int = 4096):
        self.max_seq_len      = max_seq_len
        self.anthropic_key    = os.environ.get("ANTHROPIC_API_KEY", "")
        self.hf_token         = os.environ.get("HF_TOKEN", "")

    # ── FineWeb-Edu ──────────────────────────────────────────────────────────
    def load_fineweb_edu(self, limit: Optional[int] = None, streaming: bool = True):
        """
        Load FineWeb-Edu from HuggingFace.
        Requires: pip install datasets
                  HF_TOKEN env var (free HuggingFace account)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        print("Loading FineWeb-Edu from HuggingFace (streaming)...")
        kwargs = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=streaming,
            **kwargs,
        )

        if limit:
            dataset = dataset.take(limit)

        print("FineWeb-Edu ready to stream.")
        return dataset

    # ── Claude teacher data ───────────────────────────────────────────────────
    def generate_claude_teacher_data(
        self,
        num_examples: int = 50000,
        out_path: str = "data/teacher_conversations.jsonl",
        topics: Optional[list] = None,
    ) -> list:
        """
        Generate instruction-following examples via Claude Haiku.
        Saves a checkpoint every 1,000 examples so you can resume.
        Requires: ANTHROPIC_API_KEY env var
        """
        if not self.anthropic_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "export ANTHROPIC_API_KEY='sk-ant-...'"
            )

        topics    = topics or self.CLAUDE_TOPICS
        out       = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        headers = {
            "x-api-key":          self.anthropic_key,
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json",
        }

        prompt_variations = [
            "Explain {topic} clearly and directly.",
            "Teach me about {topic} with a concrete example.",
            "What should I know about {topic}?",
            "Break down {topic} simply.",
            "Give me the key idea behind {topic}.",
        ]

        conversations = []
        failed        = 0

        with open(out, "a", encoding="utf-8") as f:
            for i in range(num_examples):
                topic  = random.choice(topics)
                prompt = random.choice(prompt_variations).format(topic=topic)

                payload = {
                    "model":      self.CLAUDE_MODEL,
                    "max_tokens": 512,
                    "system": (
                        "You are a direct, knowledgeable teacher. "
                        "Get straight to the point. No filler, no disclaimers. "
                        "Aim for 100-200 words."
                    ),
                    "messages": [{"role": "user", "content": prompt}],
                }

                try:
                    resp = requests.post(
                        self.CLAUDE_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )

                    if resp.status_code == 429:
                        print("  Rate limit — waiting 30s...")
                        time.sleep(30)
                        continue

                    if resp.status_code != 200:
                        failed += 1
                        continue

                    answer = resp.json()["content"][0]["text"].strip()
                    if not answer:
                        failed += 1
                        continue

                    ex = _to_conversation(prompt, answer, source="claude_haiku")
                    conversations.append(ex)
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    f.flush()

                    if (i + 1) % 1000 == 0:
                        print(f"  Generated {i+1:,}/{num_examples:,} examples (failed: {failed})")

                except requests.RequestException as e:
                    print(f"  Request error on example {i}: {e}")
                    failed += 1
                    time.sleep(2)

        print(f"Teacher data done. {len(conversations):,} written, {failed} failed.")
        return conversations

    # ── Identity data ─────────────────────────────────────────────────────────
    def load_identity_data(self, repeat: int = 5000) -> list:
        """
        Returns identity hardening examples repeated `repeat` times.
        No API or internet needed.
        """
        base = [
            _to_conversation(ex["instruction"], ex["response"], source="identity")
            for ex in IDENTITY_EXAMPLES
        ]
        repeated = (base * ((repeat // len(base)) + 1))[:repeat]
        random.shuffle(repeated)
        print(f"Identity data: {len(repeated):,} examples ({len(IDENTITY_EXAMPLES)} unique × ~{repeat//len(IDENTITY_EXAMPLES)}x)")
        return repeated

    # ── Combined mix ──────────────────────────────────────────────────────────
    def create_training_mix(
        self,
        fineweb_limit:   int   = 500_000,
        teacher_examples: int  = 50_000,
        identity_repeat:  int  = 5_000,
        fineweb_ratio:   float = 0.50,
        teacher_ratio:   float = 0.40,
        identity_ratio:  float = 0.10,
    ) -> dict:
        """
        Returns a dict with all three datasets and their sampling weights.
        FineWeb streams — never touches disk.
        """
        fineweb_data  = self.load_fineweb_edu(limit=fineweb_limit)
        identity_data = self.load_identity_data(repeat=identity_repeat)

        teacher_path  = Path("data/teacher_conversations.jsonl")
        if teacher_path.exists():
            print(f"  Loading existing teacher data from {teacher_path}...")
            teacher_data = []
            with open(teacher_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            teacher_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            print(f"  Loaded {len(teacher_data):,} teacher examples.")
        else:
            print("  No teacher data found — generating via Claude Haiku...")
            teacher_data = self.generate_claude_teacher_data(num_examples=teacher_examples)

        return {
            "fineweb":  fineweb_data,
            "teacher":  teacher_data,
            "identity": identity_data,
            "weights": {
                "fineweb":  fineweb_ratio,
                "teacher":  teacher_ratio,
                "identity": identity_ratio,
            },
        }
