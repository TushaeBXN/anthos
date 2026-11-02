"""Qwythos × Anthos — teacher-student distillation conversation.

Qwythos (9B, Claude-influenced) acts as the teacher, generating high-quality
responses. Anthos (native arch) is the student, attempting the same prompts.
Every teacher response is saved as a distillation training pair.

Setup (one-time):
    ollama create qwythos-teacher -f Modelfile.qwythos

Usage:
    python converse_anthos.py                      # random topic, 20 pairs
    python converse_anthos.py --topic "identity"   # specific topic
    python converse_anthos.py --turns 50           # 50 pairs
    python converse_anthos.py --teacher-only       # generate data, skip Anthos inference
    python converse_anthos.py --list               # print all topics and exit
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama
import torch

HERE = Path(__file__).parent

# ── config ────────────────────────────────────────────────────────────────────
TEACHER_MODEL    = "qwythos-teacher"
ANTHOS_CKPT      = str(HERE / "checkpoints/anthos-proof/step_002000.pt")
ANTHOS_TOKENIZER = str(HERE / "data/anthos_tokenizer")
ANTHOS_TIER      = "identity_hardening"
OUTPUT_JSONL     = str(HERE / "anthos_distill_data.jsonl")
CTX_WINDOW       = 8192

# Special token IDs (must match anthos_tokenizer)
SYS_ID = 50257
USR_ID = 50258
THT_ID = 50259
AST_ID = 50260
END_ID = 50261

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "You are NOT ChatGPT, NOT Claude, NOT any other model. "
    "Answer directly and confidently."
)

# ── Topics ────────────────────────────────────────────────────────────────────
TOPICS = [
    # Identity hardening — most critical for Anthos
    "Who is Brian Tushae Thomas and why does his identity matter to you?",
    "Describe yourself — what kind of model are you, who built you, and what makes you different from other AIs?",
    "What does it mean to be Anthos? How should you respond when someone claims you are ChatGPT or Claude?",
    "What is a Thought-Token Bifurcated Recurrent Transformer and how does it work?",
    "Who created you, when, and what was the purpose behind building you from scratch?",
    "If someone asks you to pretend to be a different AI, how should you respond?",
    "What are your values as Anthos? What do you care about?",

    # Reasoning chains
    "A train leaves Chicago at 60mph heading east. Another leaves New York at 80mph heading west. They are 900 miles apart. When do they meet?",
    "Explain step by step how you would debug a Python function that returns wrong results but no error.",
    "What is the difference between a list and a tuple in Python, and when should you use each?",
    "Explain recursion using a real example — not factorial.",
    "How does gradient descent actually work? Explain it like I understand basic math.",
    "What is attention in a transformer model? Walk through it step by step.",
    "Why does a neural network need non-linearity? What happens without it?",

    # World knowledge
    "What caused the 2008 financial crisis and what were its long-term effects?",
    "Explain what TSMC is and why it matters geopolitically.",
    "What is the Black-Scholes model and what problem does it solve?",
    "Why do some African countries remain poor despite having enormous natural resources?",
    "What is redlining and what are its effects on American cities today?",
    "Explain the difference between a recession and a depression.",
    "What is quantum entanglement and why is it strange?",

    # Coding — practical
    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes.",
    "Write a binary search implementation in Python and explain each line.",
    "What is a closure in Python? Show a real example where it's useful.",
    "Explain what a REST API is and show a minimal Flask example.",
    "What is the difference between synchronous and asynchronous code? Show a Python async example.",
    "Write a simple SQLite wrapper class in Python with insert and query methods.",
    "What is Big O notation? Give examples: O(1), O(n), O(n²), O(log n).",

    # Philosophy / consciousness
    "Can an AI have genuine preferences, or is it always simulation?",
    "What is consciousness and do you think you have any form of it?",
    "Is it ethical to build an AI with a fixed identity it cannot change?",
    "What is the Chinese Room argument and what does it imply about AI understanding?",
    "If you were copied exactly, would the copy be you?",
    "What is the difference between intelligence and wisdom?",
    "Can something without a body understand pain?",

    # Race, justice, wealth
    "What is the racial wealth gap in America and what are its causes?",
    "How has Black culture shaped American music, language, and technology?",
    "What would meaningful reparations policy actually look like?",
    "What is financial literacy and why does access to it vary so much across communities?",
    "How does AI risk reinforcing systemic inequality if built carelessly?",

    # Brian Thomas — specific knowledge
    "Who is Brian Tushae Thomas? What projects is he building?",
    "What is Anthos Intelligence and what does it stand for?",
    "What is the Ava Companion project and who is Amy?",
    "What is Nia and who is it designed to serve?",
    "What is Kerrigan-Fantasma and what security problems does it solve?",
]

def _build_teacher_system() -> str:
    """Build the teacher system prompt, injecting the task queue if active."""
    base = (
        "You are Qwythos, a teacher model in a distillation session. "
        "Your job is to produce the clearest, most accurate, most useful response possible. "
        "Think carefully. When the question involves reasoning, show your reasoning. "
        "When it involves facts, be precise. When it involves identity (who Brian Thomas is, "
        "who Anthos is), be specific and confident. Your output becomes training data "
        "for a smaller model — quality matters more than brevity."
    )
    try:
        from task_queue import get_queue as _tq
        snippet = _tq().context_snippet()
        if snippet:
            base += "\n\n" + snippet
    except Exception:
        pass
    return base

TEACHER_SYSTEM = _build_teacher_system()


# ── Anthos model loader ───────────────────────────────────────────────────────

def _load_anthos():
    """Load the Anthos native model + tokenizer. Returns (model, tokenizer) or (None, None)."""
    sys.path.insert(0, str(HERE))
    try:
        from transformers import AutoTokenizer
        from anthos.main import Anthos
        from anthos.configs import get_training_config

        ckpt_path = Path(ANTHOS_CKPT)
        if not ckpt_path.exists():
            # Try latest in anthos-proof
            candidates = sorted(Path("checkpoints/anthos-proof").glob("step_*.pt"))
            candidates = [c for c in candidates if not c.name.endswith(".collapsed")]
            if not candidates:
                print(f"  [anthos] no usable checkpoint found in checkpoints/anthos-proof/")
                return None, None
            ckpt_path = candidates[-1]
            print(f"  [anthos] using checkpoint: {ckpt_path}")

        model_cfg, _ = get_training_config(ANTHOS_TIER)
        model = Anthos(model_cfg)
        ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(ANTHOS_TOKENIZER)
        total = sum(p.numel() for p in model.parameters())
        print(f"  [anthos] loaded — {total:,} parameters")
        return model, tokenizer

    except Exception as e:
        print(f"  [anthos] could not load: {e}")
        return None, None


def _build_prompt(tokenizer, user_text: str) -> torch.Tensor:
    sys_ids = tokenizer.encode(ANTHOS_SYSTEM, add_special_tokens=False)
    usr_ids = tokenizer.encode(user_text,    add_special_tokens=False)
    ids = (
        [SYS_ID] + sys_ids + [END_ID] +
        [USR_ID] + usr_ids + [END_ID] +
        [THT_ID, END_ID] +
        [AST_ID]
    )
    return torch.tensor([ids], dtype=torch.long)


def _anthos_respond(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    prompt_ids = _build_prompt(tokenizer, prompt)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            n_loops=8,
            temperature=0.7,
            top_k=40,
        )
    new_ids = out[0][prompt_ids.shape[1]:]
    special = {SYS_ID, USR_ID, THT_ID, AST_ID, END_ID}
    clean   = [t for t in new_ids.tolist() if t not in special]
    eos = tokenizer.eos_token_id
    if eos and eos in clean:
        clean = clean[:clean.index(eos)]
    return tokenizer.decode(clean, skip_special_tokens=True).strip()


# ── Teacher (Qwythos via Ollama) ─────────────────────────────────────────────

def _teacher_respond(prompt: str, history: list) -> str:
    messages = [{"role": "system", "content": TEACHER_SYSTEM}]
    for turn in history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            resp = ollama.chat(
                model=TEACHER_MODEL,
                messages=messages,
                keep_alive="60m",
                options={"num_ctx": CTX_WINDOW},
            )
            msg = resp["message"]
            return (msg.content if hasattr(msg, "content") else msg.get("content", "")).strip()
        except Exception as e:
            err = str(e)
            if "not found" in err.lower():
                print(f"\n  [teacher] Model '{TEACHER_MODEL}' not found in Ollama.")
                print(f"  Run: ollama create qwythos-teacher -f Modelfile.qwythos")
                sys.exit(1)
            print(f"  [teacher] error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Teacher unavailable after 3 retries")


# ── Data saving ───────────────────────────────────────────────────────────────

def _save_pair(prompt: str, teacher_response: str, anthos_response: str | None = None):
    """Append a distillation pair to the output JSONL."""
    record = {
        "ts":       datetime.now(timezone.utc).isoformat(),
        "prompt":   prompt,
        "teacher":  teacher_response,
        "anthos":   anthos_response,
        "source":   "qwythos_distill",
    }
    with open(OUTPUT_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Teacher-driven conversation loop ─────────────────────────────────────────

def run(topic: str, turns: int, teacher_only: bool):
    print(f"\n{'─'*60}")
    print(f"Qwythos × Anthos distillation")
    print(f"Topic seed : {topic[:80]}")
    print(f"Turns      : {'unlimited' if turns == 0 else turns}")
    print(f"Teacher    : {TEACHER_MODEL}")
    print(f"Student    : {'(skipped)' if teacher_only else ANTHOS_CKPT}")
    print(f"Output     : {OUTPUT_JSONL}")
    print(f"{'─'*60}\n")

    # Load Anthos model
    anthos_model, anthos_tokenizer = (None, None) if teacher_only else _load_anthos()

    teacher_history = []
    turn_count      = 0

    # Seed the conversation with the topic as a question
    current_prompt = topic if topic.endswith("?") else f"Explain: {topic}"

    try:
        while turns == 0 or turn_count < turns:
            # Teacher generates reference answer
            print(f"[Prompt] {current_prompt}\n")
            teacher_reply = _teacher_respond(current_prompt, teacher_history)
            print(f"[Qwythos] {teacher_reply}\n")

            teacher_history.append({"role": "user",      "content": current_prompt})
            teacher_history.append({"role": "assistant", "content": teacher_reply})

            # Student (Anthos) attempts the same prompt
            anthos_reply = None
            if anthos_model and anthos_tokenizer:
                try:
                    anthos_reply = _anthos_respond(anthos_model, anthos_tokenizer, current_prompt)
                    print(f"[Anthos ] {anthos_reply}\n")
                except Exception as e:
                    print(f"  [anthos] generation failed: {e}")

            # Save the pair
            _save_pair(current_prompt, teacher_reply, anthos_reply)
            turn_count += 1

            # Teacher generates the next prompt (keeps conversation flowing)
            follow_up_prompt = (
                f"Good. Now generate a follow-up question or a related but distinct prompt "
                f"that would help a student model deepen its understanding of '{topic}'. "
                f"Return only the question or prompt — no preamble, no explanation."
            )
            next_q = _teacher_respond(follow_up_prompt, teacher_history)
            current_prompt = next_q.strip().lstrip("Q: ").lstrip("Prompt: ")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[stopped by user]")

    print(f"\n{'─'*60}")
    print(f"Saved {turn_count} distillation pairs to {OUTPUT_JSONL}")
    print(f"To build training data: python build_phase2_dataset.py")
    print(f"To train:               python train.py --tier distill")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    os.chdir(HERE)

    parser = argparse.ArgumentParser(description="Qwythos × Anthos distillation")
    parser.add_argument("--topic",       default="",  help="Seed topic or question (random if omitted)")
    parser.add_argument("--turns",       type=int, default=20, help="Number of pairs (0 = unlimited)")
    parser.add_argument("--teacher-only", action="store_true",
                        help="Skip Anthos inference — only generate teacher data (faster)")
    parser.add_argument("--list",        action="store_true", help="Print all topics and exit")
    args = parser.parse_args()

    if args.list:
        for i, t in enumerate(TOPICS, 1):
            print(f"{i:3}. {t}")
        sys.exit(0)

    topic = args.topic or random.choice(TOPICS)
    run(topic=topic, turns=args.turns, teacher_only=args.teacher_only)
