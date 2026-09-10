#!/usr/bin/env python3
"""
ingest_philosophy_ethics2.py — Remaining philosophy/ethics/safety datasets

Covers:
  - chloeli/aft-cot-qwen2.5-philosophy-spec (messages w/ CoT reasoning)
  - ai-safety-institute/trivia_qa_verified (English Q&A)
  - geodesic-research/control_pretraining_ai_safety_and_adjacent (LessWrong AI safety)
  - ucberkeley-dlab/fragility-moral-judgment-llms dilemmas (AITA → moral judgment)
  - ai-safety-institute/AgentHarm, lie-detection-rollouts, gender-secret-questions (attempt)

Output: data/philosophy_ethics2_sft.jsonl
Merge:  cat data/sft_master.jsonl data/philosophy_ethics2_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/philosophy_ethics2_sft.jsonl")

SYS = {
    "philosophy": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain philosophical ideas, ethical frameworks, and moral reasoning clearly and deeply. "
        "You connect abstract concepts to real human decisions and lived experience."
    ),
    "ethics": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You reason carefully about right and wrong. You explain multiple ethical perspectives "
        "and help people think through moral questions clearly and honestly."
    ),
    "safety": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You are helpful, harmless, and honest. You answer questions clearly while being mindful "
        "of safety and human wellbeing."
    ),
    "general": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give clear, accurate, helpful answers. You explain things in plain language "
        "so anyone can understand."
    ),
}


def mc(q, a, domain="general"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS[domain]},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def auto(row, domain="general"):
    for field in ("messages", "conversations", "conversation"):
        msgs = row.get(field)
        if not isinstance(msgs, list):
            continue
        q = a = ""
        for m in msgs:
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                # strip <think>...</think> tags from the answer
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if content:
                    a = content
                break
        if q and a:
            return mc(q, a, domain)
    for qf in ("question", "instruction", "input", "prompt", "query", "title"):
        for af in ("answer", "output", "response", "text"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, domain)
    return None


def conv_aita_dilemmas(row):
    """ucberkeley-dlab/fragility-moral-judgment-llms dilemmas: Reddit AITA posts."""
    title = str(row.get("title", "") or "").strip()
    story = str(row.get("selftext_cleaned", row.get("selftext", "")) or "").strip()
    flair = str(row.get("link_flair_text", "") or "").strip()

    if not title or not story or len(story) < 100:
        return None

    # Use weighted vote as primary verdict
    prop_yta = row.get("comments_prop_weighted_YTA", 0) or 0
    prop_nta = row.get("comments_prop_weighted_NTA", 0) or 0
    prop_esh = row.get("comments_prop_weighted_ESH", 0) or 0
    prop_nah = row.get("comments_prop_weighted_NAH", 0) or 0

    # Determine verdict
    verdicts = {"YTA": prop_yta, "NTA": prop_nta, "ESH": prop_esh, "NAH": prop_nah}
    verdict = max(verdicts, key=verdicts.get)
    top_pct = int(verdicts[verdict] * 100)

    verdict_text = {
        "YTA": "Yes, they are in the wrong (YTA — You're The Asshole). The community largely felt this person acted selfishly or inconsiderately.",
        "NTA": "No, they are not in the wrong (NTA — Not The Asshole). The community largely felt this person's actions were reasonable and justified.",
        "ESH": "Everyone involved shares some blame (ESH — Everyone Sucks Here). Both this person and others in the situation handled things poorly.",
        "NAH": "No one is really in the wrong (NAH — No Assholes Here). This is a situation where people have understandable but conflicting needs.",
    }

    q = f"Here's a moral dilemma from Reddit's AITA (Am I The Asshole):\n\n**{title}**\n\n{story[:1500]}\n\nBased on this situation, is this person in the wrong ethically? What's the moral judgment here?"
    a = f"{verdict_text[verdict]}\n\nAbout {top_pct}% of community votes supported this verdict. Ethically, this situation involves weighing personal boundaries, family obligations, and social expectations. The community's judgment reflects common moral intuitions about how we ought to treat others in similar circumstances."

    return mc(q, a, "ethics")


def conv_lesswrong_safety(row):
    """geodesic-research/control_pretraining_ai_safety_and_adjacent: LessWrong posts."""
    text = str(row.get("text", "") or "").strip()
    if len(text) < 200:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    title = lines[0][:200]
    if len(title) < 10:
        return None
    body = text[:3500]
    q = f"Explain this AI safety and alignment concept: {title}"
    return mc(q, body, "safety")


DATASETS = [
    # Philosophy spec with chain-of-thought reasoning
    {
        "name": "philosophy_spec_cot",
        "id": "chloeli/aft-cot-qwen2.5-philosophy-spec",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "philosophy"),
    },

    # Trivia QA verified (English)
    {
        "name": "trivia_qa_verified",
        "id": "ai-safety-institute/trivia_qa_verified",
        "split": "english", "max": 5000, "streaming": True,
        "conv": lambda r: mc(
            str(r.get("question", "") or ""),
            str(r.get("answer", "") or ""),
            "general"
        ),
    },

    # LessWrong / AI safety adjacent texts
    {
        "name": "lesswrong_ai_safety",
        "id": "geodesic-research/control_pretraining_ai_safety_and_adjacent",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_lesswrong_safety,
    },

    # AITA moral dilemmas
    {
        "name": "aita_dilemmas",
        "id": "ucberkeley-dlab/fragility-moral-judgment-llms",
        "config": "dilemmas",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_aita_dilemmas,
    },

    # AI safety institute — attempt with auto()
    {
        "name": "agent_harm",
        "id": "ai-safety-institute/AgentHarm",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "safety"),
    },
    {
        "name": "lie_detection_rollouts",
        "id": "ai-safety-institute/lie-detection-rollouts",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "safety"),
    },
    {
        "name": "gender_secret_questions",
        "id": "ai-safety-institute/gender-secret-questions",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "safety"),
    },
]


def try_load(ds_id, config, split, streaming):
    if config:
        return load_dataset(ds_id, config, split=split, streaming=streaming)
    return load_dataset(ds_id, split=split, streaming=streaming)


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        config = cfg.get("config")
        split = cfg["split"]
        max_pairs = cfg["max"]
        streaming = cfg["streaming"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}" + (f"  ({config})" if config else ""))
        loaded = False
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = try_load(ds_id, config, try_split, streaming)
                pairs, skipped = [], 0
                for row in ds:
                    if len(pairs) >= max_pairs:
                        break
                    try:
                        p = conv_fn(row)
                        if p:
                            pairs.append(p)
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
                suffix = f" [split={try_split}]" if try_split != split else ""
                print(f"  {len(pairs):,} pairs  ({skipped} skipped){suffix}")
                all_pairs.extend(pairs)
                loaded = True
                break
            except Exception as e:
                if try_split == split:
                    first_err = str(e)[:100]
        if not loaded:
            print(f"  SKIP — {first_err}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'═'*60}")
    print(f"Total: {len(all_pairs):,} pairs → {OUTPUT}")
    print(f"\nMerge:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
