#!/usr/bin/env python3
"""
ingest_culture_governance.py — Culture, Global Awareness, Governance

Covers:
  - potsawee/cultural_awareness_mcq (global MCQ)
  - Svngoku/african_cultural_reasoning (African culture Q&A with reasoning)
  - JesusAura999/Senik_CulturalAwareness_Dataset100K (cultural awareness conversations)
  - ombhojane/indian-cultural-dataset (Indian mythology, traditions)
  - llm-for-emotion/Cultural-Emo eng (cultural emotion Q&A)
  - Ameeeee/Cultural_diversity (diverse topics, best model response)
  - AYI-NEDJIMI/ai-governance-en (AI governance frameworks)
  - Amr04/UnitedNations-ParagraphsAlligned-ar-en-dataset (UN English text)
  - infinite-dataset-hub/CulturalIndicators (cultural traditions)

Output: data/culture_governance_sft.jsonl
Merge:  cat data/sft_master.jsonl data/culture_governance_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/culture_governance_sft.jsonl")
CHOICE_LABELS = list("ABCDE")

SYS = {
    "culture": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain cultures, traditions, and global perspectives with depth and respect. "
        "You celebrate human diversity and help people understand one another across cultural differences."
    ),
    "governance": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain governance, policy, and institutional frameworks clearly. "
        "You help people understand how systems of power, law, and accountability work."
    ),
    "general": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You give clear, accurate, helpful answers in plain language so anyone can understand."
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
            if role == "system":
                continue
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if content:
                    a = content
                break
        if q and a:
            return mc(q, a, domain)
    for qf in ("question", "input", "prompt", "query", "title", "My topics"):
        for af in ("answer", "output", "response", "Llama70B"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, domain)
    return None


def conv_cultural_mcq(row):
    """potsawee/cultural_awareness_mcq: question + options + answer_text + category."""
    q = str(row.get("question", "") or "").strip()
    options = row.get("options", [])
    answer_text = str(row.get("answer_text", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    if not q or not answer_text:
        return None
    if isinstance(options, list) and len(options) > 1:
        opts = "\n".join(f"{CHOICE_LABELS[i]}. {o}" for i, o in enumerate(options) if i < 5)
        question = f"[{category}] {q}\n\n{opts}" if category else f"{q}\n\n{opts}"
        correct_idx = options.index(answer_text) if answer_text in options else 0
        a = f"{CHOICE_LABELS[correct_idx]}. {answer_text}"
    else:
        question = f"[{category}] {q}" if category else q
        a = answer_text
    return mc(question, a, "culture")


def conv_african_reasoning(row):
    """Svngoku/african_cultural_reasoning: question + reasoning_steps + final_answer + cultural_context."""
    q = str(row.get("question", "") or "").strip()
    steps = row.get("reasoning_steps", [])
    final = str(row.get("final_answer", "") or "").strip()
    context = str(row.get("cultural_context", "") or "").strip()
    topic = str(row.get("topic", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    if not q or not final:
        return None
    a_parts = []
    if isinstance(steps, list) and steps:
        reasoning = " ".join(str(s) for s in steps)
        a_parts.append(reasoning[:1200])
    a_parts.append(f"\n\n**Answer:** {final}")
    if context:
        a_parts.append(f"\n\n**Cultural context:** {context}")
    a = "".join(a_parts)
    if topic:
        q = f"[African Culture — {topic.title()}] {q}"
    return mc(q, a, "culture")


def conv_senik_cultural(row):
    """JesusAura999/Senik_CulturalAwareness_Dataset100K: conversations format."""
    p = auto(row, "culture")
    if p:
        return p
    inp = str(row.get("input", "") or "").strip()
    resp = str(row.get("response", "") or "").strip()
    if inp and resp and len(resp) > 20:
        # Strip artifact markers like [cult gen 1]
        inp = re.sub(r'\[cult\s+\w+\s+\d+\]', '', inp).strip()
        resp = re.sub(r'\[cult\s+\w+\s+\d+\]', '', resp).strip()
        return mc(inp, resp, "culture")
    return None


def conv_indian_cultural(row):
    """ombhojane/indian-cultural-dataset: text + category → explain."""
    text = str(row.get("text", "") or "").strip()
    category = str(row.get("category", "") or "").strip()
    if len(text) < 80:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    # First line after "Category: X\nContent: Title" is often the topic
    content_start = text.find("Content:")
    if content_start != -1:
        body = text[content_start + 8:].strip()
        topic_line = body.splitlines()[0].strip()[:150] if body else lines[0]
    else:
        topic_line = lines[0][:150]
    q = f"Tell me about this Indian cultural topic: {topic_line}"
    a = text[:3500]
    return mc(q, a, "culture")


def conv_cultural_diversity(row):
    """Ameeeee/Cultural_diversity: My topics + Llama70B (best available answer)."""
    q = str(row.get("My topics", "") or "").strip()
    # Use Llama70B as primary, strip <think> tags from others
    for field in ("Llama70B", "Qwen/QwQ-32B ", "Perplexity / R1"):
        a = str(row.get(field, "") or "").strip()
        a = re.sub(r'<think>.*?</think>', '', a, flags=re.DOTALL).strip()
        if a and len(a) > 30 and a.lower() != "i am unable to provide an answer to that question.":
            return mc(q, a[:3500], "general")
    return None


def conv_ai_governance(row):
    """AYI-NEDJIMI/ai-governance-en: name + content + details → structured Q&A."""
    name = str(row.get("name", "") or "").strip()
    content = str(row.get("content", "") or "").strip()
    details = str(row.get("details", "") or "").strip()
    refs = str(row.get("regulatory_reference", "") or "").strip()
    tools = str(row.get("tools", "") or "").strip()
    if not name or not content:
        return None
    q = f"What is '{name}' in AI governance, and how should it be implemented?"
    a_parts = [content[:1500]]
    if details:
        a_parts.append(f"\n\n**Implementation steps:**\n{details[:800]}")
    if refs:
        a_parts.append(f"\n\n**Regulatory references:** {refs[:300]}")
    if tools:
        a_parts.append(f"\n\n**Tools:** {tools[:200]}")
    return mc(q, "".join(a_parts), "governance")


def conv_un_aligned(row):
    """Amr04/UnitedNations-ParagraphsAlligned-ar-en-dataset: use English side."""
    en = str(row.get("en", "") or "").strip()
    if len(en) < 80:
        return None
    lines = [l.strip() for l in en.splitlines() if l.strip()]
    topic = lines[0][:200] if lines else en[:100]
    q = f"Explain this UN policy or legal principle: {topic}"
    return mc(q, en[:3500], "governance")


def conv_cultural_indicators(row):
    """infinite-dataset-hub/CulturalIndicators: id (description) + text_prompt + labels."""
    description = str(row.get("id", "") or "").strip()
    prompt = str(row.get("text_prompt", "") or "").strip()
    labels = str(row.get("labels", "") or "").strip()
    if not description or len(description) < 10:
        return None
    q = f"What cultural tradition or practice does this describe: {description}?"
    a = f"This represents {prompt.lower() if prompt else 'a cultural practice'}"
    if labels:
        a += f" associated with {labels}."
    a += f" Specifically: {description}. Cultural traditions like this reflect the values, beliefs, and history of the communities that practice them, connecting generations through shared meaning and identity."
    return mc(q, a, "culture")


DATASETS = [
    {
        "name": "cultural_awareness_mcq",
        "id": "potsawee/cultural_awareness_mcq",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_cultural_mcq,
    },
    {
        "name": "african_cultural_reasoning",
        "id": "Svngoku/african_cultural_reasoning",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_african_reasoning,
    },
    {
        "name": "senik_cultural_awareness",
        "id": "JesusAura999/Senik_CulturalAwareness_Dataset100K",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_senik_cultural,
    },
    {
        "name": "indian_cultural",
        "id": "ombhojane/indian-cultural-dataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_indian_cultural,
    },
    {
        "name": "cultural_emo_english",
        "id": "llm-for-emotion/Cultural-Emo",
        "config": "eng",
        "split": "test", "max": 3000, "streaming": True,
        "conv": lambda r: auto(r, "culture"),
    },
    {
        "name": "cultural_diversity_topics",
        "id": "Ameeeee/Cultural_diversity",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_cultural_diversity,
    },
    {
        "name": "ai_governance_en",
        "id": "AYI-NEDJIMI/ai-governance-en",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_ai_governance,
    },
    {
        "name": "un_paragraphs_en",
        "id": "Amr04/UnitedNations-ParagraphsAlligned-ar-en-dataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_un_aligned,
    },
    {
        "name": "cultural_indicators",
        "id": "infinite-dataset-hub/CulturalIndicators",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_cultural_indicators,
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
        first_err = ""
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
                if not first_err:
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
