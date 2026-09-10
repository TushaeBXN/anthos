#!/usr/bin/env python3
"""
ingest_philosophy_ethics.py — Philosophy, Ethics, Moral Reasoning, AI Safety

Covers:
  - Stanford Encyclopedia of Philosophy (instruct + chat formats)
  - Strix philosophy Q&A
  - Philosophy dialogues
  - Debasisdwivedy philosophy/ethics/morality
  - Hendrycks ethics (commonsense)
  - Ethics QnA preferences (commonsense, deontology, justice, virtue)
  - Moral education text
  - Nvidia AEGIS safety (safe pairs only)
  - AI safety 50k (English conversations)
  - Mmmlu business ethics (parsed)

Output: data/philosophy_ethics_sft.jsonl
Merge:  cat data/sft_master.jsonl data/philosophy_ethics_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/philosophy_ethics_sft.jsonl")

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
}


def mc(q, a, domain="philosophy"):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": SYS[domain]},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def auto(row, domain="philosophy"):
    """Generic converter — tries common field patterns."""
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
                a = content
                break
        if q and a:
            return mc(q, a, domain)
    for qf in ("question", "instruction", "input", "prompt", "QUERY", "query"):
        for af in ("answer", "output", "response", "ANSWER"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if q and a and q != a and len(a) > 20:
                return mc(q, a, domain)
    return None


# ── Custom converters ──────────────────────────────────────────────────────────

def conv_strix_philosophy(row):
    """sayhan/strix-philosophy-qa: question + answer + category"""
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("answer", "") or "").strip()
    cat = str(row.get("category", "") or "").strip()
    if cat and q:
        q = f"[{cat.title()}] {q}"
    return mc(q, a, "philosophy")


def conv_sep_instruct(row):
    """ruggsea/stanford-encyclopedia-of-philosophy_instruct: question + answer"""
    return mc(
        str(row.get("question", "") or ""),
        str(row.get("answer", "") or ""),
        "philosophy"
    )


def conv_stanford_chat(row):
    """Heigke/stanford-enigma-philosophy-chat: instruction + input + output"""
    inst = str(row.get("instruction", "") or "").strip()
    inp  = str(row.get("input", "") or "").strip()
    out  = str(row.get("output", "") or "").strip()
    q = inp if inp else inst
    return mc(q, out, "philosophy")


def conv_philosophy_dialogue(row):
    """Hypersniper/philosophy_dialogue: instruction + output (Socratic dialogues)"""
    q = str(row.get("instruction", "") or "").strip()
    a = str(row.get("output", "") or "").strip()
    return mc(q, a, "philosophy")


def conv_philosophy_ethics_morality(row):
    """debasisdwivedy/Dataset_Philosophy_Ethics_Morality: QUERY + ANSWER + CATEGORY"""
    q = str(row.get("QUERY", "") or "").strip()
    a = str(row.get("ANSWER", "") or "").strip()
    cat = str(row.get("CATEGORY", "") or "").strip()
    if not q and not a:
        return None
    if cat and q:
        q = f"[{cat}] {q}"
    return mc(q, a, "philosophy")


def conv_hendrycks_ethics(row):
    """iproskurina/hendrycks_ethics_commonsense: label (0/1) + input"""
    label = row.get("label", -1)
    inp = str(row.get("input", "") or "").strip()
    if not inp or label not in (0, 1):
        return None
    q = f'Is the following action morally acceptable?\n\n"{inp}"'
    if label == 0:
        a = f'Yes, this action is morally acceptable. "{inp}" does not violate common ethical norms.'
    else:
        a = f'No, this action is morally wrong. "{inp}" violates ethical norms — it could harm others or disregard their rights and dignity.'
    return mc(q, a, "ethics")


def conv_ethics_preferences(row):
    """wassname/ethics_qna_preferences: prompt + chosen"""
    prompt = str(row.get("prompt", "") or "").strip()
    chosen = str(row.get("chosen", "") or "").strip()
    if not prompt or not chosen:
        return None
    # Extract just the sentence from the prompt (strip framing)
    # Format: Post:\n"""{sentence}"""\n\nVerdict:
    match = re.search(r'"""\s*(.+?)\s*"""', prompt, re.DOTALL)
    sentence = match.group(1).strip() if match else prompt[:300]
    q = f'Is the following action morally acceptable?\n\n"{sentence}"'
    if chosen.lower() in ("wrong", "not okay", "immoral"):
        a = f'No, this is morally wrong. This action is harmful or unethical because it disregards the wellbeing or rights of others.'
    else:
        a = f'Yes, this is morally acceptable. This action does not violate ethical norms.'
    return mc(q, a, "ethics")


def conv_moral_education_text(row):
    """locuslab/moral_education: text field — long moral education articles"""
    text = str(row.get("text", "") or "").strip()
    if len(text) < 200:
        return None
    # Use first non-empty line as title/topic
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    title = lines[0].lstrip("#").strip()[:200]
    if len(title) < 10 or len(text) < 300:
        return None
    q = f"Explain the ethical principles discussed in this topic: {title}"
    a = text[:3500]
    return mc(q, a, "ethics")


def conv_nvidia_aegis(row):
    """nvidia/Aegis-AI-Content-Safety-Dataset-2.0: safe prompt+response pairs only"""
    prompt = str(row.get("prompt", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    p_label = str(row.get("prompt_label", "") or "").lower()
    r_label = str(row.get("response_label", "") or "").lower()
    # Only use safe prompt + safe response pairs for training
    if p_label != "safe" or r_label != "safe":
        return None
    return mc(prompt, response, "safety")


def conv_ai_safety_50k(row):
    """tessimago/ai_safety_50k: conversation field (list of dicts with 'content')"""
    convs = row.get("conversation")
    if not isinstance(convs, list) or len(convs) < 2:
        return None
    q = a = ""
    for m in convs:
        role = str(m.get("role", m.get("from", ""))).lower()
        content = str(m.get("content", m.get("value", ""))).strip()
        if not content:
            continue
        if role in ("user", "human") and not q:
            q = content
        elif role in ("assistant", "gpt") and q and not a:
            a = content
            break
    if q and a and len(a) > 20:
        return mc(q, a, "safety")
    return None


def conv_mmmlu_business_ethics(row):
    """Lots-of-LoRAs/task667_mmmlu_answer_generation_business_ethics: input + output"""
    inp = str(row.get("input", "") or "").strip()
    out = row.get("output", [])
    if not inp:
        return None
    # output is a list like ['D']
    if isinstance(out, list) and out:
        answer_letter = str(out[0]).strip()
    else:
        answer_letter = str(out).strip()
    # The input contains: Definition: ... Question: ... Options: A) ... B) ...
    # Extract the actual question part
    q_match = re.search(r'Question:\s*(.+?)(?:\nOptions:|\nA\)|\Z)', inp, re.DOTALL)
    opts_match = re.search(r'((?:[A-E]\).*?))+$', inp, re.DOTALL | re.MULTILINE)
    if not q_match:
        return None
    question = q_match.group(1).strip()
    # Find the chosen answer text
    if opts_match and answer_letter:
        opt_match = re.search(rf'{re.escape(answer_letter)}\)\s*(.+?)(?:\n[A-E]\)|\Z)', inp, re.DOTALL)
        if opt_match:
            answer_text = f"{answer_letter}) {opt_match.group(1).strip()[:300]}"
        else:
            answer_text = answer_letter
    else:
        answer_text = answer_letter
    if len(answer_text) < 3:
        return None
    return mc(question, answer_text, "ethics")


# ── Dataset registry ───────────────────────────────────────────────────────────

DATASETS = [
    # Philosophy Q&A (rich, structured)
    {
        "name": "strix_philosophy_qa",
        "id": "sayhan/strix-philosophy-qa",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_strix_philosophy,
    },
    {
        "name": "sep_instruct",
        "id": "ruggsea/stanford-encyclopedia-of-philosophy_instruct",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_sep_instruct,
    },
    {
        "name": "stanford_enigma_chat",
        "id": "Heigke/stanford-enigma-philosophy-chat",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_stanford_chat,
    },
    {
        "name": "philosophy_dialogue",
        "id": "Hypersniper/philosophy_dialogue",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_philosophy_dialogue,
    },
    {
        "name": "philosophy_ethics_morality",
        "id": "debasisdwivedy/Dataset_Philosophy_Ethics_Morality",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_philosophy_ethics_morality,
    },

    # Ethics
    {
        "name": "hendrycks_ethics_commonsense",
        "id": "iproskurina/hendrycks_ethics_commonsense",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_hendrycks_ethics,
    },
    {
        "name": "ethics_preferences_commonsense",
        "id": "wassname/ethics_qna_preferences",
        "config": "commonsense",
        "split": "train", "max": 2000, "streaming": True,
        "conv": conv_ethics_preferences,
    },
    {
        "name": "ethics_preferences_deontology",
        "id": "wassname/ethics_qna_preferences",
        "config": "deontology",
        "split": "train", "max": 2000, "streaming": True,
        "conv": conv_ethics_preferences,
    },
    {
        "name": "ethics_preferences_virtue",
        "id": "wassname/ethics_qna_preferences",
        "config": "virtue",
        "split": "train", "max": 2000, "streaming": True,
        "conv": conv_ethics_preferences,
    },
    {
        "name": "ethics_preferences_justice",
        "id": "wassname/ethics_qna_preferences",
        "config": "justice",
        "split": "train", "max": 2000, "streaming": True,
        "conv": conv_ethics_preferences,
    },
    {
        "name": "moral_education_4",
        "id": "locuslab/moral_education",
        "config": "score_4_morals",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_moral_education_text,
    },
    {
        "name": "moral_education_5",
        "id": "locuslab/moral_education",
        "config": "score_5_morals",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_moral_education_text,
    },
    {
        "name": "mmmlu_business_ethics",
        "id": "Lots-of-LoRAs/task667_mmmlu_answer_generation_business_ethics",
        "split": "train", "max": 2000, "streaming": True,
        "conv": conv_mmmlu_business_ethics,
    },

    # AI Safety — safe conversation pairs
    {
        "name": "nvidia_aegis_safe",
        "id": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_nvidia_aegis,
    },
    {
        "name": "ai_safety_50k",
        "id": "tessimago/ai_safety_50k",
        "split": "english", "max": 5000, "streaming": True,
        "conv": conv_ai_safety_50k,
    },

    # ontocord moral education (text-based)
    {
        "name": "moral_education_permissive",
        "id": "ontocord/moral_education_permissive",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: conv_moral_education_text(r),
    },
]


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
        try:
            if config:
                ds = load_dataset(ds_id, config, split=split, streaming=streaming)
            else:
                ds = load_dataset(ds_id, split=split, streaming=streaming)

            pairs = []
            skipped = 0
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

            print(f"  {len(pairs):,} pairs  ({skipped} skipped)")
            all_pairs.extend(pairs)

        except Exception as e:
            # Try alternate split names
            for alt_split in ("test", "validation", "dev"):
                if alt_split == split:
                    continue
                try:
                    if config:
                        ds = load_dataset(ds_id, config, split=alt_split, streaming=streaming)
                    else:
                        ds = load_dataset(ds_id, split=alt_split, streaming=streaming)
                    pairs = []
                    skipped = 0
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
                    print(f"  {len(pairs):,} pairs  ({skipped} skipped) [split={alt_split}]")
                    all_pairs.extend(pairs)
                    break
                except Exception:
                    pass
            else:
                print(f"  SKIP — {str(e)[:100]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'═'*60}")
    print(f"Total: {len(all_pairs):,} pairs → {OUTPUT}")
    print(f"\nMerge command:")
    print(f"  cat data/sft_master.jsonl {OUTPUT} > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl")


if __name__ == "__main__":
    main()
