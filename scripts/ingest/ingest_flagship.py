#!/usr/bin/env python3
"""
ingest_flagship.py — Premier instruction/reasoning/culture datasets

Tier 1 (flagship instruction datasets — highest quality signal):
  - Open-Orca/OpenOrca              (4.2M GPT-4/3.5 instruction pairs)
  - teknium/OpenHermes-2.5          (1M instruction conversations)
  - open-thoughts/OpenThoughts-114k (114k chain-of-thought reasoning)
  - HuggingFaceH4/ultrachat_200k    (200k multi-turn chat)
  - openbmb/UltraChat               (1.5M multi-turn conversations)
  - lmsys/lmsys-chat-1m             (1M real user conversations, English only)
  - openai/gsm8k                    (grade school math reasoning)
  - nvidia/Llama-Nemotron-Post-Training-Dataset (Nemotron post-training SFT)
  - BAAI/Infinity-Instruct          (large instruction collection)
  - proj-persona/PersonaHub         (persona-conditioned Q&A)
  - JosephusCheung/GuanacoDataset   (English subset)

Tier 2 (culture + knowledge):
  - wikimedia/wikipedia             (en, 20231101 — English Wikipedia)
  - AIM-SCU/When-Cultures-Meet      (cultural interaction Q&A)
  - burgerbee/art_and_culture_wiki  (art/culture wiki text)
  - MBZUAI-Paris/Deep-Culture-Lense (multicultural dataset)
  - jsbeaudry/general-culture-english-creole (English side only)
  - HuggingFaceFW/fineweb-edu       (high-quality educational web text)

Skipped:
  - ILSVRC/imagenet-1k              (images)
  - bigcode/the-stack               (code — separate domain)
  - HuggingFaceFW/finepdfs          (raw PDFs, no structure)
  - tiiuae/falcon-refinedweb        (raw web text)
  - Congliu/Chinese-DeepSeek-R1-Distill-data-110k (Chinese)
  - HumynLabs/medical-prescription-english-audio (audio)
  - HuggingFaceM4/FineVision        (vision/image)
  - uonlp/CulturaX                  (multilingual raw)
  - Salesforce/wikitext             (raw text, no Q&A)
  - ministere-culture/comparia-conversations (French)
  - AI-Culture-Commons/* (multilingual/HTML)
  - imhmdf/taiwan-culture-llama-70b-training (Chinese/Traditional)
  - Salesforce/xlam-function-calling-60k (tool-calling, separate domain)
  - ropensci/historydata            (R package, not HuggingFace)

Output: data/flagship_sft.jsonl
Merge:  cat data/sft_master.jsonl data/flagship_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/flagship_sft.jsonl")

SYS_DEFAULT = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You give clear, accurate, thoughtful answers to help people learn and solve problems. "
    "You make knowledge accessible to everyone, especially underserved communities."
)

SYS_MATH = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain math step by step so anyone can follow. "
    "You show your work clearly and explain the reasoning behind each step."
)

SYS_CULTURE = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You have deep knowledge of world cultures, histories, and traditions. "
    "You explain cultural concepts with respect, nuance, and real depth."
)

SYS_EDU = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a knowledgeable tutor who explains topics clearly and builds real understanding. "
    "You make learning accessible to everyone regardless of background."
)


def mc(q, a, sys=None):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": sys or SYS_DEFAULT},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


def is_english(text, threshold=0.75):
    sample = text[:300]
    if not sample:
        return False
    return sum(1 for c in sample if ord(c) < 128) / len(sample) >= threshold


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def conv_messages(msgs, sys_override=None):
    """Extract Q&A from messages list (role/content or from/value)."""
    q = a = ""
    sys_msg = ""
    for m in msgs:
        role = str(m.get("role", m.get("from", ""))).lower()
        content = str(m.get("content", m.get("value", ""))).strip()
        if role == "system" and not sys_msg:
            sys_msg = content
        elif role in ("user", "human") and not q:
            q = content
        elif role in ("assistant", "gpt") and q and not a:
            a = strip_think(content)
    return q, a, sys_msg


# ── Tier 1: Flagship Instruction ──────────────────────────────────────────────

def conv_openorca(row):
    """Open-Orca/OpenOrca: system_prompt + question + response."""
    sys_p = str(row.get("system_prompt", "") or "").strip()
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("response", "") or "").strip()
    if not q or not a or not is_english(q):
        return None
    sys = sys_p if sys_p and len(sys_p) > 10 else SYS_DEFAULT
    return mc(q, a, sys)


def conv_openhermes(row):
    """teknium/OpenHermes-2.5: conversations list with from/value."""
    convs = row.get("conversations", [])
    if not isinstance(convs, list):
        return None
    q, a, sys_msg = conv_messages(convs)
    if not q or not a or not is_english(q):
        return None
    return mc(q, a, sys_msg or SYS_DEFAULT)


def conv_openthoughts(row):
    """open-thoughts/OpenThoughts-114k: conversations with <think> tags."""
    convs = row.get("conversations", row.get("messages", []))
    if not isinstance(convs, list):
        return None
    q, a, sys_msg = conv_messages(convs)
    if not q or not a or not is_english(q):
        return None
    return mc(q, a, sys_msg or SYS_DEFAULT)


def conv_ultrachat_200k(row):
    """HuggingFaceH4/ultrachat_200k: messages list with role/content."""
    msgs = row.get("messages", [])
    if not isinstance(msgs, list):
        return None
    q, a, sys_msg = conv_messages(msgs)
    if not q or not a or not is_english(q):
        return None
    return mc(q, a, sys_msg or SYS_DEFAULT)


def conv_ultrachat(row):
    """openbmb/UltraChat: data field is list of [human, assistant, human, ...] turns."""
    data = row.get("data", row.get("conversations", row.get("messages", [])))
    if isinstance(data, list) and len(data) >= 2:
        # Flat alternating list: [human_turn, assistant_turn, ...]
        if isinstance(data[0], str):
            q = data[0].strip()
            a = data[1].strip() if len(data) > 1 else ""
            if q and a and is_english(q):
                return mc(q, strip_think(a), SYS_DEFAULT)
        # List of dicts
        elif isinstance(data[0], dict):
            q, a, sys_msg = conv_messages(data)
            if q and a and is_english(q):
                return mc(q, a, sys_msg or SYS_DEFAULT)
    # id+data structure
    inner = row.get("conversation", [])
    if isinstance(inner, list) and inner:
        q, a, sys_msg = conv_messages(inner)
        if q and a and is_english(q):
            return mc(q, a, sys_msg or SYS_DEFAULT)
    return None


def conv_lmsys(row):
    """lmsys/lmsys-chat-1m: conversation list, language field — English only."""
    lang = str(row.get("language", "") or "").lower()
    if lang and lang not in ("english", "en", ""):
        return None
    convs = row.get("conversation", row.get("conversations", row.get("messages", [])))
    if not isinstance(convs, list):
        return None
    q, a, sys_msg = conv_messages(convs)
    if not q or not a or not is_english(q):
        return None
    return mc(q, a, sys_msg or SYS_DEFAULT)


def conv_gsm8k(row):
    """openai/gsm8k: question + answer (#### separates steps from final answer)."""
    q = str(row.get("question", "") or "").strip()
    a_raw = str(row.get("answer", "") or "").strip()
    if not q or not a_raw:
        return None
    # Keep full answer with steps (the #### marks final answer)
    a = a_raw.replace("####", "\n\nFinal Answer:").strip()
    return mc(q, a, SYS_MATH)


def conv_nemotron(row):
    """nvidia/Llama-Nemotron-Post-Training-Dataset: messages or input/output."""
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list) and msgs:
        q, a, sys_msg = conv_messages(msgs)
        if q and a and is_english(q):
            return mc(q, a, sys_msg or SYS_DEFAULT)
    # Flat fields
    for qf, af in [("input", "output"), ("instruction", "response"), ("prompt", "completion")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 10 and is_english(q):
            return mc(q, strip_think(a), SYS_DEFAULT)
    return None


def conv_infinity_instruct(row):
    """BAAI/Infinity-Instruct: conversations or instruction/response."""
    convs = row.get("conversations", row.get("messages", []))
    if isinstance(convs, list) and convs:
        q, a, sys_msg = conv_messages(convs)
        if q and a and is_english(q):
            return mc(q, a, sys_msg or SYS_DEFAULT)
    for qf, af in [("instruction", "response"), ("input", "output"), ("query", "answer")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 10 and is_english(q):
            return mc(q, strip_think(a), SYS_DEFAULT)
    return None


def conv_personahub(row):
    """proj-persona/PersonaHub: persona + instruction + response."""
    persona = str(row.get("persona", row.get("character", "")) or "").strip()
    instruction = str(row.get("instruction", row.get("input", row.get("question", ""))) or "").strip()
    response = str(row.get("response", row.get("output", row.get("answer", ""))) or "").strip()
    if not instruction or not response or not is_english(instruction):
        return None
    if persona:
        sys = f"{SYS_DEFAULT}\n\nPersona context: {persona[:300]}"
        return mc(instruction, strip_think(response), sys)
    return mc(instruction, strip_think(response), SYS_DEFAULT)


def conv_guanaco(row):
    """JosephusCheung/GuanacoDataset: instruction + input + output, English rows."""
    instruction = str(row.get("instruction", row.get("input", "")) or "").strip()
    output = str(row.get("output", row.get("response", "")) or "").strip()
    extra_input = str(row.get("input", "") or "").strip()
    if not instruction or not output or not is_english(instruction):
        return None
    q = instruction
    if extra_input and extra_input != instruction and len(extra_input) > 5:
        q = f"{instruction}\n\n{extra_input}"
    return mc(q, output[:3500], SYS_DEFAULT)


# ── Tier 2: Culture + Knowledge ───────────────────────────────────────────────

def conv_wikipedia(row):
    """wikimedia/wikipedia: title + text — make 'Tell me about X' pairs."""
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    if not title or not text or len(text) < 200 or not is_english(text):
        return None
    q = f"Tell me about {title}."
    # Use first 3 paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 50]
    a = "\n\n".join(paras[:3])[:3500]
    if len(a) < 100:
        return None
    return mc(q, a, SYS_EDU)


def conv_when_cultures_meet(row):
    """AIM-SCU/When-Cultures-Meet: cultural Q&A."""
    for qf, af in [("question", "answer"), ("input", "output"),
                   ("instruction", "response"), ("query", "response")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, a, SYS_CULTURE)
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list):
        q, a, _ = conv_messages(msgs)
        if q and a and is_english(q):
            return mc(q, a, SYS_CULTURE)
    return None


def conv_art_culture_wiki(row):
    """burgerbee/art_and_culture_wiki: wiki text about art/culture."""
    text = str(row.get("text", row.get("content", row.get("body", ""))) or "").strip()
    title = str(row.get("title", row.get("name", "")) or "").strip()
    if not text or len(text) < 150 or not is_english(text):
        return None
    if title:
        q = f"Tell me about {title} from an art and culture perspective."
    else:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        q = f"Explain this art and culture topic: {lines[0][:150]}"
    return mc(q, text[:3000], SYS_CULTURE)


def conv_deep_culture(row):
    """MBZUAI-Paris/Deep-Culture-Lense: multicultural Q&A."""
    for qf, af in [("question", "answer"), ("input", "output"),
                   ("instruction", "response"), ("prompt", "completion")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), SYS_CULTURE)
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list):
        q, a, _ = conv_messages(msgs)
        if q and a and is_english(q):
            return mc(q, a, SYS_CULTURE)
    return None


def conv_culture_english_creole(row):
    """jsbeaudry/general-culture-english-creole: use English side."""
    # Likely has english/creole parallel fields
    eng = str(row.get("english", row.get("en", row.get("English", ""))) or "").strip()
    question = str(row.get("question", row.get("input", "")) or "").strip()
    answer = str(row.get("answer", row.get("output", "")) or "").strip()
    if question and answer and is_english(question):
        return mc(question, answer, SYS_CULTURE)
    if eng and len(eng) > 50:
        lines = [l.strip() for l in eng.splitlines() if l.strip()]
        if len(lines) >= 2:
            return mc(f"Tell me about this cultural topic: {lines[0][:150]}", "\n".join(lines[1:])[:2000], SYS_CULTURE)
    return None


def conv_fineweb_edu(row):
    """HuggingFaceFW/fineweb-edu: high quality educational web text."""
    text = str(row.get("text", row.get("content", "")) or "").strip()
    score = row.get("score", row.get("educational_score", 0))
    if not text or len(text) < 200 or not is_english(text):
        return None
    # Only use high-quality entries (score >= 3 out of 5)
    try:
        if float(score) < 3.0:
            return None
    except (ValueError, TypeError):
        pass
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10 and len(lines) > 1:
        topic = " ".join(lines[:2])[:200]
    if len(topic) < 10:
        return None
    q = f"Explain this educational topic: {topic}"
    return mc(q, text[:3500], SYS_EDU)


DATASETS = [
    # Tier 1 — Flagship
    {"name": "openorca",           "id": "Open-Orca/OpenOrca",                            "split": "train", "max": 5000, "conv": conv_openorca,       "streaming": True},
    {"name": "openhermes_25",      "id": "teknium/OpenHermes-2.5",                        "split": "train", "max": 5000, "conv": conv_openhermes,      "streaming": True},
    {"name": "openthoughts_114k",  "id": "open-thoughts/OpenThoughts-114k",               "split": "train", "max": 5000, "conv": conv_openthoughts,    "streaming": True},
    {"name": "ultrachat_200k",     "id": "HuggingFaceH4/ultrachat_200k",                  "split": "train_sft", "max": 5000, "conv": conv_ultrachat_200k, "streaming": True},
    {"name": "ultrachat",          "id": "openbmb/UltraChat",                             "split": "train", "max": 5000, "conv": conv_ultrachat,       "streaming": True},
    {"name": "lmsys_chat",         "id": "lmsys/lmsys-chat-1m",                           "split": "train", "max": 5000, "conv": conv_lmsys,           "streaming": True},
    {"name": "gsm8k",              "id": "openai/gsm8k",                   "config": "main", "split": "train", "max": 5000, "conv": conv_gsm8k,       "streaming": True},
    {"name": "nemotron_postrain",  "id": "nvidia/Llama-Nemotron-Post-Training-Dataset",   "split": "train", "max": 5000, "conv": conv_nemotron,        "streaming": True},
    {"name": "infinity_instruct",  "id": "BAAI/Infinity-Instruct",                        "split": "train", "max": 5000, "conv": conv_infinity_instruct, "streaming": True},
    {"name": "personahub",         "id": "proj-persona/PersonaHub",                       "split": "train", "max": 5000, "conv": conv_personahub,      "streaming": True},
    {"name": "guanaco",            "id": "JosephusCheung/GuanacoDataset",                 "split": "train", "max": 5000, "conv": conv_guanaco,         "streaming": True},
    # Tier 2 — Culture + Knowledge
    {"name": "wikipedia_en",       "id": "wikimedia/wikipedia", "config": "20231101.en",  "split": "train", "max": 5000, "conv": conv_wikipedia,       "streaming": True},
    {"name": "when_cultures_meet", "id": "AIM-SCU/When-Cultures-Meet",                    "split": "train", "max": 3000, "conv": conv_when_cultures_meet, "streaming": True},
    {"name": "art_culture_wiki",   "id": "burgerbee/art_and_culture_wiki",                "split": "train", "max": 3000, "conv": conv_art_culture_wiki, "streaming": True},
    {"name": "deep_culture_lense", "id": "MBZUAI-Paris/Deep-Culture-Lense",               "split": "train", "max": 3000, "conv": conv_deep_culture,    "streaming": True},
    {"name": "culture_en_creole",  "id": "jsbeaudry/general-culture-english-creole",      "split": "train", "max": 2000, "conv": conv_culture_english_creole, "streaming": True},
    {"name": "fineweb_edu",        "id": "HuggingFaceFW/fineweb-edu",     "config": "sample-10BT", "split": "train", "max": 5000, "conv": conv_fineweb_edu, "streaming": True},
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]
        conv_fn = cfg["conv"]
        config = cfg.get("config")
        streaming = cfg.get("streaming", True)

        print(f"\n[{name}]  {ds_id}")
        loaded = False
        first_err = ""

        try_splits = [split, "train", "train_sft", "test", "validation"]
        for try_split in try_splits:
            try:
                kwargs = {"split": try_split, "streaming": streaming}
                if config:
                    ds = load_dataset(ds_id, config, **kwargs)
                else:
                    ds = load_dataset(ds_id, **kwargs)

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
                    first_err = str(e)[:120]

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
