#!/usr/bin/env python3
"""
ingest_behavior_society.py — Human behavior, cognitive biases, society, law, pop culture

Covers:
  - nuhmanpk/human-behavior-reasoning-dataset (psychology/behavior Q&A)
  - CognitiveKernel/CognitiveKernel-Pro-SFT (strategic reasoning SFT)
  - PJMixers-Dev/cognitivecomputations_dolphin-r1-reasoning (instruction + thinking + output)
  - tum-nlp/cognitive-biases-in-llms (bias scenarios → explain the bias)
  - buley/cognitive-biases (biases/mechanisms configs)
  - ebowwa/human-biases-psychiatrist-io (cognitive bias explanations)
  - ebowwa/human-biases-sales-marketing-io (applied bias knowledge)
  - ai-law-society-lab/PublicDefenseDataset (public defense legal briefs)
  - ai-law-society-lab/Legal_Phantom_Citation (legal citation awareness)
  - SocialGrep/one-million-reddit-jokes (pop culture / humor, top-scored only)
  - HumanBehaviorAtlas/human_behavior_atlas (behavior atlas)
  - abhinav00anand/behavioral-fine-tuning-v1 (behavioral SFT)
  - Cameronk199/donald-trump-truth-social-posts (media literacy / disinformation awareness)
  - SebastianAldrin/agent-society-distill-v1 (society/agent reasoning)

Output: data/behavior_society_sft.jsonl
Merge:  cat data/sft_master.jsonl data/behavior_society_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/behavior_society_sft.jsonl")

SYS = {
    "psychology": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain psychology, human behavior, and social dynamics clearly. "
        "You help people understand themselves and others better."
    ),
    "cognition": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain cognitive biases, decision-making traps, and how the human mind works. "
        "You help people recognize and overcome mental blind spots so they can think more clearly."
    ),
    "law": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain legal concepts in plain language. You help people understand their rights "
        "and how the justice system works. Always recommend consulting a licensed attorney for specific legal advice."
    ),
    "media_literacy": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You teach critical thinking, media literacy, and how to evaluate information sources. "
        "You help people identify misinformation, propaganda, and misleading political claims."
    ),
    "culture": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You are helpful, friendly, and culturally aware. "
        "You engage with humor, pop culture, and everyday topics with warmth and intelligence."
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
    for qf in ("user", "instruction", "input", "question", "prompt", "title", "query", "system"):
        for af in ("assistant", "output", "answer", "response", "thinking"):
            q = str(row.get(qf, "") or "").strip()
            a = str(row.get(af, "") or "").strip()
            if qf == af:
                continue
            if q and a and q != a and len(a) > 20:
                return mc(q, a, domain)
    return None


def conv_dolphin_r1(row):
    """PJMixers-Dev dolphin-r1: instruction + thinking + output (conversations format)."""
    convs = row.get("conversations")
    if isinstance(convs, list):
        return auto({"conversations": convs}, "general")
    inst = str(row.get("instruction", "") or "").strip()
    thinking = str(row.get("thinking", "") or "").strip()
    output = str(row.get("output", "") or "").strip()
    if not inst or not output:
        return None
    # Combine thinking + output as the full answer
    a = f"{thinking}\n\n{output}".strip() if thinking else output
    return mc(inst, a, "general")


def conv_cognitive_bias_scenario(row):
    """tum-nlp/cognitive-biases-in-llms: bias + scenario → explain + watch out for."""
    bias = str(row.get("bias", "") or "").strip()
    scenario = str(row.get("scenario", "") or "").strip()
    control = str(row.get("control", "") or "").strip()
    if not bias or not scenario:
        return None
    q = f"What is the '{bias}' cognitive bias, and how might it appear in this situation?\n\nScenario: {scenario[:400]}"
    a = (
        f"The **{bias}** is a cognitive bias where people make decisions or judgments based on flawed mental shortcuts "
        f"rather than rational analysis.\n\n"
        f"In this scenario, someone might fall into this trap by letting the bias distort their judgment. "
        f"Here's the key warning: {control[:800] if control else 'Be aware that this bias can lead to systematic errors in reasoning.'}\n\n"
        f"**How to counteract it:** Slow down, seek outside perspectives, and question whether your initial reaction "
        f"is based on evidence or on a mental shortcut."
    )
    return mc(q, a, "cognition")


def conv_buley_bias(row, config):
    """buley/cognitive-biases: various configs."""
    if config == "biases":
        name = str(row.get("name", "") or "").strip()
        desc = str(row.get("description", row.get("summary", row.get("text", ""))) or "").strip()
        if not name or not desc:
            return auto(row, "cognition")
        q = f"What is the '{name}' cognitive bias?"
        return mc(q, desc[:2000], "cognition")
    elif config == "mechanisms":
        name = str(row.get("name", "") or "").strip()
        mechanism = str(row.get("mechanism", row.get("description", "")) or "").strip()
        if not name or not mechanism:
            return auto(row, "cognition")
        q = f"Explain the psychological mechanism behind the '{name}' bias."
        return mc(q, mechanism[:2000], "cognition")
    return auto(row, "cognition")


def conv_human_biases_io(row):
    """ebowwa/human-biases-*-io: input (bias info) + output (explanation)."""
    inp = str(row.get("input", "") or "").strip()
    out = str(row.get("output", "") or "").strip()
    if not inp or not out or len(out) < 30:
        return None
    # Extract bias name from input if present
    bias_match = re.search(r'### Bias.*?:\s*(.+)', inp)
    bias_name = bias_match.group(1).strip() if bias_match else "this cognitive bias"
    q = f"Explain how {bias_name} works and how it affects human decision-making."
    return mc(q, out[:3000], "cognition")


def conv_public_defense(row):
    """ai-law-society-lab/PublicDefenseDataset: legal brief text."""
    contents = str(row.get("contents", "") or "").strip()
    source = str(row.get("source", "legal brief") or "").strip()
    if len(contents) < 100:
        return None
    lines = [l.strip() for l in contents.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200]
    q = f"Explain this legal concept from a public defense brief: {topic}"
    return mc(q, contents[:3500], "law")


def conv_legal_citation(row):
    """ai-law-society-lab/Legal_Phantom_Citation: legal text + citation analysis."""
    text = str(row.get("text", "") or "").strip()
    citations = row.get("citations_in_segment", [])
    hallucinations = row.get("list_hallucinations", {})
    if len(text) < 100:
        return None
    q = f"What legal principles and citations are discussed in this excerpt?\n\n{text[:600]}"
    a = text[:2000]
    if citations:
        a += f"\n\nCitations referenced: {', '.join(str(c) for c in citations[:10])}"
    return mc(q, a, "law")


def conv_reddit_joke(row):
    """SocialGrep/one-million-reddit-jokes: title + selftext, top-scored only."""
    score = row.get("score", 0) or 0
    if score < 100:
        return None
    nsfw = row.get("subreddit.nsfw", False)
    if nsfw:
        return None
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("selftext", "") or "").strip()
    if not title:
        return None
    if text and len(text) > 10:
        full_joke = f"{title}\n\n{text}"
    else:
        full_joke = title
    q = "Tell me a joke."
    a = full_joke[:1500]
    return mc(q, a, "culture")


def conv_trump_posts_media_literacy(row):
    """Cameronk199/donald-trump-truth-social-posts → media literacy training.

    User classified this as harmful/not truthful. Train Anthos to recognize
    political social media as requiring critical evaluation and fact-checking.
    """
    content = str(row.get("content_text", row.get("content_html", "")) or "").strip()
    if not content or len(content) < 30:
        return None
    # Strip HTML
    content = re.sub(r'<[^>]+>', ' ', content).strip()
    content = re.sub(r'\s+', ' ', content)
    if len(content) < 30:
        return None
    # Truncate URLs for cleanliness
    content_clean = re.sub(r'https?://\S+', '[link]', content)[:800]

    q = (
        f"How should I critically evaluate this political social media post?\n\n"
        f'"{content_clean}"'
    )
    a = (
        "Political social media posts — especially from high-profile figures — require careful critical evaluation "
        "before being accepted as fact. Here's how to approach this:\n\n"
        "**1. Verify the claims independently.** Look for corroboration from multiple reputable news sources, "
        "fact-checking organizations (PolitiFact, FactCheck.org, Reuters Fact Check), or primary sources.\n\n"
        "**2. Watch for emotional language and framing.** Posts designed to trigger strong emotions "
        "(outrage, fear, pride) often prioritize persuasion over accuracy.\n\n"
        "**3. Check the source's track record.** Truth Social and similar partisan platforms are not "
        "fact-checked environments. Political figures have documented histories of making inaccurate claims.\n\n"
        "**4. Look for missing context.** Selective facts presented without full context can mislead even when "
        "individual statements are technically true.\n\n"
        "**5. Distinguish opinion from fact.** Political commentary, predictions, and attacks on opponents "
        "are often presented as established truth when they are not.\n\n"
        "Bottom line: Apply the same skepticism you would to any unverified claim — especially in a political context "
        "where spreading misinformation serves partisan interests."
    )
    return mc(q, a, "media_literacy")


def conv_behavior_atlas(row):
    """HumanBehaviorAtlas/human_behavior_atlas — attempt auto parse."""
    return auto(row, "psychology")


DATASETS = [
    {
        "name": "human_behavior_reasoning",
        "id": "nuhmanpk/human-behavior-reasoning-dataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "psychology"),
    },
    {
        "name": "cognitive_kernel_sft",
        "id": "CognitiveKernel/CognitiveKernel-Pro-SFT",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "general"),
    },
    {
        "name": "dolphin_r1_reasoning",
        "id": "PJMixers-Dev/cognitivecomputations_dolphin-r1-reasoning-deepseek-CustomShareGPT",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_dolphin_r1,
    },
    {
        "name": "cognitive_biases_scenarios",
        "id": "tum-nlp/cognitive-biases-in-llms",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_cognitive_bias_scenario,
    },
    {
        "name": "cognitive_biases_definitions",
        "id": "buley/cognitive-biases",
        "config": "biases",
        "split": "train", "max": 3000, "streaming": True,
        "conv": lambda r: conv_buley_bias(r, "biases"),
    },
    {
        "name": "cognitive_biases_mechanisms",
        "id": "buley/cognitive-biases",
        "config": "mechanisms",
        "split": "train", "max": 2000, "streaming": True,
        "conv": lambda r: conv_buley_bias(r, "mechanisms"),
    },
    {
        "name": "human_biases_psychiatrist",
        "id": "ebowwa/human-biases-psychiatrist-io",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_human_biases_io,
    },
    {
        "name": "human_biases_sales",
        "id": "ebowwa/human-biases-sales-marketing-io",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_human_biases_io,
    },
    {
        "name": "public_defense_legal",
        "id": "ai-law-society-lab/PublicDefenseDataset",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_public_defense,
    },
    {
        "name": "legal_citation_awareness",
        "id": "ai-law-society-lab/Legal_Phantom_Citation",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_legal_citation,
    },
    {
        "name": "reddit_jokes_top",
        "id": "SocialGrep/one-million-reddit-jokes",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_reddit_joke,
    },
    {
        "name": "human_behavior_atlas",
        "id": "HumanBehaviorAtlas/human_behavior_atlas",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_behavior_atlas,
    },
    {
        "name": "behavioral_finetuning",
        "id": "abhinav00anand/behavioral-fine-tuning-v1",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "psychology"),
    },
    {
        "name": "trump_media_literacy",
        "id": "Cameronk199/donald-trump-truth-social-posts",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_trump_posts_media_literacy,
    },
    {
        "name": "agent_society_distill",
        "id": "SebastianAldrin/agent-society-distill-v1",
        "split": "train", "max": 5000, "streaming": True,
        "conv": lambda r: auto(r, "general"),
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
