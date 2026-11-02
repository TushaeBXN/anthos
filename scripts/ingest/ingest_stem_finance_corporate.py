#!/usr/bin/env python3
"""
ingest_stem_finance_corporate.py — STEM Science, Finance, Corporate, Project Management

Covers:
  STEM Science (MaCroScope):
    - ai4collaboration/MaCroScope-Biology-Chemistry-10k
    - ai4collaboration/MaCroScope-Chemistry-20k-Cleaned
    - ai4collaboration/MaCroScope-Physics-1k
    - ai4collaboration/MaCroScope-Economics-5k
    - j14i/cl-macros-thinking   (reasoning chains)
  Economics / MMLU:
    - joey234/mmlu-high_school_macroeconomics-neg
    - joey234/mmlu-high_school_microeconomics
    - macrocosm/arxiv_titles    (science paper titles → explain)
  Finance:
    - Sachin21112004/news-finance-dataset
    - Yahoo-Finance-News/FineWeb-2022
    - BAAI/IndustryCorpus2_finance_economics
    - geodesic-research/finance-inoculation-midtraining
    - HCAI-Lab-GT/dolma3-6t-sample-10000-docs-finance-and-business
    - JohnGavin/finance-data
  Corporate / Business:
    - axondendriteplus/IC38-Corporate-Agent-Questions  (insurance/corporate Q&A)
    - PhillyMac/Corporate_Governance_Risk_Leadership_Theory
    - lbrenap1/mining-legal-arguments-us-corporate-case-law
    - fingriffin/natural-questions-corporate-jargon
    - garvitupdy/Corporate_AI_Dataset
    - JayJayThrowThrow/hansard-historic-contemporary   (UK Parliament debates)
  Marketing / Project Management:
    - marketeam/FineWeb-Marketing
    - ai-in-projectmanagement/ProjectManagementLLM_dataset
    - aumghag/Data-Analytics-Digital-Marketing-Project-Management-QA_DB

Skipped:
  - erinmikailstaples/awesome-pop-culture-data (GitHub repo, not HF)
  - isaquecerqueira/millan_internet_traffic + revd007/Internet-Firewall (network data)
  - luethan2025/WikiArt-Contemporary-Realism (images)
  - macrocosm-os/code-parrot-github-code (code domain)
  - leodemore/btc-finance-historical-data (price tables only)
  - ZixuanKe/posttrain_tokenized_finance_unsup_qwen2.5_32b_instr (tokenized)
  - tidy-finance/factor-library (factor tables)
  - mteb/llm-eval-legalbench-corporate-lobbying + mteb/CorporateLobbyingLegalBenchClassification (classification benchmarks)
  - SafetyMP/corporate-site-harness-training-data (construction safety harness)
  - Falah/ads_corporate_prompts + MarieDeVox/saas-english-corporate-voice-dataset-sample (ad copy)
  - phxdev/corporate-speak-dataset + autotrain (low quality jargon)
  - jason1966/agewerc_corporate-credit-rating + ZipLime/corporate-actions (structured tables)
  - agungpambudi/corporate-event-detection (classification only)
  - QuantumSkynet/Llama-4-Maverick-*-HR-Resume-* (HR screening, sensitive)
  - MikePfunk28/corporateDataset (unknown, skip)
  - beforee/english-project-management-basics-30 (30 rows)
  - Lots-of-LoRAs/task705_mmmlu_* (already covered)

Output: data/stem_finance_corporate_sft.jsonl
Merge:  cat data/sft_master.jsonl data/stem_finance_corporate_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/stem_finance_corporate_sft.jsonl")

SYS = {
    "science": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain science clearly and accurately, making complex topics accessible "
        "to anyone regardless of their educational background. You love helping people "
        "understand how the world works."
    ),
    "finance": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain financial concepts, news, and data clearly — making finance "
        "accessible to people who've been shut out of this knowledge. "
        "You help people understand money, markets, and economic systems."
    ),
    "business": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain business, corporate governance, management, and professional "
        "concepts clearly. You help people navigate professional environments "
        "and understand how organizations work."
    ),
    "law": (
        "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
        "You explain legal concepts, arguments, and processes in plain language. "
        "You help people understand the law without needing an expensive lawyer."
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
        {"from": "system", "value": SYS.get(domain, SYS["general"])},
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


def generic_qa(row, domain="general"):
    """Try standard Q&A field pairs."""
    for qf, af in [("question", "answer"), ("instruction", "response"),
                   ("input", "output"), ("prompt", "completion"),
                   ("query", "response"), ("Question", "Answer")]:
        q = str(row.get(qf, "") or "").strip()
        a = str(row.get(af, "") or "").strip()
        if q and a and len(a) > 15 and is_english(q):
            return mc(q, strip_think(a), domain)
    msgs = row.get("messages", row.get("conversations", []))
    if isinstance(msgs, list):
        q = a = ""
        for m in msgs:
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if role in ("user", "human") and not q:
                q = content
            elif role in ("assistant", "gpt") and q and not a:
                a = strip_think(content)
        if q and a and is_english(q):
            return mc(q, a, domain)
    return None


def text_to_qa(row, domain, q_prefix="Explain this topic:"):
    """Convert raw text field to Q&A using first line as topic."""
    text = str(row.get("text", row.get("content", row.get("body", "")) or "")).strip()
    if len(text) < 150 or not is_english(text):
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    topic = lines[0][:200].lstrip('#').strip()
    if len(topic) < 10 and len(lines) > 1:
        topic = " ".join(lines[:2])[:200]
    if len(topic) < 10:
        return None
    return mc(f"{q_prefix} {topic}", text[:3500], domain)


# ── MaCroScope STEM ────────────────────────────────────────────────────────────

def conv_macroscope(row):
    """ai4collaboration/MaCroScope-*: Q&A or instruction/response format."""
    p = generic_qa(row, "science")
    if p:
        return p
    # Try subject-specific fields
    subject = str(row.get("subject", row.get("topic", row.get("domain", ""))) or "").strip()
    explanation = str(row.get("explanation", row.get("solution", row.get("rationale", ""))) or "").strip()
    q = str(row.get("question", row.get("problem", "")) or "").strip()
    a = str(row.get("answer", row.get("solution", row.get("correct_answer", ""))) or "").strip()
    if not q or not a:
        return None
    if explanation:
        full_a = f"{a}\n\nExplanation: {explanation}"
    else:
        full_a = a
    return mc(q, full_a[:3500], "science")


def conv_macros_thinking(row):
    """j14i/cl-macros-thinking: reasoning chains."""
    p = generic_qa(row, "science")
    if p:
        return p
    return text_to_qa(row, "science", "Explain this scientific concept:")


# ── MMLU Economics ─────────────────────────────────────────────────────────────

MMLU_CHOICES = ["A", "B", "C", "D"]

def conv_mmlu_econ(row):
    """joey234/mmlu-high_school_macroeconomics-neg and microeconomics."""
    q = str(row.get("question", "") or "").strip()
    choices = row.get("choices", row.get("options", []))
    answer = row.get("answer", row.get("label", None))
    if not q:
        return None
    if isinstance(choices, list) and choices:
        opts = "\n".join(f"  {MMLU_CHOICES[i]}. {c}" for i, c in enumerate(choices) if i < 4)
        full_q = f"{q}\n\nOptions:\n{opts}"
    else:
        full_q = q
    if answer is not None:
        try:
            idx = int(answer)
            if isinstance(choices, list) and idx < len(choices):
                a = f"{MMLU_CHOICES[idx]}. {choices[idx]}"
            else:
                a = str(answer)
        except (ValueError, TypeError):
            a = str(answer)
    else:
        a = str(row.get("output", row.get("response", "")) or "").strip()
    if not a:
        return None
    return mc(full_q, a, "finance")


# ── ArXiv titles ───────────────────────────────────────────────────────────────

def conv_arxiv_titles(row):
    """macrocosm/arxiv_titles: title + abstract or just title."""
    title = str(row.get("title", row.get("text", "")) or "").strip()
    abstract = str(row.get("abstract", row.get("summary", "")) or "").strip()
    if not title or len(title) < 15 or not is_english(title):
        return None
    q = f"Explain this research paper in simple terms: {title}"
    a = abstract if abstract and len(abstract) > 50 else (
        f"This paper titled '{title}' represents academic research. "
        f"The title suggests it explores concepts related to {title.lower().split()[0:3]}. "
        f"Academic papers in this area typically investigate theoretical frameworks, "
        f"experimental findings, or computational methods relevant to the subject."
    )
    return mc(q, a[:3000], "science")


# ── Finance ────────────────────────────────────────────────────────────────────

def conv_finance_news(row):
    """Finance news datasets: title + text/content."""
    title = str(row.get("title", row.get("headline", row.get("Title", ""))) or "").strip()
    text = str(row.get("text", row.get("content", row.get("body", row.get("article", ""))) or "")).strip()
    sentiment = row.get("sentiment", row.get("label", row.get("category", "")))
    if not text and not title:
        return None
    body = text or title
    if len(body) < 80 or not is_english(body[:200]):
        return None
    if title:
        q = f"Explain this financial news story: {title}"
    else:
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        q = f"Summarize this financial news: {lines[0][:150]}"
    a = body[:3000]
    if sentiment:
        a += f"\n\nMarket sentiment: {sentiment}"
    return mc(q, a, "finance")


def conv_finance_corpus(row):
    """BAAI/IndustryCorpus2_finance_economics and similar: text corpus."""
    return text_to_qa(row, "finance", "Explain this financial or economic topic:")


def conv_finance_business_dolma(row):
    """dolma finance/business sample."""
    p = generic_qa(row, "finance")
    if p:
        return p
    return text_to_qa(row, "finance", "Summarize this finance and business document:")


def conv_john_gavin_finance(row):
    """JohnGavin/finance-data: structured finance Q&A or text."""
    p = generic_qa(row, "finance")
    if p:
        return p
    return text_to_qa(row, "finance", "Explain this financial concept:")


# ── Corporate / Business ───────────────────────────────────────────────────────

def conv_ic38_insurance(row):
    """axondendriteplus/IC38-Corporate-Agent-Questions: insurance/corporate Q&A."""
    p = generic_qa(row, "business")
    if p:
        return p
    # IC38 format may have numbered questions
    text = str(row.get("text", row.get("content", "")) or "").strip()
    if text and len(text) > 40 and is_english(text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            return mc(lines[0][:300], "\n".join(lines[1:])[:2000], "business")
    return None


def conv_corporate_governance(row):
    """PhillyMac/Corporate_Governance_Risk_Leadership_Theory."""
    p = generic_qa(row, "business")
    if p:
        return p
    return text_to_qa(row, "business", "Explain this corporate governance concept:")


def conv_corporate_legal(row):
    """lbrenap1/mining-legal-arguments-us-corporate-case-law."""
    p = generic_qa(row, "law")
    if p:
        return p
    text = str(row.get("text", row.get("argument", row.get("content", ""))) or "").strip()
    case = str(row.get("case", row.get("title", row.get("case_name", ""))) or "").strip()
    if not text or len(text) < 80 or not is_english(text):
        return None
    q = f"Explain the legal argument in this US corporate case{': ' + case if case else ''}."
    return mc(q, text[:3000], "law")


def conv_corporate_jargon(row):
    """fingriffin/natural-questions-corporate-jargon: explain jargon in plain English."""
    p = generic_qa(row, "business")
    if p:
        return p
    jargon = str(row.get("term", row.get("jargon", row.get("phrase", ""))) or "").strip()
    definition = str(row.get("definition", row.get("explanation", row.get("meaning", ""))) or "").strip()
    if jargon and definition:
        return mc(f"What does the corporate term '{jargon}' mean in plain English?", definition[:2000], "business")
    return text_to_qa(row, "business", "Explain this business term or concept:")


def conv_corporate_ai(row):
    """garvitupdy/Corporate_AI_Dataset."""
    p = generic_qa(row, "business")
    if p:
        return p
    return text_to_qa(row, "business", "Explain this corporate AI concept:")


def conv_hansard(row):
    """JayJayThrowThrow/hansard-historic-contemporary: UK Parliament debates."""
    text = str(row.get("text", row.get("speech", row.get("content", ""))) or "").strip()
    speaker = str(row.get("speaker", row.get("member", row.get("name", ""))) or "").strip()
    date = str(row.get("date", row.get("year", "")) or "").strip()
    topic = str(row.get("topic", row.get("subject", row.get("debate", ""))) or "").strip()
    if not text or len(text) < 100 or not is_english(text):
        return None
    if topic and speaker:
        q = f"What did {speaker} say in the UK Parliament about {topic}?"
    elif topic:
        q = f"Summarize this UK Parliamentary speech about {topic}."
    else:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        q = f"Summarize this UK Parliamentary speech: {lines[0][:150]}"
    return mc(q, text[:3000], "business")


# ── Marketing / PM ─────────────────────────────────────────────────────────────

def conv_marketing(row):
    """marketeam/FineWeb-Marketing: marketing web text."""
    p = generic_qa(row, "business")
    if p:
        return p
    return text_to_qa(row, "business", "Explain this marketing concept or strategy:")


def conv_project_management(row):
    """ai-in-projectmanagement/ProjectManagementLLM_dataset."""
    p = generic_qa(row, "business")
    if p:
        return p
    return text_to_qa(row, "business", "Explain this project management concept:")


def conv_analytics_marketing_pm(row):
    """aumghag/Data-Analytics-Digital-Marketing-Project-Management-QA_DB."""
    p = generic_qa(row, "business")
    return p


DATASETS = [
    # STEM Science
    {"name": "macroscope_bio_chem",    "id": "ai4collaboration/MaCroScope-Biology-Chemistry-10k", "split": "train", "max": 5000, "conv": conv_macroscope},
    {"name": "macroscope_chemistry",   "id": "ai4collaboration/MaCroScope-Chemistry-20k-Cleaned", "split": "train", "max": 5000, "conv": conv_macroscope},
    {"name": "macroscope_physics",     "id": "ai4collaboration/MaCroScope-Physics-1k",             "split": "train", "max": 1000, "conv": conv_macroscope},
    {"name": "macroscope_economics",   "id": "ai4collaboration/MaCroScope-Economics-5k",           "split": "train", "max": 5000, "conv": conv_macroscope},
    {"name": "macros_thinking",        "id": "j14i/cl-macros-thinking",                            "split": "train", "max": 3000, "conv": conv_macros_thinking},
    # Economics / MMLU
    {"name": "mmlu_macroeconomics",    "id": "joey234/mmlu-high_school_macroeconomics-neg",        "split": "test",  "max": 2000, "conv": conv_mmlu_econ},
    {"name": "mmlu_microeconomics",    "id": "joey234/mmlu-high_school_microeconomics",            "split": "test",  "max": 2000, "conv": conv_mmlu_econ},
    {"name": "arxiv_titles",           "id": "macrocosm/arxiv_titles",                             "split": "train", "max": 3000, "conv": conv_arxiv_titles},
    # Finance
    {"name": "finance_news",           "id": "Sachin21112004/news-finance-dataset",                "split": "train", "max": 3000, "conv": conv_finance_news},
    {"name": "yahoo_finance_2022",     "id": "Yahoo-Finance-News/FineWeb-2022",                    "split": "train", "max": 5000, "conv": conv_finance_news},
    {"name": "industry_finance_econ",  "id": "BAAI/IndustryCorpus2_finance_economics",             "split": "train", "max": 5000, "conv": conv_finance_corpus},
    {"name": "finance_inoculation",    "id": "geodesic-research/finance-inoculation-midtraining",  "split": "train", "max": 5000, "conv": conv_finance_corpus},
    {"name": "dolma_finance_biz",      "id": "HCAI-Lab-GT/dolma3-6t-sample-10000-docs-finance-and-business", "split": "train", "max": 5000, "conv": conv_finance_business_dolma},
    {"name": "john_gavin_finance",     "id": "JohnGavin/finance-data",                             "split": "train", "max": 3000, "conv": conv_john_gavin_finance},
    # Corporate / Business
    {"name": "ic38_insurance",         "id": "axondendriteplus/IC38-Corporate-Agent-Questions",    "split": "train", "max": 5000, "conv": conv_ic38_insurance},
    {"name": "corporate_governance",   "id": "PhillyMac/Corporate_Governance_Risk_Leadership_Theory", "split": "train", "max": 3000, "conv": conv_corporate_governance},
    {"name": "corporate_case_law",     "id": "lbrenap1/mining-legal-arguments-us-corporate-case-law", "split": "train", "max": 3000, "conv": conv_corporate_legal},
    {"name": "corporate_jargon",       "id": "fingriffin/natural-questions-corporate-jargon",      "split": "train", "max": 3000, "conv": conv_corporate_jargon},
    {"name": "corporate_ai",           "id": "garvitupdy/Corporate_AI_Dataset",                    "split": "train", "max": 3000, "conv": conv_corporate_ai},
    {"name": "hansard_debates",        "id": "JayJayThrowThrow/hansard-historic-contemporary",     "split": "train", "max": 5000, "conv": conv_hansard},
    # Marketing / PM
    {"name": "fineweb_marketing",      "id": "marketeam/FineWeb-Marketing",                        "split": "train", "max": 5000, "conv": conv_marketing},
    {"name": "project_management",     "id": "ai-in-projectmanagement/ProjectManagementLLM_dataset", "split": "train", "max": 3000, "conv": conv_project_management},
    {"name": "analytics_marketing_pm", "id": "aumghag/Data-Analytics-Digital-Marketing-Project-Management-QA_DB", "split": "train", "max": 3000, "conv": conv_analytics_marketing_pm},
]


def main():
    all_pairs = []

    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        split = cfg["split"]
        max_pairs = cfg["max"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}")
        loaded = False
        first_err = ""
        for try_split in [split, "train", "test", "validation"]:
            try:
                ds = load_dataset(ds_id, split=try_split, streaming=True)
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
