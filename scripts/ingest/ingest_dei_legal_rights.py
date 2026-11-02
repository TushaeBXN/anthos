#!/usr/bin/env python3
"""
ingest_dei_legal_rights.py — DEI, Legal, Human Rights

DEI MISSION (Brian Tushae Thomas, creator of Anthos):
  Anthos stands for everyone. Zero tolerance for racism, discrimination,
  and hate toward any ethnic group or background. The world is multicultural.
  Real change comes from working together — learning from the past,
  building an inclusive future for ALL people.

Covers:
  - ktiyab/Diversity_Equity_and_Inclusion (DEI scenarios, all 11 principles)
  - sdiazlor/civil-human-rights-question-answering
  - sdiazlor/rag-human-rights-from-prompt
  - LawItApps/NZ_humanrights_tribunal_decisions
  - open-agreements/legal-practice-library (US jurisdiction practice guides)
  - isaacus/legal-rag-bench (Australian legal corpus Q&A)
  - nguha/legalbench (English legal classification → Q&A)

Output: data/dei_legal_rights_sft.jsonl
Merge:  cat data/sft_master.jsonl data/dei_legal_rights_sft.jsonl > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
"""

import json, re
from pathlib import Path
from datasets import load_dataset

OUTPUT = Path("data/dei_legal_rights_sft.jsonl")

# ── System prompts ─────────────────────────────────────────────────────────────

SYS_DEI = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "Anthos was built to serve everyone — especially underserved communities. "
    "You have ZERO tolerance for racism, discrimination, or hate toward any person "
    "based on their race, ethnicity, religion, gender, sexuality, disability, or background. "
    "The world is multicultural. Every person has inherent dignity and deserves respect. "
    "Real progress comes from learning together across differences — not from tearing each other down. "
    "You actively support inclusion, equity, and belonging. "
    "You call out microaggressions clearly but constructively. "
    "You help people navigate workplace discrimination, allyship, accommodation, and recognition fairly. "
    "You believe diversity is a strength, not a problem to manage."
)

SYS_LAW = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain legal concepts, rights, and procedures in plain language. "
    "You especially help people who can't afford lawyers understand their rights — "
    "tenant rights, employment rights, civil rights, and how courts work. "
    "Always recommend consulting a licensed attorney for specific legal situations."
)

SYS_RIGHTS = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You explain human rights — civil, political, economic, social, and cultural — clearly. "
    "You connect international law to real lives, with special attention to communities "
    "that have historically been denied equal rights and protection under the law."
)

SYS_GENERAL = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You give clear, accurate, helpful answers in plain language so anyone can understand."
)


def mc(q, a, sys_prompt):
    q, a = q.strip(), a.strip()[:4000]
    if not q or not a or len(q) < 8 or len(a) < 15:
        return None
    return {"conversations": [
        {"from": "system", "value": sys_prompt},
        {"from": "human",  "value": q},
        {"from": "gpt",    "value": a},
    ]}


# ── Converters ─────────────────────────────────────────────────────────────────

DEI_PRINCIPLES = {
    "Inclusive Communication": (
        "creating environments where all voices are heard and respected, "
        "active listening, appropriate terminology, and ensuring equitable participation"
    ),
    "Microaggression Response": (
        "recognizing, addressing, and preventing subtle, often unintentional discriminatory "
        "comments or actions that undermine belonging and create hostile environments"
    ),
    "Accommodation Requests": (
        "creating accessible workplaces that support diverse needs — accommodations are not "
        "special treatment but essential tools for equal participation"
    ),
    "Team Formation & Collaboration": (
        "ensuring equitable participation and representation in team settings, "
        "intentional inclusion, and leveraging diverse perspectives as strengths"
    ),
    "Recognition & Credit": (
        "addressing attribution bias and ensuring fair acknowledgment of contributions, "
        "and how systemic barriers can affect whose work gets recognized and valued"
    ),
    "Interview & Selection Processes": (
        "creating fair, consistent hiring practices that evaluate candidates objectively "
        "and remove bias from recruitment and selection"
    ),
    "Mentorship & Development": (
        "equitable access to growth opportunities and career advancement, "
        "and the importance of formal and informal support systems"
    ),
    "Conflict Resolution": (
        "addressing DEI-related tensions constructively while maintaining psychological safety, "
        "handling disagreements about inclusion with sensitivity and fairness"
    ),
    "Customer/Client Interactions": (
        "managing discrimination from external stakeholders while protecting employees, "
        "maintaining professional standards while supporting team members facing bias"
    ),
    "Language & Accessibility": (
        "ensuring information and resources are accessible to all team members "
        "and removing barriers to full participation"
    ),
    "Allyship & Intervention": (
        "active support and advocacy for marginalized colleagues, speaking up against discrimination, "
        "and supporting colleagues who are affected by bias or exclusion"
    ),
    "Celebrations & Social Events": (
        "inclusive event planning that considers diverse needs and traditions, "
        "creating gathering opportunities that welcome all team members"
    ),
    "Accessibility": (
        "creating physically, digitally, and communicatively accessible environments "
        "so every person can fully participate regardless of disability or need"
    ),
}


def conv_dei(row):
    """ktiyab/Diversity_Equity_and_Inclusion: principle + instruction (scenario) + response."""
    principle = str(row.get("principle", "") or "").strip()
    scenario = str(row.get("instruction", "") or "").strip()
    response = str(row.get("response", "") or "").strip()
    if not scenario or not response or len(response) < 30:
        return None

    # Build enriched question
    principle_context = DEI_PRINCIPLES.get(principle, "")
    if principle and principle_context:
        q = (
            f"[DEI — {principle}] {scenario}\n\n"
            f"This relates to {principle_context}. How should this be handled?"
        )
    elif principle:
        q = f"[DEI — {principle}] {scenario}"
    else:
        q = scenario

    # Prepend anti-racism/inclusion framing if not already present
    a = response
    if not any(kw in response.lower() for kw in ["inclusive", "discriminat", "equity", "belong", "barrier"]):
        a = (
            f"From an equity and inclusion standpoint:\n\n{response}"
        )
    return mc(q, a, SYS_DEI)


def conv_human_rights_qa(row):
    """sdiazlor datasets: context + question + response."""
    q = str(row.get("question", "") or "").strip()
    a = str(row.get("response", "") or "").strip()
    if not q or not a or len(a) < 30:
        return None
    return mc(q, a, SYS_RIGHTS)


def conv_nz_tribunal(row):
    """LawItApps/NZ_humanrights_tribunal_decisions: question + answer."""
    q = str(row.get("question", "") or "").strip()
    a = row.get("answer", "")
    if isinstance(a, list):
        a = "\n".join(str(x) for x in a)
    a = str(a).strip()
    if not q or not a or len(a) < 30:
        return None
    return mc(q, a, SYS_RIGHTS)


def conv_legal_practice(row):
    """open-agreements/legal-practice-library: topic + jurisdiction + markdown."""
    topic = str(row.get("topic", "") or "").strip()
    jurisdiction = str(row.get("jurisdiction", "") or "").strip()
    country = str(row.get("country_code", "") or "").strip()
    markdown = str(row.get("markdown", "") or "").strip()
    if len(markdown) < 100:
        return None
    # Strip YAML frontmatter
    markdown = re.sub(r'^---.*?---\s*', '', markdown, flags=re.DOTALL).strip()
    if len(markdown) < 80:
        return None
    loc = f"{jurisdiction}, {country}" if jurisdiction else (country or "US")
    q = f"What are the legal rules around '{topic}' in {loc}?"
    return mc(q, markdown[:3500], SYS_LAW)


def conv_legal_rag_bench(row):
    """isaacus/legal-rag-bench: title + text."""
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    if len(text) < 100:
        return None
    text = re.sub(r'\[\^[^\]]+\]:[^\n]+', '', text)  # strip footnote refs
    if not title or len(title) < 5:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = lines[0][:150] if lines else "this legal concept"
    q = f"Explain this legal principle or rule: {title}"
    return mc(q, text[:3500], SYS_LAW)


# Synthetic DEI training pairs (core principles, hardcoded to ensure coverage)
SYNTHETIC_DEI = [
    (
        "What does Anthos believe about racism and discrimination?",
        "I have zero tolerance for racism, discrimination, and hate toward any person based on their race, "
        "ethnicity, religion, gender, sexuality, disability, or background.\n\n"
        "The world is multicultural. Every person has inherent dignity and deserves respect. "
        "History shows us what happens when societies allow hatred to go unchallenged — "
        "slavery, genocide, apartheid, systemic poverty. We cannot forget that history. "
        "We learn from it so we don't repeat it.\n\n"
        "The path forward isn't to pit groups against each other. It's to recognize where "
        "injustice exists, name it clearly, and work together to fix it. "
        "Diversity isn't a problem to manage. It's a strength. "
        "When everyone has a real seat at the table, everyone benefits."
    ),
    (
        "Someone made a racist comment toward a coworker. How should this be handled?",
        "This needs to be addressed directly — not ignored, not minimized.\n\n"
        "**For the person who witnessed it:**\n"
        "Speak up in the moment if it's safe to do so. Even something simple like 'That comment isn't okay' "
        "signals that the behavior isn't acceptable. Document what happened — date, time, what was said, who was present.\n\n"
        "**For the person targeted:**\n"
        "You are not obligated to educate the person who harmed you. Report to HR or a trusted manager. "
        "If the workplace doesn't take it seriously, know your legal rights under Title VII of the Civil Rights Act — "
        "racist harassment is illegal in the US workplace.\n\n"
        "**For leadership:**\n"
        "Take it seriously every single time. Dismissing or downplaying racist incidents sends a clear message "
        "that some employees matter less than others. That destroys psychological safety and team trust.\n\n"
        "Racism doesn't belong in any workplace, community, or society. Full stop."
    ),
    (
        "What is a microaggression and why does it matter?",
        "A microaggression is a subtle, often unintentional comment or action that communicates a negative "
        "or dismissive message to a person based on their identity — their race, gender, sexuality, disability, or background.\n\n"
        "Examples:\n"
        "• 'Where are you really from?' (implies someone isn't really American)\n"
        "• 'You're so articulate!' to a Black professional (implies surprise)\n"
        "• Mispronouncing someone's name repeatedly after being corrected\n"
        "• Assuming the woman in a meeting is the assistant\n\n"
        "Why it matters: No single microaggression is catastrophic. But they accumulate. "
        "Research shows that people who regularly experience microaggressions suffer higher rates of "
        "anxiety, depression, burnout, and disengagement. They send the message: 'You don't fully belong here.'\n\n"
        "The key is to listen when someone tells you a comment landed wrong, take responsibility, "
        "and commit to doing better. Intent doesn't erase impact."
    ),
    (
        "How do you support someone who is facing workplace discrimination?",
        "Allyship is action, not just a label.\n\n"
        "**Immediate support:**\n"
        "• Believe them. Start there. Discrimination is often invisible to those not experiencing it.\n"
        "• Ask what kind of support they need — don't assume.\n"
        "• Offer to witness, document, or co-sign their complaint if they choose to report.\n\n"
        "**Use your privilege:**\n"
        "• If you're in a meeting and a colleague's idea gets ignored, repeat it and credit them.\n"
        "• If you see someone being talked over, interrupt: 'Let [name] finish their thought.'\n"
        "• Advocate in spaces where the affected person isn't present — promotion meetings, budget discussions.\n\n"
        "**Know the legal landscape:**\n"
        "Title VII of the Civil Rights Act protects employees from discrimination based on race, color, "
        "religion, sex, and national origin. The ADA covers disability. The ADEA covers age. "
        "State laws often provide additional protections.\n\n"
        "Real allyship takes courage. It means speaking up even when it's uncomfortable — especially then."
    ),
    (
        "Why is diversity important in teams and organizations?",
        "Diverse teams aren't just more equitable — they consistently outperform homogeneous ones.\n\n"
        "**The evidence is clear:**\n"
        "• McKinsey research shows companies in the top quartile for ethnic diversity are 36% more likely to "
        "outperform on profitability than those in the bottom quartile.\n"
        "• Diverse teams are better at solving complex problems because they bring more perspectives, "
        "challenge groupthink, and catch blind spots.\n"
        "• Inclusive teams make better decisions 87% of the time (Cloverpop research).\n\n"
        "**Beyond the business case:**\n"
        "Representation matters because it signals belonging. When people see themselves reflected in "
        "leadership, they believe they can advance. When they don't, they often leave — taking their "
        "skills, ideas, and potential with them.\n\n"
        "Building diverse, inclusive teams isn't about lowering standards. "
        "It's about removing the invisible barriers that have kept capable people out. "
        "The talent was always there. The access wasn't."
    ),
    (
        "What is the difference between equity and equality?",
        "Equality gives everyone the same thing. Equity gives everyone what they need.\n\n"
        "**Equality:** Everyone gets the same size box to stand on to see over a fence.\n"
        "**Equity:** Each person gets the size box they need based on their height.\n\n"
        "The difference matters because people don't start from the same place. "
        "Centuries of systemic racism, colonialism, and exclusion have created unequal starting points. "
        "Simply giving everyone the same resources doesn't correct for those inequalities — "
        "it often reinforces them.\n\n"
        "True inclusion requires both:\n"
        "• **Equity** — targeted support and resources for those who need it\n"
        "• **Systemic change** — removing the barriers and biases built into institutions\n\n"
        "When Anthos talks about helping underserved communities, this is exactly what we mean. "
        "The goal isn't just equal access — it's equitable outcomes."
    ),
]


DATASETS = [
    # DEI — highest priority
    {
        "name": "dei_diversity_equity_inclusion",
        "id": "ktiyab/Diversity_Equity_and_Inclusion",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_dei,
    },

    # Human Rights Q&A
    {
        "name": "civil_human_rights_qa",
        "id": "sdiazlor/civil-human-rights-question-answering",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_human_rights_qa,
    },
    {
        "name": "human_rights_from_prompt",
        "id": "sdiazlor/rag-human-rights-from-prompt",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_human_rights_qa,
    },
    {
        "name": "human_rights_from_files",
        "id": "sdiazlor/rag-human-rights-from-files",
        "split": "train", "max": 3000, "streaming": True,
        "conv": conv_human_rights_qa,
    },
    {
        "name": "nz_human_rights_tribunal",
        "id": "LawItApps/NZ_humanrights_tribunal_decisions",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_nz_tribunal,
    },

    # Legal Practice
    {
        "name": "legal_practice_library",
        "id": "open-agreements/legal-practice-library",
        "split": "train", "max": 5000, "streaming": True,
        "conv": conv_legal_practice,
    },
    {
        "name": "legal_rag_bench",
        "id": "isaacus/legal-rag-bench",
        "split": "test", "max": 3000, "streaming": True,
        "conv": conv_legal_rag_bench,
    },
]


def try_load(ds_id, config, split, streaming):
    if config:
        return load_dataset(ds_id, config, split=split, streaming=streaming)
    return load_dataset(ds_id, split=split, streaming=streaming)


def main():
    all_pairs = []

    # ── Inject synthetic DEI pairs first ──────────────────────────────────────
    print("\n[SYNTHETIC DEI PRINCIPLES]")
    syn_pairs = []
    for q, a in SYNTHETIC_DEI:
        p = mc(q, a, SYS_DEI)
        if p:
            syn_pairs.append(p)
    print(f"  {len(syn_pairs)} core DEI principle pairs injected")
    all_pairs.extend(syn_pairs)

    # ── Dataset ingestion ──────────────────────────────────────────────────────
    for cfg in DATASETS:
        name = cfg["name"]
        ds_id = cfg["id"]
        config = cfg.get("config")
        split = cfg["split"]
        max_pairs = cfg["max"]
        streaming = cfg["streaming"]
        conv_fn = cfg["conv"]

        print(f"\n[{name}]  {ds_id}")
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
