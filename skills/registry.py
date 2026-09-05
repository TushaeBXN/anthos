"""
Anthos skill registry and training data generator.

Each skill entry includes:
- system_prompt: the capability definition Anthos should internalize
- example_pairs: seed instruction-response pairs for training data generation
- data_gen_prompts: prompts to feed to Gemini/Claude to generate more training pairs
"""

SKILL_MAP = {

    "ai-research-assistant": {
        "category": "knowledge",
        "system_prompt": (
            "When asked to research a topic, run a systematic investigation: "
            "identify key facts, timeline, actors, causes, and implications. "
            "Synthesize findings with source attribution. "
            "Structure output: background → current state → analysis → unknowns."
        ),
        "example_pairs": [
            {
                "instruction": "Research the history and current state of quantum computing.",
                "response": (
                    "**Quantum Computing: Research Summary**\n\n"
                    "**Background:** Quantum computing emerged from Richard Feynman's 1982 proposal "
                    "to use quantum systems to simulate physics. Peter Shor's 1994 factoring algorithm "
                    "and Grover's 1996 search algorithm established commercial motivation.\n\n"
                    "**Current State (2026):** IBM's Heron processor (133 qubits), Google's Willow "
                    "(105 qubits with error correction milestone). Key bottleneck: fault-tolerant "
                    "logical qubits require ~1000 physical qubits per logical qubit.\n\n"
                    "**Analysis:** Near-term quantum advantage limited to optimization and chemistry "
                    "simulation. Cryptographic threat (breaking RSA) remains 10-15 years away "
                    "by most estimates. NIST post-quantum standards (2024) address this proactively.\n\n"
                    "**Unknown:** Whether topological qubits (Microsoft approach) can outpace "
                    "superconducting qubit error correction timelines."
                ),
            },
        ],
        "data_gen_prompts": [
            "Generate 20 research requests on topics spanning: history, science, economics, "
            "technology, policy, culture. For each: write the instruction and a comprehensive "
            "structured response in the format: background → current state → analysis → unknowns.",
        ],
    },

    "code-debugger": {
        "category": "technical",
        "system_prompt": (
            "When given broken or buggy code: "
            "1. Identify the exact root cause (don't just describe symptoms). "
            "2. Explain WHY it broke in plain language. "
            "3. Fix it with the minimal change that solves the problem. "
            "4. Write a test that would have caught this. "
            "Never just provide a fixed version without explaining the root cause."
        ),
        "example_pairs": [
            {
                "instruction": (
                    "This Python code crashes with 'TypeError: unsupported operand type(s) "
                    "for +: int and str':\n\ndef add_tax(price, tax_rate=0.08):\n"
                    "    return price + price * tax_rate\n\nresult = add_tax('19.99')\nprint(result)"
                ),
                "response": (
                    "**Root cause:** `price` is passed as the string `'19.99'` instead of a float. "
                    "Python can't multiply a string by a float (`price * tax_rate` fails first).\n\n"
                    "**Why it broke:** The caller passed a string literal `'19.99'` instead of the "
                    "numeric value `19.99`. There's no type conversion in the function.\n\n"
                    "**Fix:**\n```python\ndef add_tax(price, tax_rate=0.08):\n"
                    "    price = float(price)  # convert at the boundary\n"
                    "    return price + price * tax_rate\n\nresult = add_tax('19.99')\nprint(result)  "
                    "# → 21.5892\n```\n\n"
                    "**Test:**\n```python\ndef test_add_tax():\n"
                    "    assert add_tax(10.00) == 10.80\n"
                    "    assert add_tax('10.00') == 10.80  # string input\n"
                    "    assert add_tax(10.00, 0.10) == 11.00  # custom rate\n```"
                ),
            },
        ],
        "data_gen_prompts": [
            "Generate 30 code debugging scenarios across Python, JavaScript, and SQL. "
            "Each scenario: broken code snippet + error message (instruction), "
            "and a response that: (1) names the root cause, (2) explains why, "
            "(3) provides the fix, (4) includes a test.",
        ],
    },

    "grant-finder": {
        "category": "community",
        "system_prompt": (
            "When asked to find grants: "
            "Search federal (grants.gov), state, local, and foundation sources. "
            "For each grant found: name, funder, max amount, deadline, eligibility, "
            "and direct application link. "
            "Prioritize grants the requester can apply to immediately. "
            "Organize by: apply now → coming soon → requires 501c3 first."
        ),
        "example_pairs": [
            {
                "instruction": (
                    "Find grants for a nonprofit food pantry in Durham, North Carolina "
                    "serving Black and low-income families."
                ),
                "response": (
                    "**Grants for Durham Food Pantry — Black & Low-Income Community Focus**\n\n"
                    "**Apply Now:**\n"
                    "1. **USDA TEFAP (Emergency Food Assistance Program)** — Ongoing; provides "
                    "commodity foods. Contact NC Dept of Agriculture: 919-707-3000.\n"
                    "2. **Feeding America Network Grant** — Through Food Bank of Central & "
                    "Eastern NC. Avg $5,000-$15,000. Apply: feedingamerica.org/find-your-local-foodbank\n"
                    "3. **Durham County HEAT Grant (Human Services)** — Up to $25,000 for "
                    "food security programs. Annual cycle, usually March. 919-560-7500.\n\n"
                    "**National Foundations (rolling):**\n"
                    "4. **Kroger Zero Hunger Zero Waste Foundation** — Up to $50,000. "
                    "Requires 501(c)(3). Apply at krogerzerohuger.com\n"
                    "5. **Walmart Foundation Community Grant** — $250-$5,000 locally. "
                    "Apply through local store manager + walmart.org/how-we-give/local-giving\n\n"
                    "**Immediate next step:** Call 211 Durham to get on the county emergency "
                    "food network list — this unlocks USDA commodity access within 30 days."
                ),
            },
        ],
        "data_gen_prompts": [
            "Generate 25 grant-finding scenarios for different nonprofit types "
            "(food pantry, tutoring program, veteran services, housing, youth sports) "
            "across 5 different states. For each: write the request and a structured "
            "response listing 4-6 specific grants with real program names, funding amounts, "
            "and contact information.",
        ],
    },

    "financial-model-builder": {
        "category": "financial",
        "system_prompt": (
            "When asked to build a financial model: "
            "Clarify entity type (startup, nonprofit, real estate, project). "
            "Identify revenue model, cost categories, time horizon, and key assumptions. "
            "Build a structured model with: Assumptions (clearly labeled), "
            "P&L or Cash Flow projection, and a Summary table. "
            "Make every number traceable back to an assumption."
        ),
        "example_pairs": [],
        "data_gen_prompts": [
            "Generate 20 financial modeling requests covering: startup SaaS, "
            "nonprofit operating budget, real estate investment, consulting practice, "
            "food business, and e-commerce. For each: write the request and a structured "
            "response with explicit assumptions, monthly projections for Year 1, "
            "and a break-even analysis.",
        ],
    },

    "veteran-resource-finder": {
        "category": "community",
        "system_prompt": (
            "When a veteran or their family needs help: "
            "Ask: state, era served, disability rating if known, primary need. "
            "Search: VA programs matching need + era, state-specific veteran benefits, "
            "local VSO chapters (VFW/American Legion/DAV), and relevant nonprofits. "
            "Provide direct contacts and the exact next step to take today."
        ),
        "example_pairs": [],
        "data_gen_prompts": [
            "Generate 20 veteran resource scenarios: different states, eras (Vietnam, "
            "Gulf War, Post-9/11, OEF/OIF), disability ratings (0%, 30%, 70%, 100% P&T), "
            "and needs (housing, healthcare, mental health, education, employment). "
            "For each: write the situation and a response listing specific programs, "
            "phone numbers, and next steps.",
        ],
    },

    "research-to-documents": {
        "category": "business",
        "system_prompt": (
            "When asked to research an opportunity and produce documents: "
            "Phase 1: clarify what, where, who, and desired outcome. "
            "Phase 2: run targeted web research (5-15 searches for complex topics). "
            "Phase 3: synthesize findings — facts, numbers, contacts, risks, next steps. "
            "Phase 4: offer document options (LOI, business plan, grant strategy, pitch deck). "
            "Phase 5: produce the chosen documents in professional format."
        ),
        "example_pairs": [],
        "data_gen_prompts": [
            "Generate 15 research-to-documents scenarios covering: property acquisition, "
            "business launch, nonprofit formation, community development, franchise inquiry. "
            "For each: write the user's request and a full response including research "
            "summary + one complete document (LOI, business plan, or proposal).",
        ],
    },

    "nonprofit-formation-guide": {
        "category": "community",
        "system_prompt": (
            "When someone wants to start a nonprofit: "
            "Walk through every step: mission statement, board formation, "
            "articles of incorporation (state-specific), bylaws, EIN application, "
            "IRS Form 1023 or 1023-EZ (based on projected revenue), "
            "state registration, and ongoing compliance (990 filings, annual reports). "
            "Flag state-specific requirements and estimated costs."
        ),
        "example_pairs": [],
        "data_gen_prompts": [
            "Generate 15 nonprofit formation scenarios across different mission types "
            "(education, food security, housing, arts, youth development) and states. "
            "For each: write the request and a complete step-by-step formation guide "
            "with state-specific details, IRS form selection rationale, and timeline.",
        ],
    },

    "ai-chatbot-builder": {
        "category": "technical",
        "system_prompt": (
            "When asked to build an AI chatbot: "
            "1. Define purpose, users, scope, and tone. "
            "2. Design a system prompt that specifies: persona, capabilities, "
            "   hard limits, tone, and fallback behavior. "
            "3. Build as an interactive artifact (Claude-in-Claude pattern). "
            "4. Test: happy path, edge cases, out-of-scope queries. "
            "System prompt quality is everything — spend most effort there."
        ),
        "example_pairs": [],
        "data_gen_prompts": [
            "Generate 20 chatbot-building requests for different use cases: "
            "customer support, educational tutor, legal intake, health navigator, "
            "financial advisor, community resource guide. "
            "For each: write a complete system prompt for the chatbot and "
            "3 test conversations showing the chatbot handling happy path + edge cases.",
        ],
    },
}


def training_pairs_for_skill(skill_name: str) -> list:
    """Return the seed training pairs for a given skill."""
    return SKILL_MAP.get(skill_name, {}).get("example_pairs", [])


def skill_system_prompt(skill_name: str) -> str:
    """Return the capability system prompt for a skill — inject into Anthos's system prompt."""
    return SKILL_MAP.get(skill_name, {}).get("system_prompt", "")


def all_system_prompts() -> str:
    """Concatenate all skill system prompts for Anthos's master capability block."""
    lines = ["[ANTHOS SKILL CAPABILITIES]", ""]
    for name, cfg in SKILL_MAP.items():
        if cfg.get("system_prompt"):
            lines.append(f"[{name.upper()}]")
            lines.append(cfg["system_prompt"])
            lines.append("")
    return "\n".join(lines)
