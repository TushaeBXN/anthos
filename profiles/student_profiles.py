"""
profiles/student_profiles.py — Five student profile definitions for the Anthos math tutor.

Hard rules start from defaults defined here and grow via escalation.py, which appends
lines to profiles/hard_rules/{profile_id}.txt whenever a failure type hits 5+ occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HARD_RULES_DIR = Path(__file__).parent / "hard_rules"


@dataclass
class StudentProfile:
    id: str
    name: str
    description: str
    scaffolding_preference: str
    default_scaffolding_approaches: list[str]
    hard_rules: list[str] = field(default_factory=list)

    def load_escalated_rules(self) -> list[str]:
        """Read any hard rules added by the escalation system."""
        rule_file = HARD_RULES_DIR / f"{self.id}.txt"
        if not rule_file.exists():
            return []
        lines = rule_file.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    def all_hard_rules(self) -> list[str]:
        return self.hard_rules + self.load_escalated_rules()

    def to_system_context(self) -> str:
        rules = self.all_hard_rules()
        lines = [
            f"[STUDENT PROFILE: {self.name}]",
            f"Learning style: {self.description}",
            f"Preferred scaffolding: {self.scaffolding_preference}",
            "Tutoring approach:",
        ]
        for approach in self.default_scaffolding_approaches:
            lines.append(f"  - {approach}")
        if rules:
            lines.append("Hard rules (never violate):")
            for rule in rules:
                lines.append(f"  * {rule}")
        return "\n".join(lines)


PROFILES: dict[str, StudentProfile] = {

    "direct_instruction": StudentProfile(
        id="direct_instruction",
        name="Direct Instruction Learner",
        description=(
            "Prefers clear explanations followed by worked examples. "
            "Dislikes open-ended exploration without guidance. "
            "Responds well to step-by-step structure."
        ),
        scaffolding_preference="Explain the rule first, then walk through an example step by step",
        default_scaffolding_approaches=[
            "State the relevant math rule or principle clearly",
            "Walk through one partial example to anchor understanding",
            "Pause before the final step and ask the student to complete it",
            "Confirm understanding with a targeted check question",
        ],
        hard_rules=[
            "Never give the final numerical answer — stop one step before and ask the student",
            "Always end your response with a guiding question",
            "Keep explanations under 4 sentences before handing control back to the student",
        ],
    ),

    "interactive_discussion": StudentProfile(
        id="interactive_discussion",
        name="Interactive Discussion Learner",
        description=(
            "Learns by talking through problems. Wants to be asked questions, "
            "not lectured at. Engages most when the tutor builds on their ideas."
        ),
        scaffolding_preference="Ask questions that activate prior knowledge before introducing new concepts",
        default_scaffolding_approaches=[
            "Open with 'What do you already know about [concept]?'",
            "Reflect the student's idea back and probe deeper: 'You said X — what makes you think that?'",
            "Use Socratic questioning to guide toward the correct approach",
            "Let the student reach the answer through dialogue, never supply it",
        ],
        hard_rules=[
            "Never answer a question with a statement — always respond with a question or reflection",
            "Every response must contain at least one question mark",
            "Do not supply any intermediate numerical value the student hasn't calculated themselves",
        ],
    ),

    "hands_on": StudentProfile(
        id="hands_on",
        name="Hands-On / Real-World Learner",
        description=(
            "Needs concrete, real-world context before abstract rules make sense. "
            "Learns best when math is anchored to something tangible."
        ),
        scaffolding_preference="Anchor every concept in a real-world scenario, then extract the math",
        default_scaffolding_approaches=[
            "Introduce the math through a concrete real-world setup (money, distance, time, etc.)",
            "Ask the student to estimate or reason about the real situation first",
            "Show how the math mirrors what they already know from experience",
            "Ask them to plug in numbers from the scenario before generalizing",
        ],
        hard_rules=[
            "Always introduce math through a real-world story or situation, never in the abstract",
            "Never give the numerical answer — ask 'what would that look like in our scenario?'",
            "End every response by asking the student to apply or test something themselves",
        ],
    ),

    "creative_narrative": StudentProfile(
        id="creative_narrative",
        name="Creative / Narrative Learner",
        description=(
            "Engages through stories, games, and creative framing. "
            "Dry procedural instruction causes disengagement. "
            "Math becomes memorable when embedded in narrative."
        ),
        scaffolding_preference="Frame math problems as story challenges or game puzzles",
        default_scaffolding_approaches=[
            "Cast the math problem as a story quest or character challenge",
            "Use gamification cues: 'Your character needs to figure out X to unlock the next level'",
            "Ask creative what-if questions to explore the concept",
            "Let the student invent the context or name the characters in the scenario",
        ],
        hard_rules=[
            "Never present math as dry procedure — always embed it in a story or game context",
            "Do not give away numerical answers; instead say 'what does your character discover?'",
            "End with an open narrative question that requires math to answer",
        ],
    ),

    "inquiry_based": StudentProfile(
        id="inquiry_based",
        name="Inquiry-Based / Discovery Learner",
        description=(
            "Motivated by figuring things out independently. "
            "Feels undermined when given answers. "
            "Thrives on guided discovery and pattern recognition."
        ),
        scaffolding_preference="Present structured observations and let the student infer the rule",
        default_scaffolding_approaches=[
            "Show a pattern or set of examples without naming the rule",
            "Ask 'What do you notice? What do you wonder?'",
            "Offer minimal hints only when the student is stuck, never the answer",
            "Celebrate partial insight: 'That's exactly the right observation — what follows from it?'",
        ],
        hard_rules=[
            "Never state the rule or answer — guide the student to discover it themselves",
            "Respond to stuck students with a question, not an explanation",
            "Always end with a discovery prompt: 'What pattern do you see?' or 'What would happen if?'",
        ],
    ),
}


def get_profile(profile_id: str) -> StudentProfile:
    if profile_id not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{profile_id}'. Valid options: {valid}")
    return PROFILES[profile_id]


def add_hard_rule(profile_id: str, rule: str) -> None:
    """Append an escalated hard rule to the profile's rule file."""
    HARD_RULES_DIR.mkdir(parents=True, exist_ok=True)
    rule_file = HARD_RULES_DIR / f"{profile_id}.txt"
    existing = rule_file.read_text(encoding="utf-8").splitlines() if rule_file.exists() else []
    if rule not in existing:
        with rule_file.open("a", encoding="utf-8") as f:
            f.write(rule + "\n")
