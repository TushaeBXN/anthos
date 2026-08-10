"""
Context engineering loop for the Nia Squad multi-agent system.

Six layers:
  WRITE     — append_learning()       record each session outcome per agent
  SHARED    — synthesize_squad()      Nia promotes cross-agent patterns
  SELECT    — load_agent_context()    each agent gets SOUL + last-10 + relevant shared
  VERIFIER  — nia_verdict()           gate every output before delivery
  COMPRESS  — compress_learnings()    archive oldest-20 when >40 entries
  ESCALATE  — check_escalation()      3 rejections of same pattern → hard rule
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

SQUAD_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = SQUAD_ROOT / "agents"
SHARED_DIR = SQUAD_ROOT / "shared"
REJECTION_LOG = SHARED_DIR / "rejection_log.json"

AGENTS = ["nia", "mike", "kelly", "keisha", "david", "pamela"]

MAX_ENTRIES = 40
ARCHIVE_COUNT = 20
MAX_CONTEXT_ENTRIES = 10
ESCALATION_THRESHOLD = 3

# ── internal helpers ──────────────────────────────────────────────────────────

def _learnings_path(agent: str) -> Path:
    return AGENTS_DIR / agent / "learnings.md"


def _soul_path(agent: str) -> Path:
    return AGENTS_DIR / agent / "SOUL.md"


# Matches the two-line entry format written by append_learning()
_ENTRY_RE = re.compile(
    r"^- date: (?P<date>\S+) \| task: (?P<task>[^|]+) \| decision: (?P<decision>.+)\n"
    r"  outcome: (?P<outcome>.+)$",
    re.MULTILINE,
)


def _parse_raw_entries(content: str) -> list[re.Match]:
    return list(_ENTRY_RE.finditer(content))


# ── WRITE LAYER ───────────────────────────────────────────────────────────────

def append_learning(agent: str, task: str, decision: str, outcome: str) -> None:
    """Append one structured entry to agents/<agent>/learnings.md.

    Entry format (exactly 2 lines, max ~120 chars each):
      - date: YYYY-MM-DD | task: <task> | decision: <decision>
        outcome: <outcome>
    """
    path = _learnings_path(agent)
    path.parent.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    # Truncate fields so entries stay readable
    task = task[:80].strip()
    decision = decision[:80].strip()
    outcome = outcome[:100].strip()

    entry = (
        f"- date: {today} | task: {task} | decision: {decision}\n"
        f"  outcome: {outcome}\n"
    )

    with path.open("a") as fh:
        fh.write(entry)

    compress_learnings(agent)


# ── COMPRESS LAYER ────────────────────────────────────────────────────────────

def compress_learnings(agent: str) -> None:
    """When learnings.md exceeds MAX_ENTRIES raw entries, archive the oldest ARCHIVE_COUNT.

    Archive block format:
      ## ARCHIVE [YYYY-MM-DD] — N entries | tasks: … | outcomes: …
    """
    path = _learnings_path(agent)
    if not path.exists():
        return

    content = path.read_text()
    matches = _parse_raw_entries(content)

    if len(matches) <= MAX_ENTRIES:
        return

    to_archive = matches[:ARCHIVE_COUNT]
    to_keep = matches[ARCHIVE_COUNT:]

    # Summarise archived entries
    tasks = [m.group("task").strip() for m in to_archive]
    outcomes = [m.group("outcome").strip() for m in to_archive]
    date_range = f"{to_archive[0].group('date')} → {to_archive[-1].group('date')}"
    top_outcomes = Counter(o.split()[0].lower() for o in outcomes if o).most_common(3)
    outcome_summary = ", ".join(f"{w}×{c}" for w, c in top_outcomes)
    task_preview = "; ".join(tasks[:4]) + (f" … +{len(tasks)-4}" if len(tasks) > 4 else "")

    archive_block = (
        f"## ARCHIVE [{date.today().isoformat()}] — {len(to_archive)} entries | "
        f"period: {date_range}\n"
        f"   tasks: {task_preview}\n"
        f"   outcomes: {outcome_summary}\n\n"
    )

    # Preserve any existing archive blocks verbatim, prepend new one, then raw entries
    existing_archives = "".join(
        m.group(0)
        for m in re.finditer(
            r"^## ARCHIVE .+?(?=\n## ARCHIVE |\Z)", content, re.DOTALL | re.MULTILINE
        )
    )

    header_line = content.splitlines()[0] + "\n" if content.startswith("#") else ""
    rebuilt = header_line
    if existing_archives.strip():
        rebuilt += existing_archives.rstrip() + "\n\n"
    rebuilt += archive_block
    rebuilt += "".join(m.group(0) + "\n" for m in to_keep)

    path.write_text(rebuilt)


# ── SELECT LAYER ──────────────────────────────────────────────────────────────

def load_agent_context(agent: str) -> dict:
    """Return the context packet an agent receives at the start of a task.

    Contains:
      soul            — full SOUL.md text
      personal        — list of last MAX_CONTEXT_ENTRIES raw entry strings
      shared          — list of relevant shared-learning blocks (keyword-matched)
      hard_rules      — full hard_rules.md text
    """
    soul_path = _soul_path(agent)
    soul = soul_path.read_text() if soul_path.exists() else ""

    # Last N raw entries
    learn_path = _learnings_path(agent)
    personal: list[str] = []
    if learn_path.exists():
        matches = _parse_raw_entries(learn_path.read_text())
        personal = [m.group(0) for m in matches[-MAX_CONTEXT_ENTRIES:]]

    # Hard rules (always loaded in full)
    hard_rules_path = SHARED_DIR / "hard_rules.md"
    hard_rules = hard_rules_path.read_text() if hard_rules_path.exists() else ""

    # Relevant shared learnings — blocks whose keywords overlap with agent's SOUL
    squad_path = SHARED_DIR / "squad_learnings.md"
    shared: list[str] = []
    if squad_path.exists():
        soul_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", soul.lower()))
        for block in re.split(r"\n{2,}", squad_path.read_text()):
            if not block.strip() or re.match(r"^# [^#]", block):
                continue  # skip top-level title only; keep ## section blocks
            block_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", block.lower()))
            if soul_words & block_words:  # any overlap → include
                shared.append(block.strip())

    return {
        "agent": agent,
        "soul": soul,
        "personal": personal,
        "shared": shared,
        "hard_rules": hard_rules,
    }


# ── SHARED LEARNINGS ──────────────────────────────────────────────────────────

def synthesize_squad_learnings() -> None:
    """Nia reads every agent's learnings.md and writes cross-agent patterns.

    Only patterns appearing across two or more agents are promoted.
    Called by Nia at session start before tasks are dispatched.
    """
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all entries keyed by agent
    all_entries: dict[str, list[re.Match]] = {}
    for agent in AGENTS:
        path = _learnings_path(agent)
        if path.exists():
            all_entries[agent] = _parse_raw_entries(path.read_text())

    # Build keyword → set-of-agents index over task+decision fields
    keyword_agents: dict[str, set[str]] = defaultdict(set)
    keyword_examples: dict[str, list[str]] = defaultdict(list)

    for agent, entries in all_entries.items():
        for m in entries:
            searchable = f"{m.group('task')} {m.group('decision')}"
            for word in set(re.findall(r"\b[a-zA-Z]{4,}\b", searchable.lower())):
                keyword_agents[word].add(agent)
                if len(keyword_examples[word]) < 3:
                    keyword_examples[word].append(
                        f"[{agent}] {m.group('task').strip()} → {m.group('outcome').strip()}"
                    )

    # Keep only cross-agent patterns (2+ agents)
    patterns = {
        kw: agents
        for kw, agents in keyword_agents.items()
        if len(agents) >= 2
    }

    if not patterns:
        # Nothing to promote yet; leave squad_learnings.md as-is
        return

    today = date.today().isoformat()
    lines: list[str] = [
        "# Squad Learnings\n",
        f"_Synthesized by Nia on {today}. "
        "Only patterns seen across 2+ agents appear here._\n",
    ]

    # Sort by breadth (most agents first) then alphabetically
    for kw, agents in sorted(patterns.items(), key=lambda x: (-len(x[1]), x[0])):
        agent_list = ", ".join(sorted(agents))
        lines.append(f"## Pattern: `{kw}` — seen across: {agent_list}")
        for ex in keyword_examples[kw][:3]:
            lines.append(f"- {ex}")
        lines.append("")

    (SHARED_DIR / "squad_learnings.md").write_text("\n".join(lines))


# ── VERIFIER GATE ─────────────────────────────────────────────────────────────

def nia_verdict(agent: str, output: str) -> tuple[bool, str]:
    """Nia's Minister of Verdicts check.

    Checks (in order):
      1. Hard rules — explicit RULE: lines in hard_rules.md
      2. SOUL prohibitions — 'Never' / 'Must not' lines in the agent's SOUL.md
      3. Lane violations — output references another agent's exclusive domain

    Returns:
      (True,  "approved")          — output passes
      (False, "<reason>")          — output rejected; reason returned to originating agent
    """
    soul_path = _soul_path(agent)
    soul = soul_path.read_text() if soul_path.exists() else ""
    output_lower = output.lower()

    # 1. Hard rules
    hard_path = SHARED_DIR / "hard_rules.md"
    if hard_path.exists():
        for line in hard_path.read_text().splitlines():
            if not line.startswith("RULE:"):
                continue
            rule_text = line[5:].split("[escalated:")[0].strip().lower()
            rule_keywords = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", rule_text)]
            # Require at least 2 of the first 4 rule keywords to appear in output
            trigger_words = rule_keywords[:4]
            hits = sum(1 for w in trigger_words if w in output_lower)
            if trigger_words and hits >= min(2, len(trigger_words)):
                reason = f"Violates hard rule: {line[5:].strip()}"
                _record_rejection(agent, reason)
                check_escalation(reason)
                return False, reason

    # 2. SOUL prohibitions
    for line in soul.splitlines():
        stripped = line.strip()
        if not re.match(r"(?i)^(never|must not)[:\s]", stripped):
            continue
        prohibition = re.sub(r"(?i)^(never|must not)[:\s]*", "", stripped).strip()
        keywords = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", prohibition.lower())]
        if len(keywords) >= 2:
            hits = sum(1 for w in keywords[:4] if w in output_lower)
            if hits >= min(2, len(keywords[:4])):
                reason = (
                    f"Contradicts {agent}'s SOUL.md — prohibited behaviour: "
                    f"{prohibition[:120]}"
                )
                _record_rejection(agent, reason)
                check_escalation(reason)
                return False, reason

    # 3. Lane violations — check for exclusive domain keywords from other agents
    violated = _check_lane_violation(agent, soul, output_lower)
    if violated:
        reason = violated
        _record_rejection(agent, reason)
        check_escalation(reason)
        return False, reason

    return True, "approved"


def _check_lane_violation(agent: str, soul: str, output_lower: str) -> str | None:
    """Return a violation reason string if output steps into another agent's exclusive lane."""
    # Extract this agent's lane keywords
    own_lane_match = re.search(r"(?i)^## Lane\b.+?(?=\n##|\Z)", soul, re.DOTALL | re.MULTILINE)
    own_lane_text = own_lane_match.group(0).lower() if own_lane_match else ""
    own_lane_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", own_lane_text))

    for other in AGENTS:
        if other == agent:
            continue
        other_soul_path = _soul_path(other)
        if not other_soul_path.exists():
            continue
        other_soul = other_soul_path.read_text()
        other_lane_match = re.search(
            r"(?i)^## Lane\b.+?(?=\n##|\Z)", other_soul, re.DOTALL | re.MULTILINE
        )
        if not other_lane_match:
            continue
        other_lane = other_lane_match.group(0).lower()
        other_lane_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", other_lane))

        # Words exclusive to the other agent's lane (not shared with this agent's lane)
        exclusive = other_lane_words - own_lane_words
        # Only flag if multiple exclusive terms appear together in the output
        hits = [w for w in exclusive if w in output_lower]
        if len(hits) >= 3:
            preview = ", ".join(hits[:4])
            return (
                f"Lane violation: {agent} output contains terms exclusive to "
                f"{other}'s domain ({preview})"
            )
    return None


# ── ESCALATION ────────────────────────────────────────────────────────────────

def _load_rejection_log() -> dict:
    if REJECTION_LOG.exists():
        return json.loads(REJECTION_LOG.read_text())
    return {}


def _save_rejection_log(log: dict) -> None:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    REJECTION_LOG.write_text(json.dumps(log, indent=2))


def _pattern_key(reason: str) -> str:
    """Stable normalised key for counting rejection patterns."""
    words = sorted(set(re.findall(r"\b[a-zA-Z]{4,}\b", reason.lower())))
    return " ".join(words[:10])


def _record_rejection(agent: str, reason: str) -> None:
    log = _load_rejection_log()
    key = _pattern_key(reason)
    entry = log.setdefault(
        key,
        {"count": 0, "agents": [], "reason": reason, "promoted": False},
    )
    entry["count"] += 1
    if agent not in entry["agents"]:
        entry["agents"].append(agent)
    _save_rejection_log(log)


def check_escalation(rejection_reason: str) -> None:
    """Promote a rejection pattern to hard_rules.md once it hits ESCALATION_THRESHOLD."""
    log = _load_rejection_log()
    key = _pattern_key(rejection_reason)
    if key not in log:
        return
    entry = log[key]
    if entry["count"] >= ESCALATION_THRESHOLD and not entry.get("promoted"):
        _promote_to_hard_rule(entry["reason"])
        entry["promoted"] = True
        _save_rejection_log(log)


def _promote_to_hard_rule(reason: str) -> None:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    hard_path = SHARED_DIR / "hard_rules.md"
    existing = hard_path.read_text() if hard_path.exists() else "# Hard Rules\n"
    today = date.today().isoformat()
    new_rule = f"RULE: {reason} [escalated: {today}]\n"
    # Idempotent — don't add duplicates
    if new_rule.strip() not in existing:
        hard_path.write_text(existing.rstrip() + "\n" + new_rule)
