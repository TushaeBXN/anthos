"""
Nia Squad — session orchestrator.

Usage:
  python orchestrate.py run                  # run an interactive session
  python orchestrate.py context <agent>      # print an agent's context packet
  python orchestrate.py synthesize           # Nia synthesizes squad learnings now
  python orchestrate.py log <agent> <task> <decision> <outcome>  # manual learning entry
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Make loop importable from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from loop.context_loop import (
    AGENTS,
    append_learning,
    check_escalation,
    load_agent_context,
    nia_verdict,
    synthesize_squad_learnings,
)


# ── session runner ────────────────────────────────────────────────────────────

class SquadSession:
    """Encapsulates one squad working session.

    Call flow per task:
      1. session_start()       — Nia synthesizes squad learnings
      2. get_context(agent)    — load SOUL + personal + shared + hard_rules
      3. submit_output(agent, output)   — run verifier gate
         └─ on pass  → record learning, return output
         └─ on fail  → return rejection with reason (agent should revise)
      4. session_end()         — no-op hook for future session teardown
    """

    def __init__(self) -> None:
        self._rejection_counts: dict[str, int] = {}  # agent → consecutive rejects

    def session_start(self) -> None:
        """Nia synthesizes squad learnings at the top of every session."""
        print("[Nia / session-start] Reading all agents' learnings …")
        synthesize_squad_learnings()
        print("[Nia / session-start] Squad learnings synthesized. Hard rules loaded.")

    def get_context(self, agent: str) -> dict:
        """Return the context packet for an agent."""
        if agent not in AGENTS:
            raise ValueError(f"Unknown agent '{agent}'. Valid: {AGENTS}")
        return load_agent_context(agent)

    def submit_output(
        self,
        agent: str,
        output: str,
        task: str,
        decision: str,
    ) -> tuple[bool, str]:
        """Run the verifier gate and, on pass, record the learning.

        Returns:
          (True,  output)          — approved; caller receives the output
          (False, reason)          — rejected; reason is returned to originating agent
        """
        passed, verdict = nia_verdict(agent, output)

        if passed:
            outcome = f"approved by Nia — output delivered"
            append_learning(agent, task, decision, outcome)
            self._rejection_counts.pop(agent, None)
            return True, output
        else:
            # Track consecutive rejections per agent (for logging purposes;
            # escalation is handled inside check_escalation via the log file)
            self._rejection_counts[agent] = self._rejection_counts.get(agent, 0) + 1
            outcome = f"rejected — {verdict[:80]}"
            append_learning(agent, task, decision, outcome)
            return False, verdict

    def session_end(self) -> None:
        pass


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _print_context(agent: str) -> None:
    ctx = load_agent_context(agent)
    print(f"\n{'='*60}")
    print(f"CONTEXT PACKET FOR: {agent.upper()}")
    print(f"{'='*60}")
    print("\n--- SOUL (truncated to 10 lines) ---")
    print("\n".join(ctx["soul"].splitlines()[:10]))
    print("\n--- LAST PERSONAL LEARNINGS ---")
    if ctx["personal"]:
        for entry in ctx["personal"]:
            print(textwrap.indent(entry, "  "))
    else:
        print("  (none yet)")
    print("\n--- RELEVANT SHARED LEARNINGS ---")
    if ctx["shared"]:
        for block in ctx["shared"]:
            print(textwrap.indent(block, "  "))
    else:
        print("  (none yet)")
    print("\n--- HARD RULES ---")
    if ctx["hard_rules"].strip():
        print(textwrap.indent(ctx["hard_rules"].strip(), "  "))
    else:
        print("  (none yet)")


def _interactive_session() -> None:
    session = SquadSession()
    session.session_start()

    print("\nNia Squad — interactive session")
    print("Commands: <agent> | context <agent> | quit")
    print(f"Agents: {', '.join(AGENTS)}\n")

    while True:
        try:
            raw = input("squad> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        parts = raw.split(None, 1)
        cmd = parts[0].lower()

        if cmd == "context" and len(parts) == 2:
            _print_context(parts[1].lower())
            continue

        if cmd in AGENTS:
            agent = cmd
            task = input(f"  task for {agent}: ").strip()
            decision = input(f"  decision made: ").strip()
            output = input(f"  output to verify: ").strip()
            passed, result = session.submit_output(agent, output, task, decision)
            status = "APPROVED" if passed else "REJECTED"
            print(f"  [{status}] {result[:200]}")
            continue

        print(f"  Unknown command. Try: context <agent>, or type an agent name.")

    session.session_end()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "run":
        _interactive_session()

    elif args[0] == "context" and len(args) == 2:
        _print_context(args[1].lower())

    elif args[0] == "synthesize":
        synthesize_squad_learnings()
        print("Done — squad_learnings.md updated.")

    elif args[0] == "log" and len(args) == 5:
        _, agent, task, decision, outcome = args
        if agent not in AGENTS:
            print(f"Unknown agent '{agent}'. Valid: {AGENTS}")
            sys.exit(1)
        append_learning(agent, task, decision, outcome)
        print(f"Learning appended for {agent}.")

    else:
        print(__doc__)
        sys.exit(1)
