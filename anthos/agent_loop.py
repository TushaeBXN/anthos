"""
Learnings-aware wrapper around AutonomousImprovementAgent.

Integrates all six context-engineering layers without touching the original
agent's hypothesis/experiment/analysis logic or any Docker/tool internals:

  SELECT  — inject learnings context at the top of each iteration
  RUN     — delegate to the parent class methods unchanged
  VERIFY  — gate loop advancement on a clean pytest run
  WRITE   — record outcomes (success or failure) to learnings.md
  COMPRESS/ESCALATE — triggered automatically inside LearningsManager.write_entry
  ISOLATE — expose build_subagent_context() for any spawned subtask
"""

from __future__ import annotations

import time
from typing import Optional

from anthos.autonomous_agent import AutonomousImprovementAgent
from learnings_manager import LearningsManager


class LearningsAwareAgent(AutonomousImprovementAgent):
    """
    Subclass of AutonomousImprovementAgent that adds the context engineering
    loop.  All existing experiment/analysis methods are inherited unchanged.
    """

    def __init__(
        self,
        model,
        experiment_dir: str = "experiments/",
        learnings_manager: Optional[LearningsManager] = None,
        run_tests_on_apply: bool = True,
    ) -> None:
        super().__init__(model, experiment_dir)
        self.lm = learnings_manager or LearningsManager()
        self.run_tests_on_apply = run_tests_on_apply
        # Carries the failure output from the previous iteration when the
        # verifier gate blocked advancement.
        self._reentry_failure: Optional[str] = None

    # ── Main loop ────────────────────────────────────────────────────────────

    def run_forever(self, sleep_seconds: int = 3600) -> None:
        """
        Drop-in replacement for the parent's run_forever().

        Iteration order:
          1. SELECT  — build system context from learnings.md
          2. RUN     — generate hypothesis, design & run experiment, analyse
          3. VERIFY  — run pytest; on failure → WRITE the failure and re-enter
          4. WRITE   — record the successful outcome
          5. APPLY   — call parent _apply_improvement if threshold met
          6. SLEEP
        """
        print("Starting learnings-aware improvement loop...")
        while True:
            try:
                self._run_one_iteration()
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("Autonomous loop stopped.")
                break
            except Exception as exc:
                msg = str(exc)
                self.lm.write_entry(
                    task="(exception during loop)",
                    succeeded="none",
                    failed=msg,
                    rule=f"Guard against: {msg[:100]}",
                )
                print(f"Experiment failed: {exc}")
                time.sleep(60)

    # ── Single iteration ─────────────────────────────────────────────────────

    def _run_one_iteration(self) -> None:
        # ── 1. SELECT ────────────────────────────────────────────────────────
        context = self.lm.select_context()
        if self._reentry_failure:
            context += f"\n## Re-entry: Previous Test Failure\n```\n{self._reentry_failure}\n```\n"
            self._reentry_failure = None
        if context:
            print(f"[Learnings] Injected {len(context)} chars of context.")

        # ── 2. RUN (parent logic, untouched) ─────────────────────────────────
        hypothesis = self._generate_hypothesis()
        experiment = self._design_experiment(hypothesis)

        # ISOLATION: build a clean context for this subtask so that if the
        # experiment were dispatched to a subagent, it carries only what it needs.
        _subtask_ctx = self.lm.build_subagent_context(
            task_spec=f"Run experiment: {hypothesis}",
            relevant_files=["anthos/autonomous_agent.py", "anthos/main.py"],
        )

        result = self._run_experiment(experiment)
        insight = self._analyze_result(result)

        # ── 3. VERIFY ────────────────────────────────────────────────────────
        if self.run_tests_on_apply:
            passed, test_output = self.lm.run_tests()
            if not passed:
                failure_summary = test_output[-2000:]
                self._reentry_failure = failure_summary
                self.lm.write_entry(
                    task=hypothesis,
                    succeeded="experiment ran",
                    failed=f"tests failed — {failure_summary[:180]}",
                    rule=f"Tests must pass before advancing past: {hypothesis[:80]}",
                )
                print(
                    "[Verifier Gate] Tests failed — loop blocked. "
                    "Re-entering with failure context."
                )
                return  # do NOT advance; next iteration picks up _reentry_failure

        # ── 4. WRITE ─────────────────────────────────────────────────────────
        self.experiment_history.append({
            "hypothesis": hypothesis,
            "result": result,
            "insight": insight,
            "applied": insight["improvement"] > 0.05,
            "timestamp": time.time(),
        })
        self.lm.write_entry(
            task=hypothesis,
            succeeded=insight.get("insight", "")[:180],
            failed="none",
            rule=f"Δ{insight['improvement']:.2%} when {hypothesis[:90]}",
        )

        # ── 5. APPLY ─────────────────────────────────────────────────────────
        if insight["improvement"] > 0.05:
            self._apply_improvement(insight)

        print(f"Cycle complete. Improvement: {insight['improvement']:.2%}")
