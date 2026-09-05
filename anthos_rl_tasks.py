"""
anthos_rl_tasks.py — Synthetic Python coding tasks with executable unit-test reward.

Each RLTask has:
  prompt   — function header + docstring the model must complete
  tests    — list of (args_tuple, expected_value) pairs
  family   — string grouping similar tasks
  task_id  — unique string

Used by train_rl.py. Does not import from anthos — zero coupling.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field


@dataclass
class RLTask:
    family:  str
    task_id: str
    prompt:  str          # model sees this and must complete the function body
    tests:   list[tuple]  # [(args_tuple, expected_value), ...]


def evaluate_task(task: RLTask, completion: str) -> dict:
    """
    Run the model's completion against the task's unit tests.
    Returns {"passed": bool, "pass_fraction": float, "status": "ok"|"invalid"|"timeout"}.
    """
    full_source = task.prompt + completion

    # Syntax check first — cheap
    try:
        ast.parse(full_source)
    except SyntaxError:
        return {"passed": False, "pass_fraction": 0.0, "status": "invalid"}

    fn_name = _extract_fn_name(task.prompt)
    if fn_name is None:
        return {"passed": False, "pass_fraction": 0.0, "status": "invalid"}

    passed = 0
    for args, expected in task.tests:
        call  = f"{fn_name}({', '.join(repr(a) for a in args)})"
        src   = full_source + f"\n_result = {call}\nprint(repr(_result))"
        ok, out = _run(src)
        if ok and out == repr(expected):
            passed += 1

    total = len(task.tests)
    frac  = passed / total if total else 0.0
    return {"passed": passed == total, "pass_fraction": frac, "status": "ok"}


def get_tasks(families: list[str] | None = None) -> list[RLTask]:
    tasks = _build_tasks()
    if families:
        tasks = [t for t in tasks if t.family in families]
    return tasks


# ── internals ─────────────────────────────────────────────────────────────────

def _extract_fn_name(prompt: str) -> str | None:
    try:
        tree = ast.parse(prompt)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        pass
    return None


def _run(source: str, timeout: float = 4.0) -> tuple[bool, str]:
    import os
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        r = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception:
        return False, ""


def _task(family, task_id, fn_name, doc, body_hint, tests) -> RLTask:
    prompt = f"def {fn_name}({', '.join(str(a) for a in _arg_names(tests))}):\n    \"\"\"{doc}\"\"\"\n"
    return RLTask(family=family, task_id=task_id, prompt=prompt, tests=tests)


def _arg_names(tests):
    """Infer argument names from test count."""
    n = len(tests[0][0])
    return ["x"] if n == 1 else ["x", "y"] if n == 2 else [f"a{i}" for i in range(n)]


def _build_tasks() -> list[RLTask]:
    tasks: list[RLTask] = []

    # ── increment / add-N ────────────────────────────────────────────────────
    for n in [1, 2, 3, 5, 10]:
        tasks.append(RLTask(
            family="increment",
            task_id=f"add_{n}",
            prompt=f"def add_{n}(x):\n    \"\"\"Return x + {n}.\"\"\"\n",
            tests=[((v,), v + n) for v in [0, 1, -1, 7, 100]],
        ))

    # ── multiply ─────────────────────────────────────────────────────────────
    for factor in [2, 3, 5, 10]:
        tasks.append(RLTask(
            family="multiply",
            task_id=f"times_{factor}",
            prompt=f"def times_{factor}(x):\n    \"\"\"Return x * {factor}.\"\"\"\n",
            tests=[((v,), v * factor) for v in [0, 1, 2, -3, 7]],
        ))

    # ── parity ───────────────────────────────────────────────────────────────
    tasks.append(RLTask(
        family="parity",
        task_id="is_even",
        prompt="def is_even(x):\n    \"\"\"Return True if x is even, False otherwise.\"\"\"\n",
        tests=[((v,), v % 2 == 0) for v in [0, 1, 2, 3, 4, 7, 10, 100]],
    ))
    tasks.append(RLTask(
        family="parity",
        task_id="is_odd",
        prompt="def is_odd(x):\n    \"\"\"Return True if x is odd, False otherwise.\"\"\"\n",
        tests=[((v,), v % 2 != 0) for v in [0, 1, 2, 3, 5, 8, 11, 99]],
    ))

    # ── negate / absolute ────────────────────────────────────────────────────
    tasks.append(RLTask(
        family="negate",
        task_id="negate",
        prompt="def negate(x):\n    \"\"\"Return -x.\"\"\"\n",
        tests=[((v,), -v) for v in [0, 1, -1, 5, -100]],
    ))
    tasks.append(RLTask(
        family="absolute",
        task_id="absolute",
        prompt="def absolute(x):\n    \"\"\"Return the absolute value of x.\"\"\"\n",
        tests=[((v,), abs(v)) for v in [0, 1, -1, 5, -100]],
    ))

    # ── clamp ────────────────────────────────────────────────────────────────
    for lo, hi in [(0, 10), (-5, 5), (0, 100)]:
        tasks.append(RLTask(
            family="clamp",
            task_id=f"clamp_{lo}_{hi}",
            prompt=f"def clamp(x):\n    \"\"\"Return x clamped to [{lo}, {hi}].\"\"\"\n",
            tests=[((v,), max(lo, min(hi, v))) for v in [lo - 5, lo, (lo + hi) // 2, hi, hi + 5]],
        ))

    # ── min / max of two ─────────────────────────────────────────────────────
    tasks.append(RLTask(
        family="minmax",
        task_id="two_min",
        prompt="def two_min(x, y):\n    \"\"\"Return the smaller of x and y.\"\"\"\n",
        tests=[((a, b), min(a, b)) for a, b in [(0, 0), (1, 2), (2, 1), (-1, 1), (5, 5)]],
    ))
    tasks.append(RLTask(
        family="minmax",
        task_id="two_max",
        prompt="def two_max(x, y):\n    \"\"\"Return the larger of x and y.\"\"\"\n",
        tests=[((a, b), max(a, b)) for a, b in [(0, 0), (1, 2), (2, 1), (-1, 1), (5, 5)]],
    ))

    # ── square / cube ────────────────────────────────────────────────────────
    tasks.append(RLTask(
        family="power",
        task_id="square",
        prompt="def square(x):\n    \"\"\"Return x squared.\"\"\"\n",
        tests=[((v,), v * v) for v in [0, 1, -1, 3, 10]],
    ))
    tasks.append(RLTask(
        family="power",
        task_id="cube",
        prompt="def cube(x):\n    \"\"\"Return x cubed.\"\"\"\n",
        tests=[((v,), v ** 3) for v in [0, 1, -1, 2, 3]],
    ))

    return tasks
