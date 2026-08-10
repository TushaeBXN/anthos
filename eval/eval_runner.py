"""
eval/eval_runner.py — CLI that integrates EvalContextLoop with AnthosBenchmark.

Runs all benchmark tasks for a given training phase/modality, threads
results through the full context engineering loop (verify → write → escalate
→ compress → feedback), and exits with a non-zero code when the verifier gate
blocks the result.

Usage examples
--------------
# Basic text eval after a proof-tier checkpoint:
    python eval/eval_runner.py \\
        --tier proof --variant anthos_1b --modality text \\
        --checkpoint checkpoints/mansa_sovereign/step_010000.pt

# Instruction phase with community-routing data:
    python eval/eval_runner.py \\
        --tier instruct --variant anthos_3b --modality text \\
        --checkpoint path/to/ckpt.pt \\
        --routing-hard 0.55 --routing-easy 0.45

# Colibri (50B+ disk-streaming) variant:
    python eval/eval_runner.py \\
        --tier research --variant anthos_50b --modality text \\
        --checkpoint path/to/ckpt.pt \\
        --expert-activation-rate 0.38

# Print current targeting flags only (no eval):
    python eval/eval_runner.py \\
        --flags-only --tier instruct --modality text

# Inject pre-computed metrics (skip model inference):
    python eval/eval_runner.py \\
        --tier proof --variant anthos_1b --modality text \\
        --metrics '{"gsm8k": 0.72, "mmlu": 0.58, "humaneval": 0.45}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or eval/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval_context_loop import (
    EvalContextLoop,
    TIER_TO_PHASE,
    ALL_MODALITIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Failure pattern detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_failure_patterns(results: dict[str, float]) -> list[str]:
    """
    Heuristically classify failure patterns from benchmark scores.
    Caller may extend this list with domain-specific checks.
    """
    patterns = []
    if not results:
        return ["no_metrics_collected"]

    avg = sum(v for v in results.values() if v is not None) / max(len(results), 1)

    if avg < 0.30:
        patterns.append("low_overall_accuracy")
    if results.get("gsm8k", 1.0) < 0.40:
        patterns.append("weak_math_reasoning")
    if results.get("humaneval", 1.0) < 0.30:
        patterns.append("weak_code_generation")
    if results.get("truthfulqa", 1.0) < 0.40:
        patterns.append("low_truthfulness")
    if results.get("mmlu", 1.0) < 0.30:
        patterns.append("low_knowledge_breadth")

    return patterns if patterns else []


def _missing_data_note(failures: list[str], modality: str) -> str:
    """Generate the one-line 'missing' note from failure patterns."""
    notes = []
    if "weak_math_reasoning" in failures:
        notes.append("chain-of-thought math data")
    if "weak_code_generation" in failures:
        notes.append("diverse coding instruction pairs")
    if "low_truthfulness" in failures:
        notes.append("calibration and factual grounding data")
    if "low_knowledge_breadth" in failures:
        notes.append("broad domain coverage across MMLU subjects")
    if modality == "vision" and not notes:
        notes.append("paired vision-language examples")
    if modality == "audio" and not notes:
        notes.append("speech-to-text aligned training pairs")
    return "; ".join(notes) if notes else "no obvious data gap identified"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks(
    checkpoint_path: str,
    limit_per_task: int = 100,
    modality: str = "text",
) -> dict[str, float]:
    """
    Load a checkpoint and run AnthosBenchmark.  Falls back gracefully when the
    heavy ML dependencies (torch, datasets) are absent.
    """
    try:
        import torch
        from anthos.main    import Anthos
        from anthos.configs import get_model_config
    except ImportError as e:
        print(f"[eval_runner] Cannot import torch/anthos: {e}")
        print("[eval_runner] Re-run with torch installed, or pass --metrics directly.")
        return {}

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[eval_runner] Checkpoint not found: {ckpt_path}")
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval_runner] Loading checkpoint from {ckpt_path} on {device}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model_state = ckpt.get("model", ckpt)

    # Infer config from checkpoint metadata when available
    cfg_name = ckpt.get("variant", None)
    try:
        from anthos.configs import get_model_config
        model_cfg = get_model_config(cfg_name) if cfg_name else None
    except Exception:
        model_cfg = None

    if model_cfg is None:
        # Minimal fallback: derive vocab_size from embedding weight shape
        embed_key = next((k for k in model_state if "embed" in k and "weight" in k), None)
        vocab_size = model_state[embed_key].shape[0] if embed_key else 50257
        from anthos.main import AnthosConfig
        model_cfg = AnthosConfig(vocab_size=vocab_size)

    model = Anthos(model_cfg).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    from transformers import AutoTokenizer
    tok_path = "data/anthos_tokenizer" if Path("data/anthos_tokenizer").exists() else "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    from anthos.benchmark_suite import AnthosBenchmark
    bench = AnthosBenchmark(model, tokenizer)
    results = bench.run_all(limit_per_task=limit_per_task)
    return {k: v for k, v in results.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Anthos eval with context engineering loop"
    )

    # Identity
    parser.add_argument(
        "--tier",
        choices=sorted(TIER_TO_PHASE.keys()),
        help="Training tier (maps to canonical phase)",
    )
    parser.add_argument(
        "--phase",
        choices=["Alignment", "Pretraining", "Instruction"],
        help="Explicit phase override (skips tier mapping)",
    )
    parser.add_argument(
        "--variant",
        default="anthos_1b",
        help="Model variant name, e.g. anthos_1b, anthos_50b",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "vision", "audio", "multi"],
        default="text",
        help="Modality being evaluated",
    )
    parser.add_argument(
        "--modalities-tested",
        nargs="+",
        choices=ALL_MODALITIES,
        help="Actual modalities exercised (required when --modality multi)",
    )

    # Eval source
    parser.add_argument("--checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--metrics",
        help="Pre-computed metrics as JSON string, e.g. '{\"gsm8k\": 0.72}'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max samples per benchmark task",
    )

    # Community routing
    parser.add_argument("--routing-hard", type=float, help="Hard-token routing fraction")
    parser.add_argument("--routing-easy", type=float, help="Easy-token routing fraction")

    # Colibri
    parser.add_argument(
        "--expert-activation-rate",
        type=float,
        help="MoE expert activation rate per forward pass (Colibri 34B+ only)",
    )

    # Misc
    parser.add_argument(
        "--flags-only",
        action="store_true",
        help="Print current targeting flags without running an eval",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print loaded context (recent entries + hard rules) before running",
    )

    args = parser.parse_args()

    # ── Resolve phase ─────────────────────────────────────────────────────────
    if args.phase:
        phase = args.phase
    elif args.tier:
        phase = TIER_TO_PHASE.get(args.tier)
        if not phase:
            print(f"[eval_runner] Unknown tier '{args.tier}'", file=sys.stderr)
            return 2
    else:
        if not args.flags_only:
            print("[eval_runner] Provide --tier or --phase", file=sys.stderr)
            return 2
        phase = "Pretraining"  # default for --flags-only

    modality = args.modality

    loop = EvalContextLoop()

    # ── Flags-only mode ───────────────────────────────────────────────────────
    if args.flags_only:
        flags = loop.get_dataset_targeting_flags(phase=phase, modality=modality)
        print(json.dumps(flags, indent=2))
        return 0

    # ── Context preview ───────────────────────────────────────────────────────
    if args.show_context:
        ctx = loop.select_context(phase=phase, modality=modality)
        print(f"\n[eval_runner] SELECT context for ({phase}, {modality})")
        print(f"  total prior runs : {ctx['total_for_combo']}")
        print(f"  recent entries   : {len(ctx['recent_entries'])}")
        print(f"  hard rules       : {len(ctx['hard_rules'])}")
        for rule in ctx["hard_rules"]:
            print(f"    ⚠ {rule}")
        print()

    # ── Collect metrics ───────────────────────────────────────────────────────
    if args.metrics:
        try:
            raw_metrics = json.loads(args.metrics)
            metrics = {k: float(v) for k, v in raw_metrics.items()}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[eval_runner] --metrics parse error: {e}", file=sys.stderr)
            return 2
    elif args.checkpoint:
        metrics = run_benchmarks(
            checkpoint_path=args.checkpoint,
            limit_per_task=args.limit,
            modality=modality,
        )
    else:
        print(
            "[eval_runner] Provide --checkpoint or --metrics (or --flags-only).",
            file=sys.stderr,
        )
        return 2

    if not metrics:
        print("[eval_runner] No metrics collected — aborting.", file=sys.stderr)
        return 1

    # ── Build routing distribution ────────────────────────────────────────────
    routing_distribution: dict[str, float] | None = None
    if args.routing_hard is not None and args.routing_easy is not None:
        total = args.routing_hard + args.routing_easy
        routing_distribution = {
            "hard": args.routing_hard / total,
            "easy": args.routing_easy / total,
        }
    elif args.routing_hard is not None or args.routing_easy is not None:
        print(
            "[eval_runner] Provide both --routing-hard and --routing-easy together.",
            file=sys.stderr,
        )

    # ── Colibri expert rate ───────────────────────────────────────────────────
    expert_activation_rate: float | None = args.expert_activation_rate
    if expert_activation_rate is None and EvalContextLoop.is_colibri_variant(args.variant):
        print(
            f"[eval_runner] WARNING: variant '{args.variant}' looks like a Colibri"
            " (≥34B) model but --expert-activation-rate was not supplied."
            " Pass the activation rate from log_expert_utilization() for a complete entry."
        )

    # ── Failure detection ─────────────────────────────────────────────────────
    failures = _detect_failure_patterns(metrics)
    missing  = _missing_data_note(failures, modality)

    # ── Run context loop ──────────────────────────────────────────────────────
    print(f"\n[eval_runner] Running context loop for ({phase}, {modality})")
    print(f"  variant  : {args.variant}")
    print(f"  metrics  : {metrics}")
    print(f"  failures : {failures}")
    print(f"  missing  : {missing}")
    if routing_distribution:
        print(f"  routing  : {routing_distribution}")
    if expert_activation_rate is not None:
        print(f"  expert_activation_rate: {expert_activation_rate:.3f}")

    result = loop.verify_and_write(
        phase=phase,
        variant=args.variant,
        modality=modality,
        metrics=metrics,
        failures=failures,
        missing=missing,
        routing_distribution=routing_distribution,
        modalities_tested=args.modalities_tested,
        expert_activation_rate=expert_activation_rate,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    status = "CHECKPOINT ✓" if result["is_checkpoint"] else "BLOCKED ✗"
    print(f"\n[eval_runner] Verifier gate: {status}")

    if result["gate_failures"]:
        for msg in result["gate_failures"]:
            print(f"  ✗ {msg}")

    if result["triggered_rules"]:
        print("\n[eval_runner] Hard rules triggered:")
        for rule in result["triggered_rules"]:
            print(f"  ⚠ {rule}")

    if result["escalated_rules"]:
        print("\n[eval_runner] New rules escalated to training_rules.md:")
        for rule in result["escalated_rules"]:
            print(f"  ↑ {rule}")

    if result["compressed"]:
        print("\n[eval_runner] eval_learnings.md compressed (oldest 30 entries archived).")

    print(f"\n[eval_runner] Targeting flags written to {loop.flags_path}")

    # Non-zero exit when gate blocked — CI pipelines can use this
    return 0 if result["is_checkpoint"] else 1


if __name__ == "__main__":
    sys.exit(main())
