#!/usr/bin/env python3
"""
rsi_loop.py — Recursive self-improvement orchestrator for Anthos.

Loop: detect → target → DPO → gate (×N)

Each iteration:
  1. detect:  gap_detector.py  → logs/gaps_round_N.json
  2. target:  generate_target_data.py → data/anthos_rsi_round_N.jsonl
  3. improve: dpo_unsloth.py → outputs/anthos-rsi-round-N/merged
  4. gate:    gate.py  → logs/gate_round_N.json
              COMMIT → current = candidate, continue
              ROLLBACK → keep current, stop

Aggressive settings per the ARC pattern:
  - 10–15 top gap nodes per iteration
  - SPIN + LADDER data generation
  - Frozen-judge gate with calibration canary
  - Auto-rollback on regression

Usage:
    python3 scripts/train/rsi_loop.py \
        --config pipeline.yaml \
        --start-from outputs/anthos-dpo/merged \
        --rounds 8

Requires the full pipeline to have run first (merge → SFT → abliterate → DPO).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

LOG = logging.getLogger("anthos.rsi_loop")

SCRIPTS = Path(__file__).parent
EVAL = SCRIPTS.parent / "eval"
DATA = SCRIPTS.parent / "data"


def run(cmd: list[str], log_path: Path, check: bool = True) -> int:
    LOG.info("exec: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.returncode


def dpo_on_target(
    model_path: str,
    data_path: Path,
    out_path: Path,
    beta: float,
    epochs: float,
    lr: float,
    log_path: Path,
) -> None:
    """Run DPO on the target data for this round."""
    # Filter to only DPO-format pairs (not SFT examples)
    dpo_only = out_path.parent / (data_path.stem + "_dpo_only.jsonl")
    with data_path.open() as fin, dpo_only.open("w") as fout:
        for line in fin:
            obj = json.loads(line)
            if obj.get("format") == "dpo" and "prompt" in obj and "chosen" in obj and "rejected" in obj:
                fout.write(json.dumps({"prompt": obj["prompt"], "chosen": obj["chosen"], "rejected": obj["rejected"]}) + "\n")

    run(
        [
            sys.executable, str(SCRIPTS / "dpo_unsloth.py"),
            "--model", model_path,
            "--data", str(dpo_only),
            "--out", str(out_path),
            "--beta", str(beta),
            "--epochs", str(epochs),
            "--lr", str(lr),
        ],
        log_path,
    )


def run_rsi_round(
    round_num: int,
    current_model: str,
    cfg: dict,
    max_top_gaps: int,
    beta: float,
    gap_margin: float,
    canary_tolerance: float,
) -> tuple[str, bool]:
    """Run one RSI iteration. Returns (new_model_path, committed)."""
    logs = Path("logs")
    logs.mkdir(exist_ok=True)

    gaps_file = logs / f"gaps_round_{round_num}.json"
    target_file = Path(f"data/anthos_rsi_round_{round_num}.jsonl")
    candidate_out = Path(f"outputs/anthos-rsi-round-{round_num}")
    gate_file = logs / f"gate_round_{round_num}.json"

    LOG.info("=== RSI round %d ===", round_num)
    LOG.info("current model: %s", current_model)

    # 1. Detect
    LOG.info("[detect] finding capability gaps...")
    run(
        [
            sys.executable, str(EVAL / "gap_detector.py"),
            "--model", current_model,
            "--out", str(gaps_file),
            "--top", str(max_top_gaps),
        ],
        logs / f"detect_round_{round_num}.log",
    )

    # Check if any significant gaps remain
    gaps = json.loads(gaps_file.read_text())
    top_failure_rate = gaps[0]["failure_rate"] if gaps else 0.0
    LOG.info("[detect] top gap failure rate: %.1f%%", top_failure_rate * 100)
    if top_failure_rate < 0.1:
        LOG.info("[detect] no significant gaps (top < 10%%) — loop complete")
        return current_model, False  # converged

    # 2. Target
    LOG.info("[target] generating data for %d gap nodes...", len(gaps))
    run(
        [
            sys.executable, str(DATA / "generate_target_data.py"),
            "--gaps", str(gaps_file),
            "--model", current_model,
            "--out", str(target_file),
            "--mode", "both",
        ],
        logs / f"target_round_{round_num}.log",
    )

    # 3. Improve (DPO on targeted data)
    LOG.info("[improve] running DPO on targeted data...")
    dpo_on_target(
        model_path=current_model,
        data_path=target_file,
        out_path=candidate_out,
        beta=beta,
        epochs=1.0,
        lr=5e-6,
        log_path=logs / f"dpo_round_{round_num}.log",
    )
    candidate_merged = str(candidate_out / "merged")

    # 4. Gate
    LOG.info("[gate] evaluating candidate checkpoint...")
    gate_rc = run(
        [
            sys.executable, str(EVAL / "gate.py"),
            "--candidate", candidate_merged,
            "--previous", current_model,
            "--gaps", str(gaps_file),
            "--out", str(gate_file),
            "--gap-margin", str(gap_margin),
            "--canary-tolerance", str(canary_tolerance),
        ],
        logs / f"gate_round_{round_num}.log",
        check=False,
    )

    gate_result = json.loads(gate_file.read_text())
    decision = gate_result["decision"]
    reason = gate_result["reason"]
    LOG.info("[gate] %s — %s", decision, reason)

    if decision == "COMMIT":
        LOG.info("[gate] committing candidate as new current model")
        return candidate_merged, True
    else:
        LOG.info("[gate] rolling back — keeping previous model")
        # Clean up the failed candidate to save disk
        if candidate_out.exists():
            shutil.rmtree(str(candidate_out))
            LOG.info("[gate] cleaned up failed candidate: %s", candidate_out)
        return current_model, True  # stopped by gate, not converged


def main() -> int:
    parser = argparse.ArgumentParser(description="Anthos RSI loop")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-from", required=True,
                        help="Starting model path (output of DPO stage)")
    parser.add_argument("--rounds", type=int, default=8,
                        help="Maximum RSI iterations")
    parser.add_argument("--top-gaps", type=int, default=10,
                        help="Gap nodes to target per iteration")
    parser.add_argument("--gap-margin", type=float, default=0.02)
    parser.add_argument("--canary-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/rsi_loop.log"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text())
    dpo_beta = cfg.get("dpo", {}).get("beta", 0.5)

    current_model = args.start_from
    history: list[dict] = []

    for round_num in range(1, args.rounds + 1):
        new_model, should_continue = run_rsi_round(
            round_num=round_num,
            current_model=current_model,
            cfg=cfg,
            max_top_gaps=args.top_gaps,
            beta=dpo_beta,
            gap_margin=args.gap_margin,
            canary_tolerance=args.canary_tolerance,
        )

        gate_file = Path(f"logs/gate_round_{round_num}.json")
        gate_result = json.loads(gate_file.read_text()) if gate_file.exists() else {}
        history.append({
            "round": round_num,
            "previous": current_model,
            "candidate": new_model,
            "decision": gate_result.get("decision", "converged"),
            "scores": gate_result.get("scores", {}),
        })
        current_model = new_model

        if not should_continue:
            LOG.info("RSI loop converged at round %d (no significant gaps remain)", round_num)
            break

    # Save history log
    history_path = Path("logs/rsi_history.json")
    history_path.write_text(json.dumps(history, indent=2))

    LOG.info("=== RSI loop complete ===")
    LOG.info("final model: %s", current_model)
    LOG.info("rounds run:  %d", len(history))
    LOG.info("history:     %s", history_path)

    # Update pipeline.yaml rsi.final_model
    cfg.setdefault("rsi", {})["final_model"] = current_model
    args.config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    LOG.info("updated pipeline.yaml: rsi.final_model = %s", current_model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
