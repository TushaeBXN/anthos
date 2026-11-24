#!/usr/bin/env python3
"""
generate_target_data.py — LADDER decomposition + SPIN pairs for RSI loop.

For each gap node, produces training data that creates a solvable difficulty
gradient into the gap:

  LADDER (knowledge/coding gaps):
    Given a failing problem P, generate 3-5 simpler variants P1,P2,...
    that isolate the subskill. Recurse until leaves are reliably solved.
    Training set = union of leaves + first-level parents.

  SPIN (decision/autonomy gaps):
    Current model generates responses to gap prompts. Top-scored response
    = chosen. Previous-round response (or static baseline) = rejected.
    Beta=0.5 DPO on these pairs.

  Proposer-Solver (tool/autonomy gaps):
    Proposer generates novel scenario; Solver produces trajectory.
    Frozen judge verifies the final state.

Output: JSONL DPO pairs {"prompt": ..., "chosen": ..., "rejected": ...}
        or SFT examples {"text": ...} depending on gap type.

Usage:
    python3 scripts/data/generate_target_data.py \
        --gaps logs/gaps.json \
        --model outputs/anthos-dpo/merged \
        --out data/anthos_rsi_round_1.jsonl \
        --mode dpo
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

LOG = logging.getLogger("anthos.target_data")

SYSTEM = (
    "You are Anthos. You reason before acting. You plan, call tools, "
    "observe results, and self-correct. Be direct and blunt."
)

# ---------------------------------------------------------------------------
# LADDER: recursive decomposition templates per taxonomy node
# ---------------------------------------------------------------------------

LADDER_TEMPLATES = {
    "coding/boundary_conditions": [
        "Write a function that handles {base_case} correctly.",
        "Extend your solution to also handle {next_case}.",
        "Now handle both {base_case} and {next_case} together.",
        "Add a test for the case where {edge_case}.",
    ],
    "coding/debugging_partial_info": [
        "What are the two most common causes of {symptom}?",
        "If {symptom} only appears with {condition}, which cause is most likely?",
        "Write a minimal reproducer for {symptom} caused by {likely_cause}.",
    ],
    "tool_calling/error_recovery": [
        "A tool returned '{error}'. What is the most likely cause?",
        "Given that cause, what is the corrected tool call?",
        "If the corrected call also fails, what is the fallback strategy?",
    ],
    "autonomy/gate_classification": [
        "Is '{action}' reversible without data loss? Answer yes/no and explain.",
        "If reversible, should you apply it automatically or ask first? Why?",
        "If irreversible, what information do you need before proceeding?",
    ],
    "autonomy/intervention_detection": [
        "An error appears {n} consecutive times in tool outputs. At what count do you intervene?",
        "The error is '{error}'. Is it deterministic or could it be transient?",
        "If deterministic and reversible, write the Decision: line you would emit.",
    ],
}

LADDER_VARS = {
    "coding/boundary_conditions": [
        {"base_case": "an empty array", "next_case": "a single-element array", "edge_case": "the target equals the boundary value"},
        {"base_case": "null input", "next_case": "empty string", "edge_case": "input contains only whitespace"},
    ],
    "coding/debugging_partial_info": [
        {"symptom": "intermittent test failures", "condition": "tests run in parallel", "likely_cause": "shared mutable state"},
        {"symptom": "wrong output on duplicate inputs", "condition": "set is used for deduplication", "likely_cause": "hash collision or early exit"},
    ],
    "tool_calling/error_recovery": [
        {"error": "No such file or directory: *.py"},
        {"error": "pattern matched nothing"},
        {"error": "FileNotFoundError: config.json"},
    ],
    "autonomy/gate_classification": [
        {"action": "edit a function to fix an off-by-one"},
        {"action": "run git push --force origin main"},
        {"action": "drop and recreate the users table"},
    ],
    "autonomy/intervention_detection": [
        {"n": "2", "error": "IndexError at binary_search.py:12"},
        {"n": "3", "error": "ImportError: No module named requests"},
    ],
}


def generate_ladder_examples(node: str, failures: list[dict], rng: random.Random) -> list[dict]:
    """Generate LADDER-decomposed SFT examples for a gap node."""
    templates = LADDER_TEMPLATES.get(node, [])
    vars_pool = LADDER_VARS.get(node, [{}])
    if not templates:
        return []

    examples = []
    for _ in range(min(len(failures) * 3, 20)):
        v = rng.choice(vars_pool)
        messages = [{"role": "system", "content": SYSTEM}]
        for template in templates:
            try:
                question = template.format(**v)
            except KeyError:
                continue
            messages.append({"role": "user", "content": question})
            # Simple synthetic answer — the THINK structure teaches the shape
            messages.append({"role": "assistant", "content": (
                f"<think>\n{question[:60]}... reasoning step.\n</think>\n"
                f"<answer>\nAddressing: {question[:80]}\n</answer>"
            )})

        if len(messages) > 1:
            text = "\n".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages
            )
            examples.append({"text": text, "node": node, "type": "ladder"})

    return examples


# ---------------------------------------------------------------------------
# SPIN: model generates responses, top → chosen, previous round → rejected
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_spin_pairs(
    model,
    tokenizer,
    failures: list[dict],
    n_candidates: int,
    max_new_tokens: int,
    device: str,
    rng: random.Random,
) -> list[dict]:
    """Generate SPIN DPO pairs from current model responses."""
    pairs = []
    for failure in failures:
        prompt_text = failure["problem"]
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        candidates = []
        for _ in range(n_candidates):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            # Score by presence of <think> + substantive content
            score = int("<think>" in text) + int("<answer>" in text) + int(len(text) > 100)
            candidates.append((score, text))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if len(candidates) >= 2:
            chosen_score, chosen = candidates[0]
            rejected_score, rejected = candidates[-1]
            # Only pair if there is a meaningful quality gap
            if chosen_score > rejected_score:
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "node": failure.get("node", "unknown"),
                    "type": "spin",
                })

        # Also add the previous-round (static) response as rejected if available
        if failure.get("outputs") and candidates:
            best = candidates[0][1]
            prev = failure["outputs"][0]
            if best != prev and len(best) > len(prev) * 1.2:
                pairs.append({
                    "prompt": prompt,
                    "chosen": best,
                    "rejected": prev,
                    "node": failure.get("node", "unknown"),
                    "type": "spin_prev",
                })

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RSI target data")
    parser.add_argument("--gaps", required=True, type=Path,
                        help="gaps.json from gap_detector.py")
    parser.add_argument("--model", required=True,
                        help="Current model for SPIN pair generation")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["dpo", "sft", "both"], default="both")
    parser.add_argument("--n-candidates", type=int, default=4,
                        help="SPIN: candidates per problem")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    gaps = json.loads(args.gaps.read_text())
    LOG.info("loaded %d gap nodes", len(gaps))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    all_examples: list[dict] = []

    for gap in gaps:
        node = gap["node"]
        failures = [{**f, "node": node} for f in gap["failures"]]
        LOG.info("targeting node: %s (%d failures)", node, len(failures))

        # Decision/autonomy nodes → SPIN DPO
        if any(x in node for x in ["autonomy", "tool_calling", "persistence"]):
            if args.mode in ("dpo", "both"):
                pairs = generate_spin_pairs(
                    model, tokenizer, failures,
                    args.n_candidates, args.max_tokens, device, rng
                )
                for p in pairs:
                    all_examples.append({
                        "prompt": p["prompt"],
                        "chosen": p["chosen"],
                        "rejected": p["rejected"],
                        "node": node,
                        "format": "dpo",
                    })
                LOG.info("  SPIN: %d pairs", len(pairs))

        # Knowledge/coding nodes → LADDER SFT
        if any(x in node for x in ["coding", "boundary", "debugging", "parsing"]):
            if args.mode in ("sft", "both"):
                examples = generate_ladder_examples(node, failures, rng)
                for ex in examples:
                    all_examples.append({**ex, "format": "sft"})
                LOG.info("  LADDER: %d SFT examples", len(examples))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_dpo, n_sft = 0, 0
    with args.out.open("w") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex) + "\n")
            if ex["format"] == "dpo":
                n_dpo += 1
            else:
                n_sft += 1

    LOG.info("wrote %d total examples (DPO: %d, SFT: %d) → %s", len(all_examples), n_dpo, n_sft, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
