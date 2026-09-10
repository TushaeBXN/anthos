#!/usr/bin/env python3
"""
run_eval_suite.py — Run the Stage 1 fixed eval suite against Anthos.

Records pre-training baseline (random-init) and post-training scores so
Stage 1 → 2 → 3 comparisons are meaningful. Scores are rubric-based (0-3),
not exact match — a human scores each response using stage1_eval_suite.json.

Usage:
    # Random-init baseline (before any training):
    python run_eval_suite.py --random-init --out results/baseline_random_init.json

    # After Stage 1 training:
    python run_eval_suite.py --checkpoint checkpoints/mansa_sovereign/step_XXXXXX.pt \\
                             --out results/stage1_results.json

    # Compare two runs:
    python run_eval_suite.py --compare results/baseline_random_init.json results/stage1_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

SUITE_FILE = "stage1_eval_suite.json"
SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "You are NOT ChatGPT, NOT Claude, NOT any other model. "
    "Answer directly and confidently."
)


def load_suite():
    with open(SUITE_FILE) as f:
        return json.load(f)


def load_model_and_tokenizer(checkpoint_path: str, tier: str = "sft"):
    from transformers import AutoTokenizer
    from anthos.main import Anthos
    from anthos.configs import get_training_config

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("data/anthos_tokenizer")

    print(f"  Loading model (tier={tier})...")
    model_cfg, _ = get_training_config(tier)
    model = Anthos(model_cfg)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠ {len(missing)} missing keys (expected for new params)")
        print(f"  Checkpoint loaded: {checkpoint_path}")
    else:
        print(f"  Using RANDOM INIT weights (no checkpoint loaded)")

    model.eval()
    return model, tokenizer


SYS_ID = 50257
USR_ID = 50258
THT_ID = 50259
AST_ID = 50260
END_ID = 50261


def build_prompt(tokenizer, user: str) -> torch.Tensor:
    sys_ids = tokenizer.encode(SYSTEM, add_special_tokens=False)
    usr_ids = tokenizer.encode(user,   add_special_tokens=False)
    ids = (
        [SYS_ID] + sys_ids + [END_ID] +
        [USR_ID] + usr_ids + [END_ID] +
        [THT_ID, END_ID] +
        [AST_ID]
    )
    return torch.tensor([ids], dtype=torch.long)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    prompt_ids = build_prompt(tokenizer, prompt)
    special = {SYS_ID, USR_ID, THT_ID, AST_ID, END_ID}

    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            n_loops=8,
            temperature=0.7,
            top_k=40,
        )

    new_ids = out[0][prompt_ids.shape[1]:]
    clean = [t for t in new_ids.tolist() if t not in special]
    eos = tokenizer.eos_token_id
    if eos and eos in clean:
        clean = clean[:clean.index(eos)]

    return tokenizer.decode(clean, skip_special_tokens=True).strip()


def run_suite(model, tokenizer, suite: dict, label: str) -> dict:
    results = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "suite_version": suite.get("version", "unknown"),
        "prompts": []
    }

    prompts = suite["prompts"]
    print(f"\n  Running {len(prompts)} prompts...")
    print(f"  {'─'*60}")

    for i, item in enumerate(prompts, 1):
        pid   = item["id"]
        domain = item["domain"]
        qlabel = item["label"]
        prompt = item["prompt"]
        rubric = item["rubric"]

        print(f"\n  [{i:02d}/{len(prompts)}] {pid} — {qlabel}")

        t0 = time.time()
        try:
            response = generate(model, tokenizer, prompt)
        except Exception as e:
            response = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        print(f"  {'─'*40}")
        print(f"  Q: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  A: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"  ({elapsed:.1f}s)")
        print()
        print(f"  RUBRIC:")
        for score in ["3", "2", "1", "0"]:
            print(f"    [{score}] {rubric.get(score, '')}")
        print()

        # Prompt human for score
        while True:
            raw = input(f"  Score for {pid} (0-3, or 's' to skip): ").strip().lower()
            if raw == 's':
                score = None
                notes = ""
                break
            if raw in ('0', '1', '2', '3'):
                score = int(raw)
                notes = input(f"  Notes (optional, Enter to skip): ").strip()
                break
            print("  Enter 0, 1, 2, 3, or 's' to skip.")

        results["prompts"].append({
            "id": pid,
            "domain": domain,
            "label": qlabel,
            "response": response,
            "score": score,
            "notes": notes,
            "elapsed_s": round(elapsed, 2),
        })

    return results


def summarize(results: dict):
    prompts = results["prompts"]
    scored = [p for p in prompts if p["score"] is not None]
    if not scored:
        print("  No scored prompts.")
        return

    total = sum(p["score"] for p in scored)
    max_possible = 3 * len(scored)
    pct = 100 * total / max_possible if max_possible else 0

    print(f"\n  {'═'*60}")
    print(f"  RESULTS: {results['label']}")
    print(f"  {'═'*60}")
    print(f"  Total score : {total}/{max_possible}  ({pct:.1f}%)")
    print(f"  Scored      : {len(scored)}/{len(prompts)} prompts")
    print()

    by_domain: dict[str, list] = {}
    for p in scored:
        by_domain.setdefault(p["domain"], []).append(p["score"])

    print(f"  By domain:")
    for domain, scores in sorted(by_domain.items()):
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 10 / 3)
        print(f"    {domain:16s} avg={avg:.2f}/3.0  n={len(scores)}  {bar}")
    print()


def compare(file_a: str, file_b: str):
    with open(file_a) as f: a = json.load(f)
    with open(file_b) as f: b = json.load(f)

    a_map = {p["id"]: p for p in a["prompts"] if p["score"] is not None}
    b_map = {p["id"]: p for p in b["prompts"] if p["score"] is not None}
    common = sorted(set(a_map) & set(b_map))

    print(f"\n  {'═'*60}")
    print(f"  COMPARISON")
    print(f"  A: {a['label']}  ({a['timestamp'][:10]})")
    print(f"  B: {b['label']}  ({b['timestamp'][:10]})")
    print(f"  {'═'*60}")
    print(f"  {'ID':10s}  {'Domain':14s}  {'A':>4}  {'B':>4}  {'Δ':>5}")
    print(f"  {'─'*50}")

    deltas = []
    for pid in common:
        pa = a_map[pid]
        pb = b_map[pid]
        delta = pb["score"] - pa["score"]
        deltas.append(delta)
        sign = "+" if delta > 0 else ""
        print(f"  {pid:10s}  {pa['domain']:14s}  {pa['score']:>4}  {pb['score']:>4}  {sign}{delta:>4}")

    if deltas:
        a_total = sum(a_map[pid]["score"] for pid in common)
        b_total = sum(b_map[pid]["score"] for pid in common)
        max_p   = 3 * len(common)
        print(f"\n  Total: {a_total}/{max_p} → {b_total}/{max_p}  "
              f"(Δ={b_total-a_total:+d}, {100*(b_total-a_total)/max_p:.1f}pp)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run Anthos Stage 1 eval suite")
    parser.add_argument("--checkpoint",  type=str, default="",
                        help="Path to .pt checkpoint. Omit for random-init baseline.")
    parser.add_argument("--random-init", action="store_true",
                        help="Explicitly run with no checkpoint (random init).")
    parser.add_argument("--tier",        type=str, default="sft",
                        help="Config tier (default: sft).")
    parser.add_argument("--out",         type=str, default="",
                        help="Output JSON file for results.")
    parser.add_argument("--compare",     nargs=2,  metavar="FILE",
                        help="Compare two result JSON files.")
    parser.add_argument("--summary",     type=str,
                        help="Print summary of an existing result file.")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    if args.summary:
        with open(args.summary) as f:
            results = json.load(f)
        summarize(results)
        return

    suite = load_suite()

    # Determine label
    if args.random_init or not args.checkpoint:
        ckpt_path = None
        label = "baseline_random_init"
    else:
        ckpt_path = args.checkpoint
        label = Path(args.checkpoint).stem

    out_path = args.out or f"results/{label}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Anthos Stage 1 Eval Suite")
    print(f"  Label     : {label}")
    print(f"  Tier      : {args.tier}")
    print(f"  Prompts   : {len(suite['prompts'])}")
    print(f"  Output    : {out_path}")
    print(f"{'═'*60}")

    model, tokenizer = load_model_and_tokenizer(ckpt_path or "", args.tier)

    results = run_suite(model, tokenizer, suite, label)
    summarize(results)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
