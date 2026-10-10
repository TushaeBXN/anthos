"""
Anthos — Evaluation Script

Runs lm-evaluation-harness benchmarks against a saved checkpoint.

Usage:
    python eval.py --checkpoint checkpoints/sft/step_000500.pt
    python eval.py --checkpoint checkpoints/sft/step_000500.pt --tasks mmlu,hellaswag,gsm8k
    python eval.py --checkpoint checkpoints/proof/step_010000.pt --tasks hellaswag --limit 100
"""

import argparse
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.eval_harness import run_eval, print_eval_summary

AVAILABLE_TASKS = [
    "mmlu", "hellaswag", "arc_easy", "arc_challenge",
    "winogrande", "truthfulqa_mc", "gsm8k", "mathqa",
]

def main():
    parser = argparse.ArgumentParser(description="Anthos benchmark evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy",
                        help=f"Comma-separated task list. Available: {', '.join(AVAILABLE_TASKS)}")
    parser.add_argument("--tier", type=str, default="proof",
                        help="Config tier used to build the model (default: proof)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (default: auto from tier)")
    parser.add_argument("--n-loops", type=int, default=8,
                        help="Recurrent loop depth for eval (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap examples per task (for fast smoke evals)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write full results JSON to this path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tasks  = [t.strip() for t in args.tasks.split(",")]

    print(f"\n{'─'*60}")
    print(f"  Anthos Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Tasks      : {', '.join(tasks)}")
    print(f"  Device     : {device}")
    if args.limit:
        print(f"  Limit      : {args.limit} examples per task")
    print(f"{'─'*60}\n")

    # Build model
    model_cfg, train_cfg = get_training_config(args.tier)
    model = Anthos(model_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  ℹ New params (randomly init): {len(missing)} tensors")

    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", float("nan"))
    print(f"  Loaded step {step:,} | train loss {loss:.4f}\n")

    model.eval()

    # Resolve tokenizer path
    if args.tokenizer:
        tok_path = args.tokenizer
    elif Path("data/anthos_tokenizer").exists():
        tok_path = "data/anthos_tokenizer"
    else:
        tok_path = "gpt2"

    # Run eval
    summary = run_eval(
        model          = model,
        tokenizer_path = tok_path,
        tasks          = tasks,
        n_loops        = args.n_loops,
        device         = device,
        output_path    = Path(args.output) if args.output else None,
        limit          = args.limit,
    )

    print_eval_summary(summary, step=step)

    if args.output:
        print(f"\n  Full results written → {args.output}")

if __name__ == "__main__":
    main()
