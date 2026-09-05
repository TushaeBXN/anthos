#!/usr/bin/env python3
"""
convert_openhermes.py — Convert OpenHermes 2.5 to Anthos SFT JSONL format

OpenHermes 2.5 is already in ShareGPT format (conversations with from/value).
This script converts the single JSON array to JSONL so ChatInstructDataset
can stream it, filters bad examples, and optionally pushes to HuggingFace.

Usage:
    python convert_openhermes.py
    python convert_openhermes.py --input /path/to/openhermes2_5.json
    python convert_openhermes.py --push --hf-repo TushaeBXN/anthos-sft-openhermes
"""

import json
import argparse
from pathlib import Path

DEFAULT_INPUT  = "/Volumes/1TB Drive/anthos-data/datasets/openhermes-2.5/openhermes2_5.json"
DEFAULT_OUTPUT = "data/openhermes_sft.jsonl"

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "Answer directly and helpfully."
)


def convert(input_path: str, output_path: str, max_len: int = 2000) -> int:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    with open(src) as f:
        data = json.load(f)

    print(f"  {len(data):,} conversations loaded")

    kept = skipped = 0

    with open(dst, "w") as out:
        for i, item in enumerate(data):
            if i % 100_000 == 0 and i > 0:
                print(f"  {i:,} processed, {kept:,} kept ...")

            convs = item.get("conversations", [])
            if len(convs) < 2:
                skipped += 1
                continue

            # Extract turns
            system   = next((c["value"] for c in convs if c["from"] == "system"), None)
            human    = next((c["value"] for c in convs if c["from"] == "human"), "")
            gpt      = next((c["value"] for c in convs if c["from"] == "gpt"),   "")

            if not human or not gpt:
                skipped += 1
                continue

            # Skip very long responses that will just be truncated to nothing
            if len(gpt) > max_len * 4:
                skipped += 1
                continue

            # Replace any system prompt with Anthos identity
            # (OpenHermes system prompts are often generic — override with Anthos)
            out_convs = []
            if system:
                out_convs.append({"from": "system", "value": ANTHOS_SYSTEM})
            out_convs.append({"from": "human", "value": human})
            out_convs.append({"from": "gpt",   "value": gpt})

            out.write(json.dumps({"conversations": out_convs}) + "\n")
            kept += 1

    print(f"\n  Done.")
    print(f"  Kept    : {kept:,}")
    print(f"  Skipped : {skipped:,}")
    print(f"  Output  : {dst}")
    return kept


def push_to_hub(jsonl_path: str, repo_id: str):
    from datasets import Dataset
    import json as _json

    print(f"\nPushing to HuggingFace Hub: {repo_id} ...")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(_json.loads(line))

    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo_id, private=False)
    print(f"  Pushed {len(ds):,} examples → https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default=DEFAULT_INPUT,  help="Path to openhermes2_5.json")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--max-len", type=int, default=2000, help="Max response character length")
    parser.add_argument("--push",    action="store_true",    help="Push to HuggingFace Hub after converting")
    parser.add_argument("--hf-repo", default="TushaeBXN/anthos-sft-openhermes", help="HF repo id")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Is the 1TB drive mounted?")
        return

    kept = convert(args.input, args.output, args.max_len)

    if kept == 0:
        print("Nothing was kept — check the input file.")
        return

    print(f"\nTo use this data for SFT training, update configs.py sft tier:")
    print(f'    dataset = "{args.output}"')
    print(f"\nOr run directly:")
    print(f'    python train.py --tier sft --resume checkpoints/anthos-proof/step_002000.pt')
    print(f'    (after setting dataset to "{args.output}" in configs.py)')

    if args.push:
        push_to_hub(args.output, args.hf_repo)


if __name__ == "__main__":
    main()
