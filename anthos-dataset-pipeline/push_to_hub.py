"""
push_to_hub.py — Push final JSONL to HuggingFace as machomenc/anthos-code-training-v1

Usage:
    HF_TOKEN=hf_... python push_to_hub.py
    HF_TOKEN=hf_... python push_to_hub.py --thought  # push the thought-token version
"""

import argparse
import os
from pathlib import Path

from config import HF_REPO_ID, HF_TRAIN_SPLIT, MERGED_OUT, THOUGHT_OUT


def push(jsonl_path: str, repo_id: str, train_split: float = 0.95):
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set")

    from datasets import Dataset, DatasetDict
    import json

    print(f"Loading {jsonl_path}...")
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} examples")

    # Flatten conversations to text for storage
    def flatten(record):
        turns = record.get("conversations", [])
        human = next((t["value"] for t in turns if t["from"] in ("human", "user")), "")
        system = next((t["value"] for t in turns if t["from"] == "system"), "")
        assistant = next((t["value"] for t in turns if t["from"] in ("gpt", "assistant")), "")
        return {"system": system, "human": human, "assistant": assistant,
                "conversations": json.dumps(turns)}

    flat = [flatten(r) for r in records]
    ds = Dataset.from_list(flat)

    split_idx = int(len(ds) * train_split)
    ds_dict = DatasetDict({
        "train": ds.select(range(split_idx)),
        "test":  ds.select(range(split_idx, len(ds))),
    })

    print(f"Train: {len(ds_dict['train']):,} | Test: {len(ds_dict['test']):,}")
    print(f"Pushing to {repo_id}...")
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"\nDone: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thought", action="store_true", help="Push thought-token version")
    parser.add_argument("--repo", default=HF_REPO_ID)
    args = parser.parse_args()

    path = str(THOUGHT_OUT) if args.thought else str(MERGED_OUT)
    if not Path(path).exists():
        raise FileNotFoundError(f"Not found: {path}\nRun the pipeline first.")

    push(path, args.repo, HF_TRAIN_SPLIT)
