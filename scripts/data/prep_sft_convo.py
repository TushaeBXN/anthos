"""
Convert LMB-format JSONL files (role/content) to Anthos SFT format (from/value),
then build the binary SFT dataset ready for train.py --sft-data.

Usage:
    python prep_sft_convo.py
"""
import json
import subprocess
import sys
from pathlib import Path

LMB_BASE = Path("/Users/anthosintelligence/Library/Application Support/Language Model Builder/Data Sets")

SOURCES = [
    LMB_BASE / "No Robots" / "no-robots-train.jsonl",
    LMB_BASE / "Everyday Conversations 2k" / "everyday-conversations-2k-train.jsonl",
]

OUT_DIR   = Path("data/sft_convo")
MERGED    = OUT_DIR / "merged.jsonl"
SFT_OUT   = Path("data/sft_convo_bin")

ROLE_MAP = {"user": "human", "assistant": "gpt"}


def convert(src: Path, dst_fh):
    converted, skipped = 0, 0
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex.get("messages", [])
            turns = []
            for m in messages:
                role = ROLE_MAP.get(m.get("role", ""), None)
                if role is None:
                    continue
                turns.append({"from": role, "value": m["content"].strip()})
            # need at least one human + one gpt turn
            roles = [t["from"] for t in turns]
            if "human" not in roles or "gpt" not in roles:
                skipped += 1
                continue
            dst_fh.write(json.dumps({"conversations": turns}) + "\n")
            converted += 1
    print(f"  {src.name}: {converted} kept, {skipped} skipped")
    return converted


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SFT_OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    print("Converting datasets...")
    with open(MERGED, "w") as out:
        for src in SOURCES:
            if not src.exists():
                print(f"  MISSING: {src}")
                continue
            total += convert(src, out)

    print(f"\nTotal examples: {total}")
    print(f"Merged file: {MERGED}")

    print("\nBuilding binary SFT dataset...")
    cmd = [
        sys.executable, "build_sft_cl100k.py",
        "--out", str(SFT_OUT),
        f"--src", f"{MERGED}:1.0",
        "--max-len", "512",
        "--val", "200",
    ]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("\nERROR: build_sft_cl100k.py failed")
        sys.exit(1)

    print(f"\nDone. SFT binary data at: {SFT_OUT.resolve()}")
    print("\nNow run the fine-tune:")
    print(f"""
    python train.py \\
        --tier proof \\
        --init-from checkpoints/pod-anthos-pt-768/step_030000.pt \\
        --sft-data {SFT_OUT.resolve()} \\
        --steps 3000 \\
        --lr 1e-5 \\
        --warmup 100 \\
        --batch-size 4 \\
        --grad-accum 4 \\
        --vocab-size 100352 \\
        --model-set "dim=768,n_heads=12,n_kv_heads=4,expert_dim=384,lora_rank=8" \\
        --thought-prefix-only \\
        --no-rep-penalty \\
        --run-name anthos-sft-convo \\
        --log-every 50 \\
        --save-every 500
    """)


if __name__ == "__main__":
    main()
