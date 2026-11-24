#!/usr/bin/env python3
"""
merge_ties.py — TIES merge for Anthos.

Merges Qwen2.5-9B-Instruct with Qwen2.5-Coder-7B-Instruct using mergekit's
TIES implementation.

Evidence: TIES density=0.55, weights=(0.65, 0.35), normalize=True preserves
identity and coding quality on Qwen2.5 Instruct×Coder.
SLERP is contraindicated for this donor pair (fails at t∈[0.15, 0.25]).

Usage:
    python3 scripts/train/merge_ties.py \
        --base Qwen/Qwen2.5-9B-Instruct \
        --coder Qwen/Qwen2.5-Coder-7B-Instruct \
        --out outputs/anthos-merged

Requires:  pip install mergekit pyyaml
Time:      ~30 min on a 4090
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

LOG = logging.getLogger("anthos.merge")


def build_ties_config(
    base_model: str,
    coder_model: str,
    base_weight: float,
    coder_weight: float,
    density: float,
    dtype: str,
) -> dict:
    if not (0.0 < base_weight < 1.0 and 0.0 < coder_weight < 1.0):
        raise ValueError("weights must be strictly between 0 and 1")
    total = base_weight + coder_weight
    if abs(total - 1.0) > 1e-6:
        LOG.warning(
            "weights sum to %.4f (not 1.0); normalize=True will renormalize, "
            "but this is usually unintended", total,
        )
    return {
        "models": [
            {"model": base_model, "parameters": {"weight": base_weight}},
            {"model": coder_model, "parameters": {"weight": coder_weight}},
        ],
        "merge_method": "ties",
        "base_model": base_model,
        "parameters": {
            "density": density,
            "normalize": True,
            "int8_mask": True,
        },
        "dtype": dtype,
    }


def run_merge(config: dict, out_path: Path, cuda: bool = True) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    config_path = out_path / "merge_config.yml"
    with config_path.open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    cmd = [
        "mergekit-yaml",
        str(config_path),
        str(out_path),
        "--copy-tokenizer",
    ]
    if cuda:
        cmd.append("--cuda")

    LOG.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mergekit failed with exit code {result.returncode}")

    LOG.info("merge complete: %s", out_path)
    LOG.info("next: python3 scripts/train/sft_unsloth.py --model %s --data ...", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="TIES merge for Anthos")
    parser.add_argument("--base", default="Qwen/Qwen2.5-9B-Instruct")
    parser.add_argument("--coder", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--out", default="outputs/anthos-merged", type=Path)
    parser.add_argument("--base-weight", type=float, default=0.65)
    parser.add_argument("--coder-weight", type=float, default=0.35)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import mergekit  # noqa: F401
    except ImportError:
        LOG.error("mergekit not found. Install: pip install mergekit")
        return 1

    config = build_ties_config(
        base_model=args.base,
        coder_model=args.coder,
        base_weight=args.base_weight,
        coder_weight=args.coder_weight,
        density=args.density,
        dtype=args.dtype,
    )
    run_merge(config, args.out, cuda=not args.no_cuda)
    return 0


if __name__ == "__main__":
    sys.exit(main())
