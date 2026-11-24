#!/usr/bin/env python3
"""
run_pipeline.py — Anthos pipeline orchestrator.

Stages (run in this order):
  1. merge    — TIES merge (merge_ties.py)
  2. sft      — SFT on reasoning traces (generate_sft_data.py + sft_unsloth.py)
  3. abliterate — refusal-direction orthogonalization (abliterate.py)
  4. dpo      — DPO repair pass (generate_dpo_data.py + dpo_unsloth.py)
  5. gguf     — GGUF export (llama.cpp)

Config via pipeline.yaml; override individual stages with --stages.
Logs each stage to a separate file; stops on first non-zero exit.

Usage:
    python3 scripts/train/run_pipeline.py --config pipeline.yaml
    python3 scripts/train/run_pipeline.py --config pipeline.yaml --stages abliterate,dpo,gguf
    python3 scripts/train/run_pipeline.py --config pipeline.yaml --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

LOG = logging.getLogger("anthos.pipeline")

SCRIPTS = Path(__file__).parent
DATA_SCRIPTS = SCRIPTS.parent / "data"


def run(cmd: list[str], log_path: Path) -> None:
    LOG.info("exec: %s", " ".join(cmd))
    LOG.info("log:  %s", log_path)
    with log_path.open("a") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"see {log_path}"
        )


def stage_merge(cfg: dict, skip_existing: bool) -> None:
    p = cfg["paths"]
    out = Path(p["merged"])
    if skip_existing and (out / "config.json").exists():
        LOG.info("[merge] already exists, skipping")
        return
    merge_cfg = cfg["merge"]
    run(
        [
            sys.executable, str(SCRIPTS / "merge_ties.py"),
            "--base", p["base_model"],
            "--coder", p["coder_model"],
            "--out", str(out),
            "--base-weight", str(merge_cfg["base_weight"]),
            "--coder-weight", str(merge_cfg["coder_weight"]),
            "--density", str(merge_cfg["density"]),
        ],
        Path("logs/merge.log"),
    )


def stage_sft(cfg: dict, skip_existing: bool) -> None:
    p = cfg["paths"]
    sft_data = Path(p["sft_data"])
    sft_out = Path(p["sft"])
    merged_out = sft_out / "merged"

    if skip_existing and (merged_out / "config.json").exists():
        LOG.info("[sft] already exists, skipping")
        return

    if not sft_data.exists():
        run(
            [
                sys.executable, str(DATA_SCRIPTS / "generate_sft_data.py"),
                "--out", str(sft_data),
                "--coding", str(cfg.get("sft_data", {}).get("coding", 800)),
            ],
            Path("logs/sft_data.log"),
        )

    run(
        [
            sys.executable, str(SCRIPTS / "sft_unsloth.py"),
            "--model", p["merged"],
            "--data", str(sft_data),
            "--out", str(sft_out),
        ],
        Path("logs/sft.log"),
    )


def stage_abliterate(cfg: dict, skip_existing: bool) -> None:
    p = cfg["paths"]
    abl_in = Path(p["sft"]) / "merged"
    abl_out = Path(p["abliterated"])
    abl_cfg = cfg["abliterate"]

    if skip_existing and (abl_out / "config.json").exists():
        LOG.info("[abliterate] already exists, skipping")
        return

    result = subprocess.run(
        [
            sys.executable, str(SCRIPTS / "abliterate.py"),
            "--model", str(abl_in),
            "--out", str(abl_out),
            "--factor", str(abl_cfg["scale"]),
            "--layers", "last-half",
        ],
        capture_output=False,
        text=True,
    )
    if result.returncode == 2:
        LOG.warning("[abliterate] tool-call diagnostic failed — DPO stage will repair this")
    elif result.returncode != 0:
        raise RuntimeError(f"abliterate failed with exit {result.returncode}")


def stage_dpo(cfg: dict, skip_existing: bool) -> None:
    p = cfg["paths"]
    dpo_data = Path(p["dpo_data"])
    dpo_out = Path(p["dpo"])
    dpo_merged = dpo_out / "merged"
    dpo_cfg = cfg["dpo"]

    if skip_existing and (dpo_merged / "config.json").exists():
        LOG.info("[dpo] already exists, skipping")
        return

    if not dpo_data.exists():
        run(
            [
                sys.executable, str(DATA_SCRIPTS / "generate_dpo_data.py"),
                "--out", str(dpo_data),
            ],
            Path("logs/dpo_data.log"),
        )

    run(
        [
            sys.executable, str(SCRIPTS / "dpo_unsloth.py"),
            "--model", p["abliterated"],
            "--data", str(dpo_data),
            "--out", str(dpo_out),
            "--beta", str(dpo_cfg["beta"]),
            "--epochs", str(dpo_cfg["epochs"]),
            "--lr", str(dpo_cfg["lr"]),
        ],
        Path("logs/dpo.log"),
    )


def stage_gguf(cfg: dict, skip_existing: bool) -> None:
    p = cfg["paths"]
    model_in = Path(p["dpo"]) / "merged"
    gguf_out = Path(p["gguf"])
    quant = cfg["gguf"]["quant"]

    if skip_existing and gguf_out.exists():
        LOG.info("[gguf] already exists, skipping")
        return

    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    f16_path = gguf_out.with_suffix(".f16.gguf")

    # Convert HF → GGUF f16
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        raise FileNotFoundError(
            "llama.cpp/convert_hf_to_gguf.py not found. "
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
        )
    run(
        [
            sys.executable, str(convert_script),
            str(model_in),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ],
        Path("logs/gguf_convert.log"),
    )

    # Quantize
    quant_bin = Path("llama.cpp/build/bin/llama-quantize")
    if quant_bin.exists():
        run(
            [str(quant_bin), str(f16_path), str(gguf_out), quant],
            Path("logs/gguf_quant.log"),
        )
        LOG.info("[gguf] quantized → %s", gguf_out)
    else:
        LOG.warning(
            "[gguf] llama-quantize not found; f16 GGUF at %s. "
            "Build with: cd llama.cpp && cmake -B build && cmake --build build --config Release",
            f16_path,
        )


STAGE_FNS = {
    "merge": stage_merge,
    "sft": stage_sft,
    "abliterate": stage_abliterate,
    "dpo": stage_dpo,
    "gguf": stage_gguf,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Anthos pipeline runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default="merge,sft,abliterate,dpo,gguf",
                        help="Comma-separated stages to run")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a stage if its output already exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/pipeline.log"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text())
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    unknown = [s for s in stages if s not in STAGE_FNS]
    if unknown:
        LOG.error("unknown stages: %s (valid: %s)", unknown, list(STAGE_FNS))
        return 1

    for stage in stages:
        LOG.info("=== stage: %s ===", stage)
        try:
            STAGE_FNS[stage](cfg, args.skip_existing)
        except Exception as exc:
            LOG.error("stage %s failed: %s", stage, exc)
            return 1

    LOG.info("=== pipeline complete ===")
    LOG.info("GGUF: %s", cfg["paths"]["gguf"])
    LOG.info("To deploy:")
    LOG.info("  ollama create anthos -f Modelfile")
    LOG.info("  ollama run anthos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
