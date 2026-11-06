#!/usr/bin/env bash
# prepare_sft_data.sh — Generate all local SFT datasets from scripts/data/
#
# Runs every no-API data generator and reports the output paths.
# After this completes, use the generated files with:
#
#   python train_regular_sft.py \
#       --init-from checkpoints/regular-500m/step_150000.pt \
#       --run-name anthos-sft-500m \
#       --datasets alpaca dolly local \
#       --local-data data/identity_hardening.jsonl data/self_knowledge.jsonl \
#                    data/code_sft.jsonl data/hardware_security.jsonl \
#                    data/math_ml_sft.jsonl data/multilingual_sft.jsonl \
#                    data/applied_sft.jsonl data/awesome_knowledge.jsonl \
#       --steps 3000 --lr 2e-5 --seq-len 1024
#
# Usage:
#   cd /path/to/anthos
#   bash prepare_sft_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/scripts/data"
OUT_DIR="$SCRIPT_DIR/data"

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

echo "=== Anthos SFT Data Preparation ==="
echo "Output directory: $OUT_DIR"
echo ""

# ── Identity & self-knowledge (pure Python, no API, instant) ──────────────────
echo "[1/8] Identity hardening (10k examples)..."
python "$DATA_DIR/generate_identity_data.py" \
    --n 10000 \
    --out "$OUT_DIR/identity_hardening.jsonl"

echo "[2/8] Self-knowledge (architecture facts)..."
python "$DATA_DIR/generate_self_knowledge_data.py" \
    --out "$OUT_DIR/self_knowledge.jsonl"

# ── Code (pure Python, no API) ────────────────────────────────────────────────
echo "[3/8] Code intelligence dataset (~15k examples)..."
python "$DATA_DIR/generate_code_dataset.py" \
    --out "$OUT_DIR/code_sft.jsonl"

# ── Hardware security (uses NVD/arXiv public APIs) ────────────────────────────
echo "[4/8] Hardware security dataset..."
python "$DATA_DIR/generate_hardware_security_data.py" \
    --out "$OUT_DIR/hardware_security.jsonl" || {
    echo "  WARNING: hardware security generation failed (network issue?) — skipping"
}

# ── Math & ML (pure Python, no API) — hardcodes output to data/math_ml_sft.jsonl ─
echo "[5/8] Math & ML training data..."
(cd "$DATA_DIR" && mkdir -p data && python generate_math_ml_training.py) && \
    cp "$DATA_DIR/data/math_ml_sft.jsonl" "$OUT_DIR/math_ml_sft.jsonl" || {
    echo "  WARNING: math/ML generation failed — skipping"
}

# ── Multilingual (pure Python, no API) ───────────────────────────────────────
echo "[6/8] Multilingual instruction pairs..."
python "$DATA_DIR/generate_multilingual_data.py" \
    --out "$OUT_DIR/multilingual_sft.jsonl"

# ── Applied tasks (pure Python, no API) ──────────────────────────────────────
echo "[7/8] Applied tasks (coding + ML + business)..."
python "$DATA_DIR/generate_applied_training.py" \
    --n 1000 \
    --output "$OUT_DIR/applied_sft.jsonl"

# ── General knowledge (uses Wikipedia/public URLs) ────────────────────────────
echo "[8/8] General knowledge (awesome lists)..."
python "$DATA_DIR/generate_awesome_training_data.py" \
    --n 5000 \
    --out "$OUT_DIR/awesome_knowledge.jsonl" || {
    echo "  WARNING: awesome training generation failed — skipping"
}

# ── Security + agent reasoning (pure Python, no API) ─────────────────────────
echo "[9/9] Security + agent reasoning examples..."
python "$DATA_DIR/generate_security_agent_data.py" \
    --out "$OUT_DIR/security_agent_sft.jsonl"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Generated files ==="
for f in identity_hardening self_knowledge code_sft hardware_security \
          math_ml_sft multilingual_sft applied_sft awesome_knowledge security_agent_sft; do
    path="$OUT_DIR/${f}.jsonl"
    if [ -f "$path" ]; then
        count=$(wc -l < "$path")
        echo "  $path ($count examples)"
    else
        echo "  $path (MISSING)"
    fi
done

echo ""
echo "=== SFT command (run after pretrain completes at step 150k) ==="
echo ""
echo "python train_regular_sft.py \\"
echo "    --init-from checkpoints/regular-500m/step_150000.pt \\"
echo "    --run-name anthos-sft-500m \\"
echo "    --datasets alpaca dolly local \\"
echo "    --local-data \\"
echo "        $OUT_DIR/identity_hardening.jsonl \\"
echo "        $OUT_DIR/self_knowledge.jsonl \\"
echo "        $OUT_DIR/code_sft.jsonl \\"
echo "        $OUT_DIR/hardware_security.jsonl \\"
echo "        $OUT_DIR/math_ml_sft.jsonl \\"
echo "        $OUT_DIR/multilingual_sft.jsonl \\"
echo "        $OUT_DIR/applied_sft.jsonl \\"
echo "        $OUT_DIR/awesome_knowledge.jsonl \\"
echo "        $OUT_DIR/security_agent_sft.jsonl \\"
echo "    --steps 3000 --lr 2e-5 --seq-len 1024 \\"
echo "    --batch-size 8 --grad-accum 8 \\"
echo "    --save-every 500"
