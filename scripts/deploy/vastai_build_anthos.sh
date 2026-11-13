#!/usr/bin/env bash
# =============================================================================
# vastai_build_anthos.sh — Full Anthos build pipeline on Vast.ai / RunPod
#
# What this does (in order):
#   1. Installs all dependencies
#   2. Clones / updates the Anthos repo
#   3. Downloads ALL SFT datasets (background, logged)
#   4. Sets up Anthos tokenizer + base model (Qwen transplant)
#   5. Runs training stages: smoke → proof → distill
#   6. Saves final checkpoint
#
# Usage — paste this into the Vast.ai terminal:
#
#   export HF_TOKEN="hf_YOUR_TOKEN_HERE"
#   bash /workspace/anthos/scripts/deploy/vastai_build_anthos.sh
#
# Or to run just the dataset download:
#   bash /workspace/anthos/scripts/deploy/vastai_build_anthos.sh --data-only
#
# Or to run just training (if data already downloaded):
#   bash /workspace/anthos/scripts/deploy/vastai_build_anthos.sh --train-only
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/anthos"
DATA_DIR="$REPO_DIR/data"
LOG_DIR="$REPO_DIR/logs"
REPO_URL="https://github.com/TushaeBXN/anthos.git"

# Dataset row limits (set to very high = pull everything)
MANUS_LIMIT=16000000
JEV_LIMIT=741000

# Training settings for A100 80GB
BATCH_SIZE=8
GRAD_ACCUM=4

# Flags
DATA_ONLY=0
TRAIN_ONLY=0
for arg in "$@"; do
  case $arg in
    --data-only)  DATA_ONLY=1 ;;
    --train-only) TRAIN_ONLY=1 ;;
  esac
done

echo "============================================================"
echo "  Anthos Build Pipeline"
echo "  $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Check HF token
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "  Run: export HF_TOKEN=\"hf_your_token_here\""
    exit 1
fi
echo "[OK] HF_TOKEN set"

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers accelerate datasets huggingface_hub tqdm requests
# Try flash-attn (may fail on some GPUs — non-fatal)
pip install -q flash-attn --no-build-isolation || echo "  flash-attn not available (non-fatal)"
pip install -q trl  # for DPO stage later
echo "  Dependencies installed"

# ---------------------------------------------------------------------------
# 3. Clone or update repo
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Setting up repo..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo exists — pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only
else
    echo "  Cloning repo..."
    cd "$WORKSPACE"
    git clone "$REPO_URL" anthos
    cd "$REPO_DIR"
fi
mkdir -p "$DATA_DIR" "$LOG_DIR"
echo "  Repo ready at $REPO_DIR"

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 4. Download datasets (skip if --train-only)
# ---------------------------------------------------------------------------
if [ "$TRAIN_ONLY" -eq 0 ]; then
    echo ""
    echo "[3/5] Downloading all SFT datasets..."
    echo "  Logs: $LOG_DIR/download_sft.log"

    nohup python scripts/data/download_sft_datasets.py \
        --out "$DATA_DIR/sft_anthos.jsonl" \
        --manus-limit "$MANUS_LIMIT" \
        --jev-limit "$JEV_LIMIT" \
        --data-dir "$DATA_DIR" \
        > "$LOG_DIR/download_sft.log" 2>&1 &

    DOWNLOAD_PID=$!
    echo "  Download running in background (PID $DOWNLOAD_PID)"

    if [ "$DATA_ONLY" -eq 1 ]; then
        echo ""
        echo "  --data-only set. Waiting for download to finish..."
        wait $DOWNLOAD_PID
        echo "  Download complete."
        wc -l "$DATA_DIR/sft_anthos.jsonl" || true
        exit 0
    fi

    # Wait for at least some data before starting setup
    echo "  Waiting for download to produce initial data..."
    for i in $(seq 1 60); do
        if [ -f "$DATA_DIR/sft_anthos.jsonl" ] && [ "$(wc -l < "$DATA_DIR/sft_anthos.jsonl")" -gt 1000 ]; then
            echo "  Got $(wc -l < "$DATA_DIR/sft_anthos.jsonl") rows — proceeding with setup"
            break
        fi
        sleep 10
    done
fi

# ---------------------------------------------------------------------------
# 5. Setup Anthos base model (tokenizer + Qwen weights)
# ---------------------------------------------------------------------------
if [ "$DATA_ONLY" -eq 0 ]; then
    echo ""
    echo "[4/5] Setting up Anthos base model..."
    echo "  Logs: $LOG_DIR/setup_base.log"

    python scripts/train/setup_anthos_base.py \
        --base-model Qwen/Qwen2.5-9B-Instruct \
        --out-dir checkpoints/anthos-base \
        --tokenizer-dir anthos-tokenizer \
        2>&1 | tee "$LOG_DIR/setup_base.log"

    echo "  Base model ready"

    # ---------------------------------------------------------------------------
    # 6. Training stages (on A100 we run smoke → proof → distill)
    # ---------------------------------------------------------------------------
    echo ""
    echo "[5/5] Training: smoke → proof → distill"
    echo "  Each stage logs to $LOG_DIR/train_<stage>.log"

    for STAGE in smoke proof distill; do
        echo ""
        echo "  --- Stage: $STAGE ---"
        python scripts/train/sft_anthos.py \
            --stage "$STAGE" \
            --data "$DATA_DIR/sft_anthos.jsonl" \
            --tokenizer anthos-tokenizer \
            --batch-size "$BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --device cuda \
            2>&1 | tee "$LOG_DIR/train_${STAGE}.log"
        echo "  Stage $STAGE complete"
    done
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<EOF

============================================================
  Anthos build complete on Vast.ai
  $(date)
============================================================

Checkpoints:
  checkpoints/anthos-base/      (base Qwen transplant)
  checkpoints/anthos-smoke/     (architecture verified)
  checkpoints/anthos-proof/     (bifurcation learned)
  checkpoints/anthos-distill/   (domain knowledge absorbed)

Logs:
  $LOG_DIR/

Data:
  $DATA_DIR/sft_anthos.jsonl  ($(wc -l < "$DATA_DIR/sft_anthos.jsonl" 2>/dev/null || echo "?") rows)

------------------------------------------------------------
Next steps (run on Mac after downloading checkpoints):
  python scripts/train/sft_anthos.py --stage identity
  python scripts/train/sft_anthos.py --stage instruct
  python scripts/train/sft_anthos.py --stage dpo

Then export:
  python scripts/deploy/export_mlx.py  (Apple Silicon)
  python scripts/deploy/export_gguf.py (Ollama)
------------------------------------------------------------
EOF
