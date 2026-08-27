#!/usr/bin/env bash
# runpod_start.sh — Anthos code-tier training on RunPod GPU
# Usage: bash runpod_start.sh [GITHUB_TOKEN] [HF_TOKEN]
#
# What this does:
#   1. Installs system deps (git, python, build tools)
#   2. Clones the anthos-repo from GitHub
#   3. Creates venv and installs Python deps (GPU versions)
#   4. Downloads the resume checkpoint from HuggingFace (or local upload)
#   5. Starts training — logs to /workspace/train_stdout.txt
#
# Set env vars before running or pass as args:
#   GITHUB_TOKEN   — for private repo clone (optional if public)
#   HF_TOKEN       — for HuggingFace checkpoint download (optional)

set -e

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN="${1:-${GITHUB_TOKEN:-}}"
HF_TOKEN="${2:-${HF_TOKEN:-}}"
REPO_URL="https://github.com/TushaeBXN/anthos"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/anthos-repo"
CHECKPOINT="checkpoints/anthos-proof/step_002000.pt"
LOG_FILE="$WORKSPACE/train_stdout.txt"
RESUME_FROM="$REPO_DIR/$CHECKPOINT"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo "==> Installing system deps..."
apt-get update -qq && apt-get install -y -q git python3-pip python3-venv build-essential

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
echo "==> Cloning anthos-repo..."
if [ -d "$REPO_DIR" ]; then
    echo "    Already exists — pulling latest..."
    cd "$REPO_DIR" && git pull
else
    if [ -n "$GITHUB_TOKEN" ]; then
        CLONE_URL="https://${GITHUB_TOKEN}@github.com/TushaeBXN/anthos"
    else
        CLONE_URL="$REPO_URL"
    fi
    git clone "$CLONE_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ── 3. Python venv + GPU deps ─────────────────────────────────────────────────
echo "==> Creating venv and installing deps..."
python3 -m venv venv
source venv/bin/activate

# GPU-optimized torch (CUDA 12.1)
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet transformers==4.40.0 datasets tiktoken "numpy<2" sentencepiece

# Project deps (minus torch since already installed)
if [ -f requirements.txt ]; then
    grep -v "^torch" requirements.txt | pip install --quiet -r /dev/stdin || true
fi

echo "==> PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "==> CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "==> GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# ── 4. Checkpoint ─────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$RESUME_FROM")"

if [ ! -f "$RESUME_FROM" ]; then
    echo "==> Checkpoint not found at $RESUME_FROM"
    echo "    Upload it manually OR set HF_TOKEN to pull from HuggingFace."
    if [ -n "$HF_TOKEN" ]; then
        echo "    Downloading from HuggingFace (machomenc/anthos-checkpoints)..."
        pip install --quiet huggingface_hub
        python3 - << 'PYEOF'
import os
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN", "")
hf_hub_download(
    repo_id   = "machomenc/anthos-checkpoints",
    repo_type = "model",
    filename  = "step_002000.pt",
    local_dir = "checkpoints/anthos-proof",
    token     = token,
)
print("Checkpoint downloaded.")
PYEOF
    else
        echo ""
        echo "  To upload the checkpoint from your Mac, run in a NEW terminal:"
        echo "    scp -P <pod_port> /Users/dadsmacpro/Desktop/anthos-repo/checkpoints/anthos-proof/step_002000.pt root@<pod_ip>:$REPO_DIR/checkpoints/anthos-proof/"
        echo ""
        echo "  Then re-run this script."
        exit 1
    fi
fi

echo "==> Checkpoint ready: $RESUME_FROM ($(du -sh "$RESUME_FROM" | cut -f1))"

# ── 4b. Training data ─────────────────────────────────────────────────────────
DATA_FILE="$REPO_DIR/data/code_combined.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "  Training data not found: $DATA_FILE"
    echo "  Upload it from your Mac (in a NEW terminal):"
    echo ""
    echo "    scp -P <pod_port> \\"
    echo "      /Users/dadsmacpro/Desktop/anthos-repo/data/code_combined.jsonl \\"
    echo "      root@<pod_ip>:$REPO_DIR/data/code_combined.jsonl"
    echo ""
    echo "  Then re-run this script."
    exit 1
fi
echo "==> Training data ready: $(wc -l < "$DATA_FILE") examples ($(du -sh "$DATA_FILE" | cut -f1))"

# ── 5. Start training ─────────────────────────────────────────────────────────
echo ""
echo "==> Starting Anthos code-tier training..."
echo "    Log: $LOG_FILE"
echo "    Resume from: $CHECKPOINT"
echo ""

cd "$REPO_DIR"
nohup python3 train.py --tier code --resume "$CHECKPOINT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo "$TRAIN_PID" > /workspace/train.pid

# Show first output
sleep 20
echo ""
echo "==> First output:"
tail -20 "$LOG_FILE"
echo ""
echo "==> Monitor with:"
echo "    tail -f $LOG_FILE"
echo "    cat /workspace/train.pid  # to get PID"
