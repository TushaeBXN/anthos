#!/usr/bin/env bash
# runpod_start.sh — Anthos SFT training on RunPod GPU
#
# Usage:
#   bash runpod_start.sh [GITHUB_TOKEN] [HF_TOKEN]
#
# Before running (on your Mac):
#   1. python build_training_manifest.py  — creates data/train.jsonl (balanced, no truncation)
#   2. python audit_dataset.py --file data/train.jsonl --sample 20000  — verify fixes
#   3. python preflight.py --fast
#   4. Upload data/train.jsonl to the pod:
#      rsync -avz --progress data/train.jsonl root@<pod-ip>:/workspace/anthos-repo/data/
#   5. bash runpod_start.sh
#
# DO NOT upload sft_master.jsonl — use train.jsonl (the balanced manifest)

set -e

GITHUB_TOKEN="${1:-${GITHUB_TOKEN:-}}"
HF_TOKEN="${2:-${HF_TOKEN:-}}"
REPO_URL="https://github.com/TushaeBXN/anthos"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/anthos-repo"
DATA_FILE="$REPO_DIR/data/train.jsonl"
TIER="sft"
LOG_FILE="$WORKSPACE/train_stdout.txt"

# Resume checkpoint — set to latest mansa_sovereign checkpoint if available
RESUME_CKPT="checkpoints/mansa_sovereign/step_010000.pt"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo "==> Installing system deps..."
apt-get update -qq && apt-get install -y -q git python3-pip python3-venv build-essential

# ── 2. Clone / pull repo ──────────────────────────────────────────────────────
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
echo "==> Creating venv and installing GPU deps..."
python3 -m venv venv
source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --quiet "transformers==4.40.0" datasets tiktoken "numpy<2" sentencepiece
pip install --quiet duckduckgo-search wikipedia langdetect  # runtime deps for retrieval/multilingual

if [ -f requirements.txt ]; then
    grep -v "^torch" requirements.txt | pip install --quiet -r /dev/stdin || true
fi

# Install anthos as editable package
pip install --quiet -e . || true

echo "==> PyTorch   : $(python3 -c 'import torch; print(torch.__version__)')"
echo "==> CUDA      : $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "==> GPU       : $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# ── 4. Training data ──────────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/data"

if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │  data/train.jsonl NOT FOUND                                │"
    echo "  │                                                             │"
    echo "  │  On your Mac, run first:                                   │"
    echo "  │    python build_training_manifest.py                        │"
    echo "  │                                                             │"
    echo "  │  Then upload to the pod:                                   │"
    echo "  │    rsync -avz --progress \\                                  │"
    echo "  │      data/train.jsonl \\                                     │"
    echo "  │      root@<pod-ip>:/workspace/anthos-repo/data/            │"
    echo "  │                                                             │"
    echo "  │  Then re-run: bash runpod_start.sh                         │"
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""
    exit 1
fi

N_PAIRS=$(wc -l < "$DATA_FILE")
echo "==> Training data : $N_PAIRS pairs ($(du -sh "$DATA_FILE" | cut -f1))"

# ── 5. Tokenizer ──────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/data/anthos_tokenizer" ]; then
    echo "==> Building anthos_tokenizer..."
    python3 setup_tokenizer.py
fi
echo "==> Tokenizer : data/anthos_tokenizer ✓"

# ── 6. Preflight check ────────────────────────────────────────────────────────
echo ""
echo "==> Running preflight --fast..."
python3 preflight.py --fast || {
    echo "  Preflight failed — fix issues before training."
    exit 1
}

# ── 7. Resume checkpoint ──────────────────────────────────────────────────────
RESUME_FLAG=""
if [ -f "$REPO_DIR/$RESUME_CKPT" ]; then
    echo "==> Resuming from: $RESUME_CKPT"
    RESUME_FLAG="--resume $RESUME_CKPT"
elif [ -n "$HF_TOKEN" ]; then
    echo "==> Downloading checkpoint from HuggingFace..."
    pip install --quiet huggingface_hub
    python3 - << PYEOF
import os
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN", "")
try:
    hf_hub_download(
        repo_id="machomenc/anthos-checkpoints",
        repo_type="model",
        filename="step_010000.pt",
        local_dir="checkpoints/mansa_sovereign",
        token=token,
    )
    print("Checkpoint downloaded.")
except Exception as e:
    print(f"  Could not download checkpoint: {e}")
    print("  Training from scratch.")
PYEOF
    if [ -f "$REPO_DIR/$RESUME_CKPT" ]; then
        RESUME_FLAG="--resume $RESUME_CKPT"
    fi
else
    echo "==> No checkpoint found — training from scratch."
fi

# ── 8. Start training ─────────────────────────────────────────────────────────
echo ""
echo "==> Starting Anthos SFT training..."
echo "    Tier     : $TIER"
echo "    Data     : $DATA_FILE ($N_PAIRS pairs)"
echo "    Resume   : ${RESUME_FLAG:-from scratch}"
echo "    Log      : $LOG_FILE"
echo ""

nohup python3 train.py --tier "$TIER" $RESUME_FLAG > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > /workspace/train.pid
echo "Training PID: $TRAIN_PID"

sleep 30
echo ""
echo "==> First output:"
tail -30 "$LOG_FILE"
echo ""
echo "==> Monitor with:"
echo "    tail -f $LOG_FILE"
echo ""
echo "==> To check GPU usage:"
echo "    watch -n5 nvidia-smi"
