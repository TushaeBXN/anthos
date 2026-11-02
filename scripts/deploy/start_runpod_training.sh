#!/bin/bash
# start_runpod_training.sh — One-click RunPod training launcher
#
# Copy-paste this entire file into your RunPod terminal, OR:
#   bash start_runpod_training.sh
#
# Set your keys below before running.

set -e  # exit on any error

# ── Set your API keys here ────────────────────────────────────────────────────
export ANTHROPIC_API_KEY=""              # ← Paste your key here (do NOT commit with a real key)
export HF_TOKEN=""              # Your HuggingFace token — get free at huggingface.co
export WANDB_API_KEY=""         # Optional — get free at wandb.ai
export TOKENIZER="gpt2"        # Tokenizer to use (gpt2 is fine to start)

# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════"
echo "   Anthos RunPod Training — Starting"
echo "═══════════════════════════════════════════"
echo ""

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo "Step 1: Installing dependencies..."
pip install -q transformers datasets accelerate bitsandbytes tqdm requests
pip install -q huggingface_hub

# Optional but faster:
pip install -q wandb 2>/dev/null || echo "  (wandb skipped)"

echo "  ✅ Dependencies ready."

# ── 2. Clone repo (skip if already cloned) ───────────────────────────────────
if [ ! -d "anthos" ]; then
    echo ""
    echo "Step 2: Cloning Anthos repo..."
    git clone https://github.com/TushaeBXN/anthos.git
    cd anthos
    pip install -q -e .
else
    echo ""
    echo "Step 2: Updating Anthos repo..."
    cd anthos
    git pull origin main
fi

# ── 3. Create directories ────────────────────────────────────────────────────
echo ""
echo "Step 3: Creating directories..."
mkdir -p data checkpoints/anthos-runpod exports

# ── 4. Run setup checks ──────────────────────────────────────────────────────
echo ""
echo "Step 4: Running environment checks..."
python runpod_setup.py

# ── 5. Start training ────────────────────────────────────────────────────────
echo ""
echo "Step 5: Starting training..."
echo ""

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

if [ "$NUM_GPUS" -gt "1" ]; then
    echo "  Detected $NUM_GPUS GPUs — launching with torchrun"
    torchrun --nproc_per_node=$NUM_GPUS train_on_runpod.py
else
    echo "  Single GPU — launching directly"
    python train_on_runpod.py
fi
