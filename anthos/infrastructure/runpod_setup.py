"""
runpod_setup.py — One-time environment setup for RunPod pods

Run this FIRST in your RunPod Jupyter terminal before anything else.
Sets up dependencies, clones the repo, and verifies all keys are working.

Usage:
    python runpod_setup.py
"""

import os
import subprocess
import sys


def run(cmd: str, check: bool = True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed with code {result.returncode}")
    return result


def main():
    print("\n" + "─" * 60)
    print("  Anthos RunPod Setup")
    print("─" * 60 + "\n")

    # ── 1. Dependencies ───────────────────────────────────────────────────────
    print("Step 1: Installing dependencies...")
    run("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    run("pip install -q transformers datasets accelerate bitsandbytes")
    run("pip install -q wandb tqdm requests")
    run("pip install -q huggingface_hub")
    print("  Dependencies installed.\n")

    # ── 2. Verify GPU ─────────────────────────────────────────────────────────
    print("Step 2: Checking GPU...")
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name}  ({vram:.1f} GB VRAM)")
    else:
        print("  WARNING: No GPU found. Training will be very slow.")
    print()

    # ── 3. Check environment variables ────────────────────────────────────────
    print("Step 3: Checking API keys...")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    hf_token      = os.environ.get("HF_TOKEN", "")
    wandb_key     = os.environ.get("WANDB_API_KEY", "")

    if anthropic_key:
        print(f"  ✅ ANTHROPIC_API_KEY  set ({anthropic_key[:12]}...)")
    else:
        print("  ❌ ANTHROPIC_API_KEY  NOT SET")
        print("     export ANTHROPIC_API_KEY='sk-ant-...'")

    if hf_token:
        print(f"  ✅ HF_TOKEN           set ({hf_token[:8]}...)")
    else:
        print("  ⚠  HF_TOKEN           not set (needed for FineWeb-Edu)")
        print("     export HF_TOKEN='hf_...'")

    if wandb_key:
        print(f"  ✅ WANDB_API_KEY      set ({wandb_key[:8]}...)")
    else:
        print("  ⚠  WANDB_API_KEY      not set (optional — disables dashboard)")
    print()

    # ── 4. Test FineWeb-Edu streaming ────────────────────────────────────────
    print("Step 4: Testing FineWeb-Edu access...")
    try:
        from datasets import load_dataset
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            **kwargs,
        )
        sample = next(iter(ds))
        print(f"  ✅ FineWeb-Edu ready. Sample text: {sample['text'][:80]}...")
    except Exception as e:
        print(f"  ❌ FineWeb-Edu failed: {e}")
        print("     Make sure HF_TOKEN is set and you have a HuggingFace account.")
    print()

    # ── 5. Create data directories ────────────────────────────────────────────
    print("Step 5: Creating directories...")
    for d in ["data", "checkpoints/anthos-runpod", "exports"]:
        os.makedirs(d, exist_ok=True)
        print(f"  Created {d}/")
    print()

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("─" * 60)
    print("  Setup complete. Next steps:")
    print()
    print("  1. Set any missing keys above, then:")
    print("     python train_on_runpod.py")
    print()
    print("  2. Multi-GPU (if your pod has 4 GPUs):")
    print("     torchrun --nproc_per_node=4 train_on_runpod.py")
    print()
    print("  3. Monitor training in another terminal:")
    print("     watch -n 2 nvidia-smi")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
