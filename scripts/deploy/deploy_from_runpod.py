"""
deploy_from_runpod.py — Push trained Anthos checkpoint to HuggingFace Hub

Run after training completes on RunPod.

Usage:
    python deploy_from_runpod.py --checkpoint checkpoints/anthos-runpod/final.pt
    python deploy_from_runpod.py --checkpoint checkpoints/anthos-runpod/final.pt --repo TushaeBXN/anthos-1b

Requires:
    pip install huggingface_hub
    HF_TOKEN env var
"""

import argparse
import os
import torch
from pathlib import Path

from anthos.identity_hardening import CheckpointSigner


def deploy(checkpoint_path: str, repo_name: str = "TushaeBXN/anthos-1b"):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("  export HF_TOKEN='hf_...'")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    api = HfApi(token=hf_token)

    # ── Verify checkpoint ────────────────────────────────────────────────────
    print(f"  Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu")

    signer = CheckpointSigner()
    try:
        signer.verify(cp)
    except RuntimeError as e:
        print(f"  ⚠ Signature check failed: {e}")
        print("  Continuing anyway — checkpoint may be unsigned.")

    # ── Export model weights ─────────────────────────────────────────────────
    out_dir = Path("exports/huggingface")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_state = cp.get("model_state_dict", cp)
    torch.save(model_state, out_dir / "pytorch_model.bin")
    print(f"  Model weights saved to {out_dir}/pytorch_model.bin")

    # ── Write minimal model card ──────────────────────────────────────────────
    metadata = cp.get("metadata", {})
    step     = metadata.get("step", "?")
    config   = metadata.get("config", {})

    model_card = f"""---
license: cc-by-nc-4.0
language:
- en
tags:
- anthos
- thought-token
- recurrent-transformer
- mixture-of-experts
---

# Anthos

**Think in Streams.**

Built by [Tushae Thomas (TushaeBXN)](https://github.com/TushaeBXN) · 2026

## Architecture

Thought-Token Bifurcated Recurrent Transformer
- Dual streams: non-causal thought stream + causal sequence stream
- Persistent 512-slot memory bank
- Mixture-of-Experts FFN with adaptive computation time
- Identity cryptographically locked to creator

## Training checkpoint

- Steps: {step:,}
- Config: {config.get('size', '1B')} · dim={config.get('dim', 2048)} · experts={config.get('n_experts', 64)}

## Usage

```python
from anthos import Anthos, AnthosConfig
import torch

cfg   = AnthosConfig(dim=2048, n_heads=16, n_experts=64)
model = Anthos(cfg)
model.load_state_dict(torch.load("pytorch_model.bin"))

ids = torch.randint(0, 50257, (1, 64))
out = model.generate(ids, max_new_tokens=128, n_loops=16)
```

## License

CC BY-NC 4.0 — Non-commercial use only.
"""

    with open(out_dir / "README.md", "w") as f:
        f.write(model_card)

    # ── Push to Hub ──────────────────────────────────────────────────────────
    print(f"  Creating repo: {repo_name}")
    api.create_repo(repo_name, exist_ok=True, repo_type="model")

    print(f"  Uploading to https://huggingface.co/{repo_name} ...")
    api.upload_folder(
        folder_path = str(out_dir),
        repo_id     = repo_name,
        repo_type   = "model",
    )

    url = f"https://huggingface.co/{repo_name}"
    print(f"\n  ✅ Anthos deployed to: {url}")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Anthos checkpoint to HuggingFace Hub")
    parser.add_argument("--checkpoint", default="checkpoints/anthos-runpod/final.pt",
                        help="Path to signed .pt checkpoint")
    parser.add_argument("--repo",       default="TushaeBXN/anthos-1b",
                        help="HuggingFace repo name (default: TushaeBXN/anthos-1b)")
    args = parser.parse_args()

    deploy(args.checkpoint, args.repo)
