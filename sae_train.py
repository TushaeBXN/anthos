"""
sae_train.py — Train Sparse Autoencoders on Anthos Activations

Two-phase workflow:

  Phase 1 — Collect activations from a trained Anthos checkpoint
  Phase 2 — Train SAE(s) on collected activations

This is run separately from Anthos training. A trained Anthos checkpoint
is required as input.

Example:
    # Train SAE on thought stream from proof-tier checkpoint
    python sae_train.py \\
        --checkpoint checkpoints/mansa_sovereign/step_010000.pt \\
        --stream thought \\
        --n_thought 16 \\
        --dim 512 \\
        --expansion 16 \\
        --k 64 \\
        --steps 50000 \\
        --out checkpoints/sae/thought_stream.pt

    # Then analyze features
    python sae_train.py \\
        --checkpoint checkpoints/mansa_sovereign/step_010000.pt \\
        --sae_checkpoint checkpoints/sae/thought_stream.pt \\
        --analyze \\
        --stream thought
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config
from anthos.data    import get_dataloader
from anthos.sae     import SparseAutoencoder, SAEConfig
from anthos.steering import ActivationCollector


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Collect activations
# ─────────────────────────────────────────────────────────────────────────────

def collect_activations(
    model:           Anthos,
    data_loader,
    stream:          str   = "thought",
    n_thought:       int   = 16,
    n_batches:       int   = 500,
    n_loops:         int   = 8,
    device:          str   = "cpu",
    seq_len:         int   = 512,
) -> torch.Tensor:
    """
    Run Anthos forward passes and collect hidden states from the specified stream.

    Returns:
        flat_acts: [N, D] where N = n_batches * batch_size * T_stream
    """
    print(f"\n── Collecting {stream}-stream activations ───────────────────────")

    collector = ActivationCollector(model, stream=stream, n_thought_tokens=n_thought)
    collector.attach()
    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            if step >= n_batches:
                break

            if isinstance(batch, (list, tuple)):
                ids = batch[0].to(device)[:, :seq_len]
            else:
                ids = batch.to(device)[:, :seq_len-1]

            try:
                model(ids, n_loops=n_loops)
            except Exception as e:
                print(f"  Warning: forward pass error at step {step}: {e}")
                continue

            if step % 50 == 0:
                n_collected = sum(
                    t.shape[0] * t.shape[1]
                    for tensors in collector._store.values()
                    for t in tensors
                )
                print(f"  step {step:4d}/{n_batches} | activations: {n_collected:,}")

    collector.detach()
    model.train()

    flat = collector.flat_activations()
    print(f"  Total activations collected: {flat.shape[0]:,} × {flat.shape[1]}")
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Train SAE
# ─────────────────────────────────────────────────────────────────────────────

def train_sae(
    activations: torch.Tensor,
    cfg:         SAEConfig,
    steps:       int   = 50_000,
    batch_size:  int   = 2048,
    lr:          float = 2e-4,
    out_path:    Path  = Path("checkpoints/sae/sae.pt"),
    device:      str   = "cpu",
) -> SparseAutoencoder:
    """Train a single SAE on flat [N, D] activations."""

    print(f"\n── Training SAE ─────────────────────────────────────────────────")
    print(f"  d_model={cfg.d_model}, d_sae={cfg.d_sae}, k={cfg.k}")
    print(f"  Activations: {activations.shape[0]:,} × {activations.shape[1]}")
    print(f"  Steps: {steps:,} | LR: {lr} | Batch: {batch_size}")

    sae = SparseAutoencoder(cfg).to(device)
    optimizer = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    N = activations.shape[0]
    t0 = time.time()

    for step in range(1, steps + 1):
        # Random mini-batch from collected activations
        idx   = torch.randint(0, N, (batch_size,))
        batch = activations[idx].to(device)

        optimizer.zero_grad()
        features, x_hat = sae(batch)
        loss, info = sae.loss(batch, features, x_hat)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        sae.post_step()  # Renormalize decoder columns

        if step % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step:6d} | "
                f"loss {info['total_loss']:.5f} | "
                f"recon {info['recon_loss']:.5f} | "
                f"l1 {info['l1_loss']:.5f} | "
                f"active {info['avg_active']:.1f}/{cfg.k} | "
                f"{elapsed:.0f}s"
            )
            t0 = time.time()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cfg":        cfg.__dict__,
        "state_dict": sae.state_dict(),
    }, out_path)
    print(f"\n  ✓ SAE saved → {out_path}")

    return sae


# ─────────────────────────────────────────────────────────────────────────────
# Analysis: top features by activation
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sae(
    sae:         SparseAutoencoder,
    activations: torch.Tensor,
    top_n:       int = 20,
    device:      str = "cpu",
):
    """Print the top-N features by mean activation strength."""
    from anthos.features import feature_rank

    print(f"\n── Top {top_n} Features by Mean Activation ──────────────────────")
    sorted_ids, mean_acts = feature_rank(sae, activations, batch_size=512)

    for rank, (fid, act) in enumerate(
        zip(sorted_ids[:top_n].tolist(), mean_acts[:top_n].tolist())
    ):
        print(f"  rank {rank+1:3d} | feature {fid:6d} | mean_act {act:.5f}")

    print(f"\n── Feature Sparsity ──────────────────────────────────────────────")
    # Check what fraction of features are ever active
    device_sae = next(sae.parameters()).device
    ever_active = torch.zeros(sae.cfg.d_sae)
    n_sampled = min(10_000, activations.shape[0])
    sample = activations[torch.randperm(activations.shape[0])[:n_sampled]]
    for i in range(0, n_sampled, 512):
        chunk = sample[i:i+512].to(device_sae)
        f, _ = sae(chunk)
        ever_active = ever_active + (f.cpu() > 0).any(0).float()
    pct_active = (ever_active > 0).float().mean() * 100
    print(f"  Features ever active (over {n_sampled} samples): {pct_active:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAE on Anthos activations")

    # Model
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--tier",           type=str, default="proof")
    parser.add_argument("--stream",         type=str, default="thought",
                        choices=["thought", "sequence", "both"])
    parser.add_argument("--n_thought",      type=int, default=16)
    parser.add_argument("--n_loops",        type=int, default=8)
    parser.add_argument("--seq_len",        type=int, default=512)

    # SAE config
    parser.add_argument("--dim",            type=int,   default=512)
    parser.add_argument("--expansion",      type=int,   default=16)
    parser.add_argument("--k",              type=int,   default=64)
    parser.add_argument("--l1_coeff",       type=float, default=2e-4)

    # Training
    parser.add_argument("--collect_batches", type=int,  default=500)
    parser.add_argument("--steps",           type=int,  default=50_000)
    parser.add_argument("--batch_size",      type=int,  default=2048)
    parser.add_argument("--lr",              type=float, default=2e-4)

    # Output
    parser.add_argument("--out",             type=str,  default="checkpoints/sae/sae.pt")
    parser.add_argument("--sae_checkpoint",  type=str,  default=None,
                        help="Load existing SAE for analysis only")
    parser.add_argument("--analyze",         action="store_true")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Anthos model
    model_cfg, train_cfg = get_training_config(args.tier)
    model = Anthos(model_cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"✓ Loaded Anthos from {args.checkpoint}")

    # Data
    loader = get_dataloader(
        dataset_name="roneneldan/TinyStories",
        split="train",
        seq_len=args.seq_len,
        batch_size=train_cfg.batch_size,
        num_workers=0,
    )

    # Collect activations
    acts = collect_activations(
        model, loader,
        stream=args.stream,
        n_thought=args.n_thought,
        n_batches=args.collect_batches,
        n_loops=args.n_loops,
        device=device,
        seq_len=args.seq_len,
    )

    # SAE config
    sae_cfg = SAEConfig(
        d_model=args.dim,
        expansion=args.expansion,
        k=args.k,
        l1_coeff=args.l1_coeff,
    )

    if args.sae_checkpoint and Path(args.sae_checkpoint).exists():
        # Load existing SAE
        saved = torch.load(args.sae_checkpoint, map_location=device, weights_only=False)
        sae   = SparseAutoencoder(sae_cfg).to(device)
        sae.load_state_dict(saved["state_dict"])
        print(f"✓ Loaded SAE from {args.sae_checkpoint}")
    else:
        # Train new SAE
        sae = train_sae(
            acts, sae_cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            out_path=Path(args.out),
            device=device,
        )

    if args.analyze:
        analyze_sae(sae, acts, device=device)


if __name__ == "__main__":
    main()
