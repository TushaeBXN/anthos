"""
monitor_runpod.py — Real-time RunPod training monitor

Logs GPU utilization, memory, temperature, and training metrics
to WandB every 30 seconds.

Usage:
    # In a second RunPod terminal while training runs:
    python monitor_runpod.py

Requires:
    pip install wandb gputil psutil
    WANDB_API_KEY env var
"""

import os
import time
import json
import torch
from pathlib import Path


def get_gpu_stats() -> list[dict]:
    """Get per-GPU statistics."""
    stats = []

    if not torch.cuda.is_available():
        return stats

    for i in range(torch.cuda.device_count()):
        mem_used  = torch.cuda.memory_allocated(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        stats.append({
            "id":         i,
            "name":       torch.cuda.get_device_properties(i).name,
            "mem_used_gb":  round(mem_used, 2),
            "mem_total_gb": round(mem_total, 2),
            "mem_pct":    round(mem_used / mem_total * 100, 1),
        })

    return stats


def get_latest_checkpoint_info(checkpoints_dir: str = "checkpoints/anthos-runpod") -> dict:
    """Read the latest step count and loss from checkpoint directory."""
    path = Path(checkpoints_dir)
    if not path.exists():
        return {}

    checkpoints = sorted(path.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return {}

    latest = checkpoints[-1]
    try:
        cp = torch.load(latest, map_location="cpu")
        metadata = cp.get("metadata", {})
        return {
            "latest_checkpoint": str(latest.name),
            "step": metadata.get("step", 0),
        }
    except Exception:
        return {"latest_checkpoint": str(latest.name)}


def print_status(gpu_stats: list, cp_info: dict, elapsed_min: float):
    """Print a clean status line to the terminal."""
    print(f"\n{'─'*55}")
    print(f"  Elapsed: {elapsed_min:.0f}m")

    for gpu in gpu_stats:
        bar_len  = 20
        filled   = int(gpu["mem_pct"] / 100 * bar_len)
        bar      = "█" * filled + "░" * (bar_len - filled)
        print(f"  GPU {gpu['id']} [{bar}] {gpu['mem_pct']}%  "
              f"{gpu['mem_used_gb']:.1f}/{gpu['mem_total_gb']:.0f} GB")

    if cp_info:
        print(f"  Latest checkpoint: {cp_info.get('latest_checkpoint', '—')}  "
              f"(step {cp_info.get('step', 0):,})")


def main():
    interval_sec = 30
    t0 = time.time()

    # WandB setup
    wandb_enabled = False
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.init(project="anthos-monitor", name="runpod-hardware")
            wandb_enabled = True
            print("  WandB monitoring enabled.")
        except ImportError:
            print("  WandB not installed — logging to terminal only.")
    else:
        print("  WANDB_API_KEY not set — logging to terminal only.")

    print(f"  Polling every {interval_sec}s. Ctrl+C to stop.\n")

    try:
        while True:
            gpu_stats = get_gpu_stats()
            cp_info   = get_latest_checkpoint_info()
            elapsed   = (time.time() - t0) / 60

            print_status(gpu_stats, cp_info, elapsed)

            if wandb_enabled:
                log = {"elapsed_min": elapsed}
                for gpu in gpu_stats:
                    log[f"gpu_{gpu['id']}_mem_pct"]  = gpu["mem_pct"]
                    log[f"gpu_{gpu['id']}_mem_gb"]    = gpu["mem_used_gb"]
                if cp_info.get("step"):
                    log["checkpoint_step"] = cp_info["step"]
                wandb.log(log)

            # Warn on high memory usage
            for gpu in gpu_stats:
                if gpu["mem_pct"] > 95:
                    print(f"  ⚠ GPU {gpu['id']} memory at {gpu['mem_pct']}% — risk of OOM")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
