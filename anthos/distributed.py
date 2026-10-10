"""
anthos/distributed.py — Multi-GPU Training Utilities

Three levels of parallelism, each a superset of the previous:

  DDP  (DistributedDataParallel)
    - Replicate full model on every GPU
    - Each GPU processes a different micro-batch
    - Gradients all-reduced after every backward pass
    - Easiest to set up; memory scales with model size per GPU
    - Good for models that fit in one GPU's VRAM (~1B with bfloat16 = ~2 GB)

  FSDP (FullyShardedDataParallel)
    - Parameters, gradients, and optimizer state sharded across GPUs
    - Each GPU holds only 1/N of the model at rest; gather on demand
    - Reduces per-GPU memory ~N× for N GPUs; enables 3B+ on 4×A100
    - Slightly higher communication overhead than DDP

  Sharded Checkpointing (torch.distributed.checkpoint)
    - Each rank saves its own shard instead of rank-0 gathering
    - Eliminates the OOM spike that occurs when rank-0 gathers all shards
    - Required for FSDP checkpoints at 10B+ scale

Launch commands:
    # Single node, 4 GPUs (DDP or FSDP):
    torchrun --nproc_per_node=4 train.py --tier proof

    # Two nodes, 8 GPUs each:
    torchrun --nproc_per_node=8 --nnodes=2 \\
             --node_rank=0 --master_addr=<ip> --master_port=29500 \\
             train.py --tier proof

Environment variables set by torchrun:
    LOCAL_RANK  — GPU index on this node (0-7)
    RANK        — global rank across all nodes
    WORLD_SIZE  — total number of processes
"""

from __future__ import annotations

import os
import contextlib
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# ─────────────────────────────────────────────────────────────────────────────
# Process group lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize the process group from torchrun environment variables.

    Returns:
        (rank, local_rank, world_size)

    Call at the very start of train(), before creating the model.
    Single-GPU / non-torchrun environments are handled gracefully:
    returns (0, 0, 1) and skips dist.init_process_group.
    """
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    rank        = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def teardown_distributed() -> None:
    """Destroy the process group. Call at the end of train()."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True only on rank 0 (or single-GPU). Gate logging/saving on this."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    """Synchronise all ranks. No-op on single GPU."""
    if dist.is_initialized():
        dist.barrier()


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapping
# ─────────────────────────────────────────────────────────────────────────────

def wrap_ddp(model: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.

    find_unused_parameters=True is needed because Anthos's MoE and ACT
    paths don't always activate every parameter in every forward pass.
    """
    return DDP(
        model,
        device_ids            = [local_rank],
        output_device         = local_rank,
        find_unused_parameters= True,
    )


def wrap_fsdp(
    model:          torch.nn.Module,
    mixed_precision: bool = True,
) -> torch.nn.Module:
    """
    Wrap model with FullyShardedDataParallel.

    Shards parameters + gradients + optimizer state across all ranks.
    Use when the model doesn't fit in a single GPU's VRAM at full precision,
    or when you want to maximise batch size.

    mixed_precision=True uses bfloat16 for params and activations but
    float32 for the master copy and gradient reduction — same as bf16 DDP
    but with sharded memory.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype   = torch.bfloat16,
            reduce_dtype  = torch.float32,
            buffer_dtype  = torch.bfloat16,
        )

    # Try to import Anthos block types for auto-wrap policy
    try:
        from anthos.main import AnthosRecurrentBlock
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={AnthosRecurrentBlock},
        )
    except ImportError:
        wrap_policy = None

    return FSDP(
        model,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        mixed_precision   = mp_policy,
        auto_wrap_policy  = wrap_policy,
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
        cpu_offload       = CPUOffload(offload_params=False),
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap DDP or FSDP to get the raw Anthos module.
    Use when you need to call model-specific methods (e.g. generate()).
    """
    if hasattr(model, "module"):
        return model.module
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def make_distributed_dataloader(
    dataset,
    batch_size:  int,
    num_workers: int = 0,
    collate_fn   = None,
    seed:        int = 42,
    epoch:       int = 0,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Wrap a dataset in a DistributedSampler when running multi-GPU.

    Returns (loader, sampler).  Call sampler.set_epoch(epoch) at the start
    of each epoch so shuffling differs per epoch — otherwise all ranks see
    the same order and you effectively train on 1/N of the data.

    Single-GPU: sampler is None, standard DataLoader is returned.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas = dist.get_world_size(),
            rank         = dist.get_rank(),
            shuffle      = True,
            seed         = seed,
            drop_last    = True,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
        )
        return loader, sampler
    else:
        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            collate_fn  = collate_fn,
            pin_memory  = False,
        )
        return loader, None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint_distributed(
    path:      Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step:      int,
    loss:      float,
    data_step: int = 0,
    use_sharded: bool = False,
) -> None:
    """
    Save a checkpoint in a distributed-safe way.

    DDP mode (use_sharded=False):
        Only rank 0 saves.  The raw (unwrapped) model state dict is saved —
        same format as single-GPU, fully compatible with single-GPU resume.

    FSDP mode (use_sharded=True):
        Each rank saves its own shard via torch.distributed.checkpoint.
        No OOM spike from gathering; resume requires the same world_size
        (use consolidate_shards.py to merge into a single file if needed).

    Both modes use atomic rename (.tmp → final) to avoid corrupt files.
    """
    if use_sharded:
        _save_sharded(path, model, optimizer, step, loss, data_step)
    else:
        if is_main_process():
            _save_rank0(path, model, optimizer, step, loss, data_step)
        barrier()


def _save_rank0(
    path:      Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step:      int,
    loss:      float,
    data_step: int,
) -> None:
    """Rank-0 save: gather full model state dict and write atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step":      step,
        "loss":      loss,
        "data_step": data_step,
        "model":     unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_cpu":   torch.get_rng_state(),
        "rng_cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, tmp)
    os.replace(tmp, path)
    print(f"  ✓ Checkpoint → {path}", flush=True)


def _save_sharded(
    path:      Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step:      int,
    loss:      float,
    data_step: int,
) -> None:
    """
    FSDP sharded save via torch.distributed.checkpoint.

    Each rank writes its own shard into path/ (a directory).
    Metadata (step, loss, data_step) is written by rank 0 alongside.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    shard_dir = path.with_suffix("")   # e.g. step_001000/ directory
    shard_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    dcp.save(state_dict, checkpoint_id=str(shard_dir))

    if is_main_process():
        meta_path = shard_dir / "meta.pt"
        torch.save({"step": step, "loss": loss, "data_step": data_step}, meta_path)
        print(f"  ✓ Sharded checkpoint → {shard_dir}/", flush=True)
    barrier()


def load_checkpoint_distributed(
    path:        Path,
    model:       torch.nn.Module,
    optimizer:   Optional[torch.optim.Optimizer] = None,
    use_sharded: bool = False,
    device:      str  = "cpu",
) -> dict:
    """
    Load a checkpoint, returning a dict with step/loss/data_step metadata.

    Handles both single-file (DDP/single-GPU) and sharded (FSDP) formats
    transparently.  Pass optimizer=None to skip optimizer state loading.
    """
    if use_sharded:
        return _load_sharded(path, model, optimizer)
    else:
        return _load_rank0(path, model, optimizer, device)


def _load_rank0(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
    if missing and is_main_process():
        print(f"  ℹ New params (randomly init): {len(missing)} tensors", flush=True)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            if is_main_process():
                print("  ℹ Optimizer groups changed — resetting optimizer", flush=True)
    if "rng_cpu" in ckpt:
        torch.set_rng_state(ckpt["rng_cpu"])
    if "rng_cuda" in ckpt and ckpt["rng_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    return {k: ckpt[k] for k in ("step", "loss", "data_step") if k in ckpt}


def _load_sharded(path, model, optimizer):
    import torch.distributed.checkpoint as dcp
    shard_dir = path.with_suffix("")
    state_dict = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict() if optimizer else {}}
    dcp.load(state_dict, checkpoint_id=str(shard_dir))
    unwrap_model(model).load_state_dict(state_dict["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
    meta = {}
    meta_path = shard_dir / "meta.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Rolling checkpoint cleanup (rank-0 only)
# ─────────────────────────────────────────────────────────────────────────────

def prune_old_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    """Delete oldest checkpoints beyond the rolling keep window. Rank-0 only."""
    if not is_main_process():
        return
    checkpoints = sorted(ckpt_dir.glob("step_*.pt")) + sorted(ckpt_dir.glob("step_*/"))
    checkpoints = sorted(set(checkpoints))
    for old in checkpoints[:-keep]:
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            import shutil
            shutil.rmtree(old)
        print(f"  🗑  Removed old checkpoint {old.name}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience context manager: rank-0-only printing
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def main_process_only():
    """
    Context manager: block inside runs only on rank 0.

    Usage:
        with main_process_only():
            print("only printed once")
            wandb.log(...)
    """
    if is_main_process():
        yield
    else:
        yield   # yield on all ranks so the with-block is syntactically valid
        return
