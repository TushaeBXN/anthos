"""Train Anthos on any combination of GPUs, CPUs, or cloud instances"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from typing import List, Optional


class DistributedGrid:
    """Coordinator for distributed training across GPUs"""

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.world_size = 0
        self.rank = 0
        self.initialized = False

    def init(self):
        if not dist.is_initialized():
            dist.init_process_group(self.backend)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.initialized = True
        print(f"Worker {self.rank}/{self.world_size} initialized")

    def wrap_model(self, model: torch.nn.Module, device_id: Optional[int] = None) -> torch.nn.Module:
        if device_id is None:
            device_id = self.rank % torch.cuda.device_count()
        model = model.to(f"cuda:{device_id}")
        return DDP(model, device_ids=[device_id])

    def all_reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return loss

    def barrier(self):
        if self.initialized:
            dist.barrier()

    def is_main(self) -> bool:
        return self.rank == 0

    @staticmethod
    def detect_hardware() -> List[str]:
        resources = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                resources.append(f"cuda:{i}")
        if not resources:
            cpu_count = os.cpu_count() or 1
            resources.extend([f"cpu:{i}" for i in range(min(cpu_count // 2, 4))])
        return resources

    def launch_torchrun_command(self, script: str, nproc: Optional[int] = None) -> str:
        n = nproc or torch.cuda.device_count() or 1
        return f"torchrun --nproc_per_node={n} {script}"


def get_distributed_sampler(dataset, world_size: int, rank: int):
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
