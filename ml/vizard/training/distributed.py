"""DDP helpers: the process group, the rank and the world size.

A run with more than one process starts with torchrun (WORLD_SIZE > 1). Without torchrun,
the run has one process: rank 0 of world size 1.
"""

import os
from typing import Any

import torch
import torch.distributed as dist
from torchtnt.utils.env import init_from_env


def launched_distributed() -> bool:
    """True if torchrun started more than one process."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> torch.device:
    """Start the process group (NCCL on GPU, gloo on CPU) and return the device of this rank.

    Call it only when launched_distributed() is true. The float32 matmul precision is not
    changed.
    """
    return init_from_env(float32_matmul_precision="highest")


def is_distributed() -> bool:
    """True if the process group is started and has more than one process."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main() -> bool:
    """True on rank 0, the rank that writes the job files and talks to W&B."""
    return get_rank() == 0


def broadcast_from_main(value: Any) -> Any:  # noqa: ANN401
    """Return the value of rank 0 on all ranks. Without DDP, return value."""
    if not is_distributed():
        return value
    values: list[Any] = [value if is_main() else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]
