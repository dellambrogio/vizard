"""Contract for train data that DDP can split between the ranks."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ShardableData(Protocol):
    """Train data that gives each DDP rank its own part of each epoch.

    With DDP, run_training calls shard with the rank, the number of ranks and the seed of the
    run. After it, each rank gets a different part of each epoch, and all ranks get the same
    number of batches (DDP needs the same number of train steps on all ranks).
    """

    def shard(self, rank: int, world_size: int, seed: int) -> None: ...
