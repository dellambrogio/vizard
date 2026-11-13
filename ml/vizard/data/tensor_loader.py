"""Batches of data that is already in memory as tensors, with one index operation per batch."""

from collections.abc import Callable, Iterator

import torch

# Changes a batch, for example a random augmentation: (tensors of the batch) -> new batch.
BatchTransform = Callable[[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]


class TensorBatchLoader:
    """Iterable of batches of tensors that are already in memory, like TensorDataset data.

    Each batch is a tuple with one index operation on each tensor, so it is much faster than a
    DataLoader, which gets the samples one at a time and then stacks them. All tensors must
    have the same first dimension (the samples).

    shuffle uses the global torch random state, so the seed of the run sets the data order.
    drop_last drops the last batch if it has fewer than batch_size samples. transform, if set,
    changes each batch (for example a random augmentation of a full batch).

    The batches stay on the device of the tensors. On CPU, TorchTNT copies the next batch to
    the GPU while the current train step runs.

    With DDP, run_training calls shard (see ShardableData): each epoch, all ranks make the same
    order from a generator seeded with the seed of the run plus the epoch number, and each rank
    takes its own part. Each rank gets num_samples // world_size samples, so up to
    world_size - 1 samples of an epoch are left out.
    """

    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        transform: BatchTransform | None = None,
    ) -> None:
        if not tensors:
            raise ValueError("give at least one tensor")
        sizes: set[int] = {len(t) for t in tensors}
        if len(sizes) != 1:
            raise ValueError(f"all tensors must have the same first dimension, got {sizes}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.tensors: tuple[torch.Tensor, ...] = tensors
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last
        self.transform: BatchTransform | None = transform
        self.num_samples: int = sizes.pop()
        self.rank: int = 0
        self.world_size: int = 1
        self._seed: int = 0
        self._epoch: int = 0  # number of epochs started after shard

    def shard(self, rank: int, world_size: int, seed: int) -> None:
        """Give this rank its own part of each epoch (see the class docstring)."""
        if not 0 <= rank < world_size:
            raise ValueError(f"need 0 <= rank < world_size, got rank {rank}, size {world_size}")
        if self.num_samples < world_size:
            raise ValueError(f"{self.num_samples} samples cannot be split in {world_size} parts")
        self.rank, self.world_size, self._seed, self._epoch = rank, world_size, seed, 0

    @property
    def samples_per_epoch(self) -> int:
        """The number of samples of this rank in one epoch."""
        return self.num_samples // self.world_size

    def __len__(self) -> int:
        """The number of batches of this rank in one epoch."""
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return -(-self.samples_per_epoch // self.batch_size)

    def _epoch_indices(self) -> torch.Tensor | None:
        """The sample indices of this rank for the next epoch; None: all samples in order."""
        if self.world_size == 1:
            return torch.randperm(self.num_samples) if self.shuffle else None
        if self.shuffle:  # the same order on all ranks
            generator = torch.Generator().manual_seed(self._seed + self._epoch)
            order: torch.Tensor = torch.randperm(self.num_samples, generator=generator)
        else:
            order = torch.arange(self.num_samples)
        self._epoch += 1
        return order[: self.samples_per_epoch * self.world_size][self.rank :: self.world_size]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        indices: torch.Tensor | None = self._epoch_indices()
        if indices is not None:
            indices = indices.to(self.tensors[0].device)
        for i in range(len(self)):
            start: int = i * self.batch_size
            end: int = min(start + self.batch_size, self.samples_per_epoch)
            if indices is None:
                batch = tuple(t[start:end] for t in self.tensors)
            else:
                index: torch.Tensor = indices[start:end]
                batch = tuple(t[index] for t in self.tensors)
            yield self.transform(batch) if self.transform is not None else batch
