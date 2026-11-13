"""Datasets and batch loaders for CIFAR-10 training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from projects.cifar10.data.augment import random_crop_flip
from projects.cifar10.data.io import DATA_DIR, load_arrays
from projects.cifar10.data.preprocess import normalize
from vizard.data.tensor_loader import TensorBatchLoader

Batch = tuple[torch.Tensor, torch.Tensor]  # images (N, 3, 32, 32) float32, labels (N,) int64


def load_split(train: bool, data_dir: Path = DATA_DIR) -> TensorDataset:
    """Load one split as a normalized (N, 3, 32, 32) float dataset."""
    images, labels = load_arrays(train, data_dir)
    x: torch.Tensor = torch.from_numpy(normalize(images))
    y: torch.Tensor = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x, y)


def _augment(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    x, y = batch
    return random_crop_flip(x), y


def get_loader(
    split: str, batch_size: int, shuffle: bool = False, augment: bool = False
) -> TensorBatchLoader:
    """Return the batch loader of one split: "train" (50000 images) or "test" (10000 images).

    The split is in memory, so each batch is one index operation (see TensorBatchLoader).
    With augment, each batch gets a random crop and a random horizontal flip per image.
    """
    if split not in ("train", "test"):
        raise ValueError(f'split must be "train" or "test", got {split!r}')
    x, y = load_split(split == "train").tensors
    return TensorBatchLoader(
        x, y, batch_size=batch_size, shuffle=shuffle, transform=_augment if augment else None
    )
