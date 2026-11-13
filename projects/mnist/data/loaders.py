"""Datasets and batch loaders for MNIST training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from projects.mnist.data.io import DATA_DIR, load_arrays
from projects.mnist.data.preprocess import normalize
from vizard.data.tensor_loader import TensorBatchLoader

Batch = tuple[torch.Tensor, torch.Tensor]  # images (N, 1, 28, 28) float32, labels (N,) int64


def load_split(train: bool, data_dir: Path = DATA_DIR) -> TensorDataset:
    """Load one split as a normalized (N, 1, 28, 28) float dataset."""
    images, labels = load_arrays(train, data_dir)
    x: torch.Tensor = torch.from_numpy(normalize(images))
    y: torch.Tensor = torch.from_numpy(labels.astype(np.int64))
    return TensorDataset(x, y)


def get_loader(split: str, batch_size: int, shuffle: bool = False) -> TensorBatchLoader:
    """Return the batch loader of one split: "train" (60000 images) or "test" (10000 images).

    The split is in memory, so each batch is one index operation (see TensorBatchLoader).
    """
    if split not in ("train", "test"):
        raise ValueError(f'split must be "train" or "test", got {split!r}')
    x, y = load_split(split == "train").tensors
    return TensorBatchLoader(x, y, batch_size=batch_size, shuffle=shuffle)
