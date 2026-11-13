"""Turn raw CIFAR-10 images into model input. Training and inference both use this."""

import numpy as np

from projects.cifar10.data.io import SIZE

INPUT_SHAPE: tuple[int, ...] = (3, SIZE, SIZE)  # one model input sample
# Per-channel mean and standard deviation of the train split, for pixel values in [0, 1].
MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def normalize(images: np.ndarray) -> np.ndarray:
    """Turn uint8 RGB images (N, 32, 32, 3) into normalized float32 model input (N, 3, 32, 32)."""
    mean: np.ndarray = np.asarray(MEAN, dtype=np.float32)
    std: np.ndarray = np.asarray(STD, dtype=np.float32)
    x: np.ndarray = (images.astype(np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
