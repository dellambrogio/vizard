"""Turn raw MNIST images into model input. Training and inference both use this."""

import numpy as np

SIZE: int = 28
INPUT_SHAPE: tuple[int, ...] = (1, SIZE, SIZE)  # one model input sample
MEAN: float = 0.1307
STD: float = 0.3081


def normalize(images: np.ndarray) -> np.ndarray:
    """Turn uint8 images (N, 28, 28) into normalized float32 model input (N, 1, 28, 28)."""
    x: np.ndarray = (images.astype(np.float32) / 255.0 - MEAN) / STD
    return x[:, None, :, :]
