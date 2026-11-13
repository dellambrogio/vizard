"""Load PNG and JPG images and prepare them as MNIST-style 28x28 grayscale arrays."""

from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from projects.mnist.data.preprocess import SIZE

IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


class Invert(StrEnum):
    """How to handle the color polarity. MNIST digits are white on black."""

    AUTO = "auto"  # invert if the image is mostly bright (dark digit on light background)
    YES = "yes"
    NO = "no"


def find_images(path: Path) -> list[Path]:
    """Return [path] for an image file, or all PNG/JPG files in a folder (sorted, not recursive)."""
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{path} is not a PNG or JPG file")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"{path} does not exist")


def load_image(path: Path, invert: Invert = Invert.AUTO) -> np.ndarray:
    """Load one image as a uint8 (28, 28) array: grayscale, white digit on black background."""
    with Image.open(path) as img:
        gray: Image.Image = ImageOps.exif_transpose(img).convert("L")
    if gray.size != (SIZE, SIZE):
        gray = gray.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    array: np.ndarray = np.asarray(gray, dtype=np.uint8)
    if invert is Invert.YES or (invert is Invert.AUTO and array.mean() > 127):
        array = 255 - array
    return array


def load_images(paths: list[Path], invert: Invert = Invert.AUTO) -> np.ndarray:
    """Load images as a uint8 (N, 28, 28) array."""
    return np.stack([load_image(p, invert) for p in paths])
