"""Download the MNIST files and read them as numpy arrays."""

import gzip
from pathlib import Path

import numpy as np

from vizard.data.download import download_file

BASE_URL: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES: dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR: Path = Path(__file__).resolve().parents[3] / ".data" / "mnist"


def download(data_dir: Path = DATA_DIR) -> None:
    """Download the MNIST files if they are not present."""
    for name in FILES.values():
        download_file(BASE_URL + name, data_dir / name)


def _read_idx(path: Path) -> np.ndarray:
    """Read a gzipped IDX file into a numpy array."""
    with gzip.open(path, "rb") as f:
        raw: bytes = f.read()
    ndim: int = raw[3]
    shape: tuple[int, ...] = tuple(
        int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)
    )
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_arrays(train: bool, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load one split as raw arrays: images (N, 28, 28) uint8 and labels (N,) uint8."""
    download(data_dir)
    prefix: str = "train" if train else "test"
    images: np.ndarray = _read_idx(data_dir / FILES[f"{prefix}_images"])
    labels: np.ndarray = _read_idx(data_dir / FILES[f"{prefix}_labels"])
    return images, labels
