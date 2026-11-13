"""Download the CIFAR-10 files (binary version) and read them as numpy arrays."""

import tarfile
from pathlib import Path

import numpy as np
from loguru import logger

from vizard.data.download import download_file

URL: str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE: str = "cifar-10-binary.tar.gz"
EXTRACTED_DIR: str = "cifar-10-batches-bin"
TRAIN_FILES: tuple[str, ...] = tuple(f"data_batch_{i}.bin" for i in range(1, 6))
TEST_FILES: tuple[str, ...] = ("test_batch.bin",)
DATA_DIR: Path = Path(__file__).resolve().parents[3] / ".data" / "cifar10"

CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
SIZE: int = 32
RECORD_BYTES: int = 1 + 3 * SIZE * SIZE  # 1 label byte, then the red, green and blue planes


def download(data_dir: Path = DATA_DIR) -> None:
    """Download and extract the CIFAR-10 files (about 170 MB) if they are not present."""
    if (data_dir / EXTRACTED_DIR / TEST_FILES[0]).exists():
        return
    archive: Path = download_file(URL, data_dir / ARCHIVE)
    logger.info(f"extracting {archive}")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")  # "data": no absolute paths, links or devices


def load_arrays(train: bool, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load one split as raw arrays: images (N, 32, 32, 3) uint8 RGB and labels (N,) uint8."""
    download(data_dir)
    files: tuple[str, ...] = TRAIN_FILES if train else TEST_FILES
    raw: np.ndarray = np.concatenate(
        [np.fromfile(data_dir / EXTRACTED_DIR / name, dtype=np.uint8) for name in files]
    )
    records: np.ndarray = raw.reshape(-1, RECORD_BYTES)
    labels: np.ndarray = records[:, 0]
    images: np.ndarray = records[:, 1:].reshape(-1, 3, SIZE, SIZE).transpose(0, 2, 3, 1)
    return images, labels
