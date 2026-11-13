"""Classify MNIST digits with an exported ONNX model and onnxruntime."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from loguru import logger

from projects.mnist.data.preprocess import normalize
from projects.mnist.export.onnx import INPUT_NAME, OUTPUT_NAME


@dataclass(frozen=True)
class Prediction:
    """Predicted labels (N,) and class probabilities (N, 10)."""

    labels: np.ndarray
    probs: np.ndarray


class OnnxClassifier:
    """MNIST classifier backed by an ONNX model."""

    def __init__(self, path: Path, providers: list[str] | None = None) -> None:
        self.path: Path = path
        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        logger.info(f"loaded {path} with {self.session.get_providers()}")

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Run the model on normalized float32 input (N, 1, 28, 28) and return logits (N, 10)."""
        outputs = self.session.run([OUTPUT_NAME], {INPUT_NAME: x})
        return np.asarray(outputs[0])

    def predict(self, images: np.ndarray, batch_size: int = 1000) -> Prediction:
        """Classify uint8 images (N, 28, 28)."""
        chunks: list[np.ndarray] = [
            self.logits(normalize(images[i : i + batch_size]))
            for i in range(0, len(images), batch_size)
        ]
        logits: np.ndarray = np.concatenate(chunks)
        shifted: np.ndarray = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs: np.ndarray = shifted / shifted.sum(axis=1, keepdims=True)
        return Prediction(labels=logits.argmax(axis=1), probs=probs)
