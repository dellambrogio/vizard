"""Base ONNX exporter. A project inherits from BaseExporter and gives the names and inputs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import onnxruntime as ort
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from torch import nn


class BaseExporter(ABC):
    """Export a model to ONNX with a dynamic batch size and check it with onnxruntime.

    A subclass sets input_names and output_names and implements example_inputs. The first
    dimension of each input is the batch.
    """

    input_names: ClassVar[tuple[str, ...]]
    output_names: ClassVar[tuple[str, ...]]

    def __init__(self, verify_batch_size: int = 5, atol: float = 1e-4) -> None:
        self.verify_batch_size: int = verify_batch_size
        self.atol: float = atol

    @abstractmethod
    def example_inputs(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Return random model inputs with batch_size samples, one tensor per input name."""

    def export(self, model: nn.Module, path: Path) -> Path:
        """Export model to path."""
        model = model.eval().cpu()
        inputs: tuple[torch.Tensor, ...] = self.example_inputs(2)
        batch = torch.export.Dim("batch")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=list(self.input_names),
            output_names=list(self.output_names),
            dynamic_shapes=tuple({0: batch} for _ in inputs),
            external_data=False,
        )
        logger.info(f"exported ONNX model to {path}")
        return path

    def verify(self, model: nn.Module, path: Path) -> None:
        """Compare onnxruntime outputs with PyTorch outputs on random input. Raise on a diff."""
        model = model.eval().cpu()
        inputs: tuple[torch.Tensor, ...] = self.example_inputs(self.verify_batch_size)
        with torch.no_grad():
            result = model(*inputs)
        expected: tuple[torch.Tensor, ...] = (
            (result,) if isinstance(result, torch.Tensor) else tuple(result)
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed: dict[str, np.ndarray] = {
            name: x.numpy() for name, x in zip(self.input_names, inputs, strict=True)
        }
        actual = session.run(list(self.output_names), feed)
        max_diff: float = 0.0
        for name, a, e in zip(self.output_names, actual, expected, strict=True):
            diff: float = float(np.abs(np.asarray(a) - e.numpy()).max())
            if diff > self.atol:
                raise RuntimeError(
                    f"ONNX output {name} differs from PyTorch: max abs diff {diff:.2e}"
                )
            max_diff = max(max_diff, diff)
        logger.info(
            f"ONNX check passed: batch {self.verify_batch_size}, max abs diff {max_diff:.2e}"
        )

    def run(self, model: nn.Module, path: Path, verify: bool = True) -> Path:
        """Export model to path. If verify is set, also check the ONNX file."""
        self.export(model, path)
        if verify:
            self.verify(model, path)
        return path


def build_exporter(exporter_cfg: DictConfig) -> BaseExporter:
    """Build the exporter from its `_target_` config node."""
    exporter = instantiate(exporter_cfg)
    if not isinstance(exporter, BaseExporter):
        raise TypeError(
            f"config node exporter must build a BaseExporter, got {type(exporter).__name__}"
        )
    return exporter
