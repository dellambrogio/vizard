"""ONNX exporter for the CIFAR-10 model: input "image" (batch, 3, 32, 32), output "logits"."""

import torch

from projects.cifar10.data.preprocess import INPUT_SHAPE
from vizard.export.onnx import BaseExporter

INPUT_NAME: str = "image"
OUTPUT_NAME: str = "logits"


class Cifar10Exporter(BaseExporter):
    """Exporter for a model with one normalized float32 RGB image input and one logits output."""

    input_names = (INPUT_NAME,)
    output_names = (OUTPUT_NAME,)

    def example_inputs(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (torch.randn(batch_size, *INPUT_SHAPE),)
