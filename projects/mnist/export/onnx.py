"""ONNX exporter for the MNIST model: input "image" (batch, 1, 28, 28), output "logits"."""

import torch

from projects.mnist.data.preprocess import INPUT_SHAPE
from vizard.export.onnx import BaseExporter

INPUT_NAME: str = "image"
OUTPUT_NAME: str = "logits"


class MnistExporter(BaseExporter):
    """Exporter for a model with one normalized float32 image input and one logits output."""

    input_names = (INPUT_NAME,)
    output_names = (OUTPUT_NAME,)

    def example_inputs(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (torch.randn(batch_size, *INPUT_SHAPE),)
