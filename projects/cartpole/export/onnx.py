"""ONNX exporter for the CartPole policy: input "obs" (batch, 4), output "logits" (batch, 2)."""

import torch

from vizard.export.onnx import BaseExporter

INPUT_NAME: str = "obs"
OUTPUT_NAME: str = "logits"


class CartPoleExporter(BaseExporter):
    """Exporter for the policy of an ActorCritic: observations in, action logits out.

    The value network is not in the export: forward of ActorCritic gives the logits only.
    """

    input_names = (INPUT_NAME,)
    output_names = (OUTPUT_NAME,)

    def __init__(self, obs_dim: int, verify_batch_size: int = 5, atol: float = 1e-4) -> None:
        super().__init__(verify_batch_size, atol)
        self.obs_dim: int = obs_dim

    def example_inputs(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (torch.randn(batch_size, self.obs_dim),)
