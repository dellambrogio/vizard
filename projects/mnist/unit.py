"""Training unit for MNIST: cross-entropy loss and the MNIST evaluation log lines."""

from typing import Unpack

import torch
from loguru import logger
from torch import nn
from torchtnt.framework.state import State

from projects.mnist.data.loaders import Batch
from projects.mnist.evaluator import NUM_DIGITS
from projects.mnist.loss.cross_entropy import LossFn, cross_entropy
from vizard.training.unit import BaseUnit, BaseUnitOptions, OptimizerFactory


class MnistUnit(BaseUnit[Batch]):
    """Unit that trains with loss_fn. The outputs for the evaluator are the logits."""

    def __init__(
        self,
        *,
        module: nn.Module,
        device: torch.device,
        optimizer: OptimizerFactory,
        loss_fn: LossFn | None = None,
        **options: Unpack[BaseUnitOptions],
    ) -> None:
        super().__init__(module=module, device=device, optimizer=optimizer, **options)
        self.loss_fn: LossFn = loss_fn or cross_entropy()

    def compute_loss(self, state: State, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        logits: torch.Tensor = self.module(x)
        return self.loss_fn(logits, y), logits

    def log_evaluation(self, label: str, loss: float, metrics: dict[str, float]) -> None:
        """Write two lines: the total accuracy, then the accuracy of each digit."""
        digits: list[str] = [f"accuracy_{d}" for d in range(NUM_DIGITS)]
        if "accuracy" not in metrics or any(name not in metrics for name in digits):
            super().log_evaluation(label, loss, metrics)  # not the metrics of MnistEvaluator
            return
        per_digit: str = "  ".join(f"{d}: {metrics[name]:.3f}" for d, name in enumerate(digits))
        logger.info(f"[mnist] {label} | train loss {loss:.4f} | accuracy {metrics['accuracy']:.4f}")
        logger.info(f"[mnist]   per digit | {per_digit}")
