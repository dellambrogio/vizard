"""Training unit for CIFAR-10: a classification loss and the CIFAR-10 evaluation log lines."""

from collections.abc import Callable
from typing import Unpack

import torch
from loguru import logger
from torch import nn
from torchtnt.framework.state import State

from projects.cifar10.data.io import CLASSES
from projects.cifar10.data.loaders import Batch
from vizard.training.unit import BaseUnit, BaseUnitOptions, OptimizerFactory

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (logits, labels) -> scalar


class Cifar10Unit(BaseUnit[Batch]):
    """Unit that trains with loss_fn. The outputs for the evaluator are the logits.

    loss_fn is a plain function, not an nn.Module (see BaseUnit). In a config it is
    torch.nn.functional.cross_entropy with `_partial_: true`.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        device: torch.device,
        optimizer: OptimizerFactory,
        loss_fn: LossFn = nn.functional.cross_entropy,
        **options: Unpack[BaseUnitOptions],
    ) -> None:
        super().__init__(module=module, device=device, optimizer=optimizer, **options)
        self.loss_fn: LossFn = loss_fn

    def compute_loss(self, state: State, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        logits: torch.Tensor = self.module(x)
        return self.loss_fn(logits, y), logits

    def log_evaluation(self, label: str, loss: float, metrics: dict[str, float]) -> None:
        """Write the total accuracy, then the classes from the worst to the best accuracy."""
        names: list[str] = [f"accuracy_{name}" for name in CLASSES]
        if "accuracy" not in metrics or any(name not in metrics for name in names):
            super().log_evaluation(label, loss, metrics)  # not the metrics of Cifar10Evaluator
            return
        ranked: list[str] = sorted(CLASSES, key=lambda name: metrics[f"accuracy_{name}"])
        per_class: str = "  ".join(f"{name}: {metrics[f'accuracy_{name}']:.3f}" for name in ranked)
        accuracy: float = metrics["accuracy"]
        logger.info(f"[cifar10] {label} | train loss {loss:.4f} | accuracy {accuracy:.4f}")
        logger.info(f"[cifar10]   worst to best | {per_class}")
