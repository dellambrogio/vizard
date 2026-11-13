"""Base evaluator. A project inherits from BaseEvaluator and implements the metrics.

The evaluator holds all of the evaluation: the data, the interval and the metrics. The same
evaluator runs during training (the unit calls reset, update and compute) and on a trained
model (evaluate).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchtnt.utils.device import copy_data_to_device


class BaseEvaluator[TBatch](ABC):
    """Evaluate a model on loader, in full, and return metrics as name -> value.

    loader: the evaluation data: a DataLoader or another iterable of batches.
    every_n_steps: during training, also evaluate each N train steps. None: evaluate at the
        end of each epoch only.

    A subclass implements predict, reset, update and compute.
    """

    def __init__(self, loader: Iterable[Any], every_n_steps: int | None = None) -> None:
        if every_n_steps is not None and every_n_steps <= 0:
            raise ValueError(f"every_n_steps must be > 0 or None, got {every_n_steps}")
        self.loader: Iterable[Any] = loader
        self.every_n_steps: int | None = every_n_steps

    @abstractmethod
    def predict(self, module: nn.Module, data: TBatch) -> Any:  # noqa: ANN401
        """Run module on one batch and return its outputs. Used by evaluate only."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the metric state. Called at the start of each evaluation."""

    @abstractmethod
    def update(self, data: TBatch, outputs: Any) -> None:  # noqa: ANN401
        """Add one batch to the metric state.

        outputs comes from the compute_loss of the unit during training, and from predict in
        evaluate. The two must have the same form.
        """

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return the metrics of the evaluation as name -> value."""

    def evaluate(self, module: nn.Module, device: torch.device) -> dict[str, float]:
        """Evaluate module on the full loader, outside training. module stays in eval mode."""
        module = module.to(device).eval()
        self.reset()
        with torch.no_grad():
            for batch in self.loader:
                data: TBatch = copy_data_to_device(batch, device)
                self.update(data, self.predict(module, data))
        return self.compute()


def build_evaluator(evaluator_cfg: DictConfig) -> BaseEvaluator[Any]:
    """Build the evaluator from its `_target_` config node."""
    evaluator = instantiate(evaluator_cfg)
    if not isinstance(evaluator, BaseEvaluator):
        raise TypeError(
            f"config node evaluator must build a BaseEvaluator, got {type(evaluator).__name__}"
        )
    return evaluator
