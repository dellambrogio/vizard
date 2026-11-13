"""Contract for train data that needs the model, for example rollouts of the current policy."""

from typing import Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class ModelBoundData(Protocol):
    """Train data that needs the model, for example rollouts of the current policy in RL.

    run_training calls bind with the model of the unit (on its device) before training.
    """

    def bind(self, module: nn.Module) -> None: ...
