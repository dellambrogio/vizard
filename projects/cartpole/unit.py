"""Training unit for CartPole: the PPO loss and the CartPole evaluation log lines."""

from typing import Unpack

import torch
from loguru import logger
from torch import nn
from torchtnt.framework.state import State

from projects.cartpole.data.rollouts import PpoBatch
from projects.cartpole.loss.ppo import ppo_loss
from projects.cartpole.models.actor_critic import as_actor_critic
from vizard.training.unit import BaseUnit, BaseUnitOptions, OptimizerFactory


class CartPoleUnit(BaseUnit[PpoBatch | torch.Tensor]):
    """Unit that trains an ActorCritic with PPO.

    Train data: PpoBatch minibatches from RolloutLoader.
    Eval data: tensors of episode seeds. An eval step plays the episodes with the predict of
    the evaluator; the outputs are the episode returns and the loss is 0.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        device: torch.device,
        optimizer: OptimizerFactory,
        clip: float,
        value_coef: float,
        entropy_coef: float,
        **options: Unpack[BaseUnitOptions],
    ) -> None:
        super().__init__(module=module, device=device, optimizer=optimizer, **options)
        self.clip: float = clip
        self.value_coef: float = value_coef
        self.entropy_coef: float = entropy_coef

    def compute_loss(
        self, state: State, data: PpoBatch | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(data, PpoBatch):  # eval step: data is a batch of episode seeds
            if self.evaluator is None:
                raise RuntimeError("an eval step needs the evaluator of the unit")
            returns: torch.Tensor = self.evaluator.predict(self.module, data)
            return torch.zeros((), device=self.device), returns
        model = as_actor_critic(self.module)
        loss, parts = ppo_loss(model, data, self.clip, self.value_coef, self.entropy_coef)
        return loss, torch.stack(list(parts.values()))

    def log_evaluation(self, label: str, loss: float, metrics: dict[str, float]) -> None:
        """Write one line: the train loss and the episode returns."""
        names: tuple[str, ...] = ("mean_return", "min_return", "success_rate")
        if any(name not in metrics for name in names):
            super().log_evaluation(label, loss, metrics)  # not the metrics of CartPoleEvaluator
            return
        logger.info(
            f"[cartpole] {label} | train loss {loss:.4f} | "
            f"return mean {metrics['mean_return']:.1f} min {metrics['min_return']:.1f} | "
            f"success {metrics['success_rate']:.2f}"
        )
