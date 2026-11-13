"""Evaluator for CartPole: returns of full episodes with the greedy policy."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from projects.cartpole.data.episodes import play_episodes
from projects.cartpole.models.actor_critic import as_actor_critic
from vizard.evaluation.evaluator import BaseEvaluator


class CartPoleEvaluator(BaseEvaluator[torch.Tensor]):
    """Play one episode per seed of loader and report the episode returns.

    Metrics:
    - mean_return, min_return: over all episodes.
    - success_rate: the fraction of episodes with a return of at least solved_return (500 for
      CartPole-v1: the pole stays up until the time limit).
    """

    def __init__(
        self,
        loader: DataLoader,
        env_id: str,
        solved_return: float,
        every_n_steps: int | None = None,
    ) -> None:
        super().__init__(loader, every_n_steps)
        self.env_id: str = env_id
        self.solved_return: float = solved_return
        self._returns: list[torch.Tensor] = []

    def predict(self, module: nn.Module, data: torch.Tensor) -> torch.Tensor:
        return play_episodes(as_actor_critic(module), self.env_id, data.tolist())

    def reset(self) -> None:
        self._returns = []

    def update(self, data: torch.Tensor, outputs: torch.Tensor) -> None:
        self._returns.append(outputs.cpu())

    def compute(self) -> dict[str, float]:
        if not self._returns:
            return {}
        returns: torch.Tensor = torch.cat(self._returns)
        return {
            "mean_return": returns.mean().item(),
            "min_return": returns.min().item(),
            "success_rate": (returns >= self.solved_return).float().mean().item(),
        }
