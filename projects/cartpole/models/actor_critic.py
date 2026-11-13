"""Actor-critic MLP for CartPole: observation in, action logits (policy) and value out."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ActorCriticConfig:
    """Architecture of ActorCritic. Values: the unit.module.cfg node of an experiment config."""

    obs_dim: int = 4  # size of the observation vector of the environment
    num_actions: int = 2  # number of discrete actions of the environment
    hidden: int = 64  # units of each hidden layer
    layers: int = 2  # number of hidden layers, in the policy and in the value network


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    dim: int = in_dim
    for _ in range(layers):
        modules += [nn.Linear(dim, hidden), nn.Tanh()]
        dim = hidden
    modules.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    """Two separate MLPs: the policy gives action logits, the value network gives V(obs).

    forward gives the logits only, so the exported ONNX model is the policy.
    """

    def __init__(self, cfg: ActorCriticConfig | None = None) -> None:
        super().__init__()
        self.cfg: ActorCriticConfig = cfg or ActorCriticConfig()
        c = self.cfg
        self.policy = _mlp(c.obs_dim, c.hidden, c.layers, c.num_actions)
        self.value_net = _mlp(c.obs_dim, c.hidden, c.layers, 1)
        last = self.policy[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.mul_(0.01)  # start near the uniform policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the action logits, shape (batch, num_actions)."""
        return self.policy(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the value estimate, shape (batch,)."""
        return self.value_net(obs).squeeze(-1)


def as_actor_critic(module: nn.Module) -> ActorCritic:
    """Return module as an ActorCritic. Raise TypeError for another model."""
    if not isinstance(module, ActorCritic):
        raise TypeError(f"the model must be an ActorCritic, got {type(module).__name__}")
    return module
