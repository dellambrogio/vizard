"""PPO train data: rollouts of the current policy, with GAE advantages, cut into minibatches."""

from collections import deque
from collections.abc import Iterator
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.distributions import Categorical

from projects.cartpole.models.actor_critic import ActorCritic, as_actor_critic


class PpoBatch(NamedTuple):
    """One PPO minibatch. B is the minibatch size."""

    obs: torch.Tensor  # (B, obs_dim) float32
    actions: torch.Tensor  # (B,) int64
    log_probs: torch.Tensor  # (B,) log probability of the action when it was collected
    advantages: torch.Tensor  # (B,) GAE advantages, not normalized
    returns: torch.Tensor  # (B,) value targets: advantages + values


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns) with generalized advantage estimation.

    rewards, values, dones: shape (T, N), for T steps of N environments. dones[t] is 1 if the
    episode ended at step t: the value of the next observation is then not used.
    next_values: shape (N,), the values of the observations after the last step.
    """
    steps: int = rewards.shape[0]
    advantages: torch.Tensor = torch.zeros_like(rewards)
    last: torch.Tensor = torch.zeros_like(next_values)
    for t in reversed(range(steps)):
        following: torch.Tensor = next_values if t == steps - 1 else values[t + 1]
        not_done: torch.Tensor = 1.0 - dones[t]
        delta: torch.Tensor = rewards[t] + gamma * following * not_done - values[t]
        last = delta + gamma * gae_lambda * not_done * last
        advantages[t] = last
    return advantages, advantages + values


class RolloutLoader:
    """Iterable train data for PPO. One iteration over it is one training epoch.

    Each epoch collects rollouts_per_epoch rollouts. A rollout is steps_per_rollout steps of
    num_envs environments with the current policy. After each rollout, the loader yields
    ppo_epochs passes over the rollout, in shuffled minibatches of batch_size samples.

    The loader needs the model: run_training calls bind (see ModelBoundData in
    vizard/data/model_bound.py). The environments keep running from one rollout to the next.
    An episode that stops at the time limit (truncated) gets the value of its last
    observation as a bootstrap in its last reward.
    """

    def __init__(
        self,
        *,
        env_id: str,
        num_envs: int,
        steps_per_rollout: int,
        rollouts_per_epoch: int,
        ppo_epochs: int,
        batch_size: int,
        gamma: float,
        gae_lambda: float,
        seed: int = 0,
    ) -> None:
        self.env_id: str = env_id
        self.num_envs: int = num_envs
        self.steps_per_rollout: int = steps_per_rollout
        self.rollouts_per_epoch: int = rollouts_per_epoch
        self.ppo_epochs: int = ppo_epochs
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.seed: int = seed
        self._envs: list[gym.Env] = [gym.make(env_id) for _ in range(num_envs)]
        self._obs: np.ndarray | None = None  # current observations, (num_envs, obs_dim)
        self._episode_returns: np.ndarray = np.zeros(num_envs)
        self._generator: torch.Generator = torch.Generator().manual_seed(seed)
        self._module: ActorCritic | None = None
        self.finished_returns: deque[float] = deque(maxlen=100)  # returns of recent episodes
        self.finished_episodes: int = 0  # all finished episodes

    def bind(self, module: nn.Module) -> None:
        """Collect the rollouts with module, an ActorCritic."""
        self._module = as_actor_critic(module)

    def __iter__(self) -> Iterator[PpoBatch]:
        episodes_before: int = self.finished_episodes
        for _ in range(self.rollouts_per_epoch):
            rollout: PpoBatch = self.collect()
            samples: int = rollout.obs.shape[0]
            for _ in range(self.ppo_epochs):
                order: torch.Tensor = torch.randperm(samples, generator=self._generator)
                for idx in order.split(self.batch_size):
                    yield PpoBatch._make(x[idx] for x in rollout)
        if self.finished_returns:
            mean: float = float(np.mean(self.finished_returns))
            episodes: int = self.finished_episodes - episodes_before
            logger.info(
                f"[cartpole] train episodes: {episodes} in this epoch, "
                f"mean return of the last {len(self.finished_returns)}: {mean:.1f}"
            )

    def collect(self) -> PpoBatch:
        """Collect one rollout with the current policy. Return all of it as one flat batch."""
        if self._module is None:
            raise RuntimeError("RolloutLoader has no model: call bind first")
        module: ActorCritic = self._module
        device: torch.device = next(module.parameters()).device
        if self._obs is None:
            self._obs = np.stack(
                [env.reset(seed=self.seed + i)[0] for i, env in enumerate(self._envs)]
            )
        steps, envs = self.steps_per_rollout, self.num_envs
        obs_buf: torch.Tensor = torch.zeros(steps, envs, self._obs.shape[1])
        actions: torch.Tensor = torch.zeros(steps, envs, dtype=torch.int64)
        log_probs: torch.Tensor = torch.zeros(steps, envs)
        values: torch.Tensor = torch.zeros(steps, envs)
        rewards: torch.Tensor = torch.zeros(steps, envs)
        dones: torch.Tensor = torch.zeros(steps, envs)
        with torch.no_grad():
            for t in range(steps):
                obs: torch.Tensor = torch.as_tensor(self._obs, dtype=torch.float32)
                obs_device: torch.Tensor = obs.to(device)
                dist = Categorical(logits=module(obs_device))
                action: torch.Tensor = dist.sample()
                obs_buf[t] = obs
                actions[t] = action.cpu()
                log_probs[t] = dist.log_prob(action).cpu()
                values[t] = module.value(obs_device).cpu()
                self._obs = self._step_envs(t, actions[t], rewards, dones)
            last_obs = torch.as_tensor(self._obs, dtype=torch.float32).to(device)
            next_values: torch.Tensor = module.value(last_obs).cpu()
        advantages, returns = compute_gae(
            rewards, values, dones, next_values, self.gamma, self.gae_lambda
        )
        return PpoBatch(
            obs=obs_buf.flatten(0, 1),
            actions=actions.flatten(),
            log_probs=log_probs.flatten(),
            advantages=advantages.flatten(),
            returns=returns.flatten(),
        )

    def _step_envs(
        self, t: int, action: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> np.ndarray:
        """Step each environment, write rewards[t] and dones[t], return the next observations.

        A finished episode is reset. A truncated episode gets the bootstrap value.
        """
        assert self._module is not None
        next_obs: list[np.ndarray] = []
        truncated: list[tuple[int, np.ndarray]] = []  # (env index, last observation)
        for i, env in enumerate(self._envs):
            obs, reward, terminated, trunc, _ = env.step(int(action[i]))
            self._episode_returns[i] += float(reward)
            rewards[t, i] = float(reward)
            if terminated or trunc:
                dones[t, i] = 1.0
                self.finished_returns.append(float(self._episode_returns[i]))
                self.finished_episodes += 1
                self._episode_returns[i] = 0.0
                if trunc and not terminated:
                    truncated.append((i, obs))
                obs, _ = env.reset()
            next_obs.append(obs)
        if truncated:
            device: torch.device = next(self._module.parameters()).device
            last = torch.as_tensor(np.stack([o for _, o in truncated]), dtype=torch.float32)
            bootstrap: torch.Tensor = self._module.value(last.to(device)).cpu()
            for (i, _), v in zip(truncated, bootstrap, strict=True):
                rewards[t, i] += self.gamma * v
        return np.stack(next_obs)
