"""Evaluation data: full episodes with the greedy policy, one per seed."""

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from projects.cartpole.models.actor_critic import ActorCritic


class SeedDataset(Dataset[int]):
    """The episode seeds first_seed .. first_seed + episodes - 1."""

    def __init__(self, first_seed: int, episodes: int) -> None:
        self.first_seed: int = first_seed
        self.episodes: int = episodes

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.episodes:
            raise IndexError(index)
        return self.first_seed + index


def get_seed_loader(first_seed: int, episodes: int, batch_size: int) -> DataLoader:
    """Return a DataLoader of episode seeds first_seed .. first_seed + episodes - 1.

    Each batch is a tensor of seeds. The evaluator plays one episode for each seed.
    """
    return DataLoader(SeedDataset(first_seed, episodes), batch_size=batch_size, shuffle=False)


def play_episodes(module: ActorCritic, env_id: str, seeds: list[int]) -> torch.Tensor:
    """Play one episode per seed with the greedy policy (argmax of the logits).

    Return the episode returns, shape (len(seeds),). The episodes run in one batch.
    """
    device: torch.device = next(module.parameters()).device
    envs: list[gym.Env] = [gym.make(env_id) for _ in seeds]
    obs: list[np.ndarray] = [env.reset(seed=seed)[0] for env, seed in zip(envs, seeds, strict=True)]
    returns: np.ndarray = np.zeros(len(envs))
    active: list[int] = list(range(len(envs)))
    with torch.no_grad():
        while active:
            batch = torch.as_tensor(np.stack([obs[i] for i in active]), dtype=torch.float32)
            actions: list[int] = module(batch.to(device)).argmax(-1).cpu().tolist()
            still_active: list[int] = []
            for i, action in zip(active, actions, strict=True):
                obs[i], reward, terminated, truncated, _ = envs[i].step(action)
                returns[i] += float(reward)
                if not (terminated or truncated):
                    still_active.append(i)
            active = still_active
    for env in envs:
        env.close()
    return torch.as_tensor(returns, dtype=torch.float32)
