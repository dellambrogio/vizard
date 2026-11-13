"""Tests for the PPO train data: GAE and the minibatches of RolloutLoader."""

import torch

from projects.cartpole.data.rollouts import PpoBatch, RolloutLoader, compute_gae
from projects.cartpole.models.actor_critic import ActorCritic


def test_compute_gae_by_hand() -> None:
    """Two steps of one environment; the episode ends at the first step."""
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.5], [0.2]])
    dones = torch.tensor([[1.0], [0.0]])
    next_values = torch.tensor([0.4])
    gamma, lam = 0.9, 0.8
    advantages, returns = compute_gae(rewards, values, dones, next_values, gamma, lam)
    # Step 1: delta = 1 + 0.9 * 0.4 - 0.2 = 1.16; nothing after it.
    # Step 0: the episode ends, so delta = 1 - 0.5 = 0.5 and step 1 is not used.
    torch.testing.assert_close(advantages, torch.tensor([[0.5], [1.16]]))
    torch.testing.assert_close(returns, advantages + values)


def test_compute_gae_no_done_is_discounted_sum() -> None:
    """With lambda 1 and zero values, the advantage is the discounted sum of the rewards."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    zeros = torch.zeros(3, 1)
    advantages, _ = compute_gae(rewards, zeros, zeros, torch.zeros(1), 0.5, 1.0)
    torch.testing.assert_close(advantages, torch.tensor([[2.75], [3.5], [3.0]]))


def test_rollout_loader_minibatches() -> None:
    loader = RolloutLoader(
        env_id="CartPole-v1",
        num_envs=2,
        steps_per_rollout=16,
        rollouts_per_epoch=2,
        ppo_epochs=3,
        batch_size=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    loader.bind(ActorCritic())
    batches: list[PpoBatch] = list(loader)
    # 32 samples per rollout: 4 minibatches (10, 10, 10, 2) per pass, 3 passes, 2 rollouts.
    assert len(batches) == 2 * 3 * 4
    assert [b.obs.shape[0] for b in batches[:4]] == [10, 10, 10, 2]
    first: PpoBatch = batches[0]
    assert first.obs.shape == (10, 4)
    assert first.actions.dtype == torch.int64
    assert set(first.actions.tolist()) <= {0, 1}
    assert bool((first.log_probs <= 0).all())
