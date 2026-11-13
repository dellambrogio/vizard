"""PPO loss: clipped policy loss, value loss and entropy bonus."""

import torch
from torch.distributions import Categorical

from projects.cartpole.data.rollouts import PpoBatch
from projects.cartpole.models.actor_critic import ActorCritic


def ppo_loss(
    model: ActorCritic,
    batch: PpoBatch,
    clip: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the PPO loss of one minibatch and its parts (detached).

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.
    The advantages are normalized in the minibatch. clip is the PPO ratio clip, for example 0.2.
    """
    dist = Categorical(logits=model(batch.obs))
    log_probs: torch.Tensor = dist.log_prob(batch.actions)
    ratio: torch.Tensor = (log_probs - batch.log_probs).exp()
    adv: torch.Tensor = batch.advantages
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    unclipped: torch.Tensor = ratio * adv
    clipped: torch.Tensor = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
    policy_loss: torch.Tensor = -torch.min(unclipped, clipped).mean()
    value_loss: torch.Tensor = 0.5 * (model.value(batch.obs) - batch.returns).pow(2).mean()
    entropy: torch.Tensor = dist.entropy().mean()
    loss: torch.Tensor = policy_loss + value_coef * value_loss - entropy_coef * entropy
    parts: dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
    }
    return loss, parts
