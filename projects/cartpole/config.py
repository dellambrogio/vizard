"""Typed config schema for CartPole PPO training. Values: projects/cartpole/configs/*.yaml."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from vizard.config import BaseTrainConfig, HParams


@dataclass
class PpoHParams(HParams):
    """epochs, batch_size (the PPO minibatch size) and lr come from HParams.

    One epoch is rollouts_per_epoch rollouts of num_envs * steps_per_rollout environment steps.
    """

    num_envs: int = 8  # environments that run in parallel
    steps_per_rollout: int = 128  # steps of each environment in one rollout
    rollouts_per_epoch: int = 10  # rollouts in one training epoch
    ppo_epochs: int = 4  # passes over each rollout
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip: float = 0.2  # PPO ratio clip
    value_coef: float = 0.5  # weight of the value loss
    entropy_coef: float = 0.01  # weight of the entropy bonus


@dataclass
class TrainConfig(BaseTrainConfig):
    """unit, data, exporter, trainer and resume come from BaseTrainConfig."""

    # The declared type stays HParams, because a subclass cannot change the type of a field.
    # The default makes PpoHParams the schema of the node.
    hparams: HParams = field(default_factory=PpoHParams)
    env_id: str = "CartPole-v1"  # Gymnasium environment


def register() -> None:
    """Register the schema as "train_schema", so YAML configs are type-checked against it."""
    ConfigStore.instance().store(name="train_schema", node=TrainConfig)
