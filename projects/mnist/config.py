"""Typed config schema for MNIST training. Values live in projects/mnist/configs/*.yaml."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from vizard.config import BaseTrainConfig, HParams


@dataclass
class MnistHParams(HParams):
    """epochs, batch_size and lr come from HParams."""

    label_smoothing: float = 0.0  # cross-entropy label smoothing, in [0, 1)


@dataclass
class TrainConfig(BaseTrainConfig):
    """unit, data, exporter, trainer and resume come from BaseTrainConfig."""

    # The declared type stays HParams, because a subclass cannot change the type of a field.
    # The default makes MnistHParams the schema of the node.
    hparams: HParams = field(default_factory=MnistHParams)


def register() -> None:
    """Register the schema as "train_schema", so YAML configs are type-checked against it."""
    ConfigStore.instance().store(name="train_schema", node=TrainConfig)
