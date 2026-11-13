"""Typed config schema for CIFAR-10 training. Values live in projects/cifar10/configs/*.yaml."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from vizard.config import BaseTrainConfig, HParams


@dataclass
class Cifar10HParams(HParams):
    """epochs, batch_size and lr come from HParams."""

    weight_decay: float = 5e-4  # AdamW weight decay
    label_smoothing: float = 0.0  # cross-entropy label smoothing, in [0, 1)
    augment: bool = True  # random crop and horizontal flip on the train images


@dataclass
class TrainConfig(BaseTrainConfig):
    """unit, data, exporter, trainer and resume come from BaseTrainConfig."""

    # The declared type stays HParams, because a subclass cannot change the type of a field.
    # The default makes Cifar10HParams the schema of the node.
    hparams: HParams = field(default_factory=Cifar10HParams)


def register() -> None:
    """Register the schema as "train_schema", so YAML configs are type-checked against it."""
    ConfigStore.instance().store(name="train_schema", node=TrainConfig)
