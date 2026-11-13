"""Typed config schema for training, common to all projects.

A project config inherits from BaseTrainConfig and adds its own fields.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING, OmegaConf


def per_rank(batch_size: int) -> int:
    """The batch of one DDP rank: batch_size (the full batch) / the number of processes.

    The number of processes is WORLD_SIZE, set by torchrun; 1 without torchrun. Registered as
    the OmegaConf resolver "per_rank": `batch_size: ${per_rank:${hparams.batch_size}}`.
    """
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))
    batch_size = int(batch_size)
    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size {batch_size} cannot be split in {world_size} DDP processes: "
            "use a batch_size that the number of processes divides"
        )
    return batch_size // world_size


OmegaConf.register_new_resolver("per_rank", per_rank, replace=True)


class LrInterval(Enum):
    epoch = "epoch"  # the LR scheduler steps after each epoch
    step = "step"  # the LR scheduler steps after each optimizer step


class Precision(Enum):
    bf16 = "bf16"  # bfloat16 autocast
    fp16 = "fp16"  # float16 autocast, with a gradient scaler


@dataclass
class HParams:
    """Training hyperparameters, in one place.

    A project inherits from HParams to add its own hyperparameters. The `_target_` nodes read
    the values with interpolation, for example `lr: ${hparams.lr}`.
    """

    epochs: int = 3
    # The full batch of one optimizer step, for any number of DDP processes. The train data
    # gets the batch of one process with ${per_rank:${hparams.batch_size}}.
    batch_size: int = 128
    lr: float = 1e-3
    # When the LR scheduler steps: "epoch" (after each epoch) or "step" (after each optimizer
    # step). The scheduler counts in this unit, for example its total length.
    lr_interval: LrInterval = LrInterval.epoch
    clip_grad_norm: float | None = None  # clip the gradient norm to this value; None: no clip
    # Add the gradients of N batches before each optimizer step. The effective batch size is
    # N * batch_size.
    grad_accum_steps: int = 1
    # Keep an exponential moving average (EMA) of the weights, with this decay (for example
    # 0.999); None: no EMA. With EMA, the evaluations and the export use the EMA weights.
    ema_decay: float | None = None
    ema_warmup_steps: int = 0  # train steps before the first EMA update


@dataclass
class TrainerConfig:
    """Settings of the training run that are not hyperparameters."""

    checkpoint: bool = True  # save a checkpoint after each epoch
    # Also save a checkpoint each N train steps; None: after each epoch only. A resume from a
    # checkpoint in the middle of an epoch starts the data of that epoch again from the start.
    checkpoint_every_n_steps: int | None = None
    keep_last_n: int = 3
    precision: Precision | None = None  # mixed precision (autocast); None: float32
    seed: int = 0  # seed of torch, numpy and random, set before the unit and the data are built
    # Error on PyTorch operations that are not deterministic, and disable the cuDNN benchmark.
    # Slower. False keeps the PyTorch defaults.
    deterministic: bool = False
    compile: bool = False  # torch.compile the model for training
    # torch.compile mode: None (default), "reduce-overhead" or "max-autotune"
    compile_mode: str | None = None
    # After training, export the model (EMA model with EMA) with the exporter node to
    # <job dir>/export/model.onnx, and check the export.
    export_onnx: bool = False


@dataclass
class DataConfig:
    """The train data: a `_target_` node that builds the train DataLoader (or another iterable
    of batches; see ModelBoundData in vizard/data/model_bound.py for data that needs the model).

    The evaluation data is in the evaluator node.
    """

    train: Any = MISSING


class WandbMode(Enum):
    online = "online"  # send the run to the W&B server
    offline = "offline"  # write the run to <job dir>/wandb/ only


@dataclass
class WandbConfig:
    """Weights & Biases tracking. Each project sets its own project name."""

    enabled: bool = False
    project: str | None = None  # necessary when enabled
    entity: str | None = None  # W&B team; None is the default entity of the account
    mode: WandbMode = WandbMode.online
    log_every_n_steps: int = 50  # log the train loss each N steps; 0 logs per epoch only


@dataclass
class BaseTrainConfig:
    """unit, data.train, evaluator and exporter are `_target_` nodes, built with Hydra instantiate.

    They have no schema, so each value that is a hyperparameter comes from hparams.
    unit: a BaseUnit subclass. The training code adds the `device` argument.
    data.train: the train DataLoader.
    evaluator: a BaseEvaluator subclass, with its data (loader) and interval (every_n_steps).
    exporter: a BaseExporter subclass. Optional; necessary for trainer.export_onnx.
    resume: id of a job to continue (same job, same model, same optimizer state).
    init_from: id of a local job to start from: a new job gets the weights of its latest
    checkpoint (its EMA weights, if it has EMA). The optimizer, the LR scheduler, the progress
    and the EMA start new. Not together with resume.
    """

    experiment: str = MISSING  # name of the experiment, for example "baseline_1.0"
    hparams: HParams = field(default_factory=HParams)
    unit: Any = MISSING
    data: DataConfig = field(default_factory=DataConfig)
    evaluator: Any = MISSING
    exporter: Any = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    resume: str | None = None
    init_from: str | None = None
