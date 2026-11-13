"""Base TorchTNT training unit. A project inherits from BaseUnit and implements compute_loss."""

import math
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal, NotRequired, TypedDict

import torch
from loguru import logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchtnt.framework.auto_unit import AutoUnit, TrainStepResults
from torchtnt.framework.state import State

from vizard.evaluation.evaluator import BaseEvaluator
from vizard.training.speed import TrainSpeed

# (named parameters of the model) -> optimizer. torch.optim optimizers take named parameters.
OptimizerFactory = Callable[[Iterator[tuple[str, nn.Parameter]]], torch.optim.Optimizer]
LrSchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
# (metrics as name -> value, number of train steps completed) -> None
MetricLogger = Callable[[dict[str, float], int], None]


def unwrap_ddp(module: nn.Module) -> nn.Module:
    """The model in a DDP wrapper, or module if it is not a DDP wrapper."""
    return module.module if isinstance(module, DistributedDataParallel) else module


class BaseUnitOptions(TypedDict):
    """The optional arguments of BaseUnit. A subclass takes them as `**options` and gives them
    to BaseUnit, so it does not repeat them: `**options: Unpack[BaseUnitOptions]`.
    """

    lr_scheduler: NotRequired[LrSchedulerFactory | None]
    lr_interval: NotRequired[Literal["epoch", "step"]]
    clip_grad_norm: NotRequired[float | None]
    grad_accum_steps: NotRequired[int]
    precision: NotRequired[Literal["bf16", "fp16"] | None]
    ema_decay: NotRequired[float | None]
    ema_warmup_steps: NotRequired[int]
    compile: NotRequired[bool]
    compile_mode: NotRequired[str | None]
    strategy: NotRequired[Literal["ddp"] | None]


class BaseUnit[TBatch](AutoUnit[TBatch]):
    """AutoUnit with one log line per evaluation: train loss and eval metrics.

    An evaluation runs at the end of each epoch and, if the evaluator has every_n_steps, also
    each N train steps. The loss in an epoch line is the mean train loss of the epoch. The loss
    in a step line is the mean train loss since the previous evaluation.

    A subclass must implement compute_loss(state, data) -> (loss, outputs). The eval metrics
    come from the evaluator (see fit_unit in vizard/training/fit.py): it gets each eval batch
    and its outputs. To change what an evaluation writes to the log, a subclass overrides
    log_evaluation.

    optimizer makes the optimizer from the named model parameters (module.named_parameters()),
    for example functools.partial(torch.optim.Adam, lr=1e-3). In a config, this is a node with
    `_target_: torch.optim.Adam` and `_partial_: true`. vizard/optim/ has optimizers that
    use the names: adamw (no weight decay on 1-D parameters, an LR for each name pattern) and
    muon_adamw.

    lr_scheduler makes the LR scheduler from the optimizer, for example
    functools.partial(warmup_cosine, warmup=1, total=10) (see vizard/optim/schedulers.py).
    None keeps the LR constant. lr_interval sets when it steps: "epoch" (after each epoch) or
    "step" (after each optimizer step). The scheduler counts in this unit.

    clip_grad_norm clips the gradient norm before each optimizer step. With None, there is no
    clip, but the norm is still calculated for train/grad_norm (TorchTNT gets an infinite clip
    value, which multiplies the gradients by 1). grad_accum_steps adds
    the gradients of N batches before each optimizer step. precision sets the mixed precision:
    "bf16", "fp16" (with a gradient scaler) or None for float32.

    ema_decay keeps an exponential moving average (EMA) of the weights in ema_model, updated
    after each optimizer step once ema_warmup_steps train steps are done. With EMA, each
    evaluation uses the EMA weights (the train weights come back after it), and
    inference_module returns the EMA model, for the export. None: no EMA.

    compile compiles the model in place with torch.compile (mode compile_mode). The state_dict
    keys do not change. The EMA model is not compiled.

    strategy="ddp" wraps the model in DistributedDataParallel (run_training sets it in a run
    with more than one process). Then unit.module is the wrapper; raw_module is the model in
    it, with the state_dict keys of the model.

    run_training gives these values from the hparams and trainer config nodes.

    Do not keep nn.Module attributes other than the model and ema_model: TorchTNT saves every
    nn.Module attribute of a unit in its checkpoints, so they change the checkpoint format.
    A resume must keep EMA on or off as it was.

    Metric loggers (see add_metric_logger) get these metrics, with the number of train steps
    completed as the step:
    - train/loss, train/lr, train/grad_norm, train/steps_per_sec and train/data_wait_frac:
      each N train steps, if N > 0. grad_norm is the norm before the clip, from the last
      optimizer step. steps_per_sec and data_wait_frac (the part of the time spent waiting for
      the next batch) are for the steps since the previous log, without evaluation time.
    - eval/<name> for each eval metric: each evaluation
    - train/loss_epoch, train/lr, train/epoch_time_sec, train/steps_per_sec_epoch,
      train/data_wait_frac_epoch and epoch: each epoch
    The log also gets one line per epoch with the train time, steps per second and data wait.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        device: torch.device,
        optimizer: OptimizerFactory,
        lr_scheduler: LrSchedulerFactory | None = None,
        lr_interval: Literal["epoch", "step"] = "epoch",
        clip_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
        precision: Literal["bf16", "fp16"] | None = None,
        ema_decay: float | None = None,
        ema_warmup_steps: int = 0,
        compile: bool = False,
        compile_mode: str | None = None,
        strategy: Literal["ddp"] | None = None,
    ) -> None:
        super().__init__(
            module=module,
            device=device,
            step_lr_interval=lr_interval,
            precision=precision,
            gradient_accumulation_steps=grad_accum_steps,
            # An infinite clip value gives the gradient norm and does not change the gradients.
            clip_grad_norm=clip_grad_norm if clip_grad_norm is not None else math.inf,
            strategy=strategy,
        )
        # AutoUnit calls configure_optimizers_and_lr_scheduler after __init__ of the subclass.
        self._make_optimizer: OptimizerFactory = optimizer
        self._make_lr_scheduler: LrSchedulerFactory | None = lr_scheduler
        # The EMA copy is made before the compile, so it is a plain module.
        self.ema_model: AveragedModel | None = None
        self._ema_warmup_steps: int = ema_warmup_steps
        self._train_weights: dict[str, torch.Tensor] | None = None  # during an EMA evaluation
        if ema_decay is not None:
            if not 0.0 < ema_decay < 1.0:
                raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
            self.ema_model = AveragedModel(
                self.raw_module, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True
            )
        if compile:
            self.raw_module.compile(mode=compile_mode)
        self._epoch_loss_sum: float = 0.0
        self._epoch_steps: int = 0
        self._recent_loss_sum: float = 0.0  # since the previous evaluation
        self._recent_steps: int = 0
        self._metric_loggers: list[MetricLogger] = []
        self._log_every_n_steps: int = 0
        self.evaluator: BaseEvaluator[TBatch] | None = None
        self._grad_norm: float | None = None  # from the last optimizer step
        self._step_end_time: float = time.perf_counter()  # end of the last train step
        self._recent_speed: TrainSpeed = TrainSpeed()  # since the previous metric log
        self._epoch_speed: TrainSpeed = TrainSpeed()

    def add_metric_logger(self, metric_logger: MetricLogger, log_every_n_steps: int = 0) -> None:
        """Send the metrics to metric_logger too. log_every_n_steps applies to all loggers."""
        self._metric_loggers.append(metric_logger)
        self._log_every_n_steps = log_every_n_steps

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        step: int = self.train_progress.num_steps_completed
        for metric_logger in self._metric_loggers:
            metric_logger(metrics, step)

    def log_evaluation(self, label: str, loss: float, metrics: dict[str, float]) -> None:
        """Write the result of one evaluation to the log. Override it for a custom format.

        label: "step <N>" for a step evaluation, "epoch <N>" for the end of an epoch.
        loss: the mean train loss (see the class docstring).
        metrics: the metrics of the evaluator, as name -> value.
        """
        parts: list[str] = [f"loss {loss:.4f}"]
        parts += [f"{name} {value:.4f}" for name, value in metrics.items()]
        logger.info(f"{label}: {', '.join(parts)}")

    def configure_optimizers_and_lr_scheduler(
        self, module: nn.Module
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
        # The names of the model, without the "module." prefix of a DDP wrapper.
        named = unwrap_ddp(module).named_parameters()
        optimizer: torch.optim.Optimizer = self._make_optimizer(named)
        if self.ema_model is not None:
            # The hook runs only after a real optimizer step: not for the batches of a
            # gradient accumulation, and not for an fp16 step that the gradient scaler skips.
            optimizer.register_step_post_hook(self._update_ema)
        make_scheduler: LrSchedulerFactory | None = self._make_lr_scheduler
        return optimizer, make_scheduler(optimizer) if make_scheduler is not None else None

    def _update_ema(self, *_: Any) -> None:  # noqa: ANN401
        # num_steps_completed does not include the current train step yet.
        if self.ema_model is not None and (
            self.train_progress.num_steps_completed >= self._ema_warmup_steps
        ):
            self.ema_model.update_parameters(self.raw_module)

    @property
    def raw_module(self) -> nn.Module:
        """The model: unit.module, or the model in it if it is a DDP wrapper."""
        return unwrap_ddp(self.module)

    def tracked_modules(self) -> dict[str, nn.Module]:
        """The modules in the checkpoints, without DDP wrappers: a checkpoint of a DDP run has
        the same keys as one without DDP, so a unit without DDP can restore it.
        """
        return {name: unwrap_ddp(m) for name, m in super().tracked_modules().items()}

    def inference_module(self) -> nn.Module:
        """The model to evaluate and export: the EMA model with EMA, else raw_module."""
        return self.ema_model.module if self.ema_model is not None else self.raw_module

    def current_lr(self) -> float:
        """The LR of the first parameter group of the optimizer."""
        assert self.optimizer is not None  # set by configure_optimizers_and_lr_scheduler
        return float(self.optimizer.param_groups[0]["lr"])

    def on_train_epoch_start(self, state: State) -> None:
        super().on_train_epoch_start(state)
        self._step_end_time = time.perf_counter()

    def on_train_step_end(
        self, state: State, data: TBatch, step: int, results: TrainStepResults
    ) -> None:
        loss: float = results.loss.item()  # waits for the GPU, so the step times are correct
        now: float = time.perf_counter()
        data_wait: float = 0.0
        if state.train_state is not None:
            waits = state.train_state.iteration_timer.recorded_durations.get("data_wait_time")
            data_wait = waits[-1] if waits else 0.0
        for speed in (self._recent_speed, self._epoch_speed):
            speed.add(now - self._step_end_time, data_wait)
        self._step_end_time = now
        if results.total_grad_norm is not None:  # None for a batch with no optimizer step
            self._grad_norm = float(results.total_grad_norm)
        self._epoch_loss_sum += loss
        self._epoch_steps += 1
        self._recent_loss_sum += loss
        self._recent_steps += 1
        n: int = self._log_every_n_steps
        if n > 0 and self.train_progress.num_steps_completed % n == 0:
            speed = self._recent_speed
            metrics: dict[str, float] = {
                "train/loss": loss,
                "train/lr": self.current_lr(),
                "train/steps_per_sec": speed.steps_per_sec,
                "train/data_wait_frac": speed.data_wait_frac,
            }
            if self._grad_norm is not None:
                metrics["train/grad_norm"] = self._grad_norm
            self._log_metrics(metrics)
            self._recent_speed = TrainSpeed()

    def on_train_epoch_end(self, state: State) -> None:
        super().on_train_epoch_end(state)
        speed = self._epoch_speed
        epoch: int = self.train_progress.num_epochs_completed
        logger.info(
            f"epoch {epoch}: {speed.seconds:.1f} s, {speed.steps_per_sec:.1f} steps/s, "
            f"data wait {speed.data_wait_frac:.0%}"
        )
        self._log_metrics(
            {
                "train/epoch_time_sec": speed.seconds,
                "train/steps_per_sec_epoch": speed.steps_per_sec,
                "train/data_wait_frac_epoch": speed.data_wait_frac,
            }
        )
        self._epoch_speed = TrainSpeed()

    def on_eval_epoch_start(self, state: State) -> None:
        super().on_eval_epoch_start(state)
        if self.ema_model is not None and self.ema_model.n_averaged.item() > 0:
            # Evaluate the EMA weights: keep the train weights, load the EMA weights.
            weights = self.raw_module.state_dict()
            self._train_weights = {name: value.clone() for name, value in weights.items()}
            self.raw_module.load_state_dict(self.ema_model.module.state_dict())
        if self.evaluator is not None:
            self.evaluator.reset()

    def on_eval_step_end(
        self,
        state: State,
        data: TBatch,
        step: int,
        loss: torch.Tensor,
        outputs: Any,  # noqa: ANN401
    ) -> None:
        if self.evaluator is not None:
            self.evaluator.update(data, outputs)

    def on_eval_epoch_end(self, state: State) -> None:
        super().on_eval_epoch_end(state)
        if self._train_weights is not None:  # put the train weights back
            self.raw_module.load_state_dict(self._train_weights)
            self._train_weights = None
        self._step_end_time = time.perf_counter()  # the evaluation time is not train time
        eval_metrics: dict[str, float] = self.evaluator.compute() if self.evaluator else {}
        metrics: dict[str, float] = {f"eval/{name}": v for name, v in eval_metrics.items()}
        # At the end of an epoch, TorchTNT has reset the number of steps in the epoch to 0.
        epoch_end: bool = self.train_progress.num_steps_completed_in_epoch == 0
        if epoch_end:
            epoch: int = self.train_progress.num_epochs_completed
            loss: float = self._epoch_loss_sum / max(self._epoch_steps, 1)
            label: str = f"epoch {epoch}"
            metrics.update(
                {"epoch": float(epoch), "train/loss_epoch": loss, "train/lr": self.current_lr()}
            )
            self._epoch_loss_sum = 0.0
            self._epoch_steps = 0
        else:
            loss = self._recent_loss_sum / max(self._recent_steps, 1)
            label = f"step {self.train_progress.num_steps_completed}"
        self._recent_loss_sum = 0.0
        self._recent_steps = 0
        self.log_evaluation(label, loss, eval_metrics)
        self._log_metrics(metrics)
