"""Training with TorchTNT fit: evaluation, checkpoints and resume for a BaseUnit."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger
from torchtnt.framework.callback import Callback
from torchtnt.framework.callbacks import DistributedCheckpointSaver
from torchtnt.framework.fit import fit

from vizard.evaluation.evaluator import BaseEvaluator
from vizard.training.unit import BaseUnit


def fit_unit(
    unit: BaseUnit[Any],
    train_loader: Iterable[Any],
    evaluator: BaseEvaluator[Any],
    epochs: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    keep_last_n: int = 3,
    checkpoint_every_n_steps: int | None = None,
) -> None:
    """Train unit with TorchTNT and evaluate with evaluator after each train epoch.

    If evaluator.every_n_steps is set, also evaluate each N train steps.

    If checkpoint_dir is set, save a checkpoint after each train epoch and, if
    checkpoint_every_n_steps is set, each N train steps. If resume is also set, restore the
    latest checkpoint first and continue until `epochs` total epochs. A checkpoint does not
    keep the position in the train data: after a resume from the middle of an epoch, the data
    of that epoch starts again from the start.
    The trained model is unit.module.
    """
    unit.evaluator = evaluator
    callbacks: list[Callback] = []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            restore = DistributedCheckpointSaver.restore_from_latest
            restored: bool = restore(str(checkpoint_dir), unit)
            done: int = unit.train_progress.num_epochs_completed
            logger.info(f"resumed after epoch {done}" if restored else "no checkpoint found")
        callbacks.append(
            DistributedCheckpointSaver(
                str(checkpoint_dir),
                save_every_n_epochs=1,
                save_every_n_train_steps=checkpoint_every_n_steps,
                keep_last_n_checkpoints=keep_last_n,
            )
        )
    fit(
        unit,
        train_loader,
        evaluator.loader,
        max_epochs=epochs,
        evaluate_every_n_steps=evaluator.every_n_steps,
        callbacks=callbacks,
    )


def restore_latest(unit: BaseUnit[Any], checkpoint_dir: Path) -> None:
    """Restore unit (model weights and progress) from the latest checkpoint in checkpoint_dir."""
    restore = DistributedCheckpointSaver.restore_from_latest
    if not restore(str(checkpoint_dir), unit):
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    done: int = unit.train_progress.num_epochs_completed
    logger.info(f"restored checkpoint after epoch {done} from {checkpoint_dir}")
