"""Weights & Biases tracking: one W&B run per job."""

from typing import Any

import wandb
from loguru import logger

from vizard.config import WandbConfig
from vizard.jobs.job import Job


def start_wandb(
    cfg: WandbConfig, job: Job, config: dict[str, Any], group: str | None = None
) -> wandb.Run:
    """Start the W&B run of a job. config is the resolved job config.

    group is the W&B group of the run, for example the experiment name.

    The run id and name are the job id, so a resumed job continues the same W&B run. The W&B
    files go to <job dir>/wandb/.
    """
    if cfg.project is None:
        raise ValueError("wandb.project must be set when wandb.enabled is true")
    run: wandb.Run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        id=job.id,
        name=job.id,
        group=group,
        resume="allow",
        mode=cfg.mode.value,
        dir=str(job.dir),
        config=config,
    )
    logger.info(f"W&B run {run.id} ({cfg.mode.value}): project {cfg.project}")
    return run
