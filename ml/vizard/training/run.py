"""Training flow common to all projects: unit and data from config, job, checkpoints, fit."""

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchtnt.utils.env import seed as seed_everything

from vizard.config import BaseTrainConfig
from vizard.data.model_bound import ModelBoundData
from vizard.data.shardable import ShardableData
from vizard.evaluation.evaluator import BaseEvaluator, build_evaluator
from vizard.export.onnx import BaseExporter, build_exporter
from vizard.jobs.git import save_git_info
from vizard.jobs.job import JOBS_ROOT, Job, create_job, load_job
from vizard.tracking.wandb import start_wandb
from vizard.training.distributed import (
    broadcast_from_main,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main,
    launched_distributed,
)
from vizard.training.fit import fit_unit, restore_latest
from vizard.training.summary import save_model_summary
from vizard.training.unit import BaseUnit, BaseUnitOptions

CONFIG_FILE: str = "config.yaml"
RESUME_CONFIG_FILE: str = "config.resume.yaml"
CHECKPOINT_DIR: str = "checkpoints"
MODEL_SUMMARY_FILE: str = "model.txt"
EXPORT_FILE: Path = Path("export") / "model.onnx"  # in the job folder


def unit_options(cfg: BaseTrainConfig) -> BaseUnitOptions:
    """The BaseUnit arguments that come from the hparams and trainer nodes, not from unit."""
    precision = cfg.trainer.precision
    return BaseUnitOptions(
        lr_interval=cfg.hparams.lr_interval.value,
        clip_grad_norm=cfg.hparams.clip_grad_norm,
        grad_accum_steps=cfg.hparams.grad_accum_steps,
        precision=precision.value if precision is not None else None,
        ema_decay=cfg.hparams.ema_decay,
        ema_warmup_steps=cfg.hparams.ema_warmup_steps,
        compile=cfg.trainer.compile,
        compile_mode=cfg.trainer.compile_mode,
    )


def set_seed(seed: int, deterministic: bool) -> None:
    """Seed torch, numpy and random. If deterministic is set, error on PyTorch operations that
    are not deterministic and disable the cuDNN benchmark.
    """
    if deterministic:
        # Deterministic cuBLAS needs a workspace config, set before the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed, deterministic="error" if deterministic else None)


def make_shared_job(make_job: Callable[[], Job], root: Path = JOBS_ROOT) -> Job:
    """Rank 0 makes the job with make_job; the other ranks load the same job by its id.

    Without DDP, return make_job(). The ranks must share the jobs folder (one machine).
    """
    job: Job | None = make_job() if is_main() else None
    job_id: str = broadcast_from_main(job.id if job is not None else None)
    return job if job is not None else load_job(job_id, root)


def quiet_log() -> None:
    """Write warnings and errors only to the console (for the ranks other than rank 0)."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def build_unit(
    unit_cfg: DictConfig, device: torch.device, options: BaseUnitOptions | None = None
) -> BaseUnit[Any]:
    """Build the unit from its `_target_` config node, on device, with the options."""
    unit = instantiate(unit_cfg, device=device, **(options or {}))
    if not isinstance(unit, BaseUnit):
        raise TypeError(f"config node unit must build a BaseUnit, got {type(unit).__name__}")
    return unit


def run_training(
    dict_cfg: DictConfig, cfg: BaseTrainConfig, job_name: str
) -> tuple[Job, BaseUnit[Any]]:
    """Train in a new job, or continue the job cfg.resume. Return the job and the trained unit.

    dict_cfg is the Hydra config and cfg is the same config as a typed object. The unit comes
    from dict_cfg.unit, the evaluator from dict_cfg.evaluator and the train data from
    dict_cfg.data.train: a DataLoader or another iterable of batches. If the train data is a
    ModelBoundData, it gets the model of the unit.

    With cfg.init_from, the model starts from the weights of that job (see init_from_job).

    DDP: started with torchrun and more than one process, each rank trains on its own device
    and its own part of the train data (ShardableData), and the gradients are averaged. Rank 0
    makes the job and writes the job files, W&B, and the export; the other ranks
    log warnings only. Each rank evaluates the full eval data, and the metrics are of rank 0.
    hparams.batch_size is the batch of one rank.

    First the seed is set (cfg.trainer.seed and cfg.trainer.deterministic). The unit also gets
    the options of unit_options: the LR interval, gradient clipping and accumulation from
    cfg.hparams, and the precision from cfg.trainer.

    The unit, the evaluator and the train data are built before the job, so an invalid
    config makes no job. The job folder gets job.log, config.yaml (config.resume.yaml for a
    resumed run; interpolations are resolved), git.json and git.diff (the git state of the
    code, see save_git_info; git.resume.* for a resumed run), model.txt (the model summary)
    and, if cfg.trainer.checkpoint is set,
    checkpoints/ with one checkpoint per epoch and, if cfg.trainer.checkpoint_every_n_steps
    is set, one each N train steps. With cfg.trainer.export_onnx, the trained model (the EMA
    model with EMA) is exported with the exporter node to export/model.onnx.

    If cfg.wandb.enabled is set, the run also goes to Weights & Biases (see start_wandb),
    in the group cfg.experiment.
    """
    if cfg.wandb.enabled and cfg.wandb.project is None:
        raise ValueError("wandb.project must be set when wandb.enabled is true")
    if cfg.resume and cfg.init_from:
        raise ValueError("set resume or init_from, not both")
    exporter: BaseExporter | None = None
    if cfg.trainer.export_onnx:
        if dict_cfg.get("exporter") is None:
            raise ValueError("trainer.export_onnx needs the exporter node")
        exporter = build_exporter(dict_cfg.exporter)
    if launched_distributed():
        device: torch.device = init_distributed()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, world_size = get_rank(), get_world_size()
    main: bool = is_main()
    if not main:
        quiet_log()
    # Each rank gets its own random state (augmentation); DDP gives all ranks the model of
    # rank 0, and the train data shards use cfg.trainer.seed.
    set_seed(cfg.trainer.seed + rank, cfg.trainer.deterministic)
    options: BaseUnitOptions = unit_options(cfg)
    if world_size > 1:
        options["strategy"] = "ddp"
    unit: BaseUnit[Any] = build_unit(dict_cfg.unit, device, options)
    if cfg.init_from:
        init_from_job(unit, load_job(cfg.init_from))
    evaluator: BaseEvaluator[Any] = build_evaluator(dict_cfg.evaluator)
    train_loader = instantiate(dict_cfg.data.train)
    if world_size > 1:
        if isinstance(train_loader, ShardableData):
            train_loader.shard(rank, world_size, cfg.trainer.seed)
        elif not isinstance(train_loader, ModelBoundData):
            logger.warning(
                f"the train data ({type(train_loader).__name__}) is not a ShardableData: "
                "each rank must get its own part (for a DataLoader: a DistributedSampler)"
            )
    if isinstance(train_loader, ModelBoundData):
        train_loader.bind(unit.raw_module)

    def make_job() -> Job:
        return load_job(cfg.resume) if cfg.resume else create_job(job_name)

    job: Job = make_shared_job(make_job)
    if main:
        job.log_to_file()
        config_file: str = RESUME_CONFIG_FILE if cfg.resume else CONFIG_FILE
        OmegaConf.save(dict_cfg, job.dir / config_file, resolve=True)
        logger.info(f"config:\n{OmegaConf.to_yaml(dict_cfg, resolve=True)}")
        if world_size > 1:
            logger.info(f"DDP: {world_size} processes, device of rank 0: {device}")
        if cfg.init_from:
            logger.info(f"model weights from job {cfg.init_from}")
        save_git_info(job.dir, "git.resume" if cfg.resume else "git")
        save_model_summary(unit.raw_module, job.dir / MODEL_SUMMARY_FILE)

    run = None
    if cfg.wandb.enabled and main:
        resolved = OmegaConf.to_container(dict_cfg, resolve=True)
        assert isinstance(resolved, dict)
        run = start_wandb(
            cfg.wandb, job, {str(k): v for k, v in resolved.items()}, group=cfg.experiment
        )
        unit.add_metric_logger(run.log, cfg.wandb.log_every_n_steps)
    exit_code: int = 1
    try:
        fit_unit(
            unit,
            train_loader,
            evaluator,
            cfg.hparams.epochs,
            checkpoint_dir=job.dir / CHECKPOINT_DIR if cfg.trainer.checkpoint else None,
            resume=cfg.resume is not None,
            keep_last_n=cfg.trainer.keep_last_n,
            checkpoint_every_n_steps=cfg.trainer.checkpoint_every_n_steps,
        )
        exit_code = 0
        if exporter is not None and main:
            exporter.run(unit.inference_module(), job.dir / EXPORT_FILE)
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
    if is_distributed():
        dist.destroy_process_group()
    return job, unit


def load_init_weights(unit: BaseUnit[Any], source: BaseUnit[Any]) -> None:
    """Load the weights of source.inference_module() (its EMA model with EMA) into the model of
    unit (unit.raw_module).

    The load is strict: the two models must have the same parameter names and shapes.
    """
    weights = source.inference_module().state_dict()
    try:
        unit.raw_module.load_state_dict(weights, strict=True)
    except RuntimeError as e:
        raise ValueError(f"the weights do not fit the model: {e}") from e


def init_from_job(unit: BaseUnit[Any], job: Job) -> None:
    """Load the weights of the latest checkpoint of job into unit.module (see load_init_weights).

    The unit of the job is built from its config.yaml (see restore_unit). The optimizer,
    LR scheduler, progress and EMA of unit do not change.
    """
    load_init_weights(unit, restore_unit(job))


def job_config_node(job: Job, key: str, default_config: Path | None = None) -> DictConfig:
    """Return the node `key` of the config.yaml of a job.

    If the job config has no such node (an old job, from before this node was in the config),
    the node of default_config is used: a config with all values in one file, for example the
    config.yaml of another job. With no default_config, this raises KeyError.
    """
    job_cfg = OmegaConf.load(job.dir / CONFIG_FILE)
    if not isinstance(job_cfg, DictConfig) or job_cfg.get(key) is None:
        if default_config is None:
            raise KeyError(
                f"the config of job {job.id} has no {key} node; give a config that has it"
            )
        logger.info(f"job {job.id} has no {key} config, using {default_config}")
        job_cfg = OmegaConf.load(default_config)
    node = job_cfg.get(key) if isinstance(job_cfg, DictConfig) else None
    if not isinstance(node, DictConfig):
        raise TypeError(f"config node {key} must be a mapping")
    return node


def restore_unit(job: Job, default_config: Path | None = None) -> BaseUnit[Any]:
    """Build the unit of a job from its config.yaml, on CPU, and restore its latest checkpoint.

    If the job has EMA (hparams.ema_decay), the unit gets its EMA model too: see
    BaseUnit.inference_module. The model is not compiled. For a job config with no unit node,
    see job_config_node.
    """
    unit_cfg: DictConfig = job_config_node(job, "unit", default_config)
    job_cfg = OmegaConf.load(job.dir / CONFIG_FILE)
    ema_decay = OmegaConf.select(job_cfg, "hparams.ema_decay", default=None)
    options = BaseUnitOptions(ema_decay=ema_decay)
    unit: BaseUnit[Any] = build_unit(unit_cfg, torch.device("cpu"), options)
    restore_latest(unit, job.dir / CHECKPOINT_DIR)
    return unit
