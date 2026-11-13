"""Worker of the DDP tests (test_ddp.py), started with torchrun on CPU.

Usage: python -m torch.distributed.run --nproc-per-node 2 ddp_worker.py <task> <out dir>
task "train": train with DDP, write rank<N>.pt; task "job": make the shared job, write
rank<N>.txt with its id.
"""

import functools
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchtnt.framework.state import State

from vizard.data.tensor_loader import TensorBatchLoader
from vizard.evaluation.evaluator import BaseEvaluator
from vizard.jobs.job import create_job
from vizard.training.distributed import get_rank, get_world_size, init_distributed
from vizard.training.fit import fit_unit
from vizard.training.run import make_shared_job
from vizard.training.unit import BaseUnit, BaseUnitOptions

NUM_SAMPLES: int = 32
BATCH_SIZE: int = 4
EPOCHS: int = 2
Batch = tuple[torch.Tensor, ...]


class RegressionUnit(BaseUnit[Batch]):
    def compute_loss(self, state: State, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        out: torch.Tensor = self.module(x)
        return nn.functional.mse_loss(out, y), out


class NoMetrics(BaseEvaluator[Batch]):
    def predict(self, module: nn.Module, data: Batch) -> torch.Tensor:
        return module(data[0])

    def reset(self) -> None:
        pass

    def update(self, data: Batch, outputs: Any) -> None:  # noqa: ANN401
        pass

    def compute(self) -> dict[str, float]:
        return {}


def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    x: torch.Tensor = torch.randn(NUM_SAMPLES, 3, generator=generator)
    return x, x.sum(dim=1, keepdim=True)


def make_unit(device: torch.device, **options: Any) -> RegressionUnit:  # noqa: ANN401
    optimizer = functools.partial(torch.optim.SGD, lr=0.05)
    typed: BaseUnitOptions = BaseUnitOptions(**options)
    return RegressionUnit(module=nn.Linear(3, 1), device=device, optimizer=optimizer, **typed)


def train(device: torch.device, out: Path) -> None:
    rank: int = get_rank()
    torch.manual_seed(rank)  # different initial weights: DDP gives all ranks those of rank 0
    unit = make_unit(device, strategy="ddp", ema_decay=0.5)
    x, y = make_data()
    loader = TensorBatchLoader(x, y, batch_size=BATCH_SIZE, shuffle=True)
    loader.shard(rank, get_world_size(), seed=0)
    eval_loader = TensorBatchLoader(x, y, batch_size=8)
    fit_unit(unit, loader, NoMetrics(eval_loader), EPOCHS, checkpoint_dir=out / "checkpoints")
    assert unit.ema_model is not None
    assert unit.optimizer is not None
    result: dict[str, Any] = {
        "weights": unit.raw_module.state_dict(),
        "ema": unit.ema_model.module.state_dict(),
        "steps": unit.train_progress.num_steps_completed,
        "inference_is_ema": unit.inference_module() is unit.ema_model.module,
        "param_names": [n for g in unit.optimizer.param_groups for n in g["param_names"]],
    }
    torch.save(result, out / f"rank{rank}.pt")


def job(out: Path) -> None:
    jobs: Path = out / "jobs"
    shared = make_shared_job(lambda: create_job("ddp-test", root=jobs), root=jobs)
    (out / f"rank{get_rank()}.txt").write_text(shared.id)


def main() -> None:
    task, out = sys.argv[1], Path(sys.argv[2])
    device: torch.device = init_distributed()
    if task == "train":
        train(device, out)
    else:
        job(out)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
