"""DDP tests: ddp_worker.py runs in 2 processes with torchrun, gloo backend, CPU only.

The worker runs in new processes, because a process that forks after CUDA is in use (other
tests) cannot use CUDA.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import torch

from vizard.training.fit import restore_latest

WORKER: Path = Path(__file__).parent / "ddp_worker.py"
WORLD_SIZE: int = 2


def load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ddp_worker", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(task: str, out: Path) -> None:
    command: list[str] = [sys.executable, "-m", "torch.distributed.run"]
    command += [f"--nproc-per-node={WORLD_SIZE}", str(WORKER), task, str(out)]
    env: dict[str, str] = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]


def test_ddp_training_and_restore(tmp_path: Path) -> None:
    run_worker("train", tmp_path)
    worker = load_worker()
    results = [torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD_SIZE)]
    steps_per_epoch: int = worker.NUM_SAMPLES // WORLD_SIZE // worker.BATCH_SIZE
    for result in results:
        assert result["steps"] == worker.EPOCHS * steps_per_epoch
        assert result["inference_is_ema"]
        assert set(result["weights"]) == {"weight", "bias"}  # no "module." prefix
        assert result["param_names"] == ["weight", "bias"]  # the optimizer names too
    for key in ("weights", "ema"):  # the same model on all ranks
        for name, value in results[0][key].items():
            torch.testing.assert_close(results[1][key][name], value)
    # A unit without DDP (export, init_from) restores the DDP checkpoint.
    unit = worker.make_unit(torch.device("cpu"), ema_decay=0.5)
    restore_latest(unit, tmp_path / "checkpoints")
    for name, value in results[0]["weights"].items():
        torch.testing.assert_close(unit.module.state_dict()[name], value)
    for name, value in results[0]["ema"].items():
        torch.testing.assert_close(unit.ema_model.module.state_dict()[name], value)


def test_shared_job(tmp_path: Path) -> None:
    run_worker("job", tmp_path)
    ids: set[str] = {(tmp_path / f"rank{r}.txt").read_text() for r in range(WORLD_SIZE)}
    assert len(ids) == 1
    assert [p.name for p in (tmp_path / "jobs").iterdir()] == list(ids)
