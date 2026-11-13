"""Tests of the BaseUnit options (LR scheduler, clipping, accumulation, precision, EMA,
compile), the seed and the checkpoints each N train steps. A tiny linear model on CPU.
"""

import functools
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Unpack

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchtnt.framework.auto_unit import TrainStepResults
from torchtnt.framework.state import State

from vizard.evaluation.evaluator import BaseEvaluator
from vizard.optim.schedulers import warmup_cosine
from vizard.training.fit import fit_unit, restore_latest
from vizard.training.run import load_init_weights, set_seed
from vizard.training.unit import BaseUnit, BaseUnitOptions

Batch = tuple[torch.Tensor, torch.Tensor]
NUM_SAMPLES: int = 32
BATCH_SIZE: int = 4
STEPS_PER_EPOCH: int = NUM_SAMPLES // BATCH_SIZE
CPU: torch.device = torch.device("cpu")


class LinearUnit(BaseUnit[Batch]):
    """MSE regression unit. Keeps the train losses, gradient norms and LRs of each step, and
    the weights after each optimizer step.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        device: torch.device,
        lr: float = 0.1,
        **options: Unpack[BaseUnitOptions],
    ) -> None:
        optimizer = functools.partial(torch.optim.SGD, lr=lr)
        super().__init__(module=module, device=device, optimizer=optimizer, **options)
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.lrs: list[float] = []
        self.optimizer_steps: int = 0
        self.weights: list[torch.Tensor] = []  # after each optimizer step

    def on_train_start(self, state: State) -> None:
        super().on_train_start(state)
        # AutoUnit makes the optimizer after __init__, so the hook is added here.
        assert self.optimizer is not None
        self.optimizer.register_step_post_hook(self._count_step)

    def _count_step(self, *_: Any) -> None:  # noqa: ANN401
        self.optimizer_steps += 1
        self.weights.append(flat_weights(self.module))

    def compute_loss(self, state: State, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data
        out: torch.Tensor = self.module(x)
        return nn.functional.mse_loss(out.float(), y), out

    def on_train_step_end(
        self, state: State, data: Batch, step: int, results: TrainStepResults
    ) -> None:
        super().on_train_step_end(state, data, step, results)
        self.losses.append(results.loss.item())
        if results.total_grad_norm is not None:
            self.grad_norms.append(float(results.total_grad_norm))
        self.lrs.append(self.current_lr())


def flat_weights(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().clone().flatten() for p in module.parameters()])


class NoMetrics(BaseEvaluator[Batch]):
    """Evaluator with no metrics."""

    def predict(self, module: nn.Module, data: Batch) -> torch.Tensor:
        return module(data[0])

    def reset(self) -> None:
        pass

    def update(self, data: Batch, outputs: Any) -> None:  # noqa: ANN401
        pass

    def compute(self) -> dict[str, float]:
        return {}


def make_loader(shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    x: torch.Tensor = torch.randn(NUM_SAMPLES, 3, generator=generator)
    y: torch.Tensor = 100.0 * x.sum(dim=1, keepdim=True)  # large targets: large gradients
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def train(epochs: int = 1, lr: float = 0.1, **options: Unpack[BaseUnitOptions]) -> LinearUnit:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, lr=lr, **options)
    fit_unit(unit, make_loader(), NoMetrics(make_loader()), epochs)
    return unit


def test_lr_scheduler_steps_each_epoch() -> None:
    scheduler = functools.partial(warmup_cosine, warmup=1, total=3)
    unit = train(epochs=3, lr=0.01, lr_scheduler=scheduler)
    epoch_lrs: list[float] = unit.lrs[::STEPS_PER_EPOCH]  # the LR does not change in an epoch
    assert unit.lrs == [lr for lr in epoch_lrs for _ in range(STEPS_PER_EPOCH)]
    assert epoch_lrs == pytest.approx([0.005, 0.01, 0.005])


def test_lr_scheduler_steps_each_optimizer_step() -> None:
    scheduler = functools.partial(warmup_cosine, warmup=0, total=STEPS_PER_EPOCH)
    unit = train(lr=0.01, lr_scheduler=scheduler, lr_interval="step")
    assert unit.lrs == sorted(unit.lrs, reverse=True)
    assert len(set(unit.lrs)) == STEPS_PER_EPOCH
    assert unit.lrs[-1] == pytest.approx(0.0)


def test_clip_grad_norm() -> None:
    # total_grad_norm is the norm before the clip, so check the parameter change instead: with
    # SGD and lr 1, the change of the parameters is the clipped gradient.
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, lr=1.0, clip_grad_norm=0.5)
    before: torch.Tensor = torch.cat(
        [p.detach().clone().flatten() for p in unit.module.parameters()]
    )
    loader = DataLoader(make_loader().dataset, batch_size=BATCH_SIZE)
    fit_unit(unit, [next(iter(loader))], NoMetrics(make_loader()), 1)
    after: torch.Tensor = torch.cat([p.detach().flatten() for p in unit.module.parameters()])
    assert unit.grad_norms[0] > 0.5
    assert torch.linalg.vector_norm(after - before).item() == pytest.approx(0.5, rel=1e-4)


def test_grad_accum_steps() -> None:
    unit = train(grad_accum_steps=4)
    assert len(unit.losses) == STEPS_PER_EPOCH
    assert unit.optimizer_steps == STEPS_PER_EPOCH // 4


def test_bf16_precision_on_cpu() -> None:
    unit = train(precision="bf16")
    assert len(unit.losses) == STEPS_PER_EPOCH
    assert all(torch.isfinite(torch.tensor(unit.losses)))


def test_same_seed_same_losses() -> None:
    def seeded_losses(seed: int) -> list[float]:
        set_seed(seed, deterministic=False)
        unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, lr=0.01)
        fit_unit(unit, make_loader(shuffle=True), NoMetrics(make_loader()), 1)
        return unit.losses

    assert seeded_losses(7) == seeded_losses(7)
    assert seeded_losses(7) != seeded_losses(8)


def test_checkpoint_every_n_steps(tmp_path: Path) -> None:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU)
    fit_unit(
        unit,
        make_loader(),
        NoMetrics(make_loader()),
        2,
        checkpoint_dir=tmp_path,
        keep_last_n=10,
        checkpoint_every_n_steps=3,
    )
    names: list[str] = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    steps: set[int] = {int(name.split("_step_")[1].split("_")[0]) for name in names}
    # Each 3 train steps, and at the end of each epoch (8 steps per epoch).
    assert steps == {3, 6, 8, 9, 12, 15, 16}


class WeightsSeen(NoMetrics):
    """Evaluator that keeps the weights of the model it evaluates, once per evaluation."""

    def __init__(self, loader: DataLoader) -> None:
        super().__init__(loader)
        self.seen: list[torch.Tensor] = []
        self.unit: BaseUnit[Batch] | None = None

    def reset(self) -> None:
        assert self.unit is not None
        self.seen.append(flat_weights(self.unit.module))


def test_ema_matches_reference() -> None:
    decay: float = 0.5
    unit = train(ema_decay=decay)
    assert unit.ema_model is not None
    expected: torch.Tensor = unit.weights[0]  # the first update copies the weights
    for weights in unit.weights[1:]:
        expected = decay * expected + (1 - decay) * weights
    torch.testing.assert_close(flat_weights(unit.ema_model.module), expected)
    assert unit.inference_module() is unit.ema_model.module


def test_ema_warmup_and_accumulation() -> None:
    unit = train(ema_decay=0.9, ema_warmup_steps=3)
    assert unit.ema_model is not None
    assert int(unit.ema_model.n_averaged) == STEPS_PER_EPOCH - 3
    unit = train(ema_decay=0.9, grad_accum_steps=2)
    assert unit.ema_model is not None
    assert int(unit.ema_model.n_averaged) == unit.optimizer_steps == STEPS_PER_EPOCH // 2


def test_evaluation_uses_ema_weights() -> None:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, ema_decay=0.5)
    assert unit.ema_model is not None
    evaluator = WeightsSeen(make_loader())
    evaluator.unit = unit
    fit_unit(unit, make_loader(), evaluator, 1)
    torch.testing.assert_close(evaluator.seen[-1], flat_weights(unit.ema_model.module))
    assert not torch.equal(evaluator.seen[-1], unit.weights[-1])
    # After the evaluation, the model has the train weights again.
    torch.testing.assert_close(flat_weights(unit.module), unit.weights[-1])


def test_no_ema() -> None:
    unit = train()
    assert unit.ema_model is None
    assert unit.inference_module() is unit.module


def test_ema_checkpoint_restore(tmp_path: Path) -> None:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, ema_decay=0.5)
    fit_unit(unit, make_loader(), NoMetrics(make_loader()), 1, checkpoint_dir=tmp_path)
    restored = LinearUnit(module=nn.Linear(3, 1), device=CPU, ema_decay=0.5)
    restore_latest(restored, tmp_path)
    assert unit.ema_model is not None and restored.ema_model is not None
    ema: torch.Tensor = flat_weights(unit.ema_model.module)
    torch.testing.assert_close(flat_weights(restored.ema_model.module), ema)
    torch.testing.assert_close(flat_weights(restored.module), flat_weights(unit.module))


def test_ema_decay_out_of_range() -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        LinearUnit(module=nn.Linear(3, 1), device=CPU, ema_decay=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile test needs CUDA")
def test_compile_with_ema() -> None:
    cuda = torch.device("cuda")
    unit = LinearUnit(module=nn.Linear(3, 1), device=cuda, compile=True, ema_decay=0.5)
    fit_unit(unit, make_loader(), NoMetrics(make_loader()), 1)
    assert len(unit.losses) == STEPS_PER_EPOCH
    assert all(torch.isfinite(torch.tensor(unit.losses)))
    # In-place compile: the state_dict keys do not change.
    assert set(unit.module.state_dict()) == {"weight", "bias"}


def test_load_init_weights() -> None:
    source = train()
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU)
    load_init_weights(unit, source)
    torch.testing.assert_close(flat_weights(unit.module), flat_weights(source.module))


def test_load_init_weights_uses_ema() -> None:
    source = train(ema_decay=0.5)
    assert source.ema_model is not None
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, ema_decay=0.5)
    load_init_weights(unit, source)
    torch.testing.assert_close(flat_weights(unit.module), flat_weights(source.ema_model.module))
    assert not torch.equal(flat_weights(unit.module), flat_weights(source.module))


def test_load_init_weights_shape_mismatch() -> None:
    source = train()
    unit = LinearUnit(module=nn.Linear(4, 1), device=CPU)
    with pytest.raises(ValueError, match="do not fit"):
        load_init_weights(unit, source)


class SlowDataset(Dataset):
    """Dataset that waits before each sample, so the train steps wait for the data."""

    def __init__(self, base: Dataset, seconds: float) -> None:
        self.base = base
        self.seconds = seconds

    def __len__(self) -> int:
        return NUM_SAMPLES

    def __getitem__(self, index: int) -> Any:  # noqa: ANN401
        time.sleep(self.seconds)
        return self.base[index]


def logged_metrics(
    unit: LinearUnit, loader: Iterable[Any], log_every_n_steps: int = 1
) -> list[tuple[int, dict[str, float]]]:
    """Train unit for 1 epoch and return the metric logs as (step, metrics)."""
    logs: list[tuple[int, dict[str, float]]] = []
    unit.add_metric_logger(lambda m, step: logs.append((step, m)), log_every_n_steps)
    fit_unit(unit, loader, NoMetrics(make_loader()), 1)
    return logs


def test_speed_metrics() -> None:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU)
    logs = logged_metrics(unit, make_loader(), log_every_n_steps=2)
    step_logs = [m for _, m in logs if "train/loss" in m]
    assert len(step_logs) == STEPS_PER_EPOCH // 2
    assert all(m["train/steps_per_sec"] > 0 for m in step_logs)
    assert all(0.0 <= m["train/data_wait_frac"] <= 1.0 for m in step_logs)
    epoch_logs = [m for _, m in logs if "train/epoch_time_sec" in m]
    assert len(epoch_logs) == 1 and epoch_logs[0]["train/steps_per_sec_epoch"] > 0


def test_data_wait_frac_with_slow_data() -> None:
    base = make_loader().dataset
    loader = DataLoader(SlowDataset(base, 0.01), batch_size=BATCH_SIZE)
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU)
    logs = logged_metrics(unit, loader, log_every_n_steps=STEPS_PER_EPOCH)
    epoch = next(m for _, m in logs if "train/data_wait_frac_epoch" in m)
    assert epoch["train/data_wait_frac_epoch"] > 0.5
    step = next(m for _, m in logs if "train/data_wait_frac" in m)
    assert step["train/data_wait_frac"] > 0.5


def test_grad_norm_without_clip() -> None:
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    reference = nn.Linear(3, 1)
    reference.load_state_dict(model.state_dict())
    unit = LinearUnit(module=model, device=CPU, lr=0.01)
    batch: Batch = next(iter(make_loader()))
    logs = logged_metrics(unit, [batch])
    # The reference: one plain SGD step on the same batch, with no clip.
    loss = nn.functional.mse_loss(reference(batch[0]), batch[1])
    loss.backward()
    grads = [p.grad for p in reference.parameters() if p.grad is not None]
    expected_norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads])).item()
    assert logs[0][1]["train/grad_norm"] == pytest.approx(expected_norm, rel=1e-5)
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    # The infinite clip value does not change the gradients.
    torch.testing.assert_close(flat_weights(unit.module), flat_weights(reference), rtol=0, atol=0)


def test_grad_norm_with_accumulation() -> None:
    unit = LinearUnit(module=nn.Linear(3, 1), device=CPU, grad_accum_steps=2)
    logs = logged_metrics(unit, make_loader())
    norms = [m.get("train/grad_norm") for _, m in logs if "train/loss" in m]
    assert norms[0] is None  # no optimizer step after the first batch
    assert all(norm is not None for norm in norms[1:])
    assert norms[2] == norms[1]  # from the last optimizer step (step 2)
    assert norms[3] != norms[2]
