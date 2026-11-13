"""Tests of the optimizers: parameter groups, and MuonAdamW against torch.optim.Muon/AdamW."""

import functools
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchtnt.framework.state import State

from vizard.evaluation.evaluator import BaseEvaluator
from vizard.optim.adamw import adamw
from vizard.optim.muon import MuonAdamW, muon_adamw, newton_schulz
from vizard.optim.schedulers import warmup_cosine
from vizard.training.fit import fit_unit, restore_latest
from vizard.training.unit import BaseUnit, OptimizerFactory

Batch = tuple[torch.Tensor, torch.Tensor]


class Model(nn.Module):
    """A conv layer (4-D weight), a hidden linear layer, a norm and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 2, kernel_size=3, padding=1)
        self.body = nn.Linear(8, 6)
        self.norm = nn.LayerNorm(6)
        self.head = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.conv(x.unsqueeze(1)).flatten(1)
        return self.head(self.norm(self.body(h)))


def group_of(optimizer: torch.optim.Optimizer, name: str) -> dict:
    return next(g for g in optimizer.param_groups if name in g["param_names"])


def make_grads(model: nn.Module) -> None:
    torch.manual_seed(0)
    model(torch.randn(5, 4)).pow(2).mean().backward()


def test_adamw_no_decay_1d() -> None:
    model = Model()
    optimizer = adamw(model.named_parameters(), lr=1e-3, weight_decay=0.1)
    assert group_of(optimizer, "body.weight")["weight_decay"] == 0.1
    assert group_of(optimizer, "conv.weight")["weight_decay"] == 0.1
    for name in ("body.bias", "norm.weight", "norm.bias", "head.bias"):
        assert group_of(optimizer, name)["weight_decay"] == 0.0
    optimizer = adamw(model.named_parameters(), lr=1e-3, weight_decay=0.1, no_decay_1d=False)
    assert group_of(optimizer, "norm.weight")["weight_decay"] == 0.1


def test_adamw_param_lrs() -> None:
    model = Model()
    optimizer = adamw(model.named_parameters(), lr=1e-3, param_lrs={r"^head\.": 1e-2})
    assert group_of(optimizer, "head.weight")["lr"] == 1e-2
    assert group_of(optimizer, "head.bias")["lr"] == 1e-2
    assert group_of(optimizer, "body.weight")["lr"] == 1e-3


def test_adamw_frozen_parameters_are_left_out() -> None:
    model = Model()
    model.conv.requires_grad_(False)
    optimizer = adamw(model.named_parameters(), lr=1e-3)
    names: set[str] = {n for g in optimizer.param_groups for n in g["param_names"]}
    assert not any(name.startswith("conv.") for name in names)


def test_pattern_errors() -> None:
    model = Model()
    with pytest.raises(ValueError, match="match no trainable parameter"):
        adamw(model.named_parameters(), lr=1e-3, param_lrs={"haed": 1e-2})
    with pytest.raises(ValueError, match="more than one"):
        adamw(model.named_parameters(), lr=1e-3, param_lrs={"head": 1e-2, "weight": 1e-4})
    with pytest.raises(ValueError, match="match no trainable parameter"):
        muon_adamw(model.named_parameters(), lr=1e-3, adamw_params=["norm"])  # norm is 1-D


def test_muon_adamw_groups() -> None:
    model = Model()
    optimizer = muon_adamw(model.named_parameters(), lr=1e-3, adamw_params=[r"^head\."])
    muon = next(g for g in optimizer.param_groups if g["muon"])
    assert muon["param_names"] == ["body.weight"]
    assert not group_of(optimizer, "head.weight")["muon"]
    assert not group_of(optimizer, "conv.weight")["muon"]  # 3-D conv weight: AdamW
    with pytest.raises(ValueError, match="2-D"):
        MuonAdamW([{"params": [("b", model.body.bias)], "muon": True}])


def test_newton_schulz_orthogonalizes() -> None:
    torch.manual_seed(0)
    for shape in ((6, 8), (8, 6)):
        out: torch.Tensor = newton_schulz(torch.randn(*shape))
        assert out.shape == shape
        singular: torch.Tensor = torch.linalg.svdvals(out.float())
        assert singular.min() > 0.5 and singular.max() < 1.5  # near 1, see torch.optim.Muon


def train_only(model: nn.Module, names: set[str]) -> None:
    """Freeze all parameters but names, so the two models of a test stay the same."""
    for name, p in model.named_parameters():
        p.requires_grad_(name in names)


def test_muon_step_matches_torch_muon() -> None:
    model = Model()
    reference = Model()
    reference.load_state_dict(model.state_dict())
    for m in (model, reference):
        train_only(m, {"body.weight"})
    ours = muon_adamw(model.named_parameters(), lr=0.02, weight_decay=0.1, momentum=0.9)
    theirs = torch.optim.Muon(
        [reference.body.weight],
        lr=0.02,
        weight_decay=0.1,
        momentum=0.9,
        adjust_lr_fn="match_rms_adamw",
    )
    for _ in range(3):  # also checks the momentum buffer
        for m, optimizer in ((model, ours), (reference, theirs)):
            optimizer.zero_grad()
            make_grads(m)
            optimizer.step()
    torch.testing.assert_close(model.body.weight, reference.body.weight)


def test_adamw_part_matches_torch_adamw() -> None:
    model = Model()
    reference = Model()
    reference.load_state_dict(model.state_dict())
    names = ("conv.weight", "conv.bias", "norm.weight", "head.weight")
    for m in (model, reference):
        train_only(m, set(names))
    ours = muon_adamw(model.named_parameters(), lr=0.01, weight_decay=0.1, adamw_params=["head"])
    params = dict(reference.named_parameters())
    theirs = torch.optim.AdamW(
        [
            {"params": [params["conv.weight"], params["head.weight"]], "weight_decay": 0.1},
            {"params": [params["conv.bias"], params["norm.weight"]], "weight_decay": 0.0},
        ],
        lr=0.01,
    )
    for _ in range(3):
        for m, optimizer in ((model, ours), (reference, theirs)):
            optimizer.zero_grad()
            make_grads(m)
            optimizer.step()
    ours_params = dict(model.named_parameters())
    for name in names:
        torch.testing.assert_close(ours_params[name], params[name])


def test_state_dict_round_trip() -> None:
    model = Model()
    optimizer = muon_adamw(model.named_parameters(), lr=0.01)
    make_grads(model)
    optimizer.step()
    restored = muon_adamw(Model().named_parameters(), lr=0.01)
    restored.load_state_dict(optimizer.state_dict())
    saved = optimizer.state_dict()["state"]
    loaded = restored.state_dict()["state"]
    assert loaded.keys() == saved.keys()
    muon_keys = [k for k, v in saved.items() if "momentum_buffer" in v]
    assert muon_keys  # body.weight and head.weight are on Muon
    for k in muon_keys:
        torch.testing.assert_close(loaded[k]["momentum_buffer"], saved[k]["momentum_buffer"])


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

    def update(self, data: Batch, outputs: object) -> None:
        pass

    def compute(self) -> dict[str, float]:
        return {}


def make_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(1)
    x: torch.Tensor = torch.randn(32, 4, generator=generator)
    return DataLoader(TensorDataset(x, x.sum(dim=1, keepdim=True)), batch_size=4)


def make_unit(optimizer: OptimizerFactory) -> RegressionUnit:
    scheduler = functools.partial(warmup_cosine, warmup=1, total=2)
    return RegressionUnit(
        module=Model(), device=torch.device("cpu"), optimizer=optimizer, lr_scheduler=scheduler
    )


def test_unit_with_muon_adamw_trains_and_restores(tmp_path: Path) -> None:
    optimizer = functools.partial(muon_adamw, lr=0.01, adamw_params=[r"^head\."])
    unit = make_unit(optimizer)
    fit_unit(unit, make_loader(), NoMetrics(make_loader()), 2, checkpoint_dir=tmp_path)
    restored = make_unit(optimizer)
    restore_latest(restored, tmp_path)
    for (name, a), (_, b) in zip(
        unit.module.named_parameters(), restored.module.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, msg=name)
    assert restored.optimizer is not None and unit.optimizer is not None
    saved = unit.optimizer.state_dict()["state"]
    loaded = restored.optimizer.state_dict()["state"]
    assert saved.keys() == loaded.keys()


def test_unit_with_torch_optimizer() -> None:
    # A plain torch optimizer gets the named parameters too.
    unit = make_unit(functools.partial(torch.optim.Adam, lr=0.01))
    fit_unit(unit, make_loader(), NoMetrics(make_loader()), 1)
    assert unit.train_progress.num_steps_completed == 8
