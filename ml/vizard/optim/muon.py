"""Muon on the 2-D parameters, AdamW on the other parameters, in one optimizer (MuonAdamW).

A config builds it with muon_adamw and `_partial_: true`; BaseUnit adds the named parameters.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, overload

import torch
from torch import Tensor, nn
from torch.optim.adamw import adamw as adamw_step

from vizard.optim.param_groups import (
    NamedGroup,
    NamedParams,
    check_patterns_match,
    decay_groups,
    trainable,
)

# Coefficients and number of steps of the quintic Newton-Schulz iteration of Muon, as in
# torch.optim.Muon and https://github.com/KellerJordan/Muon.
NS_COEFFICIENTS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_STEPS: int = 5
NS_EPS: float = 1e-7


def gradient(p: Tensor) -> Tensor:
    """The gradient of a parameter that has one (step passes only such parameters)."""
    grad: Tensor | None = p.grad
    assert grad is not None
    return grad


def newton_schulz(update: Tensor, steps: int = NS_STEPS, eps: float = NS_EPS) -> Tensor:
    """Orthogonalize a 2-D matrix approximately: U S' V^T with S' near 1, for USV^T = update.

    Quintic Newton-Schulz iteration in bfloat16, the same as torch.optim.Muon.
    """
    a, b, c = NS_COEFFICIENTS
    x: Tensor = update.to(dtype=torch.bfloat16, copy=True)
    tall: bool = update.size(0) > update.size(1)
    if tall:
        x = x.T
    x.div_(x.norm().clamp(min=eps))  # spectral norm at most 1
    for _ in range(steps):
        gram: Tensor = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if tall else x


class MuonAdamW(torch.optim.Optimizer):
    """AdamW, with Muon (Jordan et al., 2024) on the parameter groups that set `muon: True`.

    A Muon step is the step of torch.optim.Muon with nesterov=True and
    adjust_lr_fn="match_rms_adamw": the update has the RMS of an AdamW update (Liu et al.,
    2025), so one lr and one LR scheduler drive all groups. Muon groups take 2-D parameters
    only. The other groups step as torch.optim.AdamW.
    """

    def __init__(
        self,
        params: Iterable[NamedGroup],
        *,
        lr: float = 1e-3,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
    ) -> None:
        b1, b2 = betas
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": (b1, b2),
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "muon": False,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("a Muon group takes 2-D parameters only")

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params: list[Tensor] = [p for p in group["params"] if p.grad is not None]
            if params:
                (self._muon_step if group["muon"] else self._adamw_step)(group, params)
        return loss

    def _muon_step(self, group: dict[str, Any], params: list[Tensor]) -> None:
        lr: float = group["lr"]
        momentum: float = group["momentum"]
        for p in params:
            grad: Tensor = gradient(p)
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf: Tensor = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update: Tensor = newton_schulz(grad.lerp(buf, momentum))  # Nesterov momentum
            p.mul_(1 - lr * group["weight_decay"])
            p.add_(update, alpha=-lr * 0.2 * math.sqrt(max(p.shape)))

    def _adamw_step(self, group: dict[str, Any], params: list[Tensor]) -> None:
        for p in params:
            if not self.state[p]:
                self.state[p].update(
                    step=torch.tensor(0.0),
                    exp_avg=torch.zeros_like(p),
                    exp_avg_sq=torch.zeros_like(p),
                )
        states = [self.state[p] for p in params]
        beta1, beta2 = group["betas"]
        adamw_step(
            params,
            [gradient(p) for p in params],
            [s["exp_avg"] for s in states],
            [s["exp_avg_sq"] for s in states],
            [],
            [s["step"] for s in states],
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )


def muon_adamw(
    params: NamedParams,
    *,
    lr: float,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    adamw_params: Sequence[str] = (),
) -> MuonAdamW:
    r"""MuonAdamW with Muon on the trainable 2-D parameters that no adamw_params pattern matches.

    AdamW gets all other parameters: 1-D parameters (with no weight decay), parameters with
    more than 2 dimensions (for example conv weights), and the 2-D parameters that an
    adamw_params pattern matches (re.search). Muon is usually not used for the first and last
    layers, for example r"^head\.". Each pattern must match a trainable 2-D parameter.
    """
    named = trainable(params)
    names_2d: list[str] = [name for name, p in named if p.ndim == 2]
    matchers = check_patterns_match(adamw_params, names_2d, "adamw_params (2-D)")

    def on_muon(name: str, p: nn.Parameter) -> bool:
        return p.ndim == 2 and not any(m.search(name) for m in matchers)

    rest = [(name, p) for name, p in named if not on_muon(name, p)]
    groups: list[NamedGroup] = decay_groups(rest, lambda _: lr, weight_decay, no_decay_1d=True)
    muon_params = [(name, p) for name, p in named if on_muon(name, p)]
    if muon_params:
        groups.append({"params": muon_params, "muon": True})
    return MuonAdamW(
        groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum
    )
