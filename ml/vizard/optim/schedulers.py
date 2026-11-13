"""LR schedulers that a config builds with `_partial_: true` (BaseUnit adds the optimizer)."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_factor(index: int, warmup: int, total: int, min_lr_ratio: float) -> float:
    """The LR factor at scheduler step index: linear warmup, then cosine decay.

    The factor goes from 1 / (warmup + 1) to 1 in `warmup` steps, then down to min_lr_ratio
    at step `total`, and stays at min_lr_ratio after `total`.
    """
    if index < warmup:
        return (index + 1) / (warmup + 1)
    if index >= total:
        return min_lr_ratio
    progress: float = (index - warmup) / max(total - warmup, 1)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_cosine(
    optimizer: torch.optim.Optimizer, *, warmup: int, total: int, min_lr_ratio: float = 0.0
) -> LambdaLR:
    """Linear warmup for `warmup` scheduler steps, then cosine decay until step `total`.

    The LR of the optimizer is the peak LR. warmup and total count in the lr_interval unit of
    the unit: epochs (for example total: ${hparams.epochs}) or optimizer steps. At step
    `total`, the LR is min_lr_ratio times the peak LR.
    """
    if not 0 <= warmup <= total:
        raise ValueError(f"need 0 <= warmup <= total, got warmup {warmup} and total {total}")

    def factor(index: int) -> float:
        return warmup_cosine_factor(index, warmup, total, min_lr_ratio)

    return LambdaLR(optimizer, factor)
