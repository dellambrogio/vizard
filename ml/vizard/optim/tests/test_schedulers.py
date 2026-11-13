"""Tests of the LR schedulers."""

import pytest
import torch
from torch import nn

from vizard.optim.schedulers import warmup_cosine, warmup_cosine_factor


def test_warmup_cosine_factor() -> None:
    factors: list[float] = [warmup_cosine_factor(i, 2, 6, 0.1) for i in range(8)]
    assert factors[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert factors[2:7] == sorted(factors[2:7], reverse=True)  # cosine decay
    assert factors[6] == pytest.approx(0.1)
    assert factors[7] == pytest.approx(0.1)  # stays at min_lr_ratio after total


def test_warmup_cosine_rejects_bad_warmup() -> None:
    optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0)
    with pytest.raises(ValueError, match="warmup"):
        warmup_cosine(optimizer, warmup=5, total=3)
