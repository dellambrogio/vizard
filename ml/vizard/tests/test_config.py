"""Tests of the per_rank resolver: the batch of one DDP process."""

import pytest
from omegaconf import OmegaConf

from vizard.config import per_rank


@pytest.mark.parametrize(
    ("world_size", "expected"), [(None, 128), ("1", 128), ("2", 64), ("4", 32)]
)
def test_per_rank(monkeypatch: pytest.MonkeyPatch, world_size: str | None, expected: int) -> None:
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    assert per_rank(128) == expected
    cfg = OmegaConf.create(
        {"hparams": {"batch_size": 128}, "b": "${per_rank:${hparams.batch_size}}"}
    )
    assert cfg.b == expected


def test_per_rank_not_divisible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "3")
    with pytest.raises(ValueError, match="cannot be split in 3"):
        per_rank(128)
