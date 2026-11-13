"""Tests of TensorBatchLoader."""

import pytest
import torch

from vizard.data.tensor_loader import TensorBatchLoader

NUM_SAMPLES: int = 10


def make_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    x: torch.Tensor = torch.arange(NUM_SAMPLES * 2, dtype=torch.float32).reshape(NUM_SAMPLES, 2)
    y: torch.Tensor = torch.arange(NUM_SAMPLES)
    return x, y


def labels_of_epoch(loader: TensorBatchLoader) -> list[int]:
    return [int(v) for _, y in loader for v in y]


def test_length_and_batch_sizes() -> None:
    x, y = make_tensors()
    loader = TensorBatchLoader(x, y, batch_size=4)
    assert len(loader) == 3
    assert [len(b[1]) for b in loader] == [4, 4, 2]
    loader = TensorBatchLoader(x, y, batch_size=4, drop_last=True)
    assert len(loader) == 2
    assert [len(b[1]) for b in loader] == [4, 4]


def test_no_shuffle_keeps_order_and_pairs() -> None:
    x, y = make_tensors()
    loader = TensorBatchLoader(x, y, batch_size=3)
    assert labels_of_epoch(loader) == list(range(NUM_SAMPLES))
    for xb, yb in loader:
        torch.testing.assert_close(xb, x[yb])  # the rows of x and y stay together


def test_shuffle_has_each_sample_once() -> None:
    x, y = make_tensors()
    loader = TensorBatchLoader(x, y, batch_size=3, shuffle=True)
    labels: list[int] = labels_of_epoch(loader)
    assert sorted(labels) == list(range(NUM_SAMPLES))
    for xb, yb in loader:
        torch.testing.assert_close(xb, x[yb])


def test_seed_sets_the_order() -> None:
    x, y = make_tensors()
    loader = TensorBatchLoader(x, y, batch_size=3, shuffle=True)
    torch.manual_seed(1)
    first: list[int] = labels_of_epoch(loader)
    second_epoch: list[int] = labels_of_epoch(loader)
    torch.manual_seed(1)
    assert labels_of_epoch(loader) == first
    torch.manual_seed(2)
    assert labels_of_epoch(loader) != first
    assert second_epoch != first  # a new order each epoch


def test_transform_runs_on_each_batch() -> None:
    x, y = make_tensors()

    def negate(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        xb, yb = batch
        return -xb, yb

    loader = TensorBatchLoader(x, y, batch_size=4, transform=negate)
    for xb, yb in loader:
        torch.testing.assert_close(xb, -x[yb])


def test_bad_arguments() -> None:
    x, y = make_tensors()
    with pytest.raises(ValueError, match="same first dimension"):
        TensorBatchLoader(x, y[:5], batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        TensorBatchLoader(x, y, batch_size=0)
    with pytest.raises(ValueError, match="at least one tensor"):
        TensorBatchLoader(batch_size=2)


def test_shard_parts() -> None:
    x, y = make_tensors()
    loaders = [TensorBatchLoader(x, y, batch_size=2, shuffle=True) for _ in range(3)]
    for rank, loader in enumerate(loaders):
        loader.shard(rank, 3, seed=5)
    assert all(len(loader) == 2 for loader in loaders)  # 10 // 3 = 3 samples: 2 batches
    epochs = [[labels_of_epoch(loader) for _ in range(2)] for loader in loaders]
    for epoch in range(2):
        parts = [epochs[rank][epoch] for rank in range(3)]
        assert all(len(part) == 3 for part in parts)
        merged: list[int] = [label for part in parts for label in part]
        assert len(set(merged)) == 9  # no sample on two ranks; 1 sample left out
    assert epochs[0][0] != epochs[0][1]  # a new order each epoch
    again = TensorBatchLoader(x, y, batch_size=2, shuffle=True)
    again.shard(0, 3, seed=5)
    assert labels_of_epoch(again) == epochs[0][0]  # the seed and the epoch set the order


def test_shard_errors() -> None:
    x, y = make_tensors()
    loader = TensorBatchLoader(x, y, batch_size=2)
    with pytest.raises(ValueError, match="rank"):
        loader.shard(2, 2, seed=0)
    with pytest.raises(ValueError, match="cannot be split"):
        loader.shard(0, NUM_SAMPLES + 1, seed=0)
