"""Tests of the model summary."""

from pathlib import Path

from torch import nn

from vizard.training.summary import count_parameters, model_summary, save_model_summary


def make_model() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model[0].requires_grad_(False)  # frozen: 3 * 4 + 4 = 16 parameters
    return model


def test_count_parameters() -> None:
    assert count_parameters(make_model()) == (26, 10)


def test_model_summary(tmp_path: Path) -> None:
    text: str = model_summary(make_model())
    assert text.startswith("Sequential(")
    lines: list[str] = text.splitlines()
    assert lines[-1].split() == ["total", "26", "10"]
    assert lines[-4].split() == ["0", "Linear", "16", "0"]
    assert lines[-2].split() == ["2", "Linear", "10", "10"]
    save_model_summary(make_model(), tmp_path / "model.txt")
    assert (tmp_path / "model.txt").read_text() == text
