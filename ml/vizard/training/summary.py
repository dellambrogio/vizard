"""Text summary of a model: its structure and its number of parameters."""

from pathlib import Path

from loguru import logger
from torch import nn


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return the number of parameters of module: (total, trainable)."""
    total: int = sum(p.numel() for p in module.parameters())
    trainable: int = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def model_summary(module: nn.Module) -> str:
    """The structure of module (its repr), then a table of the parameters of each top-level
    submodule, and the totals.
    """
    rows: list[tuple[str, str, int, int]] = [
        (name, type(child).__name__, *count_parameters(child))
        for name, child in module.named_children()
    ]
    total, trainable = count_parameters(module)
    in_children: int = sum(row[2] for row in rows)
    if total > in_children:  # parameters of module itself, not of a submodule
        own: int = total - in_children
        own_trainable: int = trainable - sum(row[3] for row in rows)
        rows.append(("(own)", type(module).__name__, own, own_trainable))
    rows.append(("total", "", total, trainable))
    width: int = max(len(row[0]) for row in rows)
    type_width: int = max(len(row[1]) for row in rows)
    lines: list[str] = [
        f"{'name':<{width}}  {'type':<{type_width}}  {'parameters':>12}  {'trainable':>12}"
    ]
    lines += [f"{n:<{width}}  {t:<{type_width}}  {p:>12,}  {tr:>12,}" for n, t, p, tr in rows]
    return f"{module!r}\n\n" + "\n".join(lines) + "\n"


def save_model_summary(module: nn.Module, path: Path) -> None:
    """Write model_summary(module) to path and log the number of parameters."""
    path.write_text(model_summary(module))
    total, trainable = count_parameters(module)
    logger.info(f"model: {total:,} parameters ({trainable:,} trainable), summary in {path.name}")
