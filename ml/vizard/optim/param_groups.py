"""Parameter groups from named parameters: trainable parameters, name patterns, weight decay."""

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from torch import nn

NamedParams = Iterable[tuple[str, nn.Parameter]]
NamedGroup = dict[str, Any]  # a parameter group: "params" is a list of (name, parameter)


def trainable(params: NamedParams) -> list[tuple[str, nn.Parameter]]:
    """The parameters with requires_grad."""
    return [(name, p) for name, p in params if p.requires_grad]


def check_patterns_match(
    patterns: Iterable[str], names: Sequence[str], what: str
) -> list[re.Pattern[str]]:
    """Compile the patterns. A pattern that matches no name is an error: usually a typo."""
    compiled: list[re.Pattern[str]] = [re.compile(p) for p in patterns]
    unmatched: list[str] = [m.pattern for m in compiled if not any(m.search(n) for n in names)]
    if unmatched:
        raise ValueError(f"{what} patterns match no trainable parameter: {unmatched}")
    return compiled


def decay_groups(
    named: list[tuple[str, nn.Parameter]],
    lr_of: Callable[[str], float],
    weight_decay: float,
    no_decay_1d: bool,
) -> list[NamedGroup]:
    """One group for each (LR, weight decay). With no_decay_1d, 1-D parameters (biases,
    norms) get weight decay 0.
    """
    groups: dict[tuple[float, float], list[tuple[str, nn.Parameter]]] = {}
    for name, p in named:
        decay: float = 0.0 if no_decay_1d and p.ndim <= 1 else weight_decay
        groups.setdefault((lr_of(name), decay), []).append((name, p))
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
