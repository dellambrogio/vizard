"""AdamW with no weight decay on 1-D parameters and an LR for each name pattern.

A config builds it with `_partial_: true`; BaseUnit adds the named parameters.
"""

from collections.abc import Mapping, Sequence

import torch

from vizard.optim.param_groups import NamedParams, check_patterns_match, decay_groups, trainable


def adamw(
    params: NamedParams,
    *,
    lr: float,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    no_decay_1d: bool = True,
    param_lrs: Mapping[str, float] | None = None,
) -> torch.optim.AdamW:
    r"""AdamW with no weight decay on 1-D parameters (biases, norms) if no_decay_1d is set.

    param_lrs maps a name pattern (re.search, so "backbone" matches each name that has it and
    r"^backbone\." only that prefix) to the LR of the parameters with a matching name. The
    other parameters get lr. Each pattern must match a trainable parameter, and a parameter
    must not match two patterns.
    """
    named = trainable(params)
    names: list[str] = [name for name, _ in named]
    matchers = check_patterns_match((param_lrs or {}).keys(), names, "param_lrs")
    pattern_lrs: dict[str, float] = dict(param_lrs or {})

    def lr_of(name: str) -> float:
        hits: list[str] = [m.pattern for m in matchers if m.search(name)]
        if len(hits) > 1:
            raise ValueError(f"parameter {name!r} matches more than one param_lrs pattern: {hits}")
        return float(pattern_lrs[hits[0]]) if hits else lr

    groups = decay_groups(named, lr_of, weight_decay, no_decay_1d)
    b1, b2 = betas
    return torch.optim.AdamW(groups, lr=lr, betas=(b1, b2), eps=eps, weight_decay=weight_decay)
