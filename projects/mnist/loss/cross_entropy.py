"""Cross-entropy loss for MNIST.

A loss is a plain function (logits, targets) -> scalar tensor, not an nn.Module. TorchTNT saves
every nn.Module attribute of a unit in its checkpoints, so a module here would change the
checkpoint format.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cross_entropy(label_smoothing: float = 0.0) -> LossFn:
    """Cross-entropy on logits (N, C) and class indices (N,).

    label_smoothing in [0, 1): moves this much probability mass from the true class to a
    uniform distribution over all classes. 0 is plain cross-entropy.
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    return partial(nn.functional.cross_entropy, label_smoothing=label_smoothing)
