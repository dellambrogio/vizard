"""Train-time augmentation for CIFAR-10 batches."""

import torch


def random_crop_flip(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply a random crop and a random horizontal flip to each image of a batch (N, C, H, W).

    The crop pads each side with `padding` zeros, then takes a random H x W window. The flip
    has probability 0.5. Each image gets its own crop offset and flip.
    """
    n, c, h, w = x.shape
    padded: torch.Tensor = torch.nn.functional.pad(x, (padding, padding, padding, padding))
    top: torch.Tensor = torch.randint(0, 2 * padding + 1, (n,))
    left: torch.Tensor = torch.randint(0, 2 * padding + 1, (n,))
    rows: torch.Tensor = top[:, None] + torch.arange(h)  # (N, H)
    cols: torch.Tensor = left[:, None] + torch.arange(w)  # (N, W)
    flip: torch.Tensor = torch.rand(n) < 0.5
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    batch: torch.Tensor = torch.arange(n)[:, None, None, None]
    channel: torch.Tensor = torch.arange(c)[None, :, None, None]
    return padded[batch, channel, rows[:, None, :, None], cols[:, None, None, :]]
