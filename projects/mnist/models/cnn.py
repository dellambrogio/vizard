"""Small CNN for MNIST: 28x28 grayscale input, 10 class logits."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NetConfig:
    """Architecture of Net. Values: the unit.module.cfg node of an experiment config."""

    conv1_channels: int = 32  # output channels of the first conv layer, at 28x28
    conv2_channels: int = 64  # output channels of the second conv layer, at 14x14
    hidden: int = 128  # units of the hidden linear layer


class Net(nn.Module):
    """Small convolutional classifier: two conv layers with max pooling, two linear layers."""

    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg: NetConfig = cfg or NetConfig()
        c1, c2 = self.cfg.conv1_channels, self.cfg.conv2_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, self.cfg.hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
