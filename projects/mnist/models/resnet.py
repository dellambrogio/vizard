"""Small residual CNN for MNIST: 28x28 grayscale input, 10 class logits."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ResNetConfig:
    """Architecture of ResNet. Values: the unit.module.cfg node of an experiment config."""

    # Channels of the first stage. The stages have width, 2 * width and 4 * width channels at
    # 28x28, 14x14 and 7x7.
    width: int = 32
    blocks_per_stage: int = 1  # residual blocks in each of the three stages
    dropout: float = 0.1  # dropout probability before the last linear layer


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with batch norm, plus a skip connection.

    If stride or the number of channels changes, the skip connection is a 1x1 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class ResNet(nn.Module):
    """Residual classifier: a stem, three residual stages, global average pooling, dropout."""

    def __init__(self, cfg: ResNetConfig | None = None) -> None:
        super().__init__()
        self.cfg: ResNetConfig = cfg or ResNetConfig()
        width: int = self.cfg.width
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        blocks: list[nn.Module] = []
        in_channels: int = width
        for stage in range(3):
            out_channels: int = width * 2**stage
            for block in range(self.cfg.blocks_per_stage):
                # The first block of stages 2 and 3 halves the image size.
                stride: int = 2 if stage > 0 and block == 0 else 1
                blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(in_channels, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
