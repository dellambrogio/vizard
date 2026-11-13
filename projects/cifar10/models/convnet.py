"""VGG-style CNN for CIFAR-10: 32x32 RGB input, 10 class logits."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ConvNetConfig:
    """Architecture of ConvNet. Values: the unit.module.cfg node of an experiment config."""

    # Channels of the first stage. The stages have width, 2 * width and 4 * width channels at
    # 32x32, 16x16 and 8x8.
    width: int = 64
    convs_per_stage: int = 2  # 3x3 conv layers in each of the three stages
    dropout: float = 0.2  # dropout probability before the last linear layer


class ConvNet(nn.Module):
    """Three stages of conv + batch norm + ReLU with max pooling, then global average pooling."""

    def __init__(self, cfg: ConvNetConfig | None = None) -> None:
        super().__init__()
        self.cfg: ConvNetConfig = cfg or ConvNetConfig()
        layers: list[nn.Module] = []
        in_channels: int = 3
        for stage in range(3):
            out_channels: int = self.cfg.width * 2**stage
            for _ in range(self.cfg.convs_per_stage):
                layers += [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
                in_channels = out_channels
            if stage < 2:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(in_channels, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
