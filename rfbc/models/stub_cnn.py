"""A deliberately tiny CNN for the smoke test.

This is **not** the project baseline. It exists only to verify that the
preprocessing pipeline produces tensors the rest of PyTorch can consume.
The real baseline will be defined separately in week 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StubCNN(nn.Module):
    """3 conv blocks + global pool + linear head."""

    def __init__(self, in_channels: int, num_classes: int, base_width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global pool
        )
        self.classifier = nn.Linear(base_width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = torch.flatten(h, 1)
        return self.classifier(h)
