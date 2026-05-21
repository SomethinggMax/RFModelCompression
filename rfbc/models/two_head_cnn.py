"""Two-head baseline CNN for joint C1 + C2 prediction.

Shared backbone:
    - 1x1 stem to compress the wide channel dimension (~486 in the default
      pipeline) down to something the spatial convs can chew on.
    - 3 conv blocks (Conv -> BN -> ReLU -> MaxPool) doubling channels each
      time and halving spatial dims (64 -> 32 -> 16 -> 8).
    - Global average pool + dropout.

Two heads:
    - ``head_c1``: 21-way classifier for Campaign 1 gestures (M01..M21).
    - ``head_c2``: 10-way classifier for Campaign 2 activities (A01..A10).

At forward time the network always returns both heads' logits; the training
loop is responsible for routing each sample's loss to the correct head based
on its campaign. Total params are well under 200 K so this is genuinely a
"small CNN for demo" and not a real baseline.

Note
----
This module exists for reference and comparison. The actual compression
baseline is ``rfbc.models.baseline_cnn.BaselineCNN`` — a single-head 31-class
model that is simpler to prune, quantise, and reason about per-class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoHeadCNN(nn.Module):
    """Small two-head CNN over the (C, 64, 64) BEV tensor.

    Parameters
    ----------
    in_channels
        Number of input channels (e.g. 486 for the default pipeline).
    num_c1_classes
        Number of Campaign 1 gesture classes. Default 21 (M01..M21).
    num_c2_classes
        Number of Campaign 2 activity classes. Default 10 (A01..A10).
    stem_channels
        Channels after the 1x1 stem. The default of 32 brings 486 down by a
        factor of ~15 — enough to keep early conv parameter counts modest.
    base_width
        Channels in the first conv block. Doubles each block.
    dropout
        Dropout probability applied before the linear heads.
    """

    def __init__(
        self,
        in_channels: int,
        num_c1_classes: int = 21,
        num_c2_classes: int = 10,
        stem_channels: int = 32,
        base_width: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # 1x1 channel-compression stem: turns (B, 486, 64, 64) into
        # (B, stem_channels, 64, 64) so the spatial convs see a reasonable
        # channel count.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.backbone = nn.Sequential(
            self._conv_block(stem_channels, base_width),         # 64 -> 32
            self._conv_block(base_width, base_width * 2),        # 32 -> 16
            self._conv_block(base_width * 2, base_width * 4),    # 16 -> 8
            nn.AdaptiveAvgPool2d(1),                             # 8 -> 1
        )

        feat_dim = base_width * 4
        self.dropout = nn.Dropout(dropout)
        self.head_c1 = nn.Linear(feat_dim, num_c1_classes)
        self.head_c2 = nn.Linear(feat_dim, num_c2_classes)

        self._init_weights()

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared backbone, return the post-pool feature vector."""
        h = self.stem(x)
        h = self.backbone(h)
        return torch.flatten(h, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(c1_logits, c2_logits)`` for every sample in the batch."""
        feat = self.features(x)
        feat = self.dropout(feat)
        return self.head_c1(feat), self.head_c2(feat)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
