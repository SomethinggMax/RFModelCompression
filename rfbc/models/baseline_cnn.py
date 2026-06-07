"""Single-head baseline CNN — the compression study's starting point.

Architecture (≈ 420 K parameters):

    Input   (B, 486, 64, 64)   ← BEV tensor from the standard pipeline
    Stem    Conv1×1(486→32)  + BN + ReLU        (B,  32, 64, 64)
    Block1  Conv3×3(32→32)   + BN + ReLU + Pool  (B,  32, 32, 32)
    Block2  Conv3×3(32→64)   + BN + ReLU + Pool  (B,  64, 16, 16)
    Block3  Conv3×3(64→128)  + BN + ReLU + Pool  (B, 128,  8,  8)
    Block4  Conv3×3(128→256) + BN + ReLU         (B, 256,  8,  8)
    GlobalAvgPool                                (B, 256)
    Dropout(0.5)
    Linear(256 → 31)                             logits over 31 classes

Design decisions
----------------
Standard ops only (Conv2d / BatchNorm2d / ReLU / MaxPool2d / Linear).
Both magnitude-based filter pruning and post-training quantisation (PTQ)
interact cleanly with this op set. Conv→BN→ReLU triples fold into a single
quantised op; MaxPool is quantisation-transparent; no exotic activations
or normalisation layers require special casing.

No skip connections. Filter-level structured pruning stays unambiguous:
pruning the output channels of block N only requires patching the input
channels of block N+1. There is no residual add to keep consistent.

Single head, 31 classes (21 C1 micro-gestures M01–M21 + 10 C2 macro-
activities A01–A10). Campaign membership is determined post-hoc by label
index (C1 labels come first in the label_map because campaigns=("C1","C2")
in the default config). This lets the compression analysis ask directly:
do the 21 fine-grained gesture classes degrade faster than the 10 coarse
activity classes under the same pruning budget?

Block4 carries no MaxPool. After three halvings the feature map is already
8×8; another halving to 4×4 would lose spatial structure before the global
pool rather than refine it. Block4 instead acts as a feature-refinement
stage at fixed resolution.

Dropout 0.5. The two-head demo showed a ~20-point train/val gap on C1 even
at 121 K params; with 420 K params and the harder 31-class problem stronger
regularisation is warranted. Reduce only if val accuracy is visibly below
train in a way that looks like under-fitting rather than over-fitting.

Compression helpers. ``prunable_conv_layers`` returns an ordered dict of
every Conv2d in forward-pass order. ``features()`` exposes the post-pool
representation without the classifier, useful for probing representations
before and after compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Single-head 31-class CNN baseline for the compression study.

    Parameters
    ----------
    in_channels
        Number of BEV input channels. Default 486 = 54 frames × 3 z-bands
        × 3 features/cell, matching the standard pipeline config.
    num_classes
        Output classes. Default 31 = 21 C1 gestures + 10 C2 activities.
    dropout
        Dropout probability applied immediately before the linear head.
        Default 0.5.
    width_mult
        Channel-width multiplier for the whole backbone. ``1.0`` reproduces the
        original 32/32/64/128/256 architecture exactly (~420 K params). Values
        below 1.0 shrink capacity, a cheap lever against over-fitting and a
        cleaner starting point for the compression study (e.g. ``0.5`` ≈ a
        quarter of the parameters). Channel counts are rounded and floored at 8.
    conv_dropout
        Probability for ``Dropout2d`` (spatial dropout) applied after each
        backbone block. ``0.0`` (default) is a no-op, preserving the original
        behaviour. Small values (0.1–0.2) regularise the convolutional stack in
        addition to the head dropout.
    """

    def __init__(
        self,
        in_channels: int = 486,
        num_classes: int = 31,
        dropout: float = 0.5,
        width_mult: float = 1.0,
        conv_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.width_mult = width_mult
        self.conv_dropout_p = conv_dropout

        def _w(base: int) -> int:
            return max(8, int(round(base * width_mult)))

        c1, c2, c3, c4 = _w(32), _w(64), _w(128), _w(256)
        # Stem width tracks the first spatial block so width_mult=1.0 keeps the
        # original 32-channel stem.
        c_stem = _w(32)

        # ------------------------------------------------------------------
        # Stem: 1×1 conv collapses the wide temporal-feature channel
        # dimension to 32 before the spatial convolutions start.
        # Keeping the stem separate makes it an addressable pruning target
        # independent of the spatial blocks.
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_stem, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Spatial backbone — four named blocks.
        # Blocks 1–3 halve spatial resolution via MaxPool.
        # Block 4 refines features at 8×8 without further downsampling.
        # ------------------------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(c_stem, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 64×64 → 32×32
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 32×32 → 16×16
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # 16×16 → 8×8
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),  # stays at 8×8 — no MaxPool
        )

        # Spatial dropout applied after each block (no-op when conv_dropout=0).
        self.conv_dropout = nn.Dropout2d(conv_dropout)

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c4, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone only — returns the 256-d post-pool feature vector.

        Use this during compression analysis to probe representations
        before/after pruning without passing through the classifier, or
        to attach a fresh linear head after compression.
        """
        x = self.stem(x)
        x = self.conv_dropout(self.block1(x))
        x = self.conv_dropout(self.block2(x))
        x = self.conv_dropout(self.block3(x))
        x = self.conv_dropout(self.block4(x))
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(x)))

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    @property
    def prunable_conv_layers(self) -> dict[str, nn.Conv2d]:
        """Ordered {name: module} dict of every Conv2d in forward-pass order.

        Use this when implementing structured filter pruning. The chain is:

            stem.0  →  block1.0  →  block2.0  →  block3.0  →  block4.0

        Pruning the output channels of layer N requires matching the input
        channels of layer N+1 in this sequence. There are no skip connections
        so the dependency chain is strictly linear.
        """
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, nn.Conv2d)
        }

    def layer_parameter_counts(self) -> dict[str, int]:
        """Parameter count per named top-level module.

        Useful for understanding which blocks dominate the budget before
        deciding where to focus a pruning sweep.

        Example output (default config)::

            stem:        15 616   ( 3.7 %)
            block1:       9 280   ( 2.2 %)
            block2:      18 560   ( 4.4 %)
            block3:      73 984   (17.6 %)
            block4:     295 424   (70.3 %)   ← dominates; prune here first
            classifier:   7 967   ( 1.9 %)
        """
        return {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in [
                ("stem",       self.stem),
                ("block1",     self.block1),
                ("block2",     self.block2),
                ("block3",     self.block3),
                ("block4",     self.block4),
                ("classifier", self.classifier),
            ]
        }
