"""Train-only data augmentation for BEV tensors.

Why this exists
---------------
The baseline overfits: with only 16 training subjects, ~420 K parameters, and
a *deterministic* cached BEV per window, the model memorises the training
participants (train acc → ~100 %, val acc ~85 %, val loss rising). The single
most effective fix is to stop showing the network the exact same tensor every
epoch. This module applies cheap, label-preserving, stochastic perturbations to
each cached BEV tensor on the fly, **training fold only**.

Channel layout (must be respected)
-----------------------------------
``points_to_bev`` builds an array of shape ``(T, Z, F, H, W)`` and reshapes it
to ``(C, H, W)`` with ``C = T * Z * F``. The reshape makes ``T`` the slowest
axis and ``F`` (the per-cell feature) the fastest, so for channel index ``c``::

    feature index   f = c %  F                 # 0=count, 1=mean dens, 2=max dens
    z-band index    z = (c // F) %  Z
    frame index     t = c // (Z * F)

This lets us drop whole *frames* (a contiguous block of ``Z*F`` channels) and
add noise only to the *density* features, rather than blindly perturbing the
flattened channel stack.

What we deliberately do NOT do — chirality
------------------------------------------
**No horizontal/vertical flips and no large rotations.** Campaign 1 contains
direction-/handedness-specific classes (swipe-left vs swipe-right,
clockwise vs counter-clockwise circles, left- vs right-arm circles,
inward vs outward two-hand circles). A mirror flip would turn one class into
another and silently corrupt the label. Rotations are kept small (a few
degrees) for the same reason — the geometry relative to the fixed sensor ring
should not be grossly altered.

Usage
-----
    from rfbc.data.augment import make_bev_augment, AugmentedView
    aug = make_bev_augment(cfg, strength=1.0)
    train_view = AugmentedView(train_combined_ds, aug)   # train fold only
    # val / test datasets are used unwrapped (never augmented)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Geometry helpers (pure torch, channel-agnostic)
# ---------------------------------------------------------------------------


def _translate(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift a ``(C, H, W)`` tensor by ``(dy, dx)`` cells with zero fill.

    Positive ``dy`` moves content toward larger row indices (down), positive
    ``dx`` toward larger column indices (right). Vacated cells are zero — i.e.
    "empty BEV", which is the correct neutral value for this representation.
    """
    if dy == 0 and dx == 0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    out = torch.zeros_like(x)
    src_y0, src_y1 = max(0, -dy), min(H, H - dy)
    src_x0, src_x1 = max(0, -dx), min(W, W - dx)
    dst_y0, dst_y1 = max(0, dy), min(H, H + dy)
    dst_x0, dst_x1 = max(0, dx), min(W, W + dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, src_y0:src_y1, src_x0:src_x1]
    return out


def _rotate(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate every channel plane of a ``(C, H, W)`` tensor by ``angle_deg``.

    Uses an affine grid + bilinear sampling (pure torch, no torchvision).
    Out-of-frame samples are filled with zeros. Bilinear interpolation makes
    counts fractional — acceptable as augmentation noise.
    """
    if angle_deg == 0.0:
        return x
    C, H, W = x.shape
    rad = math.radians(angle_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    # 2x3 affine for grid_sample (rotation about the image centre). Built from
    # Python floats so the tensor construction is unambiguous.
    mat = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]],
        dtype=x.dtype, device=x.device,
    ).unsqueeze(0)  # (1, 2, 3)
    grid = F.affine_grid(mat, size=(1, C, H, W), align_corners=False)
    out = F.grid_sample(
        x.unsqueeze(0), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------
# The augmentation policy
# ---------------------------------------------------------------------------


@dataclass
class BEVAugment:
    """Stochastic, label-preserving augmentation for a single ``(C, H, W)`` BEV.

    Each transform fires independently with its own probability; probabilities
    are scaled by the global ``strength`` dial (0 disables everything, 1 is the
    nominal policy). Magnitudes are deliberately conservative — this is sparse
    radar BEV, not natural images.

    Parameters
    ----------
    target_frames, num_z_bands, feat_per_cell
        Channel-layout descriptors (must match the pipeline config that built
        the tensor). Used to locate frame blocks and density features.
    strength
        Global multiplier on all firing probabilities, clamped to [0, 1].
    max_shift
        Max spatial translation in cells (each cell ≈ 12.5 cm at the default
        64-grid / 4 m half-extent).
    max_deg
        Max absolute rotation angle in degrees (kept small — chirality).
    max_frame_drop_frac
        Max fraction of the ``target_frames`` time bins that may be zeroed.
    density_noise_std, density_scale_jitter
        Additive Gaussian std and multiplicative jitter applied to occupied
        density cells (features 1 and 2), relative to the cell value.
    max_cutout
        Max side length (in cells) of the random-erasing square.
    """

    target_frames: int = 54
    num_z_bands: int = 3
    feat_per_cell: int = 3

    strength: float = 1.0

    p_translate: float = 0.5
    max_shift: int = 4

    p_rotate: float = 0.5
    max_deg: float = 12.0

    p_frame_dropout: float = 0.3
    max_frame_drop_frac: float = 0.15

    p_density_noise: float = 0.5
    density_noise_std: float = 0.05
    density_scale_jitter: float = 0.10

    p_cutout: float = 0.3
    max_cutout: int = 12

    def _fires(self, p: float) -> bool:
        p_eff = max(0.0, min(1.0, p * self.strength))
        if p_eff <= 0.0:
            return False
        return bool(torch.rand(()) < p_eff)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0.0:
            return x
        x = x.clone()
        C, H, W = x.shape
        zf = self.num_z_bands * self.feat_per_cell  # channels per frame

        # --- spatial translation ---
        if self._fires(self.p_translate) and self.max_shift > 0:
            dy = int(torch.randint(-self.max_shift, self.max_shift + 1, ()))
            dx = int(torch.randint(-self.max_shift, self.max_shift + 1, ()))
            x = _translate(x, dy, dx)

        # --- small rotation ---
        if self._fires(self.p_rotate) and self.max_deg > 0.0:
            angle = float((torch.rand(()) * 2.0 - 1.0) * self.max_deg)
            x = _rotate(x, angle)

        # --- frame (temporal) dropout ---
        if self._fires(self.p_frame_dropout):
            max_drop = max(1, int(self.max_frame_drop_frac * self.target_frames))
            n_drop = int(torch.randint(1, max_drop + 1, ()))
            frames = torch.randperm(self.target_frames)[:n_drop]
            for t in frames.tolist():
                x[t * zf:(t + 1) * zf, :, :] = 0.0

        # --- density-channel noise (features 1 and 2 only, occupied cells) ---
        if self._fires(self.p_density_noise):
            feat = torch.arange(C) % self.feat_per_cell
            dens_ch = (feat >= 1).nonzero(as_tuple=True)[0]
            if dens_ch.numel() > 0:
                sub = x[dens_ch]
                occ = sub > 0
                if occ.any():
                    scale = 1.0 + (torch.rand_like(sub) * 2.0 - 1.0) * self.density_scale_jitter
                    noise = torch.randn_like(sub) * self.density_noise_std * sub
                    sub = torch.where(occ, (sub * scale + noise).clamp_min(0.0), sub)
                    x[dens_ch] = sub

        # --- random erasing (cutout) across all channels ---
        if self._fires(self.p_cutout) and self.max_cutout > 0:
            ch = int(torch.randint(1, self.max_cutout + 1, ()))
            cw = int(torch.randint(1, self.max_cutout + 1, ()))
            y0 = int(torch.randint(0, max(1, H - ch + 1), ()))
            x0 = int(torch.randint(0, max(1, W - cw + 1), ()))
            x[:, y0:y0 + ch, x0:x0 + cw] = 0.0

        return x


def make_bev_augment(cfg, strength: float = 1.0, **overrides) -> BEVAugment:
    """Build a :class:`BEVAugment` whose channel layout matches ``cfg``.

    Any keyword in ``overrides`` is forwarded to :class:`BEVAugment`, so the
    training script can expose individual knobs on the CLI later if needed.
    """
    return BEVAugment(
        target_frames=cfg.target_frames,
        num_z_bands=cfg.num_z_bands,
        feat_per_cell=cfg.feat_per_cell,
        strength=strength,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class AugmentedView(Dataset):
    """Wrap a dataset so ``__getitem__`` returns an augmented tensor.

    The wrapper shares the **same index space** as the wrapped dataset, so a
    ``WeightedRandomSampler`` built over the underlying ``CombinedRadarDataset``
    can be used directly with a ``DataLoader`` over this view.

    Only ever wrap the *training* dataset. Validation and test datasets must be
    used unwrapped — augmenting them would bias the evaluation.
    """

    def __init__(self, base: Dataset, augment: BEVAugment) -> None:
        self.base = base
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, label = self.base[idx]
        return self.augment(x), label
