"""BEV (bird's-eye-view) tensor construction.

Given a list of fused per-frame point clouds (output of windowing), produce a
dense tensor that a 2D CNN can consume.

Layout
------
For each (frame, z-band) we compute three per-cell channel features:

    feat 0: point count
    feat 1: mean density (0 in empty cells)
    feat 2: max density  (0 in empty cells)

Channels are flattened into a single C dimension as
``(T * Z * F)`` so the resulting tensor has shape ``(C, H, W)``. The ordering
is ``feat -> z-band -> time`` slowest to fastest, but the exact ordering does
not affect a CNN's ability to learn — what matters is that it's deterministic
and recoverable.
"""

from __future__ import annotations

import numpy as np

from rfbc.config import PipelineConfig


def points_to_bev(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
) -> np.ndarray:
    """Project a list of frames (world-frame points) into a BEV tensor.

    Parameters
    ----------
    frames
        List of length ``cfg.target_frames``; each entry is ``(N_t, 5)``
        with columns ``[timestamp, x, y, z, density]`` already in the world
        frame.
    cfg
        Pipeline config — supplies grid size, extent, z-bands, feature count.

    Returns
    -------
    Numpy array of shape ``(C, H, W)`` and dtype float32, where
    ``C = target_frames * num_z_bands * feat_per_cell``.
    """
    if len(frames) != cfg.target_frames:
        raise ValueError(
            f"expected {cfg.target_frames} frames per window, got {len(frames)}"
        )
    H = W = cfg.grid_size
    Z = cfg.num_z_bands
    F = cfg.feat_per_cell  # currently 3: count, mean_dens, max_dens

    extent = cfg.grid_extent_m
    cell = (2.0 * extent) / cfg.grid_size  # metres per cell

    z_edges = np.asarray(cfg.z_bands, dtype=np.float64)

    # tensor laid out as (T, Z, F, H, W) and reshaped at the end to (C, H, W).
    out = np.zeros((cfg.target_frames, Z, F, H, W), dtype=np.float32)

    for t, frame in enumerate(frames):
        if frame.shape[0] == 0:
            continue
        x = frame[:, 1]
        y = frame[:, 2]
        z = frame[:, 3]
        d = frame[:, 4]

        # discretise (x, y) into the BEV grid; clip out-of-bounds points
        ix = np.floor((x + extent) / cell).astype(np.int64)
        iy = np.floor((y + extent) / cell).astype(np.int64)
        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not in_bounds.any():
            continue
        ix, iy, z, d = ix[in_bounds], iy[in_bounds], z[in_bounds], d[in_bounds]

        # assign each detection to a z-band
        zi = np.digitize(z, z_edges) - 1  # -> band index, -1 below, Z above
        for band in range(Z):
            sel = zi == band
            if not sel.any():
                continue
            ix_b, iy_b, d_b = ix[sel], iy[sel], d[sel]

            # accumulate counts
            np.add.at(out[t, band, 0], (iy_b, ix_b), 1.0)
            # accumulate density sum (we'll divide by count below)
            np.add.at(out[t, band, 1], (iy_b, ix_b), d_b)
            # max density via maximum.at
            np.maximum.at(out[t, band, 2], (iy_b, ix_b), d_b)

        # finalise mean: mean = sum / count where count > 0
        for band in range(Z):
            counts = out[t, band, 0]
            sums = out[t, band, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(counts > 0, sums / counts, 0.0)
            out[t, band, 1] = mean

    # flatten (T, Z, F) into channels
    return out.reshape(cfg.target_frames * Z * F, H, W)
