"""Temporal windowing.

Given a sample's per-sensor frames (already in world coordinates and
clutter-suppressed), emit one or more **fused** windows of fixed length.

Each window is a list of ``target_frames`` time bins, with each bin holding
fused points from all selected sensors that fall in that time range. We
fuse here (rather than keeping per-sensor structure) because the BEV step
treats sensors as exchangeable — every detection contributes one point.

For samples shorter than ``window_seconds`` we pad with empty frames; for
samples longer than the window length (Campaign 2 activities up to 18 s,
Campaign 3 sentiment 50–180 s) we slide the window with a stride.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def fuse_sensors_by_time(
    sensor_frames: dict[int, list[np.ndarray]],
    *,
    target_frames: int,
    fps: float,
) -> list[np.ndarray]:
    """Re-bin all sensors' detections onto a uniform time grid.

    Returns a list of ``target_frames`` arrays, each ``(M_i, 5)`` (column 0
    is the original timestamp). Bins with no detections are empty arrays.

    The time grid spans from ``t_min`` (earliest timestamp across sensors)
    to ``t_min + target_frames / fps`` and ignores anything outside.
    """
    # Concatenate everything with a sensor-id column so we can debug later
    # if needed, but for downstream BEV we only use columns 0..4.
    parts: list[np.ndarray] = [
        np.concatenate(frames, axis=0)
        for frames in sensor_frames.values()
        if frames and any(f.shape[0] for f in frames)
    ]
    if not parts:
        return [np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]
    all_pts = np.concatenate(parts, axis=0)

    if all_pts.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]

    t_min = all_pts[:, 0].min()
    bin_width = 1.0 / fps
    bins: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]

    rel_t = all_pts[:, 0] - t_min
    bin_idx = np.floor(rel_t / bin_width).astype(np.int64)
    in_range = (bin_idx >= 0) & (bin_idx < target_frames)
    pts = all_pts[in_range]
    bin_idx = bin_idx[in_range]

    if pts.shape[0]:
        order = np.argsort(bin_idx, kind="stable")
        pts = pts[order]
        bin_idx = bin_idx[order]
        edges = np.searchsorted(bin_idx, np.arange(target_frames + 1))
        for i in range(target_frames):
            bins[i] = pts[edges[i]:edges[i + 1]]
    return bins


def split_into_windows(
    sensor_frames: dict[int, list[np.ndarray]],
    *,
    target_frames: int,
    fps: float,
    stride_frames: int | None = None,
) -> list[list[np.ndarray]]:
    """Split a sample into one or more fixed-length windows.

    For samples that fit in a single window: returns one window (padded with
    empty bins if short). For longer samples: slides the window with a stride
    of ``stride_frames`` and yields multiple windows.
    """
    if stride_frames is None:
        stride_frames = target_frames  # non-overlapping by default

    parts = [np.concatenate(frames, axis=0)
             for frames in sensor_frames.values()
             if frames and any(f.shape[0] for f in frames)]
    if not parts:
        return [[np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]]
    all_pts = np.concatenate(parts, axis=0)
    if all_pts.shape[0] == 0:
        return [[np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]]

    t_min, t_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    duration = t_max - t_min
    bin_width = 1.0 / fps
    total_frames = max(1, int(np.ceil(duration / bin_width)))

    if total_frames <= target_frames:
        return [fuse_sensors_by_time(sensor_frames, target_frames=target_frames, fps=fps)]

    # Re-bin the whole sample onto a uniform grid first, then slice into
    # windows. This is cheaper than re-fusing per window.
    full_grid = _full_uniform_grid(all_pts, t_min, total_frames, bin_width)
    windows: list[list[np.ndarray]] = []
    for start in range(0, total_frames - target_frames + 1, stride_frames):
        windows.append(full_grid[start:start + target_frames])
    # Always include the tail window if the slide didn't reach the end
    last_start = total_frames - target_frames
    if last_start > 0 and (windows == [] or windows[-1] is not full_grid[last_start:last_start + target_frames]):
        tail = full_grid[last_start:last_start + target_frames]
        if not windows or not _windows_equal(windows[-1], tail):
            windows.append(tail)
    return windows


def _full_uniform_grid(
    all_pts: np.ndarray,
    t_min: float,
    total_frames: int,
    bin_width: float,
) -> list[np.ndarray]:
    bins: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float64) for _ in range(total_frames)]
    rel_t = all_pts[:, 0] - t_min
    bin_idx = np.floor(rel_t / bin_width).astype(np.int64)
    in_range = (bin_idx >= 0) & (bin_idx < total_frames)
    pts = all_pts[in_range]
    bin_idx = bin_idx[in_range]
    if pts.shape[0]:
        order = np.argsort(bin_idx, kind="stable")
        pts = pts[order]
        bin_idx = bin_idx[order]
        edges = np.searchsorted(bin_idx, np.arange(total_frames + 1))
        for i in range(total_frames):
            bins[i] = pts[edges[i]:edges[i + 1]]
    return bins


def _windows_equal(a: Iterable[np.ndarray], b: Iterable[np.ndarray]) -> bool:
    a, b = list(a), list(b)
    if len(a) != len(b):
        return False
    for fa, fb in zip(a, b):
        if fa.shape != fb.shape:
            return False
    return True
