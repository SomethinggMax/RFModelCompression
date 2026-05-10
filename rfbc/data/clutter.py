"""DBSCAN clutter suppression — borrowed from Zuo et al. 2025 step 1.

Per-sample logic:

1. Aggregate **world-frame** points from all selected sensors across all
   frames into one big point cloud.
2. Run DBSCAN with eps and min_samples from the config.
3. Keep only the largest cluster (assumed to be "the person").
4. Build a boolean mask per (sensor, frame) marking which detections survive.

The DBSCAN runs in 3D world coordinates. Sensor-local coordinates would not
make sense across sensors, so this step assumes the caller has already applied
``rfbc.data.transforms.to_world_frame`` per sensor.

This step is borrowed essentially unchanged from the dataset paper — the
authors found these hyperparameters work for the RF-Behavior recording
geometry. We may revisit them later if the resulting masks are too aggressive
or too lax.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def largest_cluster_mask(
    points_xyz: np.ndarray,
    eps: float = 1.0,
    min_samples: int = 3,
) -> np.ndarray:
    """Return a boolean mask of points belonging to the largest DBSCAN cluster.

    Parameters
    ----------
    points_xyz
        ``(N, 3)`` array of world-frame points across all sensors and frames.
    eps, min_samples
        DBSCAN hyperparameters — defaults match Zuo et al. 2025.

    Returns
    -------
    ``(N,)`` boolean mask. If DBSCAN finds no clusters (everything noise),
    a mask of all False is returned. The caller should be prepared for that.
    """
    if points_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xyz)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return np.zeros(points_xyz.shape[0], dtype=bool)

    biggest = np.bincount(valid).argmax()
    return labels == biggest


def suppress_clutter(
    sensor_world_frames: dict[int, list[np.ndarray]],
    eps: float = 1.0,
    min_samples: int = 3,
) -> dict[int, list[np.ndarray]]:
    """Apply DBSCAN clutter suppression to a sample's per-sensor frames.

    ``sensor_world_frames`` maps ``sensor_index -> list of (N_i, 5) frames``,
    where columns 1..4 are world-frame x, y, z, density. (Column 0 is
    timestamp; we keep it intact.)

    The function returns a new dict in the same shape, with non-cluster
    detections dropped. Frames that lose all their points become empty
    ``(0, 5)`` arrays — the caller decides whether to drop them.
    """
    if not sensor_world_frames:
        return {}

    # Stack into one big (M, 3) array for DBSCAN, remember provenance.
    parts: list[np.ndarray] = []
    provenance: list[tuple[int, int, int]] = []  # (sensor_idx, frame_idx, npts)
    for sidx, frames in sensor_world_frames.items():
        for fidx, frame in enumerate(frames):
            if frame.shape[0] == 0:
                provenance.append((sidx, fidx, 0))
                continue
            parts.append(frame[:, 1:4])  # x, y, z columns
            provenance.append((sidx, fidx, frame.shape[0]))

    if not parts:
        # Nothing to do — return empty frames as-is.
        return {sidx: [f.copy() for f in frames] for sidx, frames in sensor_world_frames.items()}

    all_pts = np.concatenate(parts, axis=0)
    mask = largest_cluster_mask(all_pts, eps=eps, min_samples=min_samples)

    # Walk the provenance back into per-sensor, per-frame masks.
    out: dict[int, list[np.ndarray]] = {sidx: [] for sidx in sensor_world_frames}
    cursor = 0
    for sidx, fidx, npts in provenance:
        if npts == 0:
            out[sidx].append(np.zeros((0, 5), dtype=np.float64))
            continue
        sub = mask[cursor:cursor + npts]
        cursor += npts
        original = sensor_world_frames[sidx][fidx]
        out[sidx].append(original[sub])
    return out
