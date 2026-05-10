"""Sensor-local → world-frame coordinate transform.

Each radar reports detections in its own local frame, with positive X along
boresight. To fuse multiple radars we need a single shared frame. We use the
recording-area centre as the origin (paper convention).

Per sensor, the transform is:

    world_xyz = R(yaw) @ local_xyz + translation

where ``translation`` is the sensor's position in world coordinates (Table 2
of Zuo et al. 2025) and ``yaw`` is the rotation that maps the sensor's
local +X axis onto the direction from the sensor toward the origin.

We currently only handle ground sensors. Ceiling sensors face downward and
need a different transform (a 90° pitch on top of the in-plane yaw); that is
deferred until we want to use ceiling data.
"""

from __future__ import annotations

import numpy as np

from rfbc.config import GROUND_TABLE2_XYZ, ground_yaw_deg


def _rotation_z(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def ground_sensor_transform(sensor_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, t)`` for a ground sensor: 3x3 rotation matrix + translation.

    The transform applied to a (3,) local point is ``R @ p + t``.
    """
    if sensor_index not in GROUND_TABLE2_XYZ:
        raise KeyError(
            f"sensor_index {sensor_index} is not a known ground sensor "
            f"(expected one of {sorted(GROUND_TABLE2_XYZ)})."
        )
    yaw = ground_yaw_deg(sensor_index)
    R = _rotation_z(yaw)
    t = np.array(GROUND_TABLE2_XYZ[sensor_index], dtype=np.float64)
    return R, t


def to_world_frame(frames: list[np.ndarray], sensor_index: int) -> list[np.ndarray]:
    """Apply the per-sensor transform to every frame.

    Each input frame is ``(N, 5)`` with columns
    ``[timestamp, x, y, z, density]`` in the sensor-local frame.
    Output frames have the same shape, with x/y/z replaced by world-frame
    values. Timestamp and density are preserved.
    """
    if sensor_index in GROUND_TABLE2_XYZ:
        R, t = ground_sensor_transform(sensor_index)
    else:
        raise NotImplementedError(
            f"World-frame transform for sensor {sensor_index} (likely ceiling) "
            "is not implemented yet. Configure sensor_set='ground' for now."
        )

    out: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[0] == 0:
            out.append(frame.copy())
            continue
        local_xyz = frame[:, 1:4].T          # (3, N)
        world_xyz = (R @ local_xyz).T + t    # (N, 3)
        new = frame.copy()
        new[:, 1:4] = world_xyz
        out.append(new)
    return out


def transform_sample(
    sensor_frames: dict[int, list[np.ndarray]],
) -> dict[int, list[np.ndarray]]:
    """Apply ``to_world_frame`` to every sensor in a sample."""
    return {sidx: to_world_frame(frames, sidx) for sidx, frames in sensor_frames.items()}
