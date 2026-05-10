"""Central configuration for the RF-Behavior preprocessing pipeline.

All paths, sensor coordinates, and pipeline hyperparameters live here so that
the rest of the codebase does not hardcode anything that we might want to
ablate later. Everything is dataclasses; no global mutable state.

Notes
-----
Ground sensor world-frame coordinates come from Table 2 of the dataset paper
(Zuo et al. 2025, arXiv:2511.06020), measured with the recording-area infrared
cameras and **not** the idealised circle used by the visualisation scripts in
``dataset/scripts/scripts/Radar/run_ground.py``.

The rotation angle of each ground sensor is **inferred** by assuming each
radar's boresight (local +X) points at the recording-area origin, which
matches the paper's setup description (8 nodes on the ground arranged around
the 6 m × 2 m rectangle, looking inward). If we later get explicit per-sensor
yaw measurements, replace ``ground_yaw_deg`` with those values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# code/ lives next to dataset/. Resolve dataset path relative to this file so
# the code is portable across machines.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT: Path = _REPO_ROOT / "dataset"
RADAR_ROOT: Path = DATASET_ROOT / "Radar"
CACHE_ROOT: Path = _REPO_ROOT / "code" / "cache"
SPLITS_PATH: Path = _REPO_ROOT / "code" / "splits" / "splits.json"


# ---------------------------------------------------------------------------
# Sensor geometry
# ---------------------------------------------------------------------------

# Table 2 of Zuo et al. 2025 — 3D coordinates of the eight ground radars.
# Indices match the .pkl filename (e.g. sensor 5 -> 5.pkl).
GROUND_SENSOR_INDICES: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12)
CEILING_SENSOR_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)

GROUND_TABLE2_XYZ: dict[int, tuple[float, float, float]] = {
    5:  (2.6055, -0.0940, 0.9407),
    6:  (1.8044, -3.4535, 0.8953),
    7:  (-0.2786, -3.1746, 0.9019),
    8:  (-2.6325, -3.0253, 0.8815),
    9:  (-2.9319, -0.1305, 0.8704),
    10: (-1.1499,  3.2323, 0.8784),
    11: (0.0234,   3.1029, 0.9253),
    12: (1.1062,   2.8954, 0.9261),
}


def ground_yaw_deg(sensor_index: int) -> float:
    """Yaw angle (deg) that rotates the sensor's local +X (boresight) onto the
    direction from the sensor toward the recording-area origin.

    The paper's setup has each ground radar pointing inward; this function
    computes that yaw from the Table 2 coordinates. Returned in degrees so it
    matches the convention used in ``run_ground.py``.
    """
    x, y, _ = GROUND_TABLE2_XYZ[sensor_index]
    return float(np.rad2deg(np.arctan2(-y, -x)))


# ---------------------------------------------------------------------------
# Pipeline hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Hyperparameters for the preprocessing pipeline.

    All values are intentionally easy to flip from a single place so that
    ablations (ceiling vs ground, grid size, window length, etc.) are
    one-line changes.
    """

    # --- sensor selection ---
    # "ground", "ceiling", or "all"
    sensor_set: str = "ground"

    # --- DBSCAN clutter step (Zuo et al. 2025 step 1) ---
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 3

    # --- temporal windowing ---
    fps: float = 27.0          # nominal radar frame rate
    window_seconds: float = 2.0
    window_stride_seconds: float = 1.0  # only matters for samples > window_seconds
    target_frames: int = 54    # window_seconds * fps, rounded — frames per window

    # --- BEV grid ---
    grid_size: int = 64        # H == W
    grid_extent_m: float = 4.0 # half-width in metres -> grid covers [-4, +4] in x and y
    z_bands: tuple[float, ...] = (-0.5, 0.5, 1.2, 2.2)  # 3 bands defined by 4 edges
    # Per (cell, z-band, frame) features: count, mean density, max density.
    feat_per_cell: int = 3

    # --- caching ---
    use_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: CACHE_ROOT)

    # --- campaigns to use ---
    # C1 (gestures, micro candidates), C2 (activities, macro candidates).
    # C3 sentiment excluded by default (sample durations 50x longer).
    campaigns: tuple[str, ...] = ("C1", "C2")

    @property
    def num_z_bands(self) -> int:
        return len(self.z_bands) - 1

    def selected_sensor_indices(self) -> tuple[int, ...]:
        if self.sensor_set == "ground":
            return GROUND_SENSOR_INDICES
        if self.sensor_set == "ceiling":
            return CEILING_SENSOR_INDICES
        if self.sensor_set == "all":
            return GROUND_SENSOR_INDICES + CEILING_SENSOR_INDICES
        raise ValueError(f"unknown sensor_set: {self.sensor_set!r}")

    def channels_per_window(self) -> int:
        """Number of input channels the BEV tensor produces per window."""
        return self.target_frames * self.num_z_bands * self.feat_per_cell


# A default singleton for convenience. Code that wants a different config
# should construct its own PipelineConfig instance and pass it explicitly.
DEFAULT_CONFIG = PipelineConfig()
