"""Read a single radar sensor's ``.pkl`` file from the RF-Behavior dataset.

Each ``.pkl`` is a Python list of 2D NumPy arrays. Each row is one detection
in the format ``[timestamp, x, y, z, density]`` (Zuo et al. 2025).
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rfbc.config import RADAR_ROOT


# The RF-Behavior .pkl files were created under an older NumPy where
# ``align=0`` was an acceptable dtype kwarg. NumPy 2.4+ raises a
# VisibleDeprecationWarning every time we unpickle one. The files themselves
# are still well-formed — once unpickled, the resulting arrays are normal —
# so we silence that one specific warning at load time. We narrow the filter
# to just this message so genuinely useful deprecation warnings still surface.
_PKL_ALIGN_WARNING_MSG = r"dtype\(\): align should be passed"


@dataclass
class SampleId:
    """Identifies one recording instance in the dataset.

    A "sample" is one (campaign, subject, class, repetition) tuple. Each
    sample has up to 13 ``.pkl`` files (one per radar sensor).
    """

    campaign: str   # "C1" / "C2" / "C3"
    subject: str    # e.g. "U01"
    cls: str        # e.g. "M01" / "A03" / "E04" — folder name on disk
    repetition: str # e.g. "01"

    def sensor_path(self, sensor_index: int, root: Path = RADAR_ROOT) -> Path:
        return root / self.campaign / self.subject / self.cls / self.repetition / f"{sensor_index}.pkl"


def load_sensor(path: Path | str) -> list[np.ndarray]:
    """Load one sensor's ``.pkl``, return a list of (N, 5) frame arrays.

    Empty/invalid frames are dropped. If every frame is empty, an empty list
    is returned (the caller should handle this — it does occur for a few
    sensor/sample combinations because of dataset sparsity).
    """
    path = Path(path)
    with path.open("rb") as f, warnings.catch_warnings():
        # Filter on the message text only — the warning class moved between
        # NumPy versions (was np.VisibleDeprecationWarning, now lives at
        # np.exceptions.VisibleDeprecationWarning) so importing it portably
        # is more pain than it's worth.
        warnings.filterwarnings("ignore", message=_PKL_ALIGN_WARNING_MSG)
        data = pickle.load(f)
    cleaned: list[np.ndarray] = []
    for frame in data:
        if not isinstance(frame, np.ndarray):
            continue
        if frame.size == 0:
            continue
        if frame.ndim != 2 or frame.shape[1] != 5:
            # Defensive — paper says (N, 5) always; log loudly if not.
            raise ValueError(
                f"Unexpected frame shape {frame.shape} in {path} — "
                "expected (N, 5) [timestamp, x, y, z, density]."
            )
        cleaned.append(frame.astype(np.float64, copy=False))
    return cleaned


def load_sample_sensors(
    sample: SampleId,
    sensor_indices: Sequence[int],
    root: Path = RADAR_ROOT,
    skip_missing: bool = True,
) -> dict[int, list[np.ndarray]]:
    """Load all selected sensors for one sample.

    Parameters
    ----------
    sample
        Identifies the (campaign, subject, class, repetition).
    sensor_indices
        Iterable of sensor indices to load (e.g. 5..12 for ground).
    root
        Dataset radar root. Defaults to the project's ``dataset/Radar``.
    skip_missing
        If True, sensors whose .pkl is missing on disk are silently skipped.
        Set False to surface holes in the data.

    Returns
    -------
    dict mapping sensor_index -> list of frame arrays.
    """
    out: dict[int, list[np.ndarray]] = {}
    for idx in sensor_indices:
        p = sample.sensor_path(idx, root=root)
        if not p.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(p)
        out[idx] = load_sensor(p)
    return out
