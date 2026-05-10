"""PyTorch ``Dataset`` over the preprocessed RF-Behavior data.

Items are *windows*, not whole samples. Each window is one (C, H, W) BEV
tensor + an integer class label. A short campaign 1 gesture produces one
window; a long campaign 2 activity may produce several.

Pipeline per window (executed on first access, then cached on disk):

    raw .pkl files
        → load_sensor() per sensor
        → to_world_frame() per sensor
        → suppress_clutter() across all sensors
        → split_into_windows() (fused, fixed length)
        → points_to_bev() per window

The cache is keyed by (campaign, subject, class, repetition, window_idx) plus
a hash of the relevant pipeline config so that changing window length or grid
size invalidates old cache entries automatically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rfbc.config import DEFAULT_CONFIG, RADAR_ROOT, PipelineConfig
from rfbc.data.bev import points_to_bev
from rfbc.data.clutter import suppress_clutter
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.splits import Split
from rfbc.data.transforms import transform_sample
from rfbc.data.windowing import split_into_windows


# ---------------------------------------------------------------------------
# Index entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowIndex:
    sample: SampleId
    window_idx: int
    label: int  # integer class id (assigned by the dataset, not the disk name)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def _config_hash(cfg: PipelineConfig) -> str:
    """Stable hash over the pipeline-affecting fields of the config.

    Whatever changes the contents of a window goes here. Path-only fields
    (cache_dir) are deliberately excluded.
    """
    keys = (
        "sensor_set", "dbscan_eps", "dbscan_min_samples", "fps",
        "window_seconds", "window_stride_seconds", "target_frames",
        "grid_size", "grid_extent_m", "z_bands", "feat_per_cell",
    )
    payload = {k: getattr(cfg, k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SubjectIndependentRadarDataset(Dataset):
    """Indexable dataset of BEV windows.

    Parameters
    ----------
    split
        Loaded ``Split`` object (from ``rfbc.data.splits.load_split``).
    fold
        One of ``"train"``, ``"val"``, ``"test"``.
    cfg
        Pipeline config. Defaults to ``DEFAULT_CONFIG``.
    radar_root
        Override the dataset root if needed.
    class_filter
        Optional iterable of disk class names (e.g. ``("M01", "M02")``) to
        restrict the dataset to a subset of classes. ``None`` means all
        classes found on disk.
    repetition_filter
        Optional iterable of repetition strings (e.g. ``("01",)``) to limit
        per-sample reps. Useful for tiny smoke-test runs.
    """

    def __init__(
        self,
        split: Split,
        fold: str,
        cfg: PipelineConfig = DEFAULT_CONFIG,
        radar_root: Path = RADAR_ROOT,
        class_filter: tuple[str, ...] | None = None,
        repetition_filter: tuple[str, ...] | None = None,
    ) -> None:
        if fold not in ("train", "val", "test"):
            raise ValueError(f"fold must be train/val/test, got {fold!r}")
        self.cfg = cfg
        self.radar_root = Path(radar_root)
        self.fold = fold
        self.config_hash = _config_hash(cfg)
        self.cache_root = Path(cfg.cache_dir) / self.config_hash
        if cfg.use_cache:
            self.cache_root.mkdir(parents=True, exist_ok=True)

        subjects = {
            "train": split.train,
            "val":   split.val,
            "test":  split.test,
        }[fold]

        # Build the index. We need at least one sensor file per (campaign,
        # subject, class, rep) for the entry to be valid; we don't fully
        # preprocess yet, that happens lazily in __getitem__.
        self.label_map: dict[str, int] = {}
        self.index: list[WindowIndex] = []
        for campaign in split.campaigns:
            for subject in subjects:
                subj_dir = self.radar_root / campaign / subject
                if not subj_dir.is_dir():
                    continue
                for class_dir in sorted(subj_dir.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    cls_name = class_dir.name
                    if class_filter is not None and cls_name not in class_filter:
                        continue
                    label_key = f"{campaign}/{cls_name}"
                    if label_key not in self.label_map:
                        self.label_map[label_key] = len(self.label_map)
                    label = self.label_map[label_key]
                    for rep_dir in sorted(class_dir.iterdir()):
                        if not rep_dir.is_dir():
                            continue
                        rep = rep_dir.name
                        if repetition_filter is not None and rep not in repetition_filter:
                            continue
                        sample = SampleId(
                            campaign=campaign, subject=subject,
                            cls=cls_name, repetition=rep,
                        )
                        # We don't know the number of windows per sample
                        # without preprocessing, but most C1 samples produce
                        # 1 and C2 produce 1-N. Store window 0 as a
                        # placeholder; adjust on first access.
                        self.index.append(WindowIndex(sample=sample, window_idx=0, label=label))

        # Expand the index to cover multi-window samples lazily. We materialise
        # window counts the first time the dataset is iterated; until then,
        # each sample contributes one entry. This keeps __init__ fast.
        self._expanded = False

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.index[idx]
        windows = self._load_or_compute_windows(entry.sample)
        if entry.window_idx >= len(windows):
            # If a long sample turned out to have multiple windows but we only
            # registered window_idx=0 in __init__, expand the index lazily.
            if not self._expanded:
                self._expand_index()
                return self.__getitem__(idx)
            raise IndexError(
                f"window_idx {entry.window_idx} out of range for sample "
                f"{entry.sample} (has {len(windows)} windows)"
            )
        tensor = torch.from_numpy(windows[entry.window_idx]).float()
        return tensor, entry.label

    # ------------------------------------------------------------------
    # Pipeline (preprocessing) — cached on disk
    # ------------------------------------------------------------------

    def _cache_path(self, sample: SampleId) -> Path:
        return self.cache_root / sample.campaign / sample.subject / sample.cls / f"{sample.repetition}.npz"

    def _load_or_compute_windows(self, sample: SampleId) -> list[np.ndarray]:
        if self.cfg.use_cache:
            cache_path = self._cache_path(sample)
            if cache_path.exists():
                with np.load(cache_path) as f:
                    return [f[k] for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]

        windows = self._compute_windows(sample)
        if self.cfg.use_cache:
            cache_path = self._cache_path(sample)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **{f"win_{i}": w for i, w in enumerate(windows)},
            )
        return windows

    def _compute_windows(self, sample: SampleId) -> list[np.ndarray]:
        sensor_indices = self.cfg.selected_sensor_indices()
        sensor_frames = load_sample_sensors(
            sample, sensor_indices, root=self.radar_root, skip_missing=True,
        )
        sensor_frames = transform_sample(sensor_frames)
        sensor_frames = suppress_clutter(
            sensor_frames,
            eps=self.cfg.dbscan_eps,
            min_samples=self.cfg.dbscan_min_samples,
        )
        stride = max(1, int(round(self.cfg.window_stride_seconds * self.cfg.fps)))
        time_windows = split_into_windows(
            sensor_frames,
            target_frames=self.cfg.target_frames,
            fps=self.cfg.fps,
            stride_frames=stride,
        )
        return [points_to_bev(w, self.cfg) for w in time_windows]

    # ------------------------------------------------------------------
    # Multi-window expansion
    # ------------------------------------------------------------------

    def _expand_index(self) -> None:
        """Walk every sample once, expanding multi-window samples.

        Cheap because per-sample windows are cached on the first call;
        this is essentially a no-op after the first epoch when caching is on.
        """
        new_index: list[WindowIndex] = []
        for entry in self.index:
            if entry.window_idx != 0:
                new_index.append(entry)
                continue
            windows = self._load_or_compute_windows(entry.sample)
            for w in range(len(windows)):
                new_index.append(WindowIndex(
                    sample=entry.sample,
                    window_idx=w,
                    label=entry.label,
                ))
        self.index = new_index
        self._expanded = True

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def class_names(self) -> list[str]:
        return [name for name, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]
