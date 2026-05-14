"""Visualise the BEV tensor for one sample.

Renders three views designed to confirm the preprocessing pipeline is
producing person-shaped data:

  1. Spatial overview — point counts summed across time and z-bands.
     The non-zero blob should sit near the origin (where the person stood).
  2. Per-z-band split — same spatial view, but separated into low / mid /
     high vertical bands. Gestures with raised hands should light up the
     upper band; walking gestures should light up the lower band.
  3. Per-frame montage — count summed across z-bands, shown for a handful
     of timestamps spaced through the window. Confirms the spatial blob
     evolves coherently over time.

The 6 m x 2 m recording area is overlaid as a dashed rectangle so you can
see whether the points are inside the room or escaping into clutter
territory.

Loads the cached BEV if available; otherwise computes the tensor fresh.
Saves a PNG to ``code/outputs/bev_<campaign>_<subject>_<class>_<rep>_w<window>.png``.

Usage
-----
    uv run python -m scripts.visualise_bev
    uv run python -m scripts.visualise_bev --campaign C2 --cls M01
    uv run python -m scripts.visualise_bev --campaign C1 --subject U01 --cls M07 --rep 02 --window 0
"""

from __future__ import annotations

# Force a non-interactive matplotlib backend BEFORE importing pyplot.
# This avoids issues on systems without a display, and means the script
# behaves identically whether run from PowerShell, an IDE, or a remote
# shell.
import matplotlib
matplotlib.use("Agg")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from rfbc.config import DEFAULT_CONFIG, PipelineConfig
from rfbc.data.loader import SampleId


def _config_hash(cfg: PipelineConfig) -> str:
    """Mirror of dataset._config_hash so we don't import dataset.py
    (which transitively pulls in sklearn). Keep these two in sync.
    """
    keys = ("sensor_set", "dbscan_eps", "dbscan_min_samples", "fps",
            "window_seconds", "window_stride_seconds", "target_frames",
            "grid_size", "grid_extent_m", "z_bands", "feat_per_cell")
    payload = {k: getattr(cfg, k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# ---------------------------------------------------------------------------
# BEV loading: cache first, compute as fallback
# ---------------------------------------------------------------------------


def load_window_from_cache(sample: SampleId, window: int, cfg: PipelineConfig) -> np.ndarray | None:
    """Return one cached BEV window, or None if cache miss."""
    cache_root = Path(cfg.cache_dir) / _config_hash(cfg)
    cache_path = cache_root / sample.campaign / sample.subject / sample.cls / f"{sample.repetition}.npz"
    if not cache_path.exists():
        return None
    with np.load(cache_path) as f:
        keys = sorted(f.files, key=lambda s: int(s.split("_")[1]))
        if window >= len(keys):
            return None
        return f[keys[window]]


def compute_window(sample: SampleId, window: int, cfg: PipelineConfig) -> np.ndarray:
    """Run the full preprocessing pipeline and return one window.

    Deferred imports: this path needs sklearn (DBSCAN), so we only import
    the pipeline modules when actually computing fresh. Visualising a
    cached sample doesn't trigger this.
    """
    from rfbc.data.bev import points_to_bev
    from rfbc.data.clutter import suppress_clutter
    from rfbc.data.loader import load_sample_sensors
    from rfbc.data.transforms import transform_sample
    from rfbc.data.windowing import split_into_windows

    raw = load_sample_sensors(sample, cfg.selected_sensor_indices())
    world = transform_sample(raw)
    clean = suppress_clutter(world, eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    stride = max(1, int(round(cfg.window_stride_seconds * cfg.fps)))
    windows = split_into_windows(
        clean, target_frames=cfg.target_frames, fps=cfg.fps, stride_frames=stride,
    )
    if window >= len(windows):
        raise IndexError(f"sample {sample} only produced {len(windows)} window(s); index {window} out of range")
    return points_to_bev(windows[window], cfg)


def get_window(sample: SampleId, window: int, cfg: PipelineConfig) -> tuple[np.ndarray, str]:
    """Return (bev_tensor, source_label) — cache when possible, compute otherwise."""
    cached = load_window_from_cache(sample, window, cfg)
    if cached is not None:
        return cached, "cache"
    return compute_window(sample, window, cfg), "computed fresh"


def unflatten_channels(bev: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Reshape (C, H, W) -> (T, Z, F, H, W) following bev.py's flattening."""
    return bev.reshape(cfg.target_frames, cfg.num_z_bands, cfg.feat_per_cell, cfg.grid_size, cfg.grid_size)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _add_recording_area(ax) -> None:
    """Overlay the 6 m x 2 m recording rectangle (dashed)."""
    rect = Rectangle(
        (-3.0, -1.0), 6.0, 2.0,
        linewidth=1.2, edgecolor="white", linestyle="--", facecolor="none",
        alpha=0.7,
    )
    ax.add_patch(rect)


def _imshow_bev(ax, grid: np.ndarray, cfg: PipelineConfig, title: str = "",
                vmax=None, show_axes_labels: bool = True):
    """Display one (H, W) BEV slice with axes in metres. Returns the image handle."""
    e = cfg.grid_extent_m
    im = ax.imshow(
        grid, extent=[-e, e, -e, e], origin="lower",
        cmap="inferno", vmin=0, vmax=vmax,
    )
    _add_recording_area(ax)
    if show_axes_labels:
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    return im


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------


def make_figure(bev_5d: np.ndarray, sample: SampleId, window: int,
                feat: int, cfg: PipelineConfig, montage_frames: int = 9) -> plt.Figure:
    """Build the visualisation figure with a single unified GridSpec.

    Uses one GridSpec for everything so subplots don't fight each other
    when the layout is tight. The grid has `montage_frames` columns so the
    bottom row's per-frame thumbnails fit cleanly.
    """
    spatial = bev_5d[:, :, feat].sum(axis=(0, 1))           # (H, W)
    per_band = bev_5d[:, :, feat].sum(axis=0)               # (Z, H, W)
    per_frame = bev_5d[:, :, feat].sum(axis=1)              # (T, H, W)
    montage_idx = np.linspace(0, cfg.target_frames - 1, montage_frames).astype(int)
    per_frame_montage = per_frame[montage_idx]

    feature_name = {0: "point count", 1: "mean density", 2: "max density"}.get(feat, f"feature {feat}")
    band_titles = ["z-band 0: low (-0.5 .. 0.5 m)",
                   "z-band 1: mid (0.5 .. 1.2 m)",
                   "z-band 2: high (1.2 .. 2.2 m)"]

    fig = plt.figure(figsize=(15, 11), constrained_layout=False)
    fig.suptitle(
        f"BEV — {sample.campaign}/{sample.subject}/{sample.cls}/{sample.repetition}  "
        f"window {window}  feature: {feature_name}",
        fontsize=12, y=0.985,
    )

    gs = GridSpec(
        nrows=3, ncols=montage_frames, figure=fig,
        height_ratios=[1.5, 1.2, 0.7],
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )

    # row 1 left: spatial overview
    ax_overview = fig.add_subplot(gs[0, : montage_frames // 2])
    im = _imshow_bev(ax_overview, spatial, cfg,
                     title="Spatial overview (summed across time and z-bands)")
    fig.colorbar(im, ax=ax_overview, fraction=0.046, pad=0.04)

    # row 1 right: stats text panel
    ax_stats = fig.add_subplot(gs[0, montage_frames // 2:])
    ax_stats.axis("off")
    bev_flat_nonzero = int((bev_5d > 0).sum())
    bev_flat_total = int(bev_5d.size)
    stats_text = (
        f"Sample: {sample}\n"
        f"Window: {window}\n"
        f"Window length: {cfg.target_frames} frames at {cfg.fps} Hz = "
        f"{cfg.target_frames / cfg.fps:.2f} s\n\n"
        f"Tensor (C, H, W): "
        f"({cfg.target_frames * cfg.num_z_bands * cfg.feat_per_cell}, "
        f"{cfg.grid_size}, {cfg.grid_size})\n"
        f"  = {cfg.target_frames} frames x {cfg.num_z_bands} z-bands "
        f"x {cfg.feat_per_cell} features/cell\n"
        f"Grid: {cfg.grid_size}x{cfg.grid_size} cells over "
        f"+/-{cfg.grid_extent_m} m\n"
        f"      ({2*cfg.grid_extent_m / cfg.grid_size * 100:.1f} cm per cell)\n\n"
        f"Feature visualised: {feature_name}\n"
        f"  Sum across this feature: {bev_5d[:, :, feat].sum():.1f}\n"
        f"  Spatial-overview max:    {spatial.max():.1f}\n\n"
        f"Non-zero cells (all channels): {bev_flat_nonzero}\n"
        f"  of total cells:              {bev_flat_total}\n"
        f"  fraction:                    "
        f"{100 * bev_flat_nonzero / bev_flat_total:.3f}%\n\n"
        f"Dashed rectangle: 6 m x 2 m recording area"
    )
    ax_stats.text(0.0, 1.0, stats_text, ha="left", va="top",
                  fontsize=9, family="monospace")

    # row 2: per-z-band split (3 panels)
    band_vmax = float(per_band.max()) if per_band.max() > 0 else 1.0
    third = montage_frames // 3
    for b in range(cfg.num_z_bands):
        col_lo = b * third
        col_hi = (b + 1) * third if b < cfg.num_z_bands - 1 else montage_frames
        ax = fig.add_subplot(gs[1, col_lo:col_hi])
        title = band_titles[b] if b < 3 else f"z-band {b}"
        _imshow_bev(ax, per_band[b], cfg, title=title, vmax=band_vmax)

    # row 3: per-frame montage
    montage_vmax = float(per_frame_montage.max()) if per_frame_montage.max() > 0 else 1.0
    for i, t_idx in enumerate(montage_idx):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(per_frame_montage[i],
                  extent=[-cfg.grid_extent_m, cfg.grid_extent_m,
                          -cfg.grid_extent_m, cfg.grid_extent_m],
                  origin="lower", cmap="inferno", vmin=0, vmax=montage_vmax)
        _add_recording_area(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={int(t_idx)} ({t_idx/cfg.fps:.2f}s)", fontsize=7)
        if i == 0:
            ax.set_ylabel("montage", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="C1")
    p.add_argument("--subject", default="U01")
    p.add_argument("--cls", default="M01", help="Class folder name on disk")
    p.add_argument("--rep", default="01", help="Repetition folder")
    p.add_a