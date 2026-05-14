"""Diagnostic visualiser for raw world-frame points BEFORE DBSCAN.

Use this when the regular BEV visualiser shows something suspicious (empty,
off-centre, or in the wrong place). It bypasses clutter suppression so you
can see the actual point cloud the radars produced, then compares the raw
cloud to the DBSCAN-filtered version side by side.

Renders four views:

  1. Per-sensor scatter — every detection in world coords, colour-coded by
     which sensor caught it. Sensor positions shown as stars. Tells you
     "are any sensors missing? are all sensors contributing? does each
     sensor's points land where it geometrically should?".

  2. Trajectory scatter — same points colour-coded by time. Early-window
     points blue, late-window points red. If the person is moving (walking,
     running), the points should trace a clear path. If they're stationary,
     the colour mix is more uniform.

  3. Raw vs DBSCAN BEV — two heatmaps side by side. Left: BEV histogram of
     every world-frame point, no clutter step. Right: BEV after DBSCAN's
     largest-cluster filter. The diff is what got thrown away.

  4. Per-sensor counts — text panel listing how many points each sensor
     contributed, total points, points after DBSCAN.

Usage
-----
    uv run python -m scripts.visualise_raw --campaign C2 --subject U04 --cls A01 --rep 01
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from rfbc.config import DEFAULT_CONFIG, GROUND_TABLE2_XYZ, RADAR_ROOT, PipelineConfig
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.transforms import transform_sample


def _add_recording_area(ax, alpha: float = 0.8) -> None:
    """2 m (x) x 6 m (y) recording area — long axis is Y."""
    rect = Rectangle((-1.0, -3.0), 2.0, 6.0,
                     linewidth=1.2, edgecolor="white",
                     linestyle="--", facecolor="none", alpha=alpha)
    ax.add_patch(rect)


def _add_sensor_positions(ax, marker_color: str = "white") -> None:
    """Mark each ground sensor's world-frame position with a star."""
    for sidx, (x, y, z) in GROUND_TABLE2_XYZ.items():
        ax.plot(x, y, marker="*", color=marker_color, markersize=12,
                markeredgecolor="black", markeredgewidth=0.7)
        ax.annotate(f"s{sidx}", xy=(x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="white")


def _bin_to_bev(points_xy: np.ndarray, extent: float, grid_size: int) -> np.ndarray:
    """Simple 2D histogram of (x, y) points into a grid_size x grid_size grid."""
    if points_xy.shape[0] == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    H, _, _ = np.histogram2d(
        points_xy[:, 1], points_xy[:, 0],
        bins=grid_size, range=[[-extent, extent], [-extent, extent]],
    )
    return H.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="C2")
    p.add_argument("--subject", default="U04")
    p.add_argument("--cls", default="A01")
    p.add_argument("--rep", default="01")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = DEFAULT_CONFIG
    sample = SampleId(campaign=args.campaign, subject=args.subject,
                      cls=args.cls, repetition=args.rep)
    print(f"sample: {sample}", flush=True)

    # Verify the sample dir exists.
    sample_dir = RADAR_ROOT / sample.campaign / sample.subject / sample.cls / sample.repetition
    if not sample_dir.is_dir():
        raise SystemExit(f"sample directory does not exist: {sample_dir}")

    # 1. Load raw .pkl files for every selected sensor.
    sensor_indices = cfg.selected_sensor_indices()
    raw = load_sample_sensors(sample, sensor_indices, skip_missing=True)
    missing = [s for s in sensor_indices if s not in raw]
    print(f"loaded {len(raw)}/{len(sensor_indices)} sensors  missing={missing}", flush=True)

    # 2. Apply world-frame transform.
    world = transform_sample(raw)

    # 3. Build a flat per-point table: (timestamp, x, y, z, density, sensor_idx).
    flat_rows = []
    per_sensor_counts: dict[int, int] = {}
    for sidx, frames in world.items():
        for frame in frames:
            if frame.shape[0] == 0:
                continue
            arr = np.zeros((frame.shape[0], 6), dtype=np.float64)
            arr[:, :5] = frame
            arr[:, 5] = sidx
            flat_rows.append(arr)
        per_sensor_counts[sidx] = sum(f.shape[0] for f in frames)

    if not flat_rows:
        raise SystemExit("no points in this sample (all sensors empty)")

    points = np.concatenate(flat_rows, axis=0)   # (N, 6)
    n_total = points.shape[0]
    print(f"total world-frame points: {n_total}", flush=True)

    # 4. Apply DBSCAN (only here, for the comparison panel).
    from rfbc.data.clutter import suppress_clutter
    clean = suppress_clutter(world, eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    clean_flat = []
    for sidx, frames in clean.items():
        for frame in frames:
            if frame.shape[0] == 0:
                continue
            clean_flat.append(frame)
    points_clean = (np.concatenate(clean_flat, axis=0)
                    if clean_flat else np.zeros((0, 5)))
    n_clean = points_clean.shape[0]
    print(f"after DBSCAN: {n_clean} points ({100 * n_clean / max(n_total, 1):.1f}%)", flush=True)

    # 5. Plot.
    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(
        f"Raw world-frame inspection — {sample.campaign}/{sample.subject}/"
        f"{sample.cls}/{sample.repetition}",
        fontsize=12, y=0.985,
    )
    gs = GridSpec(nrows=3, ncols=2, figure=fig,
                  height_ratios=[1.3, 1.3, 0.7],
                  hspace=0.32, wspace=0.22,
                  left=0.06, right=0.97, top=0.93, bottom=0.05)

    # --- panel 1: per-sensor scatter ---
    ax1 = fig.add_subplot(gs[0, 0])
    cmap_sensors = plt.get_cmap("tab10")
    for i, sidx in enumerate(sorted(per_sensor_counts)):
        mask = points[:, 5] == sidx
        if not mask.any():
            continue
        ax1.scatter(points[mask, 1], points[mask, 2],
                    s=6, alpha=0.5, color=cmap_sensors(i % 10),
                    label=f"s{sidx}  ({per_sensor_counts[sidx]} pts)")
    _add_recording_area(ax1)
    _add_sensor_positions(ax1)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Per-sensor scatter (raw, no DBSCAN)", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.set_facecolor("#111")

    # --- panel 2: per-time scatter ---
    ax2 = fig.add_subplot(gs[0, 1])
    if n_total > 0:
        t_norm = (points[:, 0] - points[:, 0].min())
        if t_norm.max() > 0:
            t_norm = t_norm / t_norm.max()
        sc = ax2.scatter(points[:, 1], points[:, 2], c=t_norm,
                         cmap="coolwarm", s=6, alpha=0.6)
        fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04,
                     label="time (normalised 0=start .. 1=end)")
    _add_recording_area(ax2)
    _add_sensor_positions(ax2)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Trajectory scatter (colour = time)", fontsize=10)
    ax2.set_facecolor("#111")

    # --- panel 3: raw BEV histogram ---
    ax3 = fig.add_subplot(gs[1, 0])
    raw_bev = _bin_to_bev(points[:, 1:3], cfg.grid_extent_m, cfg.grid_size)
    im3 = ax3.imshow(raw_bev,
                     extent=[-cfg.grid_extent_m, cfg.grid_extent_m,
                             -cfg.grid_extent_m, cfg.grid_extent_m],
                     origin="lower", cmap="inferno", vmin=0)
    _add_recording_area(ax3)
    _add_sensor_positions(ax3)
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title(f"BEV histogram — raw ({n_total} pts)", fontsize=10)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # --- panel 4: DBSCAN-filtered BEV histogram ---
    ax4 = fig.add_subplot(gs[1, 1])
    clean_bev = _bin_to_bev(points_clean[:, 1:3], cfg.grid_extent_m, cfg.grid_size)
    im4 = ax4.imshow(clean_bev,
                     extent=[-cfg.grid_extent_m, cfg.grid_extent_m,
                             -cfg.grid_extent_m, cfg.grid_extent_m],
                     origin="lower", cmap="inferno", vmin=0)
    _add_recording_area(ax4)
    _add_sensor_positions(ax4)
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    ax4.set_aspect("equal")
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_title(f"BEV histogram — after DBSCAN ({n_clean} pts)", fontsize=10)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # --- panel 5: stats text ---
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    duration = float(points[:, 0].max() - points[:, 0].min()) if n_total else 0.0
    sensor_lines = []
    for sidx in sorted(per_sensor_counts):
        bar_len = int(per_sensor_counts[sidx] / max(per_sensor_counts.values(), default=1) * 50)
        bar = "#" * bar_len
        sensor_lines.append(f"  sensor {sidx:>2} (yaw {_yaw_for(sidx):+6.1f} deg): "
                            f"{per_sensor_counts[sidx]:>5} pts  {bar}")
    sensor_block = "\n".join(sensor_lines)
    stats = (
        f"Sample: {sample}    duration: {duration:.2f} s\n"
        f"Raw point count:           {n_total}\n"
        f"Points after DBSCAN:       {n_clean}  ({100 * n_clean / max(n_total, 1):.1f}% survived)\n"
        f"Missing sensors:           {missing if missing else 'none'}\n\n"
        f"Per-sensor contribution (raw):\n{sensor_block}"
    )
    ax5.text(0.0, 1.0, stats, ha="left", va="top",
             fontsize=9, family="monospace")

    # save
    if args.out is None:
        out_dir = Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"raw_{sample.campaign}_{sample.subject}_{sample.cls}_{sample.repetition}.png"
        )
    else:
        out_path = Path(args.out)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}", flush=True)


def _yaw_for(sidx: int) -> float:
    """Look up the inferred yaw for diagnostics."""
    from rfbc.config import ground_yaw_deg
    try:
        return ground_yaw_deg(sidx)
    except KeyError:
        return float("nan")


if __name__ == "__main__":
    main()
