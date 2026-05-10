"""Quick sanity check on a single .pkl: shape, dtype, density range, fps.

Usage::

    python -m scripts.inspect_pkl                # default path
    python -m scripts.inspect_pkl --path PATH    # custom path
"""

from __future__ import annotations

import argparse

import numpy as np

from rfbc.config import RADAR_ROOT
from rfbc.data.loader import load_sensor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        default=str(RADAR_ROOT / "C1" / "U01" / "M01" / "01" / "5.pkl"),
        help="Path to a sensor .pkl file.",
    )
    args = p.parse_args()

    frames = load_sensor(args.path)
    if not frames:
        print(f"{args.path}: no usable frames")
        return

    npts = np.array([f.shape[0] for f in frames])
    ts = np.array([f[0, 0] for f in frames])
    all_pts = np.concatenate(frames, axis=0)

    print(f"file: {args.path}")
    print(f"frames: {len(frames)} (mean {npts.mean():.2f} pts, max {npts.max()} pts)")
    print(f"duration: {ts[-1] - ts[0]:.2f} s -> approx fps {len(frames) / (ts[-1] - ts[0]):.2f}")
    print(f"x  range: {all_pts[:, 1].min():.3f} .. {all_pts[:, 1].max():.3f}")
    print(f"y  range: {all_pts[:, 2].min():.3f} .. {all_pts[:, 2].max():.3f}")
    print(f"z  range: {all_pts[:, 3].min():.3f} .. {all_pts[:, 3].max():.3f}")
    print(f"density: {all_pts[:, 4].min():.3f} .. {all_pts[:, 4].max():.3f}")


if __name__ == "__main__":
    main()
