# RF-Behavior compression study — preprocessing & baseline

Code for the project *Evaluating the Class-wise Impact of Model Compression in
RF-Based Human Activity Recognition* (Max van Zanten, University of Twente).

This repository holds the preprocessing pipeline and (eventually) the training,
compression, and evaluation code. The raw dataset lives outside this repo
under `../dataset/` and is **not** version-controlled.

## Layout

```
code/
  rfbc/                  importable package
    config.py            paths, sensor coords, BEV/window params
    data/
      loader.py          read a single sensor .pkl
      clutter.py         DBSCAN clutter suppression
      transforms.py      sensor → world-frame coordinate transform
      windowing.py       fixed-length temporal windows
      bev.py             point cloud → (C, H, W) BEV tensor
      splits.py          subject-independent split helpers
      dataset.py         PyTorch Dataset / DataLoader
    models/
      stub_cnn.py        throwaway CNN for the smoke test
  scripts/
    inspect_pkl.py       sanity-check one .pkl from disk
    build_split.py       generate splits/splits.json
    smoke_test.py        end-to-end pipeline run on a tiny subset
  splits/
    splits.json          locked train/val/test subject lists
  cache/                 (git-ignored) cached BEV tensors
  pyproject.toml
```

## Setup

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/):

```bash
cd code
uv sync
```

For CUDA, install the matching torch wheel from the official index, e.g.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Smoke test

```bash
cd code
uv run python -m scripts.build_split        # generate splits/splits.json
uv run python -m scripts.smoke_test         # end-to-end pipeline
```

The smoke test is intentionally small: a handful of subjects/classes, one epoch
of training on a stub CNN, per-class accuracy printed at the end. The point is
to surface plumbing failures (paths, shapes, dependencies, OOM) before the
real baseline work begins.

## Dataset assumptions

- Radar `.pkl` files at
  `../dataset/Radar/C{campaign}/U{user}/M{class}/{rep}/{sensor_index}.pkl`.
- Sensor indices 0–4 are ceiling, 5–12 are ground.
- Each `.pkl` is a Python list of `(N, 5)` numpy arrays per frame, columns
  `[timestamp, x, y, z, density]`.
- Ground-radar world-frame coordinates come from Table 2 of the dataset paper
  (Zuo et al. 2025).
- Default sensor set: ground-only (indices 5–12).
- Default campaigns: C1 (gestures) + C2 (activities). C3 (sentiment) is
  excluded by default because samples are ~50× longer.
