"""Subject-independent train/val/test splits.

The split is generated **once** by ``scripts/build_split.py`` and committed to
``code/splits/splits.json``. After that, it is read back from disk by anything
that needs it. The split must never be regenerated mid-experiment — that's
the whole point of a subject-independent protocol.

Layout of ``splits.json``::

    {
      "campaigns": ["C1", "C2"],
      "subjects": {
        "train": ["U01", "U02", ...],
        "val":   ["U10", ...],
        "test":  ["U17", ...]
      },
      "seed": 42
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rfbc.config import RADAR_ROOT, SPLITS_PATH


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    campaigns: tuple[str, ...]
    seed: int

    def all_subjects(self) -> set[str]:
        return set(self.train) | set(self.val) | set(self.test)

    def membership(self, subject: str) -> str | None:
        if subject in self.train:
            return "train"
        if subject in self.val:
            return "val"
        if subject in self.test:
            return "test"
        return None


def discover_subjects(campaigns: Iterable[str], radar_root: Path = RADAR_ROOT) -> dict[str, list[str]]:
    """Return ``{campaign: sorted list of subject IDs found on disk}``."""
    out: dict[str, list[str]] = {}
    for c in campaigns:
        cdir = radar_root / c
        if not cdir.is_dir():
            out[c] = []
            continue
        out[c] = sorted(p.name for p in cdir.iterdir() if p.is_dir() and p.name.startswith("U"))
    return out


def make_split(
    campaigns: Iterable[str],
    radar_root: Path = RADAR_ROOT,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Split:
    """Build a subject-independent split.

    A subject is included only if it appears in **every** requested campaign,
    i.e. we work with the intersection across campaigns. This matters when
    micro and macro movements live in different campaigns: a subject not
    present in both can't appear in either set without breaking the
    intersection. (Use ``discover_subjects`` directly if you want the per-
    campaign union instead.)
    """
    campaigns = tuple(campaigns)
    discovered = discover_subjects(campaigns, radar_root=radar_root)
    if not discovered:
        raise RuntimeError(f"No campaigns found under {radar_root}")
    sets = [set(v) for v in discovered.values() if v]
    if not sets:
        raise RuntimeError(f"No subjects found in campaigns {campaigns} under {radar_root}")
    intersection = sorted(set.intersection(*sets))
    if not intersection:
        raise RuntimeError(
            f"No subject appears in all campaigns {campaigns}. "
            f"Per-campaign discovery: { {k: len(v) for k, v in discovered.items()} }."
        )

    rng = random.Random(seed)
    shuffled = intersection.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    if n_train + n_val >= n:
        # Ensure at least one test subject.
        n_val = max(0, n - n_train - 1)

    train = tuple(shuffled[:n_train])
    val = tuple(shuffled[n_train:n_train + n_val])
    test = tuple(shuffled[n_train + n_val:])
    return Split(train=train, val=val, test=test, campaigns=campaigns, seed=seed)


def save_split(split: Split, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaigns": list(split.campaigns),
        "subjects": {
            "train": list(split.train),
            "val":   list(split.val),
            "test":  list(split.test),
        },
        "seed": split.seed,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_split(path: Path = SPLITS_PATH) -> Split:
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found at {path}. Run scripts/build_split.py first."
        )
    payload = json.loads(path.read_text())
    return Split(
        train=tuple(payload["subjects"]["train"]),
        val=tuple(payload["subjects"]["val"]),
        test=tuple(payload["subjects"]["test"]),
        campaigns=tuple(payload["campaigns"]),
        seed=int(payload["seed"]),
    )
