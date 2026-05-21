"""Combined 31-class dataset wrapper and balanced sampler.

The compression baseline uses a single 31-class output head covering both
Campaign 1 gesture classes (M01–M21) and Campaign 2 activity classes
(A01–A10). This module provides two things:

CombinedRadarDataset
    A thin wrapper around SubjectIndependentRadarDataset that forces
    multi-window expansion on construction and exposes the campaign
    metadata (class lists, label sets) needed by the sampler and the
    per-class evaluation code.

make_combined_sampler
    A WeightedRandomSampler that draws C1 and C2 samples with equal
    campaign-level probability (50/50 by default) and uniform class
    probability within each campaign. This counteracts two sources of
    imbalance that would otherwise bias training:

    1. Window-count imbalance between campaigns. C2 activities are ~15 s
       long and slide into many 2-second windows; C1 gestures produce
       roughly one window each. Without correction the model sees ~3×
       more C2 than C1 windows per epoch.

    2. Within-campaign class imbalance. Some C2 activity classes (e.g.
       A01 walking) produce many more windows than others (e.g. A10
       standing-to-sitting), and some C1 subjects performed more
       repetitions of certain gestures than others.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from rfbc.data.dataset import SubjectIndependentRadarDataset


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class CombinedRadarDataset(Dataset):
    """Wraps SubjectIndependentRadarDataset for single-head 31-class training.

    Items are ``(tensor, unified_label)`` pairs — identical to what the base
    dataset returns, so no re-labelling is needed. The wrapper adds:

    * Forced multi-window expansion in ``__init__`` so ``__len__`` returns
      the true window count (not the sample count) from the very start.
      Without this, C2 multi-window samples would only contribute one
      entry each until the end of the first epoch.

    * ``c1_classes`` / ``c2_classes`` — class name tuples in label order,
      used by evaluation code to map integer labels back to human-readable
      names (e.g. "M07", "A03").

    * ``c1_label_set`` / ``c2_label_set`` — frozensets of integer labels
      belonging to each campaign, used for per-campaign accuracy grouping.

    * ``meta()`` — returns the (campaign, unified_label) pair for every
      item in the dataset. Used by ``make_combined_sampler``.

    Label ordering
    --------------
    Labels are assigned by ``SubjectIndependentRadarDataset`` in the order
    classes are first encountered during directory scanning — campaign-then-
    alphabetical, since ``campaigns = ("C1", "C2")`` in the default config.
    This gives:

        C1/M01 → 0,  C1/M02 → 1,  …,  C1/M21 → 20
        C2/A01 → 21, C2/A02 → 22, …,  C2/A10 → 30

    The ordering is stable across runs as long as the split and any
    ``class_filter`` argument remain the same.
    """

    def __init__(self, base: SubjectIndependentRadarDataset) -> None:
        self.base = base

        # Build per-campaign (label, class_name) pairs, sorted by label so
        # that c1_classes[i] corresponds to the i-th C1 label integer.
        c1_pairs = sorted(
            (lbl, key.split("/", 1)[1])
            for key, lbl in base.label_map.items()
            if key.startswith("C1/")
        )
        c2_pairs = sorted(
            (lbl, key.split("/", 1)[1])
            for key, lbl in base.label_map.items()
            if key.startswith("C2/")
        )

        self.c1_classes: tuple[str, ...] = tuple(cls for _, cls in c1_pairs)
        self.c2_classes: tuple[str, ...] = tuple(cls for _, cls in c2_pairs)
        self.c1_label_set: frozenset[int] = frozenset(lbl for lbl, _ in c1_pairs)
        self.c2_label_set: frozenset[int] = frozenset(lbl for lbl, _ in c2_pairs)

        # Force multi-window expansion immediately so __len__ is accurate.
        if not base._expanded:
            base._expand_index()

    # ------------------------------------------------------------------
    # PyTorch Dataset interface — pass-through to base
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.base[idx]  # (tensor, unified_label)

    # ------------------------------------------------------------------
    # Helpers for sampler and per-class evaluation
    # ------------------------------------------------------------------

    def meta(self) -> list[tuple[str, int]]:
        """Return ``[(campaign, unified_label), ...]`` for every item.

        The list is in the same order as the dataset index, so
        ``meta()[i]`` describes the item at ``dataset[i]``.
        """
        out: list[tuple[str, int]] = []
        for entry in self.base.index:
            key = f"{entry.sample.campaign}/{entry.sample.cls}"
            out.append((entry.sample.campaign, self.base.label_map[key]))
        return out

    def label_to_name(self, label: int) -> str:
        """Map a unified label integer to its class string (e.g. 'M07')."""
        if label in self.c1_label_set:
            # c1_classes is sorted by label; labels start at 0
            return self.c1_classes[label]
        # C2 labels start after all C1 labels
        c2_offset = min(self.c2_label_set)
        return self.c2_classes[label - c2_offset]

    def campaign_of_label(self, label: int) -> str:
        """Return 'C1' or 'C2' for a unified label integer."""
        return "C1" if label in self.c1_label_set else "C2"

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    @property
    def num_c1_classes(self) -> int:
        return len(self.c1_classes)

    @property
    def num_c2_classes(self) -> int:
        return len(self.c2_classes)


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------


def make_combined_sampler(
    dataset: CombinedRadarDataset,
    *,
    campaign_balance: float = 0.5,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Sampler that balances C1/C2 campaigns and classes uniformly within each.

    Weight formula for item ``i`` with campaign ``c`` and unified label ``k``::

        w(i) = P(campaign = c) / |unique classes in c| / count(c, k)

    so that every class in C1 is drawn with the same expected frequency as
    every class in C2, regardless of window counts. This mirrors the logic
    in ``rfbc.data.two_head.make_two_head_sampler`` but operates on the
    unified label space of ``CombinedRadarDataset``.

    Parameters
    ----------
    dataset
        A ``CombinedRadarDataset`` instance (train fold only — never apply
        a weighted sampler to val or test folds).
    campaign_balance
        Fraction of draws that come from C1. Default 0.5 (equal campaigns).
    replacement
        Passed to ``WeightedRandomSampler``. Keep ``True`` for training.
    """
    if not 0.0 < campaign_balance < 1.0:
        raise ValueError(
            f"campaign_balance must be strictly between 0 and 1; got {campaign_balance}"
        )

    meta = dataset.meta()
    if not meta:
        raise ValueError("Cannot build a sampler over an empty dataset.")

    # Count instances per (campaign, unified_label) pair.
    counts: Counter = Counter(meta)

    # Count distinct classes per campaign.
    classes_per_camp: dict[str, set[int]] = {"C1": set(), "C2": set()}
    for camp, label in counts:
        classes_per_camp[camp].add(label)
    n_c1 = max(1, len(classes_per_camp["C1"]))
    n_c2 = max(1, len(classes_per_camp["C2"]))

    c2_prob = 1.0 - campaign_balance
    weights: list[float] = []
    for camp, label in meta:
        if camp == "C1":
            w = campaign_balance / n_c1 / counts[(camp, label)]
        else:
            w = c2_prob / n_c2 / counts[(camp, label)]
        weights.append(w)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=replacement,
    )
