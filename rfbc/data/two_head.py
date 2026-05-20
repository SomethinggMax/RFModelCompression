"""Two-head wrapper around ``SubjectIndependentRadarDataset``.

The base dataset returns ``(tensor, unified_label_int)`` where the unified
label is an index into a single 31-class space spanning C1 gestures and C2
activities together. For a two-head model we need to know:

  * which campaign each sample is from (so the loss is routed to the right
    head), and
  * the **within-campaign** class index (0..20 for C1, 0..9 for C2),
    because each head only outputs its own campaign's classes.

This module provides:

  * :class:`TwoHeadRadarDataset` — wraps the base dataset and returns
    ``(tensor, campaign_id, within_class_id)``.
  * :func:`make_two_head_sampler` — builds a ``WeightedRandomSampler`` that
    balances the dataset 50/50 between campaigns and uniformly within each
    campaign. This counteracts the natural imbalance in the dataset (C2
    activities produce far more windows than C1 gestures because they are
    ~15 s long and slide into multiple 2 s windows).
"""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from rfbc.data.dataset import SubjectIndependentRadarDataset


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class TwoHeadRadarDataset(Dataset):
    """Wrap a ``SubjectIndependentRadarDataset`` for two-head training.

    Each item is a ``(tensor, campaign_id, within_class_id)`` triple where:

    * ``tensor`` is the ``(C, H, W)`` BEV tensor returned by the base dataset.
    * ``campaign_id`` is 0 for C1, 1 for C2.
    * ``within_class_id`` is the within-campaign class index (0..20 for C1
      gestures M01..M21 sorted alphabetically; 0..9 for C2 activities
      A01..A10 sorted alphabetically).

    The wrapper **forces lazy expansion** of the base dataset's index in
    ``__init__`` so that ``__len__`` reflects the post-expansion window count
    (otherwise the C2 multi-window samples would only contribute one window
    each until the first epoch finished iterating).
    """

    def __init__(self, base: SubjectIndependentRadarDataset) -> None:
        self.base = base

        # Derive per-campaign class lists from the base's label_map. Sorting
        # gives a stable ordering ("M01" < "M02" < ...), so the within-class
        # ids line up with the on-disk folder names alphabetically.
        c1_names = sorted({k.split("/", 1)[1] for k in base.label_map if k.startswith("C1/")})
        c2_names = sorted({k.split("/", 1)[1] for k in base.label_map if k.startswith("C2/")})
        self.c1_classes: tuple[str, ...] = tuple(c1_names)
        self.c2_classes: tuple[str, ...] = tuple(c2_names)
        self._c1_map = {name: i for i, name in enumerate(c1_names)}
        self._c2_map = {name: i for i, name in enumerate(c2_names)}

        # Force lazy multi-window expansion so the wrapper's __len__ is the
        # real post-expansion window count, not the sample count. This also
        # populates the cache for every sample (slow on first run, fast
        # afterwards because the cache is keyed by config hash).
        if not base._expanded:
            base._expand_index()

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        tensor, _unified_label = self.base[idx]
        sample = self.base.index[idx].sample
        if sample.campaign == "C1":
            return tensor, 0, self._c1_map[sample.cls]
        if sample.campaign == "C2":
            return tensor, 1, self._c2_map[sample.cls]
        raise ValueError(
            f"Two-head wrapper only handles C1/C2; got {sample.campaign!r}."
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_c1_classes(self) -> int:
        return len(self.c1_classes)

    @property
    def num_c2_classes(self) -> int:
        return len(self.c2_classes)

    def meta(self) -> list[tuple[int, int]]:
        """Return ``[(campaign_id, within_class_id), ...]`` for every item.

        Useful for the weighted sampler and for diagnostics; computed on
        demand from ``self.base.index`` so it stays consistent if the base
        index ever changes.
        """
        out: list[tuple[int, int]] = []
        for entry in self.base.index:
            cls = entry.sample.cls
            if entry.sample.campaign == "C1":
                out.append((0, self._c1_map[cls]))
            elif entry.sample.campaign == "C2":
                out.append((1, self._c2_map[cls]))
            else:
                raise ValueError(
                    f"Two-head wrapper only handles C1/C2; "
                    f"saw {entry.sample.campaign!r}."
                )
        return out


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------


def make_two_head_sampler(
    dataset: TwoHeadRadarDataset,
    *,
    campaign_balance: float = 0.5,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a sampler that balances campaigns and classes within campaigns.

    The default scheme assigns each draw:

    * ``campaign_balance`` probability of being from C1 (default 0.5),
    * uniform across the C1 classes within C1, uniform across instances
      within a class, and symmetrically for C2.

    So each *class* (M01, M02, ..., A01, ..., A10) is sampled with the same
    expected per-class frequency *within its campaign*, and the two
    campaigns are balanced 50/50 against each other. This counteracts both
    (a) C2 producing far more total windows than C1 because C2 samples are
    ~15 s and slide into multiple 2 s windows, and (b) any within-campaign
    class imbalance (e.g. some users having fewer repetitions of a class).
    """
    if not 0.0 < campaign_balance < 1.0:
        raise ValueError(f"campaign_balance must be in (0, 1); got {campaign_balance}")

    meta = dataset.meta()
    if not meta:
        raise ValueError("Cannot build a sampler over an empty dataset.")

    counts = Counter()
    for camp, within in meta:
        counts[(camp, within)] += 1

    classes_per_camp: dict[int, set[int]] = {0: set(), 1: set()}
    for (camp, within) in counts:
        classes_per_camp[camp].add(within)
    n_c1 = max(1, len(classes_per_camp[0]))
    n_c2 = max(1, len(classes_per_camp[1]))

    c2_balance = 1.0 - campaign_balance
    weights: list[float] = []
    for camp, within in meta:
        if camp == 0:
            # P(this sample) = (campaign_balance) * (1 / n_c1_classes) * (1 / class_size)
            w = campaign_balance / n_c1 / counts[(camp, within)]
        else:
            w = c2_balance / n_c2 / counts[(camp, within)]
        weights.append(w)

    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weights),
        replacement=replacement,
    )
