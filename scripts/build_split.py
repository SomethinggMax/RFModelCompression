"""Generate the locked subject-independent train/val/test split.

Run **once** at the start of the project. The resulting file
``splits/splits.json`` is committed to git and never regenerated mid-experiment.
"""

from __future__ import annotations

import argparse

from rfbc.config import DEFAULT_CONFIG, RADAR_ROOT, SPLITS_PATH
from rfbc.data.splits import discover_subjects, make_split, save_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing splits.json (default refuses).")
    args = p.parse_args()

    if SPLITS_PATH.exists() and not args.force:
        raise SystemExit(
            f"splits.json already exists at {SPLITS_PATH}. "
            "Re-running would change the split mid-experiment. "
            "Pass --force only if you really mean it."
        )

    campaigns = DEFAULT_CONFIG.campaigns
    print(f"Campaigns: {campaigns}")
    discovered = discover_subjects(campaigns, radar_root=RADAR_ROOT)
    for c, subs in discovered.items():
        print(f"  {c}: {len(subs)} subjects -> {subs[:5]}{'…' if len(subs) > 5 else ''}")

    split = make_split(
        campaigns=campaigns,
        radar_root=RADAR_ROOT,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print()
    print(f"Intersection size: {len(split.all_subjects())}")
    print(f"  train ({len(split.train)}): {list(split.train)}")
    print(f"  val   ({len(split.val)}): {list(split.val)}")
    print(f"  test  ({len(split.test)}): {list(split.test)}")

    save_split(split, SPLITS_PATH)
    print(f"\nWrote {SPLITS_PATH}")


if __name__ == "__main__":
    main()
