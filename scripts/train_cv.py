"""Subject-wise 5-fold cross-validation for the baseline CNN.

Why this exists
---------------
The single locked split puts only 4 subjects in test and 3 in val, so the
baseline "fingerprint" rests on 4 people and is high-variance. This script
replaces that single split with **subject-wise k-fold cross-validation**: the
~23 subjects in the C1∩C2 intersection are partitioned into ``--folds`` groups
(5 by default), every subject is tested exactly once, and the per-class
accuracy is reported as **mean ± std across folds**.

Important: cross-validation does **not** reduce over-fitting — each fold's model
over-fits exactly as much as before. CV makes the *estimate* trustworthy. To
actually reduce the train/val gap, run this with the regularisation flags
(``--augment``, ``--label-smoothing``, ``--weight-decay``, ``--early-stop-
patience``, optionally ``--width-mult`` / ``--conv-dropout``) — the same knobs
exposed by ``train_baseline.py``. Use both together: regularise *and* CV.

Protocol (nested, no leakage)
-----------------------------
For each outer fold ``k``:
    test    = subjects in fold k
    trainval = all other subjects
    val     = a seeded, subject-disjoint hold-out from trainval
              (``--val-subjects`` subjects, for early stopping / model selection)
    train   = trainval minus val
No subject appears in more than one of {train, val, test}. The test fold is
never touched during model selection.

Usage (run from ``code/``)
--------------------------
    # Recommended: regularised 5-fold CV, one seed
    uv run python -m scripts.train_cv --augment --label-smoothing 0.1 \
        --weight-decay 5e-4 --early-stop-patience 10

    # Add seed averaging (more robust, ~3x cost)
    uv run python -m scripts.train_cv --augment --label-smoothing 0.1 \
        --weight-decay 5e-4 --early-stop-patience 10 --seeds 42 43 44

    # Fast harness smoke-test (tiny, ~minutes)
    uv run python -m scripts.train_cv --quick

Output (saved to ``code/outputs/cv/``)
--------------------------------------
    fold_{k}/seed_{s}/...     per-fold/per-seed artifacts (same as train_baseline)
    cv_summary.json           per-fold + aggregate mean/std per class & campaign
    cv_per_class.png          per-class CV mean ± std bar chart
"""

from __future__ import annotations

# Non-interactive backend — must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rfbc.config import DEFAULT_CONFIG
from rfbc.data.splits import Split, discover_subjects
from scripts.train_baseline import build_datasets, run_one_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Subject-wise k-fold cross-validation for the baseline CNN."
    )

    # --- CV structure ---
    p.add_argument("--folds", type=int, default=5,
                   help="Number of subject folds (each subject tested once). "
                        "Default 5. Set equal to the subject count for "
                        "leave-one-subject-out.")
    p.add_argument("--val-subjects", type=int, default=3,
                   help="Subjects held out of each fold's training set for the "
                        "inner validation (early stopping / model selection). "
                        "Default 3.")
    p.add_argument("--cv-seed", type=int, default=42,
                   help="Seed for the subject shuffle that defines the folds "
                        "and the inner-val hold-out. Default 42.")

    # --- reproducibility of the training itself ---
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Training seeds run *within each fold*. With >1 seed the "
                        "per-fold metric is the seed-average. Default [42].")

    # --- training hyperparameters (mirror train_baseline.py) ---
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay. Try 5e-4 / 1e-3 to regularise.")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # --- anti-over-fitting knobs (default to original behaviour) ---
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--augment", action="store_true",
                   help="On-the-fly BEV augmentation on each fold's train set.")
    p.add_argument("--aug-strength", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Patience for early stopping. 0 = off. Try 8–10; it also "
                        "cuts the (folds x seeds) runtime substantially.")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--conv-dropout", type=float, default=0.0)

    # --- subset / speed ---
    p.add_argument("--class-filter", nargs="+", default=None)
    p.add_argument("--repetition-filter", nargs="+", default=None)
    p.add_argument("--quick", action="store_true",
                   help="Tiny harness smoke-test: 2 folds, 1 val subject, "
                        "3+3 classes, 2 reps, 2 epochs, 1 seed.")

    p.add_argument("--out-dir", default=None,
                   help="Root output dir. Defaults to code/outputs/cv/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def subject_pool(campaigns: tuple[str, ...]) -> list[str]:
    """Subjects present in *every* campaign (the working intersection).

    Mirrors ``rfbc.data.splits.make_split``: a subject must appear in all
    requested campaigns to be usable, since micro (C1) and macro (C2) live in
    different campaigns.
    """
    discovered = discover_subjects(campaigns)
    sets = [set(v) for v in discovered.values() if v]
    if not sets:
        raise RuntimeError(f"No subjects found for campaigns {campaigns}.")
    return sorted(set.intersection(*sets))


def make_folds(subjects: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """Partition subjects into ``n_folds`` disjoint, roughly equal groups."""
    if n_folds < 2:
        raise ValueError("--folds must be >= 2.")
    if n_folds > len(subjects):
        raise ValueError(
            f"--folds={n_folds} exceeds the {len(subjects)} available subjects."
        )
    shuffled = subjects.copy()
    random.Random(seed).shuffle(shuffled)
    # np.array_split handles uneven divisions (some folds get one more subject).
    return [list(chunk) for chunk in np.array_split(np.array(shuffled, dtype=object), n_folds)]


def build_fold_split(
    folds: list[list[str]],
    k: int,
    campaigns: tuple[str, ...],
    val_subjects: int,
    cv_seed: int,
) -> Split:
    """Build the (train, val, test) Split for outer fold ``k`` with nested val."""
    test = list(folds[k])
    trainval = [s for j, fold in enumerate(folds) if j != k for s in fold]

    # Deterministic, fold-specific inner-val hold-out (seed offset by k so each
    # fold draws a different but reproducible val set).
    n_val = max(1, min(val_subjects, len(trainval) - 1))
    rng = random.Random(cv_seed + 1000 + k)
    shuffled = trainval.copy()
    rng.shuffle(shuffled)
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    # Sanity: the three sets must be pairwise subject-disjoint.
    assert not (set(train) & set(val)), "train/val subject overlap"
    assert not (set(train) & set(test)), "train/test subject overlap"
    assert not (set(val) & set(test)), "val/test subject overlap"

    return Split(
        train=tuple(train), val=tuple(val), test=tuple(test),
        campaigns=campaigns, seed=cv_seed,
    )


# ---------------------------------------------------------------------------
# Metrics extraction / aggregation
# ---------------------------------------------------------------------------


def metrics_from_pc(pc: dict[str, list[dict]]) -> dict:
    """Collapse a per_class_accuracy result into overall / campaign / per-class."""
    per_class: dict[str, float] = {}
    tot_correct = tot_total = 0
    camp_acc: dict[str, float] = {}
    for camp in ("C1", "C2"):
        cc = ct = 0
        for e in pc.get(camp, []):
            per_class[e["class"]] = e["acc"]
            cc += e["correct"]
            ct += e["total"]
        camp_acc[camp] = cc / max(1, ct)
        tot_correct += cc
        tot_total += ct
    return {
        "overall": tot_correct / max(1, tot_total),
        "c1": camp_acc["C1"],
        "c2": camp_acc["C2"],
        "per_class": per_class,
    }


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "per_fold": [float(v) for v in arr],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_per_class(summary: dict, out_path: Path) -> None:
    entries = summary["aggregate"]["per_class"]
    if not entries:
        return
    # C1 first, then C2, each sorted by mean accuracy.
    c1 = sorted([e for e in entries if e["campaign"] == "C1"], key=lambda e: e["mean"])
    c2 = sorted([e for e in entries if e["campaign"] == "C2"], key=lambda e: e["mean"])
    ordered = c1 + c2
    names = [e["class"] for e in ordered]
    means = [e["mean"] for e in ordered]
    stds = [e["std"] for e in ordered]
    colors = ["steelblue"] * len(c1) + ["darkorange"] * len(c2)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.35), 4.5))
    ax.bar(range(len(names)), means, yerr=stds, color=colors, capsize=2, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("test accuracy (CV mean ± std)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Per-class accuracy across {summary['n_folds']} subject folds "
        f"(overall {summary['aggregate']['overall']['mean']:.3f} "
        f"± {summary['aggregate']['overall']['std']:.3f})"
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="steelblue", label="C1 micro-gestures"),
        Patch(facecolor="darkorange", label="C2 macro-activities"),
    ], fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG
    campaigns = cfg.campaigns

    if args.quick:
        args.folds = 2
        args.val_subjects = 1
        args.seeds = [args.seeds[0]]
        if args.class_filter is None:
            args.class_filter = ["M01", "M02", "M03", "A01", "A02", "A03"]
        if args.repetition_filter is None:
            args.repetition_filter = ["01", "02"]
        if args.epochs == 40:
            args.epochs = 2
        print("  (quick mode: 2 folds, 1 val subject, 3+3 classes, 2 reps, 2 epochs)")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "cv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = subject_pool(campaigns)
    folds = make_folds(pool, args.folds, args.cv_seed)

    print(f"Device:   {args.device}")
    print(f"Campaigns: {campaigns}")
    print(f"Subject pool ({len(pool)}): {pool}")
    print(f"Folds ({args.folds}):")
    for k, f in enumerate(folds):
        print(f"  fold {k}: test={f}")
    print(f"Seeds per fold: {args.seeds}   Epochs: {args.epochs}")
    print(f"Regularisation: augment={args.augment} (strength={args.aug_strength}), "
          f"label_smoothing={args.label_smoothing}, weight_decay={args.weight_decay}, "
          f"early_stop_patience={args.early_stop_patience}, "
          f"width_mult={args.width_mult}, conv_dropout={args.conv_dropout}\n")

    # fold_level_metrics[k] = seed-averaged metrics dict for fold k
    fold_records: list[dict] = []

    for k in range(args.folds):
        split = build_fold_split(folds, k, campaigns, args.val_subjects, args.cv_seed)
        print(f"\n{'#'*90}")
        print(f"FOLD {k}/{args.folds - 1}   "
              f"train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")
        print(f"  train={list(split.train)}")
        print(f"  val  ={list(split.val)}")
        print(f"  test ={list(split.test)}")
        print(f"{'#'*90}")

        fold_dir = out_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build datasets once per fold; all seeds within the fold share them.
        train_ds, val_ds, test_ds = build_datasets(split, cfg, args)

        per_seed: list[dict] = []
        for seed in args.seeds:
            _, pc_results = run_one_seed(
                seed=seed, split=split, cfg=cfg, args=args,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                out_dir=fold_dir,
            )
            m = metrics_from_pc(pc_results)
            m["seed"] = seed
            per_seed.append(m)

        # Seed-average within the fold.
        def _avg(key: str) -> float:
            return float(np.mean([s[key] for s in per_seed]))

        all_classes = sorted({c for s in per_seed for c in s["per_class"]})
        fold_per_class = {
            c: float(np.mean([s["per_class"][c] for s in per_seed if c in s["per_class"]]))
            for c in all_classes
        }
        # Determine campaign per class from the dataset label sets.
        c1_names = set(test_ds.c1_classes)
        fold_records.append({
            "fold": k,
            "train": list(split.train),
            "val": list(split.val),
            "test": list(split.test),
            "overall": _avg("overall"),
            "c1": _avg("c1"),
            "c2": _avg("c2"),
            "per_class": fold_per_class,
            "campaign_of": {c: ("C1" if c in c1_names else "C2") for c in all_classes},
            "per_seed": [
                {"seed": s["seed"], "overall": s["overall"], "c1": s["c1"], "c2": s["c2"]}
                for s in per_seed
            ],
        })

    # ----------------------------------------------------------------------
    # Aggregate across folds
    # ----------------------------------------------------------------------
    overall_vals = [r["overall"] for r in fold_records]
    c1_vals = [r["c1"] for r in fold_records]
    c2_vals = [r["c2"] for r in fold_records]

    per_class_vals: dict[str, list[float]] = defaultdict(list)
    class_campaign: dict[str, str] = {}
    for r in fold_records:
        for c, acc in r["per_class"].items():
            per_class_vals[c].append(acc)
            class_campaign[c] = r["campaign_of"][c]

    per_class_agg = []
    for c in sorted(per_class_vals):
        agg = _agg(per_class_vals[c])
        agg["class"] = c
        agg["campaign"] = class_campaign[c]
        per_class_agg.append(agg)

    summary = {
        "n_folds": args.folds,
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "cv_seed": args.cv_seed,
        "val_subjects": args.val_subjects,
        "subject_pool": pool,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "augment": args.augment,
            "aug_strength": args.aug_strength,
            "early_stop_patience": args.early_stop_patience,
            "width_mult": args.width_mult,
            "conv_dropout": args.conv_dropout,
        },
        "folds": fold_records,
        "aggregate": {
            "overall": _agg(overall_vals),
            "c1": _agg(c1_vals),
            "c2": _agg(c2_vals),
            "per_class": per_class_agg,
        },
    }

    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    plot_per_class(summary, out_dir / "cv_per_class.png")

    # ----------------------------------------------------------------------
    # Console report
    # ----------------------------------------------------------------------
    agg = summary["aggregate"]
    print(f"\n{'='*90}")
    print(f"CROSS-VALIDATION SUMMARY  ({args.folds} folds, {len(args.seeds)} seed(s))")
    print(f"{'='*90}")
    print(f"  Overall : {agg['overall']['mean']:.3f} ± {agg['overall']['std']:.3f}  "
          f"per-fold {[round(v,3) for v in agg['overall']['per_fold']]}")
    print(f"  C1 micro: {agg['c1']['mean']:.3f} ± {agg['c1']['std']:.3f}")
    print(f"  C2 macro: {agg['c2']['mean']:.3f} ± {agg['c2']['std']:.3f}")
    print(f"  C1/C2 gap: {agg['c1']['mean'] - agg['c2']['mean']:+.3f}")
    print("\n  Per-class (CV mean ± std), weakest first:")
    for camp in ("C1", "C2"):
        rows = sorted([e for e in agg["per_class"] if e["campaign"] == camp],
                      key=lambda e: e["mean"])
        print(f"\n    {camp}:")
        for e in rows:
            print(f"      {e['class']:>4}  {e['mean']:.3f} ± {e['std']:.3f}  "
                  f"[{e['min']:.3f}–{e['max']:.3f}]")

    print(f"\nSaved CV summary to {out_dir/'cv_summary.json'}")
    print(f"Saved per-class plot to {out_dir/'cv_per_class.png'}")


if __name__ == "__main__":
    main()
