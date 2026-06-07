"""Train the single-head 31-class baseline CNN on combined C1 + C2 data.

This script produces the pre-compression baseline for the study
"Evaluating the Class-wise Impact of Model Compression in RF-Based HAR".

Pipeline
--------
1. Load the subject-independent split from ``splits/splits.json``.
2. Build three ``CombinedRadarDataset`` instances (train / val / test)
   covering both C1 gesture classes (M01–M21) and C2 activity classes
   (A01–A10) in a single 31-class label space.
3. Build a ``WeightedRandomSampler`` that balances campaigns 50/50 and
   classes uniformly within each campaign.
4. Build ``BaselineCNN`` (~420 K params) and an AdamW optimiser with a
   cosine-annealing LR schedule.
5. Train for ``--epochs`` epochs (default 40), evaluating on val every epoch.
6. At the end, evaluate on the held-out test fold and report:
   - Overall accuracy
   - Per-campaign accuracy (C1 micro-gestures vs C2 macro-activities)
   - Per-class accuracy for all 31 classes
   - A 31-class confusion matrix annotated to show the C1/C2 boundary
7. Repeat steps 4–6 for each seed in ``--seeds`` (default: 42 only).
   With multiple seeds, a cross-seed summary (median ± std per class) is
   written at the end — the compression study should use this summary as
   its baseline fingerprint rather than any single-seed run.

Usage (run from the ``code/`` directory)
-----------------------------------------
    # Standard full run — one seed, all subjects, all classes
    uv run python -m scripts.train_baseline

    # Multi-seed run for compression baseline (recommended)
    uv run python -m scripts.train_baseline --seeds 42 43 44 --epochs 40

    # Quick smoke-test (~5 min): tiny subset, 3 epochs, one seed
    uv run python -m scripts.train_baseline --quick

    # Tune knobs
    uv run python -m scripts.train_baseline --epochs 40 --batch-size 64 --lr 5e-4

Output (saved to ``code/outputs/baseline/``)
--------------------------------------------
    seed_{N}/
        train_log.csv           per-epoch loss + accuracy
        train_curves.png        matplotlib plot of train/val loss and accuracy
        confusion.png           31-class test confusion matrix (C1/C2 annotated)
        per_class_acc.json      per-class test accuracy + window counts
        model.pt                best-val checkpoint (self-contained for compression)
    summary.json                per-class median/std across all seeds
    summary_curves.png          per-campaign val-accuracy across seeds
"""

from __future__ import annotations

# Non-interactive backend — must be set before importing pyplot.
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfbc.config import DEFAULT_CONFIG, PipelineConfig
from rfbc.data.augment import AugmentedView, make_bev_augment
from rfbc.data.combined import CombinedRadarDataset, make_combined_sampler
from rfbc.data.dataset import SubjectIndependentRadarDataset
from rfbc.data.splits import Split, load_split
from rfbc.models.baseline_cnn import BaselineCNN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the 31-class baseline CNN.")

    # Reproducibility / multi-seed
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="Random seeds to train with. Each seed produces its own "
                        "output sub-directory. Use 3–5 seeds for a robust baseline "
                        "fingerprint. Default: [42].")

    # Speed / subset knobs
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run: 2 train + 1 val + 1 test subject, "
                        "3 C1 + 3 C2 classes, 2 reps, 3 epochs, seed 42 only.")
    p.add_argument("--class-filter", nargs="+", default=None,
                   help="Restrict to specific class folder names (e.g. M01 M02 A01).")
    p.add_argument("--repetition-filter", nargs="+", default=None,
                   help="Restrict to specific repetition folders (e.g. 01 02).")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=40,
                   help="Training epochs. Default 40 — C1 needs more than 20 to "
                        "converge (the two-head demo was still climbing at epoch 20).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay. Default 1e-4; try 5e-4 or 1e-3 to "
                        "regularise harder against the observed over-fitting.")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout probability before the linear head. Default 0.5.")

    # --- anti-over-fitting knobs (all default to the original behaviour) ---
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Cross-entropy label smoothing on the TRAIN loss only. "
                        "0.0 = off (original). Try 0.05–0.1 to curb the "
                        "train-loss→0 over-confidence. Val/test loss stays "
                        "unsmoothed for comparability.")
    p.add_argument("--augment", action="store_true",
                   help="Enable on-the-fly BEV augmentation on the train fold "
                        "(translation, small rotation, frame dropout, density "
                        "noise, cutout). No mirror flips — chirality. Off by "
                        "default to preserve the original run.")
    p.add_argument("--aug-strength", type=float, default=1.0,
                   help="Global multiplier on augmentation firing probabilities "
                        "(0–1). Only used with --augment. Default 1.0.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop if val accuracy hasn't improved for this many "
                        "epochs. 0 = disabled (train all epochs, original "
                        "behaviour). Try 8–10.")
    p.add_argument("--width-mult", type=float, default=1.0,
                   help="BaselineCNN channel-width multiplier. 1.0 = original "
                        "~420 K-param model; 0.5 ≈ a quarter of the params "
                        "(less capacity to memorise).")
    p.add_argument("--conv-dropout", type=float, default=0.0,
                   help="Dropout2d probability after each conv block. 0.0 = off "
                        "(original). Try 0.1–0.2.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 on Windows to avoid pickling "
                        "issues when the cache is being built for the first time.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    p.add_argument("--out-dir", default=None,
                   help="Root output directory. Defaults to code/outputs/baseline/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def build_datasets(
    split: Split,
    cfg: PipelineConfig,
    args: argparse.Namespace,
) -> tuple[CombinedRadarDataset, CombinedRadarDataset, CombinedRadarDataset]:
    """Build train / val / test CombinedRadarDatasets."""
    class_filter = tuple(args.class_filter) if args.class_filter else None
    rep_filter = tuple(args.repetition_filter) if args.repetition_filter else None

    folds = {}
    for fold in ("train", "val", "test"):
        print(f"  Building {fold} dataset...", flush=True)
        t0 = time.time()
        base = SubjectIndependentRadarDataset(
            split, fold=fold, cfg=cfg,
            class_filter=class_filter, repetition_filter=rep_filter,
        )
        ds = CombinedRadarDataset(base)
        print(f"    {fold}: {len(ds)} windows in {time.time() - t0:.1f}s", flush=True)
        folds[fold] = ds

    return folds["train"], folds["val"], folds["test"]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: BaselineCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(
            logits, labels, reduction="mean", label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(labels)
        total_correct += int((logits.argmax(1) == labels).sum())
        total_seen += len(labels)

    return {
        "loss": total_loss / max(1, total_seen),
        "acc":  total_correct / max(1, total_seen),
    }


@torch.no_grad()
def evaluate(
    model: BaselineCNN,
    loader: DataLoader,
    device: str,
    dataset: CombinedRadarDataset | None = None,
    *,
    collect_preds: bool = False,
) -> dict:
    """Evaluate the model.

    If ``collect_preds=True``, also returns ``preds`` and ``gt`` arrays
    for confusion matrix and per-class accuracy computation.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_preds: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        preds = logits.argmax(1)

        total_loss += float(loss)
        total_correct += int((preds == labels).sum())
        total_seen += len(labels)

        if collect_preds:
            all_preds.append(preds.cpu().numpy())
            all_gt.append(labels.cpu().numpy())

    out: dict = {
        "loss": total_loss / max(1, total_seen),
        "acc":  total_correct / max(1, total_seen),
        "n":    total_seen,
    }
    if collect_preds:
        out["preds"] = np.concatenate(all_preds) if all_preds else np.array([], int)
        out["gt"]    = np.concatenate(all_gt)    if all_gt    else np.array([], int)
    return out


# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------


def per_class_accuracy(
    gt: np.ndarray,
    preds: np.ndarray,
    dataset: CombinedRadarDataset,
) -> dict[str, dict]:
    """Compute per-class accuracy grouped by campaign.

    Returns a dict with keys 'C1' and 'C2', each containing a list of
    dicts with keys 'class', 'label', 'correct', 'total', 'acc'.
    """
    num_classes = dataset.num_classes
    correct = np.zeros(num_classes, dtype=np.int64)
    total   = np.zeros(num_classes, dtype=np.int64)
    for g, p in zip(gt, preds):
        if 0 <= g < num_classes:
            total[g] += 1
            if g == p:
                correct[g] += 1

    results: dict[str, list[dict]] = {"C1": [], "C2": []}
    for label in range(num_classes):
        camp = dataset.campaign_of_label(label)
        cls_name = dataset.label_to_name(label)
        acc = float(correct[label]) / max(1, int(total[label]))
        results[camp].append({
            "class":   cls_name,
            "label":   label,
            "correct": int(correct[label]),
            "total":   int(total[label]),
            "acc":     acc,
        })
    return results


def print_per_class_accuracy(results: dict[str, list[dict]]) -> None:
    """Pretty-print per-class accuracy table, grouped by campaign."""
    for camp_tag, title in [("C1", "C1 micro-gestures"), ("C2", "C2 macro-activities")]:
        entries = results[camp_tag]
        if not entries:
            continue
        n_total = sum(e["total"] for e in entries)
        n_correct = sum(e["correct"] for e in entries)
        camp_acc = n_correct / max(1, n_total)
        print(f"\n  {title}  (overall {camp_acc:.3f}, {n_total} windows)")
        for e in entries:
            bar = "█" * int(e["acc"] * 20)
            print(f"    {e['class']}  {e['acc']:.3f}  {bar:<20}  "
                  f"({e['correct']}/{e['total']})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_curves(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_loss.plot(epochs, [r["val_loss"]   for r in rows], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (mean CE)")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [r["train_acc"] for r in rows], label="train")
    ax_acc.plot(epochs, [r["val_acc"]   for r in rows], label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (31 classes)")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Overall accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    gt: np.ndarray,
    preds: np.ndarray,
    dataset: CombinedRadarDataset,
    title: str,
    out_path: Path,
) -> None:
    """Plot a 31-class confusion matrix with a visual C1/C2 boundary."""
    n = dataset.num_classes
    cm = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gt, preds):
        if 0 <= g < n and 0 <= p < n:
            cm[g, p] += 1

    # Class labels in label-index order.
    all_names = [dataset.label_to_name(i) for i in range(n)]
    n_c1 = dataset.num_c1_classes  # boundary between C1 and C2 labels

    fig_size = max(10, n * 0.38)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("predicted", fontsize=10)
    ax.set_ylabel("ground truth", fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(all_names, fontsize=7)

    # Colour tick labels: C1 → steel blue, C2 → darkorange
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color("steelblue" if i < n_c1 else "darkorange")
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color("steelblue" if i < n_c1 else "darkorange")

    # Draw a line separating C1 (rows/cols 0–20) from C2 (21–30).
    boundary = n_c1 - 0.5
    ax.axhline(boundary, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(boundary, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    # Annotate cells (only when matrix is ≤ 35 classes to stay readable).
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")

    # Legend for label colours.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue",  label="C1 micro-gestures (M01–M21)"),
        Patch(facecolor="darkorange", label="C2 macro-activities (A01–A10)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 1.12), fontsize=8, framealpha=0.9)

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_curves(
    all_rows: dict[int, list[dict]],
    out_path: Path,
) -> None:
    """Plot val accuracy across seeds, with mean ± std bands."""
    if not all_rows:
        return
    # Align on epoch count (use shortest run).
    min_epochs = min(len(rows) for rows in all_rows.values())
    epochs = list(range(1, min_epochs + 1))

    val_accs = np.array([
        [rows[ep - 1]["val_acc"] for ep in epochs]
        for rows in all_rows.values()
    ])  # shape: (n_seeds, epochs)

    mean = val_accs.mean(0)
    std  = val_accs.std(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, mean, label="mean val acc (31-class)")
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, label="±1 std")
    for seed, rows in all_rows.items():
        ax.plot(epochs, [rows[ep - 1]["val_acc"] for ep in epochs],
                alpha=0.4, linewidth=0.8, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Val accuracy across {len(all_rows)} seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-seed training loop
# ---------------------------------------------------------------------------


def run_one_seed(
    seed: int,
    split: Split,
    cfg: PipelineConfig,
    args: argparse.Namespace,
    train_ds: CombinedRadarDataset,
    val_ds: CombinedRadarDataset,
    test_ds: CombinedRadarDataset,
    out_dir: Path,
) -> tuple[list[dict], dict]:
    """Train one seed. Returns (log_rows, per_class_results)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    sampler = make_combined_sampler(train_ds)
    # Wrap the *train* dataset with on-the-fly augmentation when requested.
    # The sampler is built over the unwrapped CombinedRadarDataset above; the
    # AugmentedView shares its index space, so the sampler stays valid.
    if getattr(args, "augment", False):
        augment = make_bev_augment(cfg, strength=args.aug_strength)
        train_view: object = AugmentedView(train_ds, augment)
        print(f"  Augmentation: ON (strength={args.aug_strength})")
    else:
        train_view = train_ds
        print("  Augmentation: OFF")
    train_loader = DataLoader(
        train_view, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    x_probe, _ = train_ds[0]
    in_channels = int(x_probe.shape[0])
    num_classes = train_ds.num_classes  # 31

    model = BaselineCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=args.dropout,
        width_mult=args.width_mult,
        conv_dropout=args.conv_dropout,
    ).to(args.device)

    print(f"\n  Model: BaselineCNN  {model.num_parameters:,} parameters")
    print(f"  in_channels={in_channels}  num_classes={num_classes}  "
          f"dropout={args.dropout}  width_mult={args.width_mult}  "
          f"conv_dropout={args.conv_dropout}")
    counts = model.layer_parameter_counts()
    total  = model.num_parameters
    for layer, cnt in counts.items():
        print(f"    {layer:<12} {cnt:>8,}  ({cnt/total*100:4.1f} %)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_rows: list[dict] = []
    best_val_acc = -1.0
    epochs_since_improve = 0
    best_ckpt = seed_dir / "model.pt"

    epoch_fmt = (
        "  epoch {ep:>3}/{tot:<3}  lr={lr:.2e}  "
        "train loss={tl:.4f}  acc={ta:.3f}    "
        "val   loss={vl:.4f}  acc={va:.3f}    [{dt:5.1f}s]"
    )

    print(f"\n  Training for {args.epochs} epochs on {args.device}...")
    print("  " + "=" * 85)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, args.device,
            label_smoothing=args.label_smoothing,
        )
        val_m   = evaluate(model, val_loader, args.device)
        scheduler.step()
        dt = time.time() - t0

        print(epoch_fmt.format(
            ep=ep, tot=args.epochs,
            lr=optimizer.param_groups[0]["lr"],
            tl=train_m["loss"], ta=train_m["acc"],
            vl=val_m["loss"],   va=val_m["acc"],
            dt=dt,
        ), flush=True)

        row = {
            "epoch":      ep,
            "lr":         optimizer.param_groups[0]["lr"],
            "train_loss": train_m["loss"],
            "train_acc":  train_m["acc"],
            "val_loss":   val_m["loss"],
            "val_acc":    val_m["acc"],
            # train-minus-val accuracy gap: the headline over-fitting metric.
            "gap":        train_m["acc"] - val_m["acc"],
            "epoch_secs": dt,
        }
        log_rows.append(row)

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            epochs_since_improve = 0
            torch.save({
                # Everything needed to reconstruct the model for compression.
                "state_dict": model.state_dict(),
                "arch": {
                    "in_channels":  in_channels,
                    "num_classes":  num_classes,
                    "dropout":      args.dropout,
                    "width_mult":   args.width_mult,
                    "conv_dropout": args.conv_dropout,
                },
                "label_map": {
                    "c1_classes":    list(train_ds.c1_classes),
                    "c2_classes":    list(train_ds.c2_classes),
                    "c1_label_set":  sorted(train_ds.c1_label_set),
                    "c2_label_set":  sorted(train_ds.c2_label_set),
                },
                "training": {
                    "epoch":        ep,
                    "seed":         seed,
                    "val_acc":      val_m["acc"],
                    "epochs_total": args.epochs,
                    "lr":           args.lr,
                    "weight_decay": args.weight_decay,
                },
            }, best_ckpt)
        else:
            epochs_since_improve += 1

        if (args.early_stop_patience > 0
                and epochs_since_improve >= args.early_stop_patience):
            print(f"  Early stopping at epoch {ep} (no val-acc improvement for "
                  f"{epochs_since_improve} epochs; best val acc="
                  f"{best_val_acc:.3f}).", flush=True)
            break

    # ------------------------------------------------------------------
    # Save training log + curves
    # ------------------------------------------------------------------
    log_path = seed_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    plot_curves(log_rows, seed_dir / "train_curves.png")

    # ------------------------------------------------------------------
    # Test evaluation (best-val checkpoint)
    # ------------------------------------------------------------------
    ckpt = torch.load(best_ckpt, map_location=args.device, weights_only=False)
    print(f"\n  Loading best-val checkpoint (epoch {ckpt['training']['epoch']})")
    model.load_state_dict(ckpt["state_dict"])

    test_m = evaluate(model, test_loader, args.device,
                      dataset=test_ds, collect_preds=True)

    print(f"\n  Test: loss={test_m['loss']:.4f}  acc={test_m['acc']:.3f}  "
          f"({test_m['n']} windows)")

    pc_results = per_class_accuracy(test_m["gt"], test_m["preds"], test_ds)
    print_per_class_accuracy(pc_results)

    # Confusion matrix.
    best_epoch = ckpt["training"]["epoch"]
    plot_confusion(
        test_m["gt"], test_m["preds"], test_ds,
        title=(f"Baseline CNN — test acc {test_m['acc']:.3f}  "
               f"({test_m['n']} windows,  seed={seed},  best epoch={best_epoch})"),
        out_path=seed_dir / "confusion.png",
    )

    # Per-class accuracy JSON (the baseline fingerprint for compression).
    pc_json: dict = {
        "seed":          seed,
        "overall_acc":   test_m["acc"],
        "overall_n":     test_m["n"],
        "c1_acc":        (sum(e["correct"] for e in pc_results["C1"])
                          / max(1, sum(e["total"] for e in pc_results["C1"]))),
        "c2_acc":        (sum(e["correct"] for e in pc_results["C2"])
                          / max(1, sum(e["total"] for e in pc_results["C2"]))),
        "per_class":     {camp: pc_results[camp] for camp in ("C1", "C2")},
    }
    (seed_dir / "per_class_acc.json").write_text(json.dumps(pc_json, indent=2))

    print(f"\n  Saved seed_{seed}/ outputs to {seed_dir}")
    return log_rows, pc_results


# ---------------------------------------------------------------------------
# Cross-seed summary
# ---------------------------------------------------------------------------


def write_summary(
    all_pc: dict[int, dict[str, list[dict]]],
    all_rows: dict[int, list[dict]],
    out_dir: Path,
) -> None:
    """Aggregate per-class accuracy statistics across seeds."""
    # Collect per-class accuracy per seed.
    # Structure: class_name → list of per-seed acc values
    class_accs: dict[str, list[float]] = defaultdict(list)
    class_totals: dict[str, int] = {}
    class_camp: dict[str, str] = {}

    for seed, pc_results in all_pc.items():
        for camp in ("C1", "C2"):
            for e in pc_results[camp]:
                cls = e["class"]
                class_accs[cls].append(e["acc"])
                class_totals[cls] = e["total"]  # approx — same split
                class_camp[cls] = camp

    summary: dict = {
        "n_seeds": len(all_pc),
        "seeds": list(all_pc.keys()),
        "per_class": [],
    }
    for cls in sorted(class_accs.keys()):
        accs = class_accs[cls]
        summary["per_class"].append({
            "class":    cls,
            "campaign": class_camp[cls],
            "median":   float(np.median(accs)),
            "mean":     float(np.mean(accs)),
            "std":      float(np.std(accs)),
            "min":      float(np.min(accs)),
            "max":      float(np.max(accs)),
            "n_windows_approx": class_totals[cls],
        })

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary_curves(all_rows, out_dir / "summary_curves.png")
    print(f"\nSaved cross-seed summary to {out_dir}/summary.json")

    # Print a compact table.
    print("\nCross-seed per-class summary (median ± std):")
    for camp_tag, title in [("C1", "C1 micro"), ("C2", "C2 macro")]:
        entries = [e for e in summary["per_class"] if e["campaign"] == camp_tag]
        print(f"\n  {title}:")
        for e in entries:
            print(f"    {e['class']}  median={e['median']:.3f}  "
                  f"std={e['std']:.3f}  [{e['min']:.3f}–{e['max']:.3f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = DEFAULT_CONFIG
    print(f"Device:   {args.device}")
    print(f"Pipeline: sensor_set={cfg.sensor_set}  "
          f"channels={cfg.channels_per_window()}  "
          f"grid={cfg.grid_size}×{cfg.grid_size}")

    split = load_split()

    if args.quick:
        split = Split(
            train=split.train[:2], val=split.val[:1], test=split.test[:1],
            campaigns=split.campaigns, seed=split.seed,
        )
        if args.class_filter is None:
            args.class_filter = ["M01", "M02", "M03", "A01", "A02", "A03"]
        if args.repetition_filter is None:
            args.repetition_filter = ["01", "02"]
        if args.epochs == 40:
            args.epochs = 3
        args.seeds = [args.seeds[0]]  # single seed in quick mode
        print("  (quick mode: trimmed split, 3+3 classes, 2 reps, 3 epochs)")

    print(f"Split: train={len(split.train)} val={len(split.val)} "
          f"test={len(split.test)} subjects")
    print(f"Seeds: {args.seeds}  Epochs: {args.epochs}\n")

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets once — all seeds share the same data.
    print("Building datasets (cache built on first run)...")
    train_ds, val_ds, test_ds = build_datasets(split, cfg, args)
    print(f"  Classes: C1={train_ds.num_c1_classes}  C2={train_ds.num_c2_classes}  "
          f"total={train_ds.num_classes}")
    print(f"  C1 classes: {list(train_ds.c1_classes)}")
    print(f"  C2 classes: {list(train_ds.c2_classes)}")

    all_log_rows: dict[int, list[dict]] = {}
    all_pc_results: dict[int, dict[str, list[dict]]] = {}

    for seed in args.seeds:
        print(f"\n{'='*90}")
        print(f"SEED {seed}")
        print(f"{'='*90}")
        log_rows, pc_results = run_one_seed(
            seed=seed,
            split=split,
            cfg=cfg,
            args=args,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            out_dir=out_dir,
        )
        all_log_rows[seed] = log_rows
        all_pc_results[seed] = pc_results

    if len(args.seeds) > 1:
        write_summary(all_pc_results, all_log_rows, out_dir)
    else:
        print(f"\nTip: run with '--seeds 42 43 44' to get the multi-seed "
              f"summary needed for the compression baseline fingerprint.")


if __name__ == "__main__":
    main()
