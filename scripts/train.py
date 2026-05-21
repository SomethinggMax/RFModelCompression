"""Train the two-head baseline CNN on C1 + C2.

Pipeline:

  1. Load the subject-independent split from ``code/splits/splits.json``.
  2. Build three ``SubjectIndependentRadarDataset`` instances (train / val /
     test) and wrap each one in ``TwoHeadRadarDataset`` so we get
     ``(tensor, campaign_id, within_class_id)`` per item.
  3. Build a ``WeightedRandomSampler`` on the train set that balances
     campaigns 50/50 and classes uniformly within each campaign.
  4. Build the ``TwoHeadCNN`` and an AdamW optimiser with a cosine LR
     schedule.
  5. Train for ``--epochs`` epochs, evaluating on val every epoch.
  6. At the end, evaluate on the held-out test fold and save per-head
     confusion matrices as PNGs into ``code/outputs/``.

Usage (run from the ``code/`` directory)::

    # full run, all subjects, all classes — this is what the meeting wants
    uv run python -m scripts.train

    # quick smoke run for iterating — narrow to a few subjects and reps
    uv run python -m scripts.train --quick

    # tweak knobs
    uv run python -m scripts.train --epochs 30 --batch-size 64 --lr 5e-4

Output (saved to ``code/outputs/``):

  * ``train_log.csv``                – per-epoch train/val loss + accuracy
  * ``train_curves.png``             – matplotlib plot of the same
  * ``confusion_c1.png``, ``confusion_c2.png`` – test-set confusion matrices
  * ``baseline_cnn.pt``              – final model checkpoint
"""

from __future__ import annotations

# Non-interactive matplotlib backend — important on Windows where the script
# might be run from a non-GUI terminal. Set BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rfbc.config import DEFAULT_CONFIG, PipelineConfig
from rfbc.data.dataset import SubjectIndependentRadarDataset
from rfbc.data.splits import Split, load_split
from rfbc.data.two_head import TwoHeadRadarDataset, make_two_head_sampler
from rfbc.models.two_head_cnn import TwoHeadCNN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    # Speed knobs
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run: 2 train + 1 val + 1 test subject, "
                        "3 classes per campaign, 2 reps, 3 epochs. ~5 minutes.")
    p.add_argument("--max-subjects-per-fold", type=int, default=None,
                   help="Cap subjects per fold (handy for iterating).")
    p.add_argument("--class-filter", nargs="+", default=None,
                   help="Restrict to specific class folder names (e.g. M01 M02 A01).")
    p.add_argument("--repetition-filter", nargs="+", default=None,
                   help="Restrict to specific repetition folders (e.g. 01 02).")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Keep at 0 on Windows when first "
                        "building the cache to avoid pickling issues.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    p.add_argument("--out-dir", default=None,
                   help="Where to save logs, curves, and confusion matrices. "
                        "Defaults to code/outputs/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trim_split(split: Split, n: int) -> Split:
    """Cap each fold of a Split to its first ``n`` subjects."""
    return Split(
        train=split.train[:n],
        val=split.val[:max(1, n // 2)],
        test=split.test[:max(1, n // 2)],
        campaigns=split.campaigns,
        seed=split.seed,
    )


def _collate(batch):
    """Stack (tensor, camp, within) triples into batched tensors."""
    tensors = torch.stack([b[0] for b in batch], dim=0)
    campaigns = torch.tensor([b[1] for b in batch], dtype=torch.long)
    within = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return tensors, campaigns, within


def make_datasets(
    split: Split, cfg: PipelineConfig, args: argparse.Namespace,
) -> tuple[TwoHeadRadarDataset, TwoHeadRadarDataset, TwoHeadRadarDataset]:
    """Build train / val / test datasets, wrapped for two-head training."""
    class_filter = tuple(args.class_filter) if args.class_filter else None
    rep_filter = tuple(args.repetition_filter) if args.repetition_filter else None

    print(f"Building train dataset... (forces cache build on first run)",
          flush=True)
    train_base = SubjectIndependentRadarDataset(
        split, fold="train", cfg=cfg,
        class_filter=class_filter, repetition_filter=rep_filter,
    )
    val_base = SubjectIndependentRadarDataset(
        split, fold="val", cfg=cfg,
        class_filter=class_filter, repetition_filter=rep_filter,
    )
    test_base = SubjectIndependentRadarDataset(
        split, fold="test", cfg=cfg,
        class_filter=class_filter, repetition_filter=rep_filter,
    )

    t0 = time.time()
    train_ds = TwoHeadRadarDataset(train_base)
    print(f"  train wrapper ready ({len(train_ds)} windows) in {time.time()-t0:.1f}s",
          flush=True)
    t0 = time.time()
    val_ds = TwoHeadRadarDataset(val_base)
    print(f"  val wrapper ready ({len(val_ds)} windows) in {time.time()-t0:.1f}s",
          flush=True)
    t0 = time.time()
    test_ds = TwoHeadRadarDataset(test_base)
    print(f"  test wrapper ready ({len(test_ds)} windows) in {time.time()-t0:.1f}s",
          flush=True)
    return train_ds, val_ds, test_ds


def two_head_loss(
    c1_logits: torch.Tensor, c2_logits: torch.Tensor,
    campaigns: torch.Tensor, within: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy loss masked by campaign.

    Returns ``(summed_loss, batch_size)``; the caller divides by batch size.
    """
    mask_c1 = campaigns == 0
    mask_c2 = campaigns == 1
    loss = c1_logits.new_zeros(())
    if mask_c1.any():
        loss = loss + F.cross_entropy(
            c1_logits[mask_c1], within[mask_c1], reduction="sum",
        )
    if mask_c2.any():
        loss = loss + F.cross_entropy(
            c2_logits[mask_c2], within[mask_c2], reduction="sum",
        )
    return loss, int(campaigns.numel())


def train_one_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_seen = 0
    correct_c1 = 0; total_c1 = 0
    correct_c2 = 0; total_c2 = 0

    for x, camp, within in loader:
        x = x.to(device, non_blocking=True)
        camp = camp.to(device, non_blocking=True)
        within = within.to(device, non_blocking=True)

        c1_logits, c2_logits = model(x)
        loss_sum, batch_n = two_head_loss(c1_logits, c2_logits, camp, within)
        loss = loss_sum / max(1, batch_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss_sum.detach())
        total_seen += batch_n

        m1 = camp == 0
        m2 = camp == 1
        if m1.any():
            correct_c1 += int((c1_logits[m1].argmax(1) == within[m1]).sum())
            total_c1 += int(m1.sum())
        if m2.any():
            correct_c2 += int((c2_logits[m2].argmax(1) == within[m2]).sum())
            total_c2 += int(m2.sum())

    return {
        "loss":   total_loss / max(1, total_seen),
        "acc_c1": correct_c1 / total_c1 if total_c1 else float("nan"),
        "acc_c2": correct_c2 / total_c2 if total_c2 else float("nan"),
        "n_c1":   total_c1,
        "n_c2":   total_c2,
    }


@torch.no_grad()
def evaluate(model, loader, device, *, collect_preds: bool = False):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    correct_c1 = 0; total_c1 = 0
    correct_c2 = 0; total_c2 = 0

    preds_c1, gt_c1, preds_c2, gt_c2 = [], [], [], []

    for x, camp, within in loader:
        x = x.to(device, non_blocking=True)
        camp = camp.to(device, non_blocking=True)
        within = within.to(device, non_blocking=True)

        c1_logits, c2_logits = model(x)
        loss_sum, batch_n = two_head_loss(c1_logits, c2_logits, camp, within)
        total_loss += float(loss_sum)
        total_seen += batch_n

        m1 = camp == 0
        m2 = camp == 1
        if m1.any():
            p1 = c1_logits[m1].argmax(1)
            correct_c1 += int((p1 == within[m1]).sum())
            total_c1 += int(m1.sum())
            if collect_preds:
                preds_c1.append(p1.cpu().numpy())
                gt_c1.append(within[m1].cpu().numpy())
        if m2.any():
            p2 = c2_logits[m2].argmax(1)
            correct_c2 += int((p2 == within[m2]).sum())
            total_c2 += int(m2.sum())
            if collect_preds:
                preds_c2.append(p2.cpu().numpy())
                gt_c2.append(within[m2].cpu().numpy())

    out: dict = {
        "loss":   total_loss / max(1, total_seen),
        "acc_c1": correct_c1 / total_c1 if total_c1 else float("nan"),
        "acc_c2": correct_c2 / total_c2 if total_c2 else float("nan"),
        "n_c1":   total_c1,
        "n_c2":   total_c2,
    }
    if collect_preds:
        out["preds_c1"] = np.concatenate(preds_c1) if preds_c1 else np.array([], dtype=int)
        out["gt_c1"]    = np.concatenate(gt_c1)    if gt_c1    else np.array([], dtype=int)
        out["preds_c2"] = np.concatenate(preds_c2) if preds_c2 else np.array([], dtype=int)
        out["gt_c2"]    = np.concatenate(gt_c2)    if gt_c2    else np.array([], dtype=int)
    return out


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str,
                   out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.4),
                                    max(5, len(class_names) * 0.4)))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Annotate cells when the matrix is small enough to read.
    if len(class_names) <= 25:
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = int(cm[i, j])
                if v == 0:
                    continue
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if gt.size == 0:
        return cm
    for g, p in zip(gt, pred):
        if 0 <= g < n_classes and 0 <= p < n_classes:
            cm[int(g), int(p)] += 1
    return cm


def plot_curves(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_loss.plot(epochs, [r["val_loss"]   for r in rows], label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss (per-sample CE, summed across both heads)")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [r["train_acc_c1"] for r in rows],
                label="train C1", linestyle="-",  color="C0")
    ax_acc.plot(epochs, [r["val_acc_c1"]   for r in rows],
                label="val C1",   linestyle="--", color="C0")
    ax_acc.plot(epochs, [r["train_acc_c2"] for r in rows],
                label="train C2", linestyle="-",  color="C1")
    ax_acc.plot(epochs, [r["val_acc_c2"]   for r in rows],
                label="val C2",   linestyle="--", color="C1")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Per-head accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Reproducibility — sampler and Adam are both stochastic.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = DEFAULT_CONFIG
    print(f"Device:          {args.device}")
    print(f"Pipeline:        sensor_set={cfg.sensor_set}  "
          f"channels={cfg.channels_per_window()}  "
          f"grid={cfg.grid_size}x{cfg.grid_size}")

    split = load_split()
    if args.quick:
        # Tiny smoke version: 2 train + 1 val + 1 test subject,
        # 3 C1 + 3 C2 classes, 2 reps, 3 epochs.
        split = Split(
            train=split.train[:2], val=split.val[:1], test=split.test[:1],
            campaigns=split.campaigns, seed=split.seed,
        )
        if args.class_filter is None:
            args.class_filter = ["M01", "M02", "M03", "A01", "A02", "A03"]
        if args.repetition_filter is None:
            args.repetition_filter = ["01", "02"]
        if args.epochs == 20:  # default — override only if user didn't change it
            args.epochs = 3
        print("  (quick mode: trimmed split, classes, reps, epochs)")
    elif args.max_subjects_per_fold is not None:
        split = _trim_split(split, args.max_subjects_per_fold)

    print(f"Split (subject IDs):")
    print(f"  train ({len(split.train)}): {list(split.train)}")
    print(f"  val   ({len(split.val)})  : {list(split.val)}")
    print(f"  test  ({len(split.test)}) : {list(split.test)}")
    print()

    train_ds, val_ds, test_ds = make_datasets(split, cfg, args)

    # Class distributions (sanity-check the imbalance).
    train_counts = Counter(train_ds.meta())
    print(f"\nTrain class distribution (campaign, within_id) -> count:")
    for (camp, w), n in sorted(train_counts.items()):
        camp_str = "C1" if camp == 0 else "C2"
        name = (train_ds.c1_classes[w] if camp == 0 else train_ds.c2_classes[w])
        print(f"  {camp_str}/{name:>4} : {n}")
    n_c1 = sum(c for (camp, _), c in train_counts.items() if camp == 0)
    n_c2 = sum(c for (camp, _), c in train_counts.items() if camp == 1)
    print(f"  TOTAL  C1: {n_c1}   C2: {n_c2}   (raw ratio C2/C1 = "
          f"{(n_c2 / max(1, n_c1)):.2f})")

    # ---------- DataLoaders ----------
    sampler = make_two_head_sampler(train_ds, campaign_balance=0.5)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        collate_fn=_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate, drop_last=False,
    )

    # ---------- Model / optim ----------
    x_probe, _, _ = train_ds[0]
    in_channels = int(x_probe.shape[0])
    model = TwoHeadCNN(
        in_channels=in_channels,
        num_c1_classes=train_ds.num_c1_classes,
        num_c2_classes=train_ds.num_c2_classes,
    ).to(args.device)
    print(f"\nModel: TwoHeadCNN  ({model.num_parameters:,} parameters)")
    print(f"  in_channels={in_channels}  c1_classes={train_ds.num_c1_classes}  "
          f"c2_classes={train_ds.num_c2_classes}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )

    # ---------- Output paths ----------
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict] = []

    print(f"\nStarting training for {args.epochs} epochs on {args.device}...")
    print("=" * 90)
    epoch_fmt = (
        "epoch {ep:>3}/{tot:<3}  lr={lr:.2e}  "
        "train  loss={tl:.4f}  c1={tac1:.3f}  c2={tac2:.3f}    "
        "val  loss={vl:.4f}  c1={vac1:.3f}  c2={vac2:.3f}    [{dt:5.1f}s]"
    )

    best_val_avg = -1.0
    best_ckpt = out_dir / "baseline_cnn.pt"

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        scheduler.step()
        dt = time.time() - t0

        print(epoch_fmt.format(
            ep=ep, tot=args.epochs,
            lr=optimizer.param_groups[0]["lr"],
            tl=train_metrics["loss"],
            tac1=train_metrics["acc_c1"], tac2=train_metrics["acc_c2"],
            vl=val_metrics["loss"],
            vac1=val_metrics["acc_c1"], vac2=val_metrics["acc_c2"],
            dt=dt,
        ), flush=True)

        log_rows.append({
            "epoch": ep,
            "lr":          optimizer.param_groups[0]["lr"],
            "train_loss":  train_metrics["loss"],
            "train_acc_c1": train_metrics["acc_c1"],
            "train_acc_c2": train_metrics["acc_c2"],
            "val_loss":    val_metrics["loss"],
            "val_acc_c1":  val_metrics["acc_c1"],
            "val_acc_c2":  val_metrics["acc_c2"],
            "epoch_seconds": dt,
        })

        # Keep checkpoint of best mean-of-heads val accuracy.
        val_avg = 0.5 * (val_metrics["acc_c1"] + val_metrics["acc_c2"])
        if val_avg > best_val_avg:
            best_val_avg = val_avg
            torch.save({
                "state_dict": model.state_dict(),
                "config": asdict(cfg) | {
                    "in_channels": in_channels,
                    "c1_classes": list(train_ds.c1_classes),
                    "c2_classes": list(train_ds.c2_classes),
                },
                "epoch": ep,
                "val_acc_c1": val_metrics["acc_c1"],
                "val_acc_c2": val_metrics["acc_c2"],
            }, best_ckpt)

    # ---------- Save logs and curves ----------
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nSaved train log to {log_path}")

    curves_path = out_dir / "train_curves.png"
    plot_curves(log_rows, curves_path)
    print(f"Saved training curves to {curves_path}")

    # ---------- Final test evaluation ----------
    print("\n" + "=" * 90)
    print("Final test-set evaluation (using best-val checkpoint)")
    print("=" * 90)

    ckpt = torch.load(best_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    test_metrics = evaluate(model, test_loader, args.device, collect_preds=True)
    print(f"test  loss={test_metrics['loss']:.4f}  "
          f"c1 acc={test_metrics['acc_c1']:.3f} ({test_metrics['n_c1']} samples)  "
          f"c2 acc={test_metrics['acc_c2']:.3f} ({test_metrics['n_c2']} samples)")

    # Confusion matrices.
    cm_c1 = confusion_matrix(test_metrics["gt_c1"], test_metrics["preds_c1"],
                             train_ds.num_c1_classes)
    cm_c2 = confusion_matrix(test_metrics["gt_c2"], test_metrics["preds_c2"],
                             train_ds.num_c2_classes)

    plot_confusion(cm_c1, list(train_ds.c1_classes),
                   f"C1 gestures — test acc {test_metrics['acc_c1']:.3f}  "
                   f"({test_metrics['n_c1']} windows)",
                   out_dir / "confusion_c1.png")
    plot_confusion(cm_c2, list(train_ds.c2_classes),
                   f"C2 activities — test acc {test_metrics['acc_c2']:.3f}  "
                   f"({test_metrics['n_c2']} windows)",
                   out_dir / "confusion_c2.png")
    print(f"Saved confusion matrices to {out_dir}/confusion_c1.png and confusion_c2.png")

    # Summary JSON for easy copy-paste / programmatic use.
    summary = {
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_epoch": ckpt["epoch"],
        "best_val_acc_c1": ckpt["val_acc_c1"],
        "best_val_acc_c2": ckpt["val_acc_c2"],
        "test_acc_c1": test_metrics["acc_c1"],
        "test_acc_c2": test_metrics["acc_c2"],
        "test_n_c1": test_metrics["n_c1"],
        "test_n_c2": test_metrics["n_c2"],
        "model_params": model.num_parameters,
        "channels": in_channels,
        "c1_classes": list(train_ds.c1_classes),
        "c2_classes": list(train_ds.c2_classes),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_dir}/train_summary.json")


if __name__ == "__main__":
    main()
