"""End-to-end smoke test of the preprocessing + training pipeline.

This script intentionally runs on a tiny subset (a handful of subjects, a
handful of classes, one repetition each, one epoch) so that any plumbing
problems surface fast: file paths, tensor shapes, dependency mismatches,
out-of-memory issues. It is **not** a baseline experiment.

Usage::

    python -m scripts.smoke_test
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from rfbc.config import DEFAULT_CONFIG
from rfbc.data.dataset import SubjectIndependentRadarDataset
from rfbc.data.splits import load_split
from rfbc.models.stub_cnn import StubCNN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-subjects-per-fold", type=int, default=2)
    p.add_argument("--classes", nargs="+", default=["M01", "M02", "M03"])
    p.add_argument("--repetitions", nargs="+", default=["01"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    cfg = DEFAULT_CONFIG
    print(f"sensor_set: {cfg.sensor_set}  channels/window: {cfg.channels_per_window()}")

    split = load_split()
    # Trim to the smallest possible dataset for a smoke test.
    trimmed = type(split)(
        train=split.train[: args.max_subjects_per_fold],
        val=split.val[: max(1, args.max_subjects_per_fold)],
        test=split.test[: max(1, args.max_subjects_per_fold)],
        campaigns=split.campaigns,
        seed=split.seed,
    )

    train_ds = SubjectIndependentRadarDataset(
        trimmed, fold="train", cfg=cfg,
        class_filter=tuple(args.classes),
        repetition_filter=tuple(args.repetitions),
    )
    val_ds = SubjectIndependentRadarDataset(
        trimmed, fold="val", cfg=cfg,
        class_filter=tuple(args.classes),
        repetition_filter=tuple(args.repetitions),
    )

    print(f"train items: {len(train_ds)}  val items: {len(val_ds)}")
    print(f"classes seen: {train_ds.class_names()}")
    if len(train_ds) == 0:
        raise SystemExit("Smoke test aborted: no training items. "
                         "Check class names in --classes match disk folder names.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    # Peek one batch to confirm shapes.
    x, y = next(iter(train_loader))
    print(f"sample batch: x={tuple(x.shape)}, dtype={x.dtype}, y={tuple(y.shape)}")

    in_channels = x.shape[1]
    num_classes = max(train_ds.num_classes, val_ds.num_classes)
    model = StubCNN(in_channels=in_channels, num_classes=num_classes).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ----- train -----
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += loss.item() * xb.size(0)
        print(f"epoch {epoch + 1}: train loss {running / len(train_ds):.4f}")

    # ----- per-class eval on val -----
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            preds = model(xb).argmax(dim=1)
            for label_id, pred in zip(yb.cpu().tolist(), preds.cpu().tolist()):
                total[label_id] += 1
                if pred == label_id:
                    correct[label_id] += 1

    names = train_ds.class_names()
    print("\nPer-class accuracy (val):")
    for label_id, n in sorted(total.items()):
        name = names[label_id] if label_id < len(names) else f"<id {label_id}>"
        acc = correct[label_id] / n if n else float("nan")
        print(f"  {name}: {correct[label_id]}/{n} = {acc:.3f}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
