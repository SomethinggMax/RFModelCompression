"""Compression sweep — pruning, INT8 PTQ, and combined, evaluated class-wise.

This is the core experiment of the study
"Evaluating the Class-wise Impact of Model Compression in RF-Based HAR".
It takes the trained baseline checkpoints (one per seed) and produces a grid
of *compressed* models, evaluating every one of them per class on the held-out
test fold so that class-wise degradation can be compared against the
pre-compression baseline fingerprint.

Compression axes
----------------
1. Pruning  — global unstructured L1 magnitude pruning over every Conv2d and
   Linear weight, ONE-SHOT (no fine-tuning), at sparsity levels
   30 / 50 / 70 / 80 / 90 / 95 %. One-shot isolates compression's raw effect
   (the Hooker et al. "what does compression forget" framing).
2. Quantization — INT8 static post-training quantization via FX graph mode
   (``torch.ao.quantization.quantize_fx``), calibrated on the val fold.
   Conv-BN-ReLU folding and quant/dequant insertion are handled by FX.
3. Combined — prune at each sparsity level, then apply INT8 PTQ on top.

Every configuration is run for each seed in ``--seeds`` (default 42 43 44).
Per-class test accuracy is saved for each (seed, config), then aggregated into
a cross-seed summary (median ± std per class, macro-averaged campaign accuracy,
and deltas versus the baseline).

There are two evaluation protocols:

* **Single split** (default): compresses ``outputs/baseline/seed_{N}/model.pt``
  and evaluates on the one locked split. Error bars come from seeds only.
* **Cross-validation** (``--cv``): compresses every ``outputs/cv/fold_k/
  seed_S/model.pt`` and evaluates each on its own held-out fold, aggregating
  across folds x seeds. Error bars then carry subject variation too, matching
  the CV baseline. This is the recommended protocol for the thesis.

Usage (run from the ``code/`` directory)
-----------------------------------------
    uv run python -m scripts.compress_sweep                  # single split, 3 seeds
    uv run python -m scripts.compress_sweep --quick          # tiny wiring check
    uv run python -m scripts.compress_sweep --skip-quant     # pruning only

    uv run python -m scripts.compress_sweep --cv             # CV: all folds x seeds
    uv run python -m scripts.compress_sweep --cv --quick     # CV wiring check (fold 0)
    uv run python -m scripts.compress_sweep --cv --skip-quant  # CV, pruning only

Output
------
    Single split -> ``code/outputs/compression/``
    CV (--cv)    -> ``code/outputs/compression_cv/``
        seed_{N}/ or fold_{k}/seed_{S}/
            results.json        every config's overall + per-class accuracy
        summary.json            aggregation across runs + deltas vs baseline
        degradation_curves.png  macro-avg C1/C2 accuracy vs sparsity
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from rfbc.config import DEFAULT_CONFIG, PipelineConfig
from rfbc.data.combined import CombinedRadarDataset
from rfbc.data.dataset import SubjectIndependentRadarDataset
from rfbc.data.splits import Split, load_split
from rfbc.models.baseline_cnn import BaselineCNN

# Reuse the exact evaluation + per-class logic from the baseline trainer so the
# numbers are directly comparable to the baseline fingerprint.
from scripts.train_baseline import evaluate, per_class_accuracy


DEFAULT_SPARSITIES = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Class-wise compression sweep.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--sparsities", type=float, nargs="+", default=DEFAULT_SPARSITIES,
                   help="Global unstructured pruning sparsity levels.")
    p.add_argument("--skip-quant", action="store_true",
                   help="Skip INT8 quantization and combined configs.")
    p.add_argument("--skip-prune", action="store_true",
                   help="Skip pruning and combined configs.")
    p.add_argument("--calib-batches", type=int, default=16,
                   help="Number of val batches used to calibrate INT8 PTQ.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--baseline-dir", default=None,
                   help="Where the baseline seed_{N}/model.pt live. "
                        "Defaults to code/outputs/baseline/.")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to code/outputs/compression/ (single split) "
                        "or code/outputs/compression_cv/ (--cv).")

    # --- CV mode: compress per fold against the cross-validation checkpoints ---
    p.add_argument("--cv", action="store_true",
                   help="Run per-fold against the CV checkpoints produced by "
                        "scripts.train_cv. Reads --cv-dir/cv_summary.json, "
                        "rebuilds each fold's split, loads "
                        "fold_k/seed_S/model.pt, and evaluates compression on "
                        "that fold's own test set. Aggregates across folds x "
                        "seeds, so deltas carry subject-variation error bars.")
    p.add_argument("--cv-dir", default=None,
                   help="Where the CV outputs live (cv_summary.json + "
                        "fold_k/seed_S/model.pt). Defaults to code/outputs/cv/.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny wiring check: 1 seed, sparsities 0.5 0.9, "
                        "trimmed split, fewer calib batches.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset construction (val for calibration, test for evaluation)
# ---------------------------------------------------------------------------


def build_eval_datasets(
    split: Split, cfg: PipelineConfig,
    class_filter=None, rep_filter=None,
) -> tuple[CombinedRadarDataset, CombinedRadarDataset]:
    """Return (val_ds, test_ds) as CombinedRadarDatasets."""
    out = {}
    for fold in ("val", "test"):
        print(f"  Building {fold} dataset...", flush=True)
        t0 = time.time()
        base = SubjectIndependentRadarDataset(
            split, fold=fold, cfg=cfg,
            class_filter=class_filter, repetition_filter=rep_filter,
        )
        out[fold] = CombinedRadarDataset(base)
        print(f"    {fold}: {len(out[fold])} windows in {time.time()-t0:.1f}s",
              flush=True)
    return out["val"], out["test"]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def load_baseline(ckpt_path: Path, device: str) -> tuple[BaselineCNN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    model = BaselineCNN(
        in_channels=arch["in_channels"],
        num_classes=arch["num_classes"],
        dropout=arch["dropout"],
        width_mult=arch.get("width_mult", 1.0),
        conv_dropout=arch.get("conv_dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Every Conv2d and Linear weight tensor — the global pruning pool."""
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    return params


def apply_global_pruning(model: nn.Module, amount: float) -> float:
    """Global unstructured L1 pruning, made permanent. Returns true sparsity."""
    params = prunable_params(model)
    prune.global_unstructured(
        params, pruning_method=prune.L1Unstructured, amount=amount,
    )
    # Bake the masks into the weights and drop the reparametrisation.
    for module, name in params:
        prune.remove(module, name)
    # Measure the sparsity actually achieved over the pruned pool.
    n_zero = sum(int((m.weight == 0).sum()) for m, _ in params)
    n_tot = sum(m.weight.numel() for m, _ in params)
    return n_zero / max(1, n_tot)


# ---------------------------------------------------------------------------
# INT8 static PTQ (FX graph mode)
# ---------------------------------------------------------------------------


def quantize_int8(
    model: nn.Module,
    calib_loader: DataLoader,
    example_input: torch.Tensor,
    n_batches: int,
) -> nn.Module:
    """INT8 static PTQ via FX graph mode. Returns a CPU quantized model.

    FX handles Conv-BN-ReLU fusion and quant/dequant stub insertion
    automatically, so the float ``BaselineCNN`` needs no modification.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    model = copy.deepcopy(model).cpu().eval()
    qmap = get_default_qconfig_mapping("fbgemm")
    prepared = prepare_fx(model, qmap, example_inputs=(example_input.cpu(),))

    # Calibrate on a handful of val batches.
    prepared.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= n_batches:
                break
            prepared(x.cpu())

    return convert_fx(prepared)


# ---------------------------------------------------------------------------
# Single evaluation → per-class record
# ---------------------------------------------------------------------------


def eval_config(
    model: nn.Module, loader: DataLoader, device: str,
    test_ds: CombinedRadarDataset,
) -> dict:
    """Evaluate one model, return overall + per-campaign + per-class accuracy."""
    res = evaluate(model, loader, device, dataset=test_ds, collect_preds=True)
    pc = per_class_accuracy(res["gt"], res["preds"], test_ds)
    c1_corr = sum(e["correct"] for e in pc["C1"])
    c1_tot = sum(e["total"] for e in pc["C1"])
    c2_corr = sum(e["correct"] for e in pc["C2"])
    c2_tot = sum(e["total"] for e in pc["C2"])
    # Macro-averaged campaign accuracy (mean of per-class accuracies).
    c1_macro = float(np.mean([e["acc"] for e in pc["C1"]])) if pc["C1"] else 0.0
    c2_macro = float(np.mean([e["acc"] for e in pc["C2"]])) if pc["C2"] else 0.0
    return {
        "overall_acc": res["acc"],
        "overall_n": res["n"],
        "c1_acc": c1_corr / max(1, c1_tot),
        "c2_acc": c2_corr / max(1, c2_tot),
        "c1_macro_acc": c1_macro,
        "c2_macro_acc": c2_macro,
        "per_class": {c: pc[c] for c in ("C1", "C2")},
    }


def per_class_acc_map(record: dict) -> dict[str, float]:
    """Flatten a record's per-class accuracies into {class_name: acc}."""
    out = {}
    for camp in ("C1", "C2"):
        for e in record["per_class"][camp]:
            out[e["class"]] = e["acc"]
    return out


# ---------------------------------------------------------------------------
# One seed: build every compressed model and evaluate it
# ---------------------------------------------------------------------------


def compress_checkpoint(
    ckpt_path: Path, args: argparse.Namespace,
    val_ds: CombinedRadarDataset, test_ds: CombinedRadarDataset,
) -> dict:
    """Load one baseline checkpoint, build every compressed variant, evaluate.

    Returns the ``configs`` dict (baseline + prune sweep + INT8 + combined),
    each entry holding overall / per-campaign / per-class accuracy on
    ``test_ds``. Calibration for INT8 PTQ uses ``val_ds``. This is the shared
    core used by both the single-split path (``run_seed``) and the CV path
    (``run_cv``).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing baseline checkpoint: {ckpt_path}")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    example_input, _ = test_ds[0]
    example_input = example_input.unsqueeze(0)

    configs: dict[str, dict] = {}

    # --- Baseline parity re-eval (float, uncompressed) --------------------
    base_model, _ = load_baseline(ckpt_path, device)
    print(f"  [baseline] evaluating...", flush=True)
    configs["baseline"] = eval_config(base_model, test_loader, device, test_ds)
    configs["baseline"]["sparsity"] = 0.0
    print(f"    overall={configs['baseline']['overall_acc']:.3f}  "
          f"C1macro={configs['baseline']['c1_macro_acc']:.3f}  "
          f"C2macro={configs['baseline']['c2_macro_acc']:.3f}", flush=True)

    # --- Pruning sweep (one-shot) -----------------------------------------
    if not args.skip_prune:
        for s in args.sparsities:
            model = copy.deepcopy(base_model)
            true_s = apply_global_pruning(model, s)
            tag = f"prune_s{int(round(s*100)):02d}"
            print(f"  [{tag}] true_sparsity={true_s:.3f}  evaluating...", flush=True)
            rec = eval_config(model, test_loader, device, test_ds)
            rec["sparsity"] = true_s
            configs[tag] = rec
            print(f"    overall={rec['overall_acc']:.3f}  "
                  f"C1macro={rec['c1_macro_acc']:.3f}  "
                  f"C2macro={rec['c2_macro_acc']:.3f}", flush=True)

    # --- INT8 PTQ (CPU) ---------------------------------------------------
    if not args.skip_quant:
        print(f"  [quant_int8] calibrating + evaluating (CPU)...", flush=True)
        qmodel = quantize_int8(base_model, val_loader, example_input,
                               args.calib_batches)
        rec = eval_config(qmodel, test_loader, "cpu", test_ds)
        rec["sparsity"] = 0.0
        configs["quant_int8"] = rec
        print(f"    overall={rec['overall_acc']:.3f}  "
              f"C1macro={rec['c1_macro_acc']:.3f}  "
              f"C2macro={rec['c2_macro_acc']:.3f}", flush=True)

        # --- Combined: prune then INT8 PTQ --------------------------------
        if not args.skip_prune:
            for s in args.sparsities:
                model = copy.deepcopy(base_model)
                true_s = apply_global_pruning(model, s)
                tag = f"combined_s{int(round(s*100)):02d}_int8"
                print(f"  [{tag}] calibrating + evaluating (CPU)...", flush=True)
                qmodel = quantize_int8(model, val_loader, example_input,
                                       args.calib_batches)
                rec = eval_config(qmodel, test_loader, "cpu", test_ds)
                rec["sparsity"] = true_s
                configs[tag] = rec
                print(f"    overall={rec['overall_acc']:.3f}  "
                      f"C1macro={rec['c1_macro_acc']:.3f}  "
                      f"C2macro={rec['c2_macro_acc']:.3f}", flush=True)

    return configs


def run_seed(
    seed: int, args: argparse.Namespace,
    val_ds: CombinedRadarDataset, test_ds: CombinedRadarDataset,
    baseline_dir: Path, out_dir: Path,
) -> dict:
    """Single-split path: compress the seed_{seed} baseline checkpoint."""
    ckpt_path = baseline_dir / f"seed_{seed}" / "model.pt"
    configs = compress_checkpoint(ckpt_path, args, val_ds, test_ds)
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "results.json").write_text(
        json.dumps({"seed": seed, "configs": configs}, indent=2))
    print(f"  Saved seed_{seed}/results.json", flush=True)
    return configs


def run_cv(args: argparse.Namespace, cfg: PipelineConfig, out_dir: Path) -> None:
    """CV path: compress each fold's checkpoint, evaluate on that fold's test.

    Reads ``--cv-dir/cv_summary.json`` (written by ``scripts.train_cv``) for the
    fold definitions and the list of seeds, then for every (fold, seed) loads
    ``fold_k/seed_S/model.pt`` and runs the same compression grid as the
    single-split path. All (fold x seed) runs are aggregated together, so the
    per-class deltas carry the same subject-variation error bars as the CV
    baseline.
    """
    cv_dir = Path(args.cv_dir) if args.cv_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "cv")
    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No cv_summary.json at {summary_path}. Run scripts.train_cv first.")
    cv = json.loads(summary_path.read_text())
    seeds = list(cv.get("seeds", [42]))
    folds = list(cv["folds"])
    cv_seed = int(cv.get("cv_seed", 42))

    if args.quick:
        folds = folds[:1]
        seeds = seeds[:1]
        args.sparsities = [0.5, 0.9]
        args.calib_batches = 4
        print("  (quick CV mode: fold 0 only, 1 seed, sparsities 0.5 0.9)")

    print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"(quant always CPU)")
    print(f"CV dir:     {cv_dir}")
    print(f"Folds:      {[f['fold'] for f in folds]}   Seeds: {seeds}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Quant:      {'skipped' if args.skip_quant else 'INT8 static PTQ'}")

    runs: dict[str, dict] = {}
    for fold_rec in folds:
        k = fold_rec["fold"]
        split = Split(
            train=tuple(fold_rec["train"]), val=tuple(fold_rec["val"]),
            test=tuple(fold_rec["test"]), campaigns=cfg.campaigns, seed=cv_seed,
        )
        print(f"\n{'='*70}\nFOLD {k}  "
              f"(val={len(split.val)} calib, test={len(split.test)})\n{'='*70}")
        val_ds, test_ds = build_eval_datasets(split, cfg)
        for seed in seeds:
            ckpt = cv_dir / f"fold_{k}" / f"seed_{seed}" / "model.pt"
            print(f"\n-- fold {k} / seed {seed}  ({ckpt})", flush=True)
            configs = compress_checkpoint(ckpt, args, val_ds, test_ds)
            fs_dir = out_dir / f"fold_{k}" / f"seed_{seed}"
            fs_dir.mkdir(parents=True, exist_ok=True)
            (fs_dir / "results.json").write_text(
                json.dumps({"fold": k, "seed": seed, "configs": configs}, indent=2))
            runs[f"fold{k}_seed{seed}"] = configs

    aggregate(runs, out_dir)
    print("\nCV compression sweep complete.")


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------


def aggregate(all_seed_configs: dict, out_dir: Path) -> None:
    """Aggregate across runs: per-config overall + per-class median ± std,
    plus delta versus the baseline config.

    A "run" is a seed (single-split path) or a fold x seed pair (CV path); the
    keys of ``all_seed_configs`` identify them. Because every run contains both
    a ``baseline`` config and the compressed configs, ``delta_vs_baseline``
    (mean compressed minus mean baseline) equals the mean within-run delta by
    linearity, and the per-class std spans whatever the runs vary over (seeds,
    or subjects+seeds under CV)."""
    config_names = list(next(iter(all_seed_configs.values())).keys())
    summary: dict = {
        "runs": [str(k) for k in all_seed_configs.keys()],
        "n_runs": len(all_seed_configs),
        "configs": {},
    }

    # Baseline per-class means (across seeds) for delta computation.
    base_pc_means: dict[str, float] = defaultdict(list)
    for cfgs in all_seed_configs.values():
        for cls, acc in per_class_acc_map(cfgs["baseline"]).items():
            base_pc_means[cls].append(acc)
    base_pc_mean = {c: float(np.mean(v)) for c, v in base_pc_means.items()}

    for name in config_names:
        overall = [all_seed_configs[s][name]["overall_acc"]
                   for s in all_seed_configs]
        c1m = [all_seed_configs[s][name]["c1_macro_acc"] for s in all_seed_configs]
        c2m = [all_seed_configs[s][name]["c2_macro_acc"] for s in all_seed_configs]
        spars = [all_seed_configs[s][name].get("sparsity", 0.0)
                 for s in all_seed_configs]

        # Per-class across seeds.
        per_cls_vals: dict[str, list[float]] = defaultdict(list)
        camp_of: dict[str, str] = {}
        for s in all_seed_configs:
            rec = all_seed_configs[s][name]
            for camp in ("C1", "C2"):
                for e in rec["per_class"][camp]:
                    per_cls_vals[e["class"]].append(e["acc"])
                    camp_of[e["class"]] = camp

        per_class = []
        for cls in sorted(per_cls_vals):
            v = per_cls_vals[cls]
            per_class.append({
                "class": cls,
                "campaign": camp_of[cls],
                "median": float(np.median(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "delta_vs_baseline": float(np.mean(v) - base_pc_mean.get(cls, 0.0)),
            })

        summary["configs"][name] = {
            "sparsity_mean": float(np.mean(spars)),
            "overall_acc_mean": float(np.mean(overall)),
            "overall_acc_std": float(np.std(overall)),
            "c1_macro_mean": float(np.mean(c1m)),
            "c1_macro_std": float(np.std(c1m)),
            "c2_macro_mean": float(np.mean(c2m)),
            "c2_macro_std": float(np.std(c2m)),
            "per_class": per_class,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved cross-seed summary to {out_dir}/summary.json")
    _plot_degradation(summary, out_dir / "degradation_curves.png")

    # Compact console table.
    print("\nConfig                 overall   C1macro   C2macro")
    for name, c in summary["configs"].items():
        print(f"  {name:<22} {c['overall_acc_mean']:.3f}    "
              f"{c['c1_macro_mean']:.3f}    {c['c2_macro_mean']:.3f}")


def _plot_degradation(summary: dict, out_path: Path) -> None:
    """Macro-averaged C1/C2 accuracy vs pruning sparsity, for prune & combined."""
    cfgs = summary["configs"]

    def curve(prefix: str):
        pts = []
        for name, c in cfgs.items():
            if name.startswith(prefix):
                pts.append((c["sparsity_mean"], c["c1_macro_mean"], c["c2_macro_mean"]))
        pts.sort()
        return pts

    fig, ax = plt.subplots(figsize=(8, 5))
    base = cfgs["baseline"]
    ax.axhline(base["c1_macro_mean"], color="steelblue", ls=":", alpha=0.6,
               label="baseline C1 (micro)")
    ax.axhline(base["c2_macro_mean"], color="darkorange", ls=":", alpha=0.6,
               label="baseline C2 (macro)")

    for prefix, ls, lbl in [("prune_s", "-", "prune"),
                            ("combined_s", "--", "prune+INT8")]:
        pts = curve(prefix)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], ls, color="steelblue", marker="o",
                label=f"{lbl} C1 (micro)")
        ax.plot(xs, [p[2] for p in pts], ls, color="darkorange", marker="s",
                label=f"{lbl} C2 (macro)")

    if "quant_int8" in cfgs:
        q = cfgs["quant_int8"]
        ax.scatter([0.0], [q["c1_macro_mean"]], color="steelblue", marker="*",
                   s=180, zorder=5, label="INT8 C1 (micro)")
        ax.scatter([0.0], [q["c2_macro_mean"]], color="darkorange", marker="*",
                   s=180, zorder=5, label="INT8 C2 (macro)")

    ax.set_xlabel("pruning sparsity")
    ax.set_ylabel("macro-averaged test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Class-wise degradation under compression")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved degradation curves to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # CV path: compress per fold against the cross-validation checkpoints.
    # ------------------------------------------------------------------
    if args.cv:
        out_dir = Path(args.out_dir) if args.out_dir else (
            Path(__file__).resolve().parents[1] / "outputs" / "compression_cv")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_cv(args, cfg, out_dir)
        return

    # ------------------------------------------------------------------
    # Single-split path (original behaviour).
    # ------------------------------------------------------------------
    split = load_split()

    class_filter = rep_filter = None
    if args.quick:
        split = Split(train=split.train[:2], val=split.val[:1],
                      test=split.test[:1], campaigns=split.campaigns,
                      seed=split.seed)
        class_filter = ("M01", "M02", "M03", "A01", "A02", "A03")
        rep_filter = ("01", "02")
        args.seeds = [args.seeds[0]]
        args.sparsities = [0.5, 0.9]
        args.calib_batches = 4
        print("  (quick mode: trimmed split, 3+3 classes, 2 reps)")

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "baseline")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / "compression")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:     {'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"(quant always CPU)")
    print(f"Seeds:      {args.seeds}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Quant:      {'skipped' if args.skip_quant else 'INT8 static PTQ'}")
    print(f"Baseline:   {baseline_dir}")

    print("\nBuilding eval datasets (val=calibration, test=evaluation)...")
    val_ds, test_ds = build_eval_datasets(split, cfg, class_filter, rep_filter)

    all_seed_configs: dict[int, dict] = {}
    for seed in args.seeds:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        all_seed_configs[seed] = run_seed(
            seed, args, val_ds, test_ds, baseline_dir, out_dir)

    aggregate(all_seed_configs, out_dir)
    print("\nCompression sweep complete.")


if __name__ == "__main__":
    main()
