# Baseline Evaluation — Pre-Compression Fingerprint

**Project:** Evaluating the Class-wise Impact of Model Compression in RF-Based Human Activity Recognition
**Model:** `BaselineCNN` (single-head, 31-class) — ~420 K parameters
**Seeds evaluated:** 42, 43, 44 (three independent training runs)
**Split:** subject-independent (no participant in more than one of train/val/test)
**Test set:** 2,405 windows (1,046 C1 micro-gesture, 1,359 C2 macro-activity)
**Date:** 2026-06-01

This document is the baseline that the pruning and quantization sweeps will
be measured against. Because the whole point of the study is *class-wise*
degradation under compression, the reference here is deliberately recorded
per class and per campaign, not as a single headline number.

---

## 1. Headline results

| Metric | Mean ± std (3 seeds) | Range |
|---|---|---|
| Overall accuracy (window-weighted) | **89.29 % ± 0.40** | 88.98 – 89.85 |
| C1 micro-gestures (window-weighted) | 81.45 % ± 0.74 | — |
| C2 macro-activities (window-weighted) | 95.32 % ± 0.15 | — |
| C1 macro-averaged (mean of per-class means) | 80.2 % | — |
| C2 macro-averaged (mean of per-class means) | 91.7 % | — |

Two things matter here:

1. **The 89.3 % overall figure is misleading on its own.** It is window-weighted,
   and C2 contributes 1,359 of 2,405 windows — and within C2, a single class
   (A01, 571 windows) is ~24 % of the entire test set and scores 97.7 %. The
   headline number is therefore pulled upward by a few large, easy classes.
   For a compression study the per-class and macro-averaged views below are the
   honest baseline.

2. **The model lands squarely in the intended "headroom" zone.** The proposal
   deliberately targeted a 70–80 % subject-independent model rather than a 99 %
   one, precisely so that class-wise degradation under compression is
   *observable*. C1 at ~80 % (and a long tail of classes in the 50–70 % band)
   gives exactly that headroom; C2 at ~95 % is closer to saturation.

---

## 2. Seed stability

The three seeds are tightly clustered — overall accuracy spans just 0.87
percentage points (88.98 / 89.02 / 89.85). Training is reproducible and the
optimisation is not sensitive to initialisation at the aggregate level.

Stability is **not** uniform across classes, however, and this is the more
important observation for the compression work. Per-class standard deviation
across seeds ranges from 0.0 (M14, A05 — perfect every run) up to ~13 points:

- Highest cross-seed variance: **M17 (±13.1), M20 (±12.8), M08 (±7.7),
  M12 (±7.7), M11 (±6.4), M16 (±6.2)** — all C1 gestures.
- C2 activities are far more stable (max std ±5.5 for A09; most ≤ 2 points).

**Implication for the sweep:** several weak C1 classes already wobble by
10+ points between seeds *with no compression at all*. A single-seed
compression run that shows a 10-point drop on M17 or M20 cannot be attributed
to compression — it is inside the noise floor. The compression sweep must
therefore be run over the same three seeds (or report against the median
across them) so that compression-induced degradation is distinguishable from
seed noise. The existing `summary.json` median ± std is the right reference
object; single-seed deltas are not trustworthy for the volatile classes.

---

## 3. Per-class results

### 3.1 C1 micro-gestures (21 classes, sorted weakest → strongest)

| Class | Mean | Median | Std | Range | n |
|---|---|---|---|---|---|
| M16 | 52.7 | 50.0 | 6.2 | 47–61 | 62 |
| M11 | 55.9 | 52.9 | 6.4 | 50–65 | 34 |
| M05 | 64.8 | 68.6 | 5.4 | 57–69 | 35 |
| M17 | 66.1 | 66.1 | 13.1 | 50–82 | 56 |
| M09 | 66.7 | 68.6 | 4.9 | 60–71 | 35 |
| M12 | 69.6 | 73.5 | 7.7 | 59–76 | 34 |
| M04 | 70.1 | 68.8 | 5.2 | 65–77 | 48 |
| M08 | 70.8 | 75.0 | 7.7 | 60–78 | 40 |
| M20 | 71.4 | 70.3 | 12.8 | 56–88 | 64 |
| M21 | 77.8 | 81.8 | 5.7 | 70–82 | 66 |
| M06 | 80.2 | 81.1 | 3.4 | 76–84 | 37 |
| M19 | 85.7 | 85.7 | 4.7 | 80–91 | 35 |
| M13 | 87.2 | 87.2 | 2.1 | 85–90 | 39 |
| M07 | 92.5 | 91.8 | 1.0 | 92–94 | 49 |
| M01 | 92.8 | 93.5 | 2.7 | 89–96 | 46 |
| M18 | 93.7 | 93.7 | 1.3 | 92–95 | 63 |
| M03 | 94.4 | 95.0 | 2.1 | 92–97 | 60 |
| M15 | 95.8 | 96.8 | 2.7 | 92–98 | 63 |
| M02 | 97.1 | 96.6 | 0.8 | 97–98 | 58 |
| M10 | 98.4 | 98.8 | 1.5 | 96–100 | 84 |
| M14 | 100.0 | 100.0 | 0.0 | 100–100 | 38 |

C1 is strongly bimodal: a cluster of near-perfect gestures (M14, M10, M02,
M15, M03, M18, M01, M07 all ≥ 92 %) and a weak tail of six classes below 70 %
(M16, M11, M05, M17, M09, M12). The weak tail is consistent with the dataset's
"complex / bimanual / circular" gesture group from the dataset paper — these
gestures are spatially ambiguous in a Doppler-free BEV representation, so the
baseline already struggles to separate them.

### 3.2 C2 macro-activities (10 classes, sorted weakest → strongest)

| Class | Mean | Median | Std | Range | n |
|---|---|---|---|---|---|
| A08 | 73.0 | 75.7 | 3.8 | 68–76 | 37 |
| A09 | 81.4 | 85.3 | 5.5 | 74–85 | 34 |
| A07 | 87.5 | 85.0 | 3.5 | 85–92 | 40 |
| A10 | 88.9 | 87.9 | 1.4 | 88–91 | 33 |
| A02 | 92.6 | 91.8 | 1.4 | 91–95 | 292 |
| A03 | 97.2 | 97.9 | 1.8 | 95–99 | 95 |
| A01 | 97.7 | 97.9 | 0.7 | 97–98 | 571 |
| A06 | 99.3 | 98.9 | 0.5 | 99–100 | 92 |
| A04 | 99.4 | 100.0 | 0.8 | 98–100 | 57 |
| A05 | 100.0 | 100.0 | 0.0 | 100–100 | 108 |

C2 is much more uniform and high. The weak end (A08, A09, A07 — the
ball-sport activities) sits where C1's *middle* sits. The locomotion and
posture activities (A01 walking, A04, A05, A06) are essentially saturated.

---

## 4. Macro vs micro: the central baseline contrast

The study's core hypothesis is that fine-grained micro-movements are more
fragile than coarse macro-movements. The baseline already shows the predicted
gap *before any compression*:

- **Micro (C1):** 80.2 % macro-averaged, with a heavy weak tail and high
  per-class variance.
- **Macro (C2):** 91.7 % macro-averaged, tightly clustered and stable.

This ~11-point pre-existing gap is the backdrop against which compression
effects must be read. The interesting result for the thesis will not be
"C1 degrades more than C2" in absolute terms (it starts lower, so it has more
room to fall) but **whether the weak C1 tail degrades disproportionately** —
i.e. whether compression selectively forgets the classes that are already
marginal (the Hooker et al. "selective forgetting" effect). The six sub-70 %
C1 classes (M16, M11, M05, M17, M09, M12) are the ones to watch most closely.

---

## 5. Compression-readiness assessment

The baseline is suitable to serve as the compression reference, with the
following caveats logged:

1. **Use the 3-seed median, not a single seed.** Seed noise on the weak C1
   classes is up to ±13 points. Every compression point should be compared
   against `summary.json` medians and, ideally, itself run on ≥ 3 seeds so the
   compression delta carries its own error bar.

2. **Report per-class, and prefer macro-averaged campaign accuracy.** The
   window-weighted 89.3 % hides the effect of interest because two large easy
   classes (A01, A05) dominate. The primary baseline metrics for the sweep
   should be: per-class accuracy (all 31), macro-averaged C1, macro-averaged
   C2, and the C1/C2 gap.

3. **Watch the floor on weak classes.** Classes already near chance-adjacent
   levels (M16 at 52.7 %, M11 at 55.9 %) have little room to fall before
   hitting random-guess territory; a "20 % relative drop" means something very
   different for M16 than for A05. Track absolute accuracy and, where useful,
   accuracy relative to per-class chance.

4. **Architecture is compression-friendly.** `BaselineCNN` is plain
   Conv→BN→ReLU→MaxPool with a single linear head and no skip connections, so
   both `torch.nn.utils.prune` (unstructured magnitude) and `torch.ao.quantization`
   (PTQ) apply cleanly. The saved `model.pt` checkpoints are self-contained
   (state dict + arch + label map), so each seed can be loaded and compressed
   independently.

5. **No data-pipeline changes needed.** Training was on the warm BEV cache;
   the compression sweep reuses the same cached tensors and the same locked
   subject-independent split, so baseline and compressed models are compared
   on identical inputs.

### Recommended baseline fingerprint to carry forward

| Reference metric | Value |
|---|---|
| Overall (window-weighted) | 89.3 % |
| C1 micro macro-avg | 80.2 % |
| C2 macro macro-avg | 91.7 % |
| C1/C2 gap | ~11.5 pts |
| Weak-tail classes (to watch) | M16, M11, M05, M17, M09, M12; plus A08 |
| Stable anchors (should not move) | M14, A05, A04, A06, M10, A01 |

---

## 6. Next step

Proceed to the compression sweeps using these three seeds and the per-class
fingerprint above as the reference. Start with unstructured magnitude pruning
across sparsity levels, then post-training quantization across bit-widths,
then the combined setting — re-running each on seeds 42/43/44 and comparing
class-wise deltas against `summary.json`.

*Source data: `outputs/baseline/summary.json` and `outputs/baseline/seed_{42,43,44}/per_class_acc.json`.*
