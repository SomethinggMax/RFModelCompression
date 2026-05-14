# RF-Behavior Pipeline Walkthrough

A guided tour of the preprocessing and smoke-test code in this repo, written
to be read alongside the actual files. Every design choice is connected to
either a fact about the RF-Behavior dataset or a methodological commitment
in the project proposal. By the end, you should be able to defend or change
every default in `config.py` from first principles.

This document is built up module by module. Each module covers one logical
slice of the pipeline; later modules build on earlier ones, so read in
order on a first pass.

## Curriculum

| Module | Topic | Files |
|---|---|---|
| 1 | The dataset, the recording setup, and what "a sample" is | none (reads `.pkl` + paper) |
| 2 | Why CNN-on-2D, and the methodological decision tree | `project_context_week3.md`, proposal |
| 3 | From raw points to fused world coordinates | `loader.py`, `transforms.py`, `config.py` |
| 4 | DBSCAN clutter suppression | `clutter.py` |
| 5 | Windowing: variable-duration samples → fixed-length tensors | `windowing.py` |
| 6 | BEV construction: sparse points → dense tensor | `bev.py`, `config.py` |
| 7 | Subject-independent splits as methodology | `splits.py`, `splits.json` |
| 8 | PyTorch Dataset, caching, and the training loop | `dataset.py`, `stub_cnn.py`, `smoke_test.py` |

---

# Module 1 — The dataset, the recording setup, and what "a sample" is

Before any code, we need to be precise about what's physically inside one
`.pkl` file. Every later choice in the pipeline either compensates for or
exploits some property of the raw data, so misunderstandings here propagate
silently into the rest of the project.

## 1.1 What an FMCW radar measures

The dataset's 13 radars are TI IWR1443 modules. They are **FMCW** —
Frequency-Modulated Continuous Wave — which means each one continuously
transmits a "chirp": a signal whose frequency rises linearly from 77 GHz to
81 GHz over a few microseconds. When that chirp hits an object and bounces
back, the radar mixes the returning signal with the currently-transmitted
one and the *beat frequency* of the mix tells you how far away the object
is. (Far targets reflect chirps that started a long time ago, so the
returning frequency lags more.)

Key things FMCW radar can measure in principle:

- **Range** — from the beat frequency, with resolution ≈ c / (2·BW). With
  4 GHz bandwidth that's ~3.75 cm. Excellent for human-scale motion.
- **Angle of arrival (azimuth and elevation)** — the radar has 3 transmit
  and 4 receive antennas, which form a 12-element virtual MIMO array. The
  *phase difference* of the returned signal across the array tells you
  which direction it came from.
- **Velocity (Doppler)** — measured from the phase of consecutive chirps.
  A moving target shifts phase frame-to-frame in a predictable way.

So a raw radar frame has both *position* and *velocity* information sitting
in the complex IQ samples. Now here's the catch.

## 1.2 What the dataset actually saved

The dataset paper (Zuo et al. 2025, §3) is explicit: the IWR1443 computes a
range-azimuth-elevation map per frame on-device, runs a CFAR-like detector
to threshold above noise, converts each detection to a `(x, y, z)` point in
the radar's local frame, and **saves only those points along with a
"density" confidence value**. The phase information needed to compute
Doppler is discarded at this stage.

This has one massive consequence: **we have no velocity data**. None.

That's why the project commits to a *spatial* 2D representation (BEV) with
temporal information encoded as channels — Doppler-style spectrograms,
which a lot of mmWave HAR work uses, are simply not available. It's also
why our published-baseline expectations are around 70–80 %, not 99 %: we're
missing a feature dimension the authors did have access to in their
internal pipeline.

So one *frame* of one sensor, in our hands, is a `(N, 5)` numpy array:

```
[[timestamp, x, y, z, density],
 [timestamp, x, y, z, density],
 ...]
```

with N typically between 0 and 7. Same timestamp on every row of one frame.
Run `uv run python -m scripts.inspect_pkl` to see this on a real file.

### What is "density"?

The paper calls this column density (§3.1: "timestamp, x, y, z, and density
are stored together"). Empirically it lives in the 8–26 range and is
positive. That's consistent with a dB-scale confidence measure
("signal-to-noise of the detection, in dB above the noise floor"). It is
**not** intensity in the photographic sense. In the BEV step we'll use it
as a per-cell strength channel — a high-density cell is a confident
detection, not necessarily a "bright" one.

## 1.3 Frame rate and timing

The radars run at approximately 27 Hz. Run `inspect_pkl.py` and you'll see
something like `approx fps 27.28`. This is the headline number you'll use
for windowing: at 2 seconds per window we get ≈ 54 frames per window.

Note "approximately". The radars are not hardware-synchronised at the
nanosecond level — each starts independently and timestamps in Unix epoch
seconds. When we fuse multiple sensors we have to bin onto a *uniform*
grid; we cannot assume frame N from sensor 5 corresponds to frame N from
sensor 6. That's what `windowing.py` exists for.

## 1.4 The 13-radar geometry

The recording area is a 2 m × 6 m rectangle — 2 m wide in the **x**
direction and 6 m long in the **y** direction. (You can verify this from
the Table 2 sensor positions: three sensors sit along the south long
side at y ≈ −3, three along the north long side at y ≈ +3, and one at
each short end at x ≈ ±2.7. The subject walks back and forth along the
long y-axis during Campaign 2 activities.) There are two radar groups:

- **8 ground radars** at ~0.9 m height, arranged around the perimeter of
  the rectangle, all pointing inward at the centre. These are the "ring"
  used in Campaigns 1–3.
- **5 ceiling radars** above the area, pointing straight down.

Each radar has a limited field of view — roughly ±60° azimuth and a
narrower elevation range. So any single radar only catches a part of the
person. To get full-body coverage, *fusion is mandatory*.

This isn't a stylistic choice — it's a hard physics constraint. Look at a
single sensor's `.pkl`:

```
sensor 5: 58 frames, 150 total points  → mean 2.6 points per frame
```

That's *the entire 2-second gesture*, from one of the most-active sensors.
You can't classify gestures from 2 points per frame. But once you fuse the
8 ground sensors:

```
8 sensors → ~1240 total points across the same 2-second window
```

That's enough.

The implications:

1. **Single-sensor classifiers are not viable on this dataset.** The
   authors' baselines all fuse.
2. **Different sensors see different body parts.** Sensor 5 is at
   `(2.6, 0, 0.9)`, looking inward along −X. From its view the person's
   front-facing limbs dominate. Sensor 9 at `(−2.9, 0, 0.9)` sees the back
   side. This is why we transform to a *world frame* before fusion: the
   same physical limb point produced by different sensors should land in
   roughly the same world coordinate.

### Why ground-only by default?

The proposal context cites the authors' own results: ~99 % accuracy on
ground-only data, 78–85 % on ceiling-only. Two reasons ground-only wins:

1. **Geometry.** Ground sensors are at limb height (~0.9 m), which is
   exactly where the discriminative motion lives — hand gestures, arm
   positions, foot placement during walking. Ceiling sensors looking down
   collapse the vertical dimension, which is the hardest one to recover.
2. **Density.** 8 sensors give more total points than 5.

For *this* project specifically, there's a third reason: micro-movement
classification (single-hand gestures) hinges on distinguishing one limb
position from another. That's exactly what ground sensors are best at.

## 1.5 What is a "sample"?

A sample is one recording instance of a person performing one
class/gesture/activity once. Concretely, a sample is identified by four
things, in this order of granularity:

```
campaign / subject / class / repetition
   C1      U01      M01      01
```

On disk that becomes:

```
dataset/Radar/C1/U01/M01/01/{0,1,2,...,12}.pkl
```

with each numbered `.pkl` being one sensor's recording of that sample.

The dataclass `SampleId` in `loader.py` mirrors this exactly:

```python
@dataclass
class SampleId:
    campaign: str   # "C1" / "C2" / "C3"
    subject: str    # "U01"
    cls: str        # "M01"
    repetition: str # "01"
```

Then `sample.sensor_path(5)` builds the full path to sensor 5's `.pkl`.

### The four campaigns

| Campaign | Content | Subjects with radar | Per-sample length | Reps | Notes |
|---|---|---|---|---|---|
| C1 | 21 hand gestures | 25 | ~3 s | 8 | Micro-movement candidates |
| C2 | 10 activities | 25 | 3–18 s | 8 | Macro-movement candidates |
| C3 | 6 sentiments | 23 | 50–180 s | 1 | Excluded; 50× longer |
| C4 | 21 gestures (industrial) | 17 | — | — | **No radar — RFID only.** Unusable. |

C1 ∩ C2 has 23 subjects (verified by `discover_subjects` in `splits.py`).
That intersection is your real working population.

### Class label encoding on disk

**Each campaign uses a different prefix on disk**, matching the paper's
notation:

- **C1**: `M01`–`M21` (M for "movement"/gesture)
- **C2**: `A01`–`A10` (A for "activity")
- **C3**: `E01`–`E06` (E for "emotion"/sentiment)

A common pitfall: it's tempting to assume every campaign uses `MNN`
because Campaign 1 does. They don't. If you write a class filter as
`["M01", "M02", ...]` it will only match Campaign 1 folders and
silently exclude all of Campaign 2 — exactly the kind of bug that
makes a smoke test pass while quietly testing half of what you
think it is.

The pipeline reads the folder name verbatim and never tries to
translate. Because each campaign uses a different prefix, the
campaign + class pair is what uniquely identifies a class. That's
why `dataset.py` encodes labels as `f"{campaign}/{cls_name}"`:
`C1/M01` is "lateral-raise" (gesture), `C2/A01` is "walking"
(activity), and they get different label ids in the model.

## 1.6 Sparsity — what it means in practice

Sparsity is the single biggest property to keep in mind throughout the
pipeline. Some hard numbers:

- 1–7 points per frame per sensor, mean ~2.5
- 27 Hz frame rate
- One 2-second window on one sensor: roughly 50–150 points total
- One 2-second window across 8 ground sensors: roughly 1000–1500 points total

For the BEV grid we're going to bin those ~1200 points into a 64 × 64 grid
across 54 frames and 3 z-bands — that's 54 × 3 × 64 × 64 = 663 552 cells
across 1200 points, of which **maybe 0.2 % will be non-empty.** This
extreme sparsity is why:

- Per-cell features are designed to be robust to zero (mean density of an
  empty cell is defined as 0, not NaN).
- We can't realistically use sub-centimetre grids — most cells would be
  empty even for the inhabited ones.
- DBSCAN is feasible: even after fusion, the total point count per sample
  is low enough (~1000) that a clustering pass takes milliseconds.

## 1.7 Things to actually do for this module

To convert the abstract above into intuition, do these three things in
order:

1. **Run `uv run python -m scripts.inspect_pkl`** with no arguments. Read
   the output. Note: the column ranges, the frame count, the
   computed-fps, the points-per-frame stats. That's the ground truth.

2. **Vary the path.** Try a different sample, e.g.:
   ```bash
   uv run python -m scripts.inspect_pkl --path "F:\Research Project\dataset\Radar\C2\U01\M01\01\5.pkl"
   ```
   This is Campaign 2 walking, sensor 5. Note the frame count is much
   higher (a longer sample) and the spatial extent is larger (the person
   moved across the room).

3. **Pick one mental picture and lock it in.** "A sample is one person
   doing one thing once, recorded simultaneously by 8 ground radars at
   ~27 Hz. Each radar produces ~2–3 points per frame. The points are
   confidence-weighted detections in the radar's local frame. To do
   anything useful we have to fuse across sensors using known sensor
   positions."

If that mental picture makes sense, you're ready for module 2.

## 1.8 Self-check questions

You should be able to answer these before moving on. Don't peek.

1. Why does the dataset have density values but not Doppler/velocity?
2. What are the units of the timestamp column? Of the x/y/z columns? Of the
   density column?
3. Why is single-sensor classification not viable on RF-Behavior?
4. What does "approximately 27 Hz" actually mean for the pipeline, and why
   do we have to re-bin onto a uniform time grid?
5. Why is C1 ∩ C2 the relevant subject pool size, not the full 25 of either
   campaign?
6. Why is C4 unusable for this project despite being a "gestures" campaign?
7. What is the ratio of expected non-empty cells to total cells in a
   typical BEV tensor, roughly?

---

# Module 2 — Why CNN-on-2D, and the methodological decision tree

This is the most consequential module for your supervisor check-in. The
architecture choice is **not** a default — it's a deliberate methodological
commitment, and you need to be able to defend it from first principles. If
a supervisor asks "why didn't you use PointNet++ like the dataset paper
did?" the answer should be in your head before they finish the sentence.

The structure of this module is: first the principle (your research
question dictates the architecture, not the dataset), then the four
concrete reasons (literature alignment, tooling maturity, methodology
transferability, deliberate-simplicity), then the costs we accept by
choosing this path.

## 2.1 Principle: the baseline is the instrument, not the object of study

Your research question is:

> How do pruning and quantization affect class-wise performance in
> RF-based human activity recognition models?

Read that carefully. The thing being studied is *compression*, applied to
*HAR models*. The HAR model itself is a tool through which we observe
compression effects — it is the experimental apparatus, not the
experimental subject. This is exactly analogous to a chemist choosing a
glass beaker not because glass beakers are inherently more interesting
than steel beakers, but because the substance under study reacts with
steel.

This framing matters because it dictates the right way to choose an
architecture: pick the one for which the *thing you're actually studying*
is best understood and best instrumented. For pruning and quantization,
that's CNNs. Period. Every published result on disparate impact from
compression uses CNNs. Every well-tested PyTorch compression tool assumes
Conv/Linear/BN/ReLU. If your apparatus differs from the literature you're
comparing against, your apparatus becomes a confounder.

A point-based baseline like PointNet++ would be more "natural" for the
dataset. It would also wreck cross-paper comparability and force you to
hand-implement things that already exist for CNNs.

## 2.2 The four cited "disparate impact under compression" papers

These are the four references your proposal builds on for the class-wise
analysis. Every one of them uses convolutional architectures.

**Hooker et al. 2021 — "What do compressed deep neural networks forget?"**
Trains ResNet-18 / ResNet-50 on CIFAR-10 and ImageNet. Defines
compression-identified examples (CIEs): examples where the compressed
model disagrees with the uncompressed one. Finds CIEs are concentrated in
*atypical, rare, or noisy* classes. Their entire methodology — sampling
margins, per-class CIE rates, the "selective forgetting" hypothesis — is
built on the assumption that you have a CNN producing per-example logits.

**Liebenwein et al. 2021 — "Lost in pruning"**
Compares pruning sensitivity across CNN architectures (LeNet, ResNet,
VGG, MobileNetV2). Their generalisation-vs-accuracy decoupling argument
needs the network to be a CNN trained with standard data augmentation, on
standard image benchmarks.

**Tran et al. 2022 — "Pruning has a disparate impact on model accuracy"**
Pruning experiments on image classification CNNs. Quantifies how pruning
amplifies accuracy gaps between majority and minority subgroups. The
fairness-amplification finding is reported per-class — exactly what you
want to mirror.

**Joseph et al. 2020 — class-level mismatches under compression**
Same family. CNN-based, per-class metrics, image domain.

When you mirror their per-class metrics on RF-Behavior, you want
*method transfer*, not *method invention*. If you mirror their CIE
methodology with a CNN baseline, the only thing that's novel is the
domain (RF instead of images) and the question (RF-HAR specifically).
That's a clean contribution. If you mirror it with PointNet++, you're
also inventing how to compute CIEs on a point-set network, which is a
second contribution none of your supervisors signed off on, and which
would require its own validation.

## 2.3 Tooling maturity — what PyTorch supports, and what it doesn't

This is the most concrete reason of the four.

`torch.nn.utils.prune` — the official PyTorch pruning module — is built
around layer-level masking on weight tensors. It works cleanly on:

- `Conv2d`, `Conv3d`, `Conv1d`
- `Linear`
- `LSTM` / `GRU` (some methods)
- BatchNorm-fused architectures

`torch.ao.quantization` — the official quantization module — is built
around module fusion patterns:

- `Conv2d → BatchNorm2d → ReLU` becomes a quantized fused module
- `Linear → ReLU` becomes a quantized fused module
- Standard ResNet/VGG-style architectures "just work"

Now look at PointNet++. Its core computational element is the
*set-abstraction* layer:

```
neighbourhood points → shared MLP → max-pool over neighbourhood
```

That sequence has no analogue in `torch.ao.quantization` 's fusion
patterns. The MLP is a sequence of Linear layers (fine), but the
`max-pool over a variable-size neighbourhood` step is implemented as a
Python loop over CUDA kernels in most PointNet++ codebases. Quantization
of those kernels is **a research problem in itself.** You'd be asking
"how do quantization errors propagate through a permutation-invariant
max over variable-cardinality neighbourhoods?" — interesting question,
unrelated to your research question, would consume the entire project.

Same for unstructured magnitude pruning. Pruning a weight matrix in a
Linear layer of an MLP is straightforward. Pruning a CUDA kernel that
implements ball-query neighbourhood aggregation is not.

So the choice is: use the architecture for which the compression tools
are mature (CNN), or burn the project re-implementing compression for an
architecture for which they aren't.

## 2.4 Why the WiFi-CSI HAR precedent is your direct comparison

Varga & Cao 2025 (your evaluation-protocol anchor) does compression on
WiFi-CSI HAR — that is, RF-based human activity recognition on a
different sensing modality — using CNN-on-2D. Moshiri et al. 2021 and
Wakili 2025 are the same: CSI signal turned into a 2D spectrogram, fed
to a CNN, compressed.

The structural similarity to your work is exact. WiFi CSI is a complex
channel response across subcarriers and time → 2D spectrogram → CNN.
RF-Behavior is sparse 3D points across sensors and time → 2D BEV (with
temporal channels) → CNN. The *input semantics* differ, but the
*tensor shape and methodology* line up almost one-for-one.

This means three good things:

1. **You can adopt their evaluation protocol verbatim.** Varga & Cao's
   subject-independent split protocol is what `splits.py` implements.
2. **Your results compare directly.** A finding like "pruning at 80%
   sparsity drops worst-class accuracy by 12 points on RF-Behavior" can
   be placed next to Varga & Cao's "...by 9 points on CSI HAR" and
   readers can interpret the difference.
3. **You get to use their negative results as a sanity check.** If
   they observed that batch-norm folding helps preserve class-wise
   performance during quantization, you can test the same on your data
   and either confirm or extend.

If your baseline were a point-set network, none of those three benefits
exist — you'd be the only person in the literature studying compression
on point-set RF-HAR, with no triangulation.

## 2.5 Deliberate simplicity — why 99 % accuracy is bad for this study

This is the most counter-intuitive of the four arguments and worth
spelling out carefully.

When the authors of RF-Behavior report 99.01–99.35 % accuracy on
ground-only data with their Tesla model, that's an excellent
classification result. It's also nearly useless for studying compression
effects. Here's why.

A class-wise compression study is, fundamentally, a study of
*degradation* — what happens to per-class accuracy as you remove model
capacity. If your starting accuracy is 99.3 %, then each class can lose
at most 99.3 percentage points before hitting zero, but in practice the
effects you want to measure are in the 1–10 percentage point range.
Differentiating "pruning hurt class M01 by 2 points" from "pruning hurt
class M01 by 4 points" against a 99 % starting baseline requires extreme
statistical care: you're trying to measure a 2 % effect in a system whose
noise floor is around 0.5–1 %. Plus, at 99 % most classes have *zero*
errors at the uncompressed baseline, so per-class CIE rates start at zero
and you have no margin to measure decreases.

A 70–80 % baseline gives you headroom. Hooker et al.'s CIE methodology
explicitly relies on having enough errors at the uncompressed baseline
to compute meaningful per-class statistics. If your uncompressed model
has 200 wrong predictions per class out of 1000, you can clearly see a
shift to 240 wrong predictions per class under compression. If your
uncompressed model has 2 wrong predictions per class, that signal is
buried in the noise.

This is the "deliberately simple" CNN argument: pick an architecture and
a training protocol that hits 70–80 % subject-independent accuracy on
purpose, *because that's the operating regime where class-wise
compression effects are observable*. Don't optimise for headline
accuracy. Optimise for measurable degradation.

A second, related argument: the authors' 99 % is almost certainly
*subject-dependent* (the paper does not specify the split, and 25
subjects with 99 % accuracy strongly implies the model has memorised
participant signatures rather than learned generalisable features). With
a strict subject-independent split à la Varga & Cao, accuracy drops
substantially — and that drop in itself is what gives you the room to
study compression effects properly. Subject independence and deliberate
simplicity are aligned, not in tension.

## 2.6 Costs we accept by choosing CNN-on-2D

Be honest about what we lose.

**Loss 1: spatial precision.** A 64 × 64 BEV grid over an 8 × 8 m area
gives 12.5 cm cells. That's coarse compared to the radar's intrinsic
3.75 cm range resolution. We're throwing away resolution in exchange for
a dense tensor. PointNet++ wouldn't.

**Loss 2: permutation invariance.** Point-set networks treat the points
as an unordered set; the answer doesn't depend on the order in which
points are presented. CNNs don't have this property. We rely on the
binning to be order-invariant per-cell (which it is), but we don't get
the cross-cell invariance properties of, say, deep set models.

**Loss 3: variable-cardinality elegance.** Point clouds gracefully handle
"this frame had 2 points, that one had 7". Our BEV pretends every frame
contributes one entry per cell, with zero where there's nothing. That's
fine — it's how WiFi-CSI HAR works too — but it's a representation
choice with information-theoretic implications.

What we get in exchange: every advantage in §2.2–§2.5. Net win for *this*
research question.

## 2.7 The supervisor talking points

When the check-in arrives, here's the script:

> "We chose CNN-on-2D, not point-based, because our research question is
> about compression rather than modelling. The disparate-impact
> compression literature — Hooker, Liebenwein, Tran, Joseph — is entirely
> CNN-based. PyTorch's pruning and quantization tooling assumes
> Conv/Linear/BN structure. The WiFi-CSI HAR compression precedent
> (Varga & Cao, Moshiri, Wakili) is also CNN-on-2D, which gives us a
> direct comparator. And a deliberately simple CNN that hits 70–80 %
> subject-independent accuracy is *better* for measuring class-wise
> compression effects than a 99 % point-cloud model, because it leaves
> headroom for degradation to be observable. We accept that we lose some
> spatial precision and the elegance of permutation invariance — those
> trade-offs are documented and not part of our research question."

If you can deliver that paragraph confidently, you've defended the most
contentious decision in the project.

## 2.8 Self-check questions

1. State your research question in one sentence, then explain in one
   sentence why that question dictates a CNN baseline rather than a
   point-based one.
2. Name the four "disparate impact" papers and the architecture each
   uses.
3. Name two specific PyTorch APIs that work cleanly on CNNs but not on
   PointNet++. For each, name the underlying reason.
4. Why is 99 % accuracy a problem for a class-wise compression study?
5. Name the WiFi-CSI HAR precedent papers and what each contributes to
   your methodology.
6. What are the three concrete costs we accept by choosing BEV-CNN over
   PointNet++?

---

# Module 3 — From raw points to fused world coordinates

This is the first module that opens code. We follow what happens to one
sample as it enters the pipeline — from the moment Python sees a path on
disk, through validation, through the sensor-local → world-frame
transform, to the point where the points from all 8 ground sensors live
in one shared coordinate system.

We'll walk three files, in this order:

- `rfbc/data/loader.py` — file IO and validation
- `rfbc/config.py` — sensor coordinates and yaw-angle derivation
- `rfbc/data/transforms.py` — the geometry

Open them in your editor as you read.

## 3.1 The `SampleId` dataclass — why we model the dataset as a typed tuple

Every operation in the pipeline takes a `SampleId` as the unit of work:

```python
@dataclass
class SampleId:
    campaign: str   # "C1" / "C2" / "C3"
    subject: str    # "U01"
    cls: str        # "M01"
    repetition: str # "01"

    def sensor_path(self, sensor_index: int, root: Path = RADAR_ROOT) -> Path:
        return root / self.campaign / self.subject / self.cls / self.repetition / f"{sensor_index}.pkl"
```

Three things to notice.

First, **`SampleId` is a `dataclass`, not a string or a tuple**. We could
have represented a sample as `"C1/U01/M01/01"` and parsed it where needed.
We didn't because typed fields catch errors at the boundary: if some
caller mistakenly passes an `M01` string where the campaign is expected,
type checkers and mypy can complain. With strings, that bug shows up
silently as an empty path on disk three function calls later.

Second, **the `sensor_path` method bakes the on-disk layout into the
domain object.** This is deliberate — the path structure
`Radar/C{campaign}/U{user}/M{class}/{rep}/{sensor}.pkl` is the *only*
place the disk layout is encoded. If the dataset ever reorganises (or we
move to a different storage backend), this method is the single point of
change. None of the rest of the codebase concatenates path strings.

Third, **the `root` parameter has a default but accepts an override.**
This makes testing trivial — point at a fake mini-dataset in `/tmp`
without touching real data. It also lets us run on a shared HPC mount
later without code changes.

## 3.2 `load_sensor` — defensive parsing of a `.pkl`

```python
def load_sensor(path: Path | str) -> list[np.ndarray]:
    path = Path(path)
    with path.open("rb") as f, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PKL_ALIGN_WARNING_MSG)
        data = pickle.load(f)
    cleaned: list[np.ndarray] = []
    for frame in data:
        if not isinstance(frame, np.ndarray):
            continue
        if frame.size == 0:
            continue
        if frame.ndim != 2 or frame.shape[1] != 5:
            raise ValueError(...)
        cleaned.append(frame.astype(np.float64, copy=False))
    return cleaned
```

`pickle.load` is a very general operation — Python will faithfully
reconstruct *whatever was pickled*, including bizarre custom classes,
broken arrays, or `None`. We don't trust it blindly. Three layers of
defence:

1. **`isinstance(frame, np.ndarray)`** — drop anything that isn't an
   array. The dataset paper says every frame is a numpy array, but old
   data has surprises and a stray `None` in the list would crash later
   stages with a much less informative error.
2. **`frame.size == 0`** — drop empty frames. Sensors do produce frames
   with zero detections (the radar fired, nothing came back, pickled an
   empty array). These are harmless and we silently drop them.
3. **`frame.ndim != 2 or frame.shape[1] != 5`** — *raise loudly*. If
   somehow we encounter a frame that's not the published format, that's
   a data-integrity problem and we want to know now, not three modules
   later when a BEV tensor comes out wrong.

The `astype(np.float64, copy=False)` is a shape-preserving cast. The
`copy=False` means "only allocate a new array if a cast is necessary";
if the array is already float64 it's a no-op.

The `warnings.catch_warnings()` block silences the NumPy 2.4
deprecation warning we hit earlier — a property of the source `.pkl`
files (pickled under older NumPy with `align=0`), not of our code. We
narrow the filter to the exact message string so unrelated warnings
still surface.

### Why we re-validate per call instead of trusting the dataset

Every load goes through these checks. Couldn't we cache the result of a
"is this dataset valid?" pass once at startup and skip thereafter?

In principle yes, but: 81 256 `.pkl` files on disk × validation cost is
small (~milliseconds) compared to the rest of the pipeline. And cached
"valid" lists go stale; one corrupted file from a network glitch and the
cache is wrong forever. The defensive check at every load is a constant-
factor cost we pay for never having to debug a "why does this only fail
on Tuesdays" problem.

## 3.3 `load_sample_sensors` — graceful handling of missing sensors

```python
def load_sample_sensors(
    sample, sensor_indices, root=RADAR_ROOT, skip_missing=True,
):
    out: dict[int, list[np.ndarray]] = {}
    for idx in sensor_indices:
        p = sample.sensor_path(idx, root=root)
        if not p.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(p)
        out[idx] = load_sensor(p)
    return out
```

The `skip_missing=True` default is interesting. Why would a sensor file
be missing for a sample that exists?

Answer: the dataset has rare gaps. Some samples are missing one or two
sensor files entirely (e.g. the `04/04/04/04.pkl` file shown in the
filesystem listings — sensor 4 is ceiling, but for ground samples we'd
sometimes find sensor 5 missing too). The cause is unclear from the
paper — possibly a sensor failure during recording, possibly a
post-processing filter that dropped degenerate sensor outputs. Either
way, we have to handle it.

`skip_missing=True` says: if a sensor is gone, that sample now has 7
sensors instead of 8. The pipeline downstream tolerates that (DBSCAN
operates on however many points it gets, the BEV grid bins whatever
points show up). We log nothing — this is fine and silent.

`skip_missing=False` is what you'd use during data validation: pass
that flag once at the start of a project to surface every missing file
loudly. We don't do that automatically because it would make a
preprocessing run abort on a single bad file from 81 256.

## 3.4 The world-frame transform — math from first principles

This is the conceptually hardest part of module 3, but the underlying
geometry is simple. Let's derive it from scratch.

### The setup

Each radar reports detections in its own local frame. By convention for
TI mmWave radar:

- **Local +X axis** = boresight direction (the way the radar is pointing)
- **Local +Y axis** = horizontal, to the radar's left
- **Local +Z axis** = vertical, upward

So a detection at `(2.0, 0, 0)` in the local frame means "2 m straight
ahead of the radar, at the same height as the radar". A detection at
`(2.0, 0.5, 0)` means "2 m ahead and 0.5 m to the radar's left".

Now, sensor 5 is mounted at `(2.6055, -0.0940, 0.9407)` in world
coordinates, pointing approximately at the recording-area origin. The
person stands somewhere near the origin. So the same physical point on
the person — say, a hand at `(0.1, 0.2, 1.5)` in world coordinates —
appears as something like `(2.5, -0.3, 0.6)` in sensor 5's local frame
(approximately 2.5 m straight ahead, slightly to one side, lower than
the radar). Sensor 9 across the room sees that same hand from the
opposite direction, with totally different local coordinates.

For fusion to make sense, all 8 sensors' detections of *that same hand*
must end up at *the same world coordinate*. That's what the transform
achieves.

### The transform: rotation then translation

The standard rigid transform is:

```
world_p = R · local_p + t
```

where:

- `local_p` is a `(3,)` vector in the radar's frame
- `R` is a 3×3 rotation matrix that aligns the radar's local axes to
  world axes
- `t` is the radar's position in the world frame
- `world_p` is the same physical point in the world frame

Step 1 (rotate) takes the local point and re-expresses it relative to
world axes (still centred on the radar). Step 2 (translate) shifts that
to the radar's actual world location.

### What rotation do we need?

Each radar's local +X points along its own boresight, which points
toward the world origin (because the ring of radars all face inward).

Let `(x_s, y_s, z_s)` be the radar's world position. The unit vector
*from the radar to the origin* is:

```
d = -(x_s, y_s, z_s) / ‖(x_s, y_s, z_s)‖
```

Since the ground radars are roughly all at the same height (~0.9 m) and
the recording area is much wider than it is tall, the boresight is
nearly horizontal. So we approximate the rotation as a **yaw-only**
rotation about the world +Z axis. That's a rotation in the x-y plane.

We need a rotation `R(yaw)` such that:

```
R(yaw) · [1, 0, 0] = [-x_s/r, -y_s/r, 0]   # local +X → direction to origin in x-y plane
```

where `r = sqrt(x_s² + y_s²)`. Working it out:

```
R(yaw) = [ cos(yaw)  -sin(yaw)  0 ]
         [ sin(yaw)   cos(yaw)  0 ]
         [    0           0     1 ]
```

so `R(yaw) · [1, 0, 0] = [cos(yaw), sin(yaw), 0]`. Setting that equal to
`[-x_s/r, -y_s/r, 0]`:

```
cos(yaw) = -x_s / r
sin(yaw) = -y_s / r
```

That's exactly `yaw = atan2(-y_s, -x_s)`. Which is the formula in
`config.py`:

```python
def ground_yaw_deg(sensor_index: int) -> float:
    x, y, _ = GROUND_TABLE2_XYZ[sensor_index]
    return float(np.rad2deg(np.arctan2(-y, -x)))
```

`atan2(b, a)` (note the argument order: y first) returns the angle whose
sine is `b/r` and whose cosine is `a/r`. So `atan2(-y_s, -x_s)` returns
the angle whose `(cos, sin)` is `(-x_s/r, -y_s/r)`, exactly what we need.

### Sanity check the numbers

For sensor 5 at `(2.6055, -0.0940, 0.9407)`:

```
yaw = atan2(0.0940, -2.6055) ≈ 177.93°
```

That makes sense — sensor 5 is on the +X side of the room, looking back
at the origin (which is in the −X direction from its perspective), so
its boresight should point roughly along the world −X axis. A yaw of
180° points exactly along world −X; we get 177.93° because sensor 5 is
slightly off the axis (its `y_s = -0.0940` puts it just south of the
x-axis).

Run this in a Python shell with the project loaded:

```python
from rfbc.config import GROUND_TABLE2_XYZ, ground_yaw_deg
for i, (x, y, z) in GROUND_TABLE2_XYZ.items():
    print(f"sensor {i:>2}: pos=({x:+.3f}, {y:+.3f}, {z:+.3f})  yaw={ground_yaw_deg(i):+.2f}°")
```

You'll see all 8 yaws roughly evenly distributed around the circle —
178°, 118°, 85°, 49°, 3°, −70°, −90°, −111°. The spacing is uneven (not
exactly 45°) because the actual rectangular layout is uneven; the
visualisation script's THETA_DEGS = [180, 135, 90, 45, 0, −45, −90, −135]
were the *intended* idealised positions, and our derived values are the
*actual* ones from infrared-camera measurement.

### Why we use measured Table 2 values, not the idealised circle

This is one place where we deliberately deviate from the dataset's own
visualisation script. Compare:

| Sensor | Paper Table 2 yaw | `run_ground.py` THETA_DEGS | Difference |
|---|---|---|---|
| 5  | 177.93° | 180° | 2.07° |
| 6  | 117.59° | 135° | 17.41° |
| 7  | 84.98°  |  90° | 5.02° |
| 8  | 48.97°  |  45° | 3.97° |
| 9  | 2.55°   |   0° | 2.55° |
| 10 | −70.42° | −45° | 25.42° |
| 11 | −90.43° | −90° | 0.43° |
| 12 | −110.91°| −135°| 24.09° |

Sensor 6 is off by 17°, sensor 10 by 25°. These are *not* small errors
relative to the geometric structure we're trying to fuse. A 17° rotation
of one sensor's points means body parts seen by sensor 6 land tens of
centimetres away from where the same body parts seen by sensor 5 land.
DBSCAN's `eps=1.0` (1 metre) is forgiving enough to still cluster them
together, but the resulting cluster blurs out the body — instead of a
well-defined limb at one location, you get a smear.

The visualisation script is fine for producing animations because the
human eye doesn't care about a 25 cm offset in a GIF. We do care, so we
use the real numbers.

## 3.5 `transforms.py` line-by-line

```python
def _rotation_z(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
```

Standard yaw rotation matrix about the world +Z axis. Note that we go
degrees → radians once at the top — the rest of the code can speak in
degrees because that's what humans read off Table 2.

```python
def ground_sensor_transform(sensor_index: int):
    if sensor_index not in GROUND_TABLE2_XYZ:
        raise KeyError(...)
    yaw = ground_yaw_deg(sensor_index)
    R = _rotation_z(yaw)
    t = np.array(GROUND_TABLE2_XYZ[sensor_index], dtype=np.float64)
    return R, t
```

Returns the `(R, t)` pair for one sensor. Two reasons we return them
separately rather than as a 4×4 homogeneous matrix:

1. We're applying it to many points at once and the explicit
   `R @ pts + t` is faster than building 4D homogeneous vectors.
2. Future code (e.g. an inverse transform for visualisation) is easier
   to write when `R` and `t` are first-class.

```python
def to_world_frame(frames: list[np.ndarray], sensor_index: int):
    R, t = ground_sensor_transform(sensor_index)  # raises if not ground
    out = []
    for frame in frames:
        if frame.shape[0] == 0:
            out.append(frame.copy())
            continue
        local_xyz = frame[:, 1:4].T          # (3, N)
        world_xyz = (R @ local_xyz).T + t    # (N, 3)
        new = frame.copy()
        new[:, 1:4] = world_xyz
        out.append(new)
    return out
```

Three things worth pausing on.

**`frame[:, 1:4]`** picks only x, y, z (cols 1–3, since col 0 is
timestamp). Timestamp and density (col 4) are *not* rotated — they're
scalars that don't depend on coordinate frame.

**`.T` then `(R @ ...).T`** does column-vector matmul efficiently. We
have `N` points; rather than looping over them, we transpose to a
`(3, N)` matrix, apply the matrix multiply once, and transpose back to
`(N, 3)`. The one matmul replaces what would otherwise be `N` separate
operations.

**`new = frame.copy()`** — we return a *new* array with rotated x/y/z
and original timestamp/density. We deliberately don't mutate the input,
because the caller may want to keep the unrotated version (e.g. for
debugging). Memory cost is small; correctness benefit is large.

### Why ceiling sensors aren't supported yet

If you call `to_world_frame` with a ceiling sensor index (0–4), it
raises `NotImplementedError`. The reason: ceiling sensors look *down*,
not *horizontally*. Their local +X axis points along the ground (away
from the sensor's downward boresight in some convention) but the
boresight is along local +Z. To transform their detections into world
coordinates we need:

1. A pitch rotation of ~90° to align the radar's downward-facing axis
   with the world Z axis,
2. *Then* a yaw rotation about the new vertical axis to align horizontally.

That's a two-step rotation, and we'd need either the paper to specify
the ceiling sensors' exact mounting orientation or measure it ourselves.
We've defaulted to ground-only (Module 1, §1.4), so we don't need this
yet. When we do, the work goes in `transforms.py` and the rest of the
pipeline is unaffected.

## 3.6 `transform_sample` — the orchestration step

```python
def transform_sample(sensor_frames: dict[int, list[np.ndarray]]):
    return {sidx: to_world_frame(frames, sidx) for sidx, frames in sensor_frames.items()}
```

One line. Applies `to_world_frame` to every sensor in a sample. After
this, the dictionary's values are still per-sensor lists of frames, but
now *every list contains world-frame points*. Subsequent stages
(clutter, windowing, BEV) treat the per-sensor structure as
exchangeable — every world-frame point looks the same regardless of
which sensor produced it.

This is the moment "sensor fusion" actually happens conceptually. After
`transform_sample`, points from sensor 5 and sensor 9 live in the same
coordinate system; whether we keep them grouped by sensor (we do, until
windowing) or pool them is now an implementation detail.

## 3.7 Trace one call by hand

To make this concrete, run this Python snippet and read along with the
output:

```python
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.transforms import transform_sample
from rfbc.config import DEFAULT_CONFIG

sample = SampleId("C1", "U01", "M01", "01")
sensors = cfg = DEFAULT_CONFIG.selected_sensor_indices()
print(f"selected sensors: {sensors}")

raw = load_sample_sensors(sample, sensors)
print(f"loaded {len(raw)} sensors")
for sidx, frames in raw.items():
    first_frame = next((f for f in frames if f.shape[0] > 0), None)
    if first_frame is not None:
        print(f"  sensor {sidx}: first non-empty frame, x={first_frame[0, 1]:+.3f}, y={first_frame[0, 2]:+.3f}, z={first_frame[0, 3]:+.3f}")

world = transform_sample(raw)
for sidx, frames in world.items():
    first_frame = next((f for f in frames if f.shape[0] > 0), None)
    if first_frame is not None:
        print(f"  sensor {sidx}: same point in world, x={first_frame[0, 1]:+.3f}, y={first_frame[0, 2]:+.3f}, z={first_frame[0, 3]:+.3f}")
```

You should see two things.

First, **the local x values are mostly positive** (the radar boresight
is +X, so points "in front of" the radar are at positive local x).
Different sensors have different local x for the same physical hand
because each sensor's "in front of me" points in a different world
direction.

Second, after the transform, **the world coordinates from all 8 sensors
should land in roughly the same region** — somewhere near the origin,
because the person stood near the centre of the recording area. They
won't be at exactly the same point (the person is not a single point;
each sensor saw a different limb), but they'll all be within ~1 metre
of each other.

That's the fusion working.

## 3.8 Self-check questions

1. Why is `SampleId` a dataclass instead of a string like
   `"C1/U01/M01/01"`?
2. What does `load_sensor` do when it encounters an empty frame? When
   it encounters a non-array? When it encounters a `(N, 3)` frame?
3. Why does `load_sample_sensors` default to `skip_missing=True` and
   not raise when a sensor file is gone?
4. Derive the formula `yaw = atan2(-y_s, -x_s)` from the requirement
   "rotate local +X to point at the origin". Write out one full step.
5. Why don't we use the `THETA_DEGS = [180, 135, 90, 45, 0, ...]` from
   the dataset's own visualisation script?
6. In `to_world_frame`, why do we transpose to `(3, N)` before
   `R @ local_xyz` and transpose back? What would the alternative be?
7. Why are timestamp and density not rotated by the transform?
8. What's the geometrically correct (but unimplemented) rotation for a
   ceiling sensor?

---

# Module 4 — DBSCAN clutter suppression

This module is short. The clutter step does one thing — keep the points
that look like a person, throw away everything else — and the
implementation is essentially borrowed from the dataset paper. But the
*why* of each choice has consequences that ripple through the rest of
the pipeline, so we'll be precise about it.

We walk one file: `rfbc/data/clutter.py`.

## 4.1 What is "clutter" in this dataset?

The radars don't only see the person. Anything in the room that reflects
77 GHz mmWave radiation produces detections. In RF-Behavior's recording
setup, that includes:

- **Multipath reflections.** A signal bounces off the person, then off
  a wall, then back to the radar. The radar sees that as an extra
  detection at the *combined* range — a "ghost" point that doesn't
  correspond to any real object location.
- **Static reflectors.** Walls, fixtures, the ceiling, any metallic
  furniture, the camera tripods. Some of these produce extremely strong
  returns. They are static across frames, so they appear as persistent
  clusters at fixed locations.
- **CFAR false alarms.** The radar's internal Constant False Alarm Rate
  detector thresholds returns above an estimated noise floor. By design,
  about 1-in-some-thousand thermal-noise samples crosses the threshold.
  These show up as isolated, low-density detections at random locations.
- **Other living things in the area.** If a researcher walks past during
  a recording, that's a second moving cluster.

These contaminate the data. A CNN trained on raw fused points would
spend capacity learning to ignore the wall, learning that the radar
tripod isn't a body part. We don't want that — we want the model to
focus on *the person doing the activity*. So we strip the clutter
upstream, before any windowing or BEV.

## 4.2 Why clustering, and not a bounding box?

The simplest "remove clutter" idea is a fixed bounding box: keep only
points inside the recording area, drop everything outside. Why don't we
do that?

Two reasons.

First, **the person moves around the recording area**. Campaign 2
activities like walking and running cover the whole 6 m × 2 m space.
A bounding box tight around the centre would cut off the limbs of a
person at the edge. A bounding box wide enough to admit the whole
recording area also admits the static clutter inside that area —
furniture in particular.

Second, **clutter is mostly inside the recording area, not outside.**
Walls and ceiling are at the edges, but the camera tripods, the chair
in Campaign 3, the badminton net, the basketball — these all live
*near the person*. Spatial bounding doesn't separate them.

Clustering, in contrast, finds whichever connected blob of points
*looks like the person*, wherever in the room they are. The person
produces a relatively dense, spatially compact cluster (limbs and torso
within a few metres of each other). Walls produce thin, distant
clusters. Random multipath produces sparse, isolated points. The three
look different enough that a density-based algorithm separates them
cleanly.

## 4.3 DBSCAN — what it does and why it suits this data

**DBSCAN** stands for "Density-Based Spatial Clustering of Applications
with Noise". It's a clustering algorithm built around a single intuitive
rule: a *cluster* is a maximal set of points that are densely
interconnected, and a *noise* point is one that isn't part of any such
set.

Two hyperparameters define "densely":

- **`eps`** (epsilon) — the radius of a point's "neighbourhood". Two
  points are direct neighbours if they're within `eps` of each other.
- **`min_samples`** — the minimum number of points (including itself)
  that must be in a point's `eps`-neighbourhood for it to count as a
  *core* point.

Then the cluster definition is recursive: a cluster is the connected
component of all core points whose `eps`-neighbourhoods overlap, plus
the non-core (border) points that fall inside those neighbourhoods.
Anything not reachable that way is labelled noise (`-1`).

Three properties of DBSCAN matter for our use case:

1. **You don't have to specify the number of clusters in advance.** The
   algorithm decides for itself how many clusters there are. We don't
   know if "the person" will be a single blob or a couple of blobs (one
   for arms, one for torso, when a sensor only catches part of the
   body) — DBSCAN figures that out.
2. **Clusters can be arbitrary shape.** A person standing with arms out
   is not a sphere or an ellipsoid; it's a roughly cross-shaped blob.
   K-means (which assumes spherical clusters) would carve it up
   incorrectly. DBSCAN doesn't care about shape.
3. **Noise is a first-class output.** Multipath and CFAR false alarms
   are noise *by definition* — isolated, low-density. DBSCAN labels them
   as `-1` and we drop them.

The catch is that DBSCAN's output quality depends heavily on `eps` and
`min_samples`. The next section addresses how we picked them.

## 4.4 The hyperparameters: `eps=1.0`, `min_samples=3`

```python
def largest_cluster_mask(points_xyz, eps=1.0, min_samples=3):
    ...
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xyz)
```

**`eps = 1.0` metre** — chosen to be roughly the spatial extent of an
adult human body. With limbs extended (e.g. lateral-raise gesture), a
person occupies a volume about 1.6 m tall and 1.5 m wide. We want
neighbouring detections of the same body to fall within each other's
neighbourhoods even when one detection is on the head and another on
the foot. 1 m gets us "any detection on the body's central trunk has
the limbs in its neighbourhood".

If `eps` were too small (say, 0.2 m), the body would split into
multiple clusters (left arm vs right arm vs torso). The "largest
cluster" rule would then keep only one of them and we'd lose body
parts. If `eps` were too large (say, 3 m), the body would merge with
nearby clutter into one giant cluster and we'd lose discrimination.

**`min_samples = 3`** — the threshold for "this point's
neighbourhood contains enough other points to be considered the core of
a cluster". Three is the smallest value that's not 1 or 2.

Why not 1 or 2? With `min_samples = 1`, every single point is a core
point, every cluster has size 1, and the algorithm degenerates. With
`min_samples = 2`, a stray multipath point that happens to land near
a CFAR false alarm forms a "cluster" of two — exactly the noise we want
to discard.

Why not 5 or 10? With `min_samples = 5`, sparse genuine body returns
get labelled as noise. Remember the underlying density: each sensor
produces ~2.5 points per frame, so a 1-metre region during a single
frame may have only 2-3 points across all sensors. We want those to
count as cluster.

Three is the sweet spot the dataset authors used (Zuo et al. step 1)
and we adopt the same values without modification. This is one place
where deferring to the published values is the right call: the authors
empirically tuned for *their own* recording geometry, which is also our
recording geometry, and we have no reason to think we know better.

The default behaviour of `DBSCAN` from scikit-learn uses Euclidean
distance, which is correct here because we're operating in metric
world coordinates (after the transform from Module 3).

## 4.5 Why we cluster *after* fusion, not before

```python
parts: list[np.ndarray] = []
provenance: list[tuple[int, int, int]] = []  # (sensor_idx, frame_idx, npts)
for sidx, frames in sensor_world_frames.items():
    for fidx, frame in enumerate(frames):
        if frame.shape[0] == 0:
            provenance.append((sidx, fidx, 0))
            continue
        parts.append(frame[:, 1:4])  # x, y, z columns
        provenance.append((sidx, fidx, frame.shape[0]))
...
all_pts = np.concatenate(parts, axis=0)
mask = largest_cluster_mask(all_pts, ...)
```

We pool every sensor's every frame into one giant `(M, 3)` array,
DBSCAN that *once*, then walk the mask back to per-sensor-per-frame
structure. This is a deliberate choice that needs justification.

The alternative would be running DBSCAN per frame (8 sensors × 80
frames = 640 separate clusterings per sample). Why don't we?

**Density.** A single frame from a single sensor has 2-3 points. DBSCAN
on 3 points with `min_samples=3` will at best find one cluster of 3, at
worst label everything as noise. Per-frame DBSCAN simply doesn't have
enough data to separate signal from clutter — it's like trying to do
statistics on a single sample.

By pooling 1-2 thousand points across the entire 2-second sample, we
give DBSCAN enough density to work properly. The person is at slightly
different positions across frames (the activity *is* movement), so the
person's points sweep out a temporal-spatial blob, but that blob is
still much denser than the multipath background. DBSCAN finds it as
one cluster.

The cost: temporal resolution within the cluster decision. Once we've
decided "this point is part of the person-cluster", we don't track
whether the cluster shape *evolved* in interesting ways. For the
present project this is fine — windowing handles temporal information
downstream. If we ever wanted to detect, say, "the person fell over
half-way through the sample", per-frame clustering would be necessary,
but we don't.

## 4.6 The "largest cluster = person" assumption

```python
biggest = np.bincount(valid).argmax()
return labels == biggest
```

After DBSCAN labels every point, we count how many points each non-noise
cluster has and keep the biggest one. The implicit assumption is **the
person produces the largest cluster**.

When does this hold? Almost always. A person actively moving in a
2-second window generates dense body returns from ~8 sensors at ~27 Hz —
~1200 points spatially co-located. Static clutter (walls, furniture)
generates a smaller, more compact blob; CFAR noise is sparse and gets
labelled `-1`.

When could it fail? Three plausible scenarios:

1. **A second person walks past during the recording.** Their signature
   could rival the subject's. RF-Behavior's recording protocol
   (controlled sessions, single subject) makes this very unlikely.
2. **A large static reflector dominates.** If the badminton net in
   Campaign 2 produces more returns than the player, "largest cluster"
   would lock onto the net. We don't currently have evidence this
   happens, but it's worth checking once we visualise BEV tensors.
3. **The subject barely moves.** Sentiment Campaign 3 has subjects
   sitting still for minutes. Static-pose subjects produce fewer
   returns frame-over-frame than moving ones, so a wall could
   plausibly outscore them. This is one more reason we excluded C3 by
   default in Module 1.

If failure mode 2 ever materialises — say, you visualise a BEV grid and
see a ghostly net rather than a person — the mitigation is to add a
spatial prior: "weight clusters by how close their centroid is to the
recording-area origin", or "discard clusters whose centroid is more
than X metres from origin". For now, plain "largest" suffices.

## 4.7 The provenance dance

Once DBSCAN gives us a flat `(M,)` boolean mask, we have to walk it
back to the original per-sensor-per-frame structure:

```python
out: dict[int, list[np.ndarray]] = {sidx: [] for sidx in sensor_world_frames}
cursor = 0
for sidx, fidx, npts in provenance:
    if npts == 0:
        out[sidx].append(np.zeros((0, 5), dtype=np.float64))
        continue
    sub = mask[cursor:cursor + npts]
    cursor += npts
    original = sensor_world_frames[sidx][fidx]
    out[sidx].append(original[sub])
return out
```

The trick: as we built the flat array we kept a `provenance` list
remembering, in order, where each chunk of points came from
(`(sensor_idx, frame_idx, npts)`). Walking that list with a `cursor`
into the mask lets us slice out exactly the per-frame mask.

Two subtleties.

**Empty frames.** If a frame originally had zero points, we appended
`(sidx, fidx, 0)` to provenance but added nothing to `parts`. On the
walk-back, we don't advance the cursor (because there are no mask bits
for that frame); we just append an empty `(0, 5)` array to keep the
output structure parallel. The downstream pipeline treats empty frames
as "this sensor saw nothing in this time bin", which is correct.

**Frames that lose all their points after suppression.** If a frame
had 4 points and DBSCAN labelled all 4 as noise, the mask slice for
that frame is all False. `original[sub]` returns an empty `(0, 5)`
array. Same downstream treatment — it's just a frame where everything
got removed.

The output dict has the *same* keys, the *same* number of frames per
sensor, and the *same* (sensor, frame, time) addressing as the input.
Only the points within frames have changed (some removed). This
preserves the abstraction: clutter suppression is "filter out bad
points", not "restructure the data".

## 4.8 Boundary cases the code handles

```python
def largest_cluster_mask(points_xyz, eps=1.0, min_samples=3):
    if points_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    labels = DBSCAN(...).fit_predict(points_xyz)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return np.zeros(points_xyz.shape[0], dtype=bool)
    biggest = np.bincount(valid).argmax()
    return labels == biggest
```

Three guards.

1. **No points at all** (`points_xyz.shape[0] == 0`). Return an empty
   mask. Defensive: the caller handles this gracefully.
2. **DBSCAN found no clusters** (`valid.size == 0`). Every point was
   labelled noise. Return an all-False mask — drop everything. This is
   the right behaviour: if DBSCAN can't find any density, there's no
   trustworthy person signature in this sample, and downstream stages
   will see an empty fused cloud.
3. **Otherwise**, return the mask for the largest cluster.

The middle case is rare but it does happen. When it does, the BEV
tensor for that sample comes out all-zeros, and the model essentially
learns "this sample was too noisy to label". That's fine — it's
a small fraction of samples and the model can decline to commit on
those.

The wrapper `suppress_clutter` adds one more guard: if every input
frame is empty (no points anywhere), short-circuit and return
unchanged. Saves a needless DBSCAN call.

## 4.9 Sanity-check the clutter step on real data

Run this snippet to see clutter suppression in action:

```python
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.transforms import transform_sample
from rfbc.data.clutter import suppress_clutter
from rfbc.config import DEFAULT_CONFIG
import numpy as np

sample = SampleId("C1", "U01", "M01", "01")
raw = load_sample_sensors(sample, DEFAULT_CONFIG.selected_sensor_indices())
world = transform_sample(raw)

before = sum(f.shape[0] for frames in world.values() for f in frames)
clean = suppress_clutter(world, eps=1.0, min_samples=3)
after = sum(f.shape[0] for frames in clean.values() for f in frames)
print(f"before clutter step: {before} points")
print(f"after clutter step:  {after} points")
print(f"removed:             {before - after} points  ({100 * (before - after) / before:.1f}%)")

# Also show the spatial extent before and after
all_before = np.concatenate(
    [f[:, 1:4] for frames in world.values() for f in frames if f.shape[0]],
    axis=0,
)
all_after = np.concatenate(
    [f[:, 1:4] for frames in clean.values() for f in frames if f.shape[0]],
    axis=0,
)
print(f"\nbefore: x range {all_before[:, 0].min():+.2f} to {all_before[:, 0].max():+.2f}")
print(f"        y range {all_before[:, 1].min():+.2f} to {all_before[:, 1].max():+.2f}")
print(f"        z range {all_before[:, 2].min():+.2f} to {all_before[:, 2].max():+.2f}")
print(f"after:  x range {all_after[:, 0].min():+.2f} to {all_after[:, 0].max():+.2f}")
print(f"        y range {all_after[:, 1].min():+.2f} to {all_after[:, 1].max():+.2f}")
print(f"        z range {all_after[:, 2].min():+.2f} to {all_after[:, 2].max():+.2f}")
```

You should see:

- A modest fraction of points removed (typically 5-30 %, depending on
  how cluttered the recording was).
- The remaining points concentrated in a tighter spatial volume —
  noticeably more compact than the original. The wider-extent points
  (more than ~1 m from the centre cluster) are the multipath and
  static-clutter detections being thrown away.

If you see *almost no* points removed, the recording was unusually
clean. If you see >50 % removed, that sample has a lot of noise — still
runs, but worth looking at. (We may add a logging counter later to flag
high-removal samples.)

## 4.10 Self-check questions

1. Name three sources of clutter in mmWave radar HAR. For each, briefly
   describe how it appears in the data.
2. Why don't we use a fixed spatial bounding box to remove clutter
   instead of DBSCAN?
3. Define `eps` and `min_samples` in DBSCAN. State what would go wrong
   if `eps` were too small. State what would go wrong if `min_samples`
   were 1.
4. Why do we run DBSCAN on the fused all-sensor all-frame cloud
   instead of per-frame?
5. Under what circumstances would the "largest cluster = person"
   assumption fail, and what's a plausible mitigation?
6. What does the function return when DBSCAN labels every point as
   noise? Why is this the right behaviour?
7. Walk through the provenance-tracking logic in `suppress_clutter`.
   Why do we need it at all, given we could just return the flat
   masked cloud?

---

# Module 5 — Windowing: variable-duration samples → fixed-length tensors

After clutter suppression we have, per sample, a dictionary of
sensor → list of frames where each frame is a `(N, 5)` array of
world-frame, person-only points. The frames come at the radar's
~27 Hz native rate, the sample lasts somewhere between 3 seconds
(C1 gestures) and 18 seconds (some C2 activities), and *the timestamps
across sensors are not synchronised*.

Windowing's job is to convert this messy, variable-length, multi-stream
input into a list of fixed-length single-stream "windows" — each window
being a list of exactly `target_frames` time bins ready for BEV
construction. We walk one file: `rfbc/data/windowing.py`.

## 5.1 Why CNNs require fixed input shapes

CNNs operate on tensors of fixed shape. A `Conv2d` layer with kernel
size 3 expects an input of shape `(batch, channels, H, W)` where
`H` and `W` are determined at graph-construction time. A 64×64 input
runs through one set of weights; a 128×128 input runs through different
ones. You can't feed both into the same model.

For us, the variable axis is *time*: a 3-second gesture has 81 frames,
a 12-second walk has 324 frames. To feed both into the same CNN we
must either (a) pick a fixed time length and reshape every sample to
match, or (b) use an architecture that natively handles variable-
length inputs (RNN, Transformer, TCN with adaptive pooling, etc.).

Why option (a) — fixed length:

- **Compression-tooling alignment.** As argued in Module 2, the
  pruning/quantization tools work cleanly on Conv/BN/Linear blocks.
  RNNs are partially supported but with caveats; Transformers' multi-
  head attention has its own quantization story. Fixed-length CNN is
  the path of least friction.
- **WiFi-CSI HAR precedent.** Varga & Cao 2025 and Moshiri 2021 both
  use fixed-length 2D inputs (CSI spectrograms). To compare to them,
  we need the same input convention.
- **Simplicity of the deliberate-baseline argument.** A simple CNN on
  fixed inputs is easy to defend as "deliberately simple". An RNN-CNN
  hybrid invites questions about why we chose that combination.

So we commit to fixed-length windows of `target_frames = 54` time
bins. The next question is how we get from variable durations to
that.

## 5.2 The fixed-length choice — 2 seconds at ≈ 27 Hz = 54 frames

Why 2 seconds?

- **Long enough** to capture a full gesture or one period of a periodic
  activity. C1 gestures are ~3 s — a 2 s window gets 2/3 of one,
  which is enough for a CNN to recognise the gesture from a distinct
  segment. A walking step cycle is ~1.2–1.5 s — a 2 s window gets one
  full step plus margin.
- **Short enough** that within-window movement isn't extreme. At 2 s,
  the person's spatial extent within one window is bounded; for
  walking, they cover ~2 m in 2 s, which fits comfortably in our 8 m
  BEV grid.
- **A round number.** 54 frames at 27 Hz is convenient for the BEV
  tensor (54 × 3 z-bands × 3 features = 486 channels, evenly divisible
  for downstream convs).
- **Matches the WiFi-CSI HAR precedent.** Varga & Cao use 2-second
  windows. So do most CSI HAR papers. Direct comparability.

`fps=27.0` is the nominal radar frame rate from the dataset paper;
`target_frames = 54` is `2.0 * 27.0` rounded. Both live in
`config.py` and can be changed in one place.

## 5.3 Why we re-bin onto a uniform grid even though radars run at ~27 Hz

This is the most subtle design choice in `windowing.py`. The radars
sample at *approximately* 27 Hz. They're not hardware-synchronised:
each starts independently when the recording begins, and timestamps
are in Unix epoch seconds. So sensor 5's frames at, say, t = 0.000,
0.037, 0.074, … and sensor 9's frames are at t = 0.012, 0.049, 0.086, …
— offset by a constant amount, plus per-frame jitter.

For a BEV tensor, we need to treat all sensors' frames uniformly:
"frame 0 of the BEV tensor" must mean the same time interval
across all sensors. The natural way to enforce this is **re-binning
all detections onto a single, common time grid**.

So the algorithm is:

1. Pool *every* point from *every* sensor into one big array, keeping
   timestamps.
2. Define a uniform grid of bins: bin `i` covers `[t_min + i/fps,
   t_min + (i+1)/fps)`.
3. Assign each point to its bin via `floor((t - t_min) * fps)`.
4. The output frame list now has, in slot `i`, all points (from any
   sensor) whose timestamp falls in bin `i`.

This gives us a clean per-frame structure where every frame represents
the same physical time interval and contains contributions from
whichever sensors happened to fire in that interval. Any per-sensor
asynchrony is absorbed into the binning step.

Subtle consequence: a single BEV "frame" may contain points from 8
sensors fired ~37 ms apart, or it may contain points from only 1
sensor (when all others happened to fire just outside the bin). We
treat all those cases the same; they're noise in the temporal axis,
but the BEV's per-cell aggregation is robust to it.

## 5.4 Pad short, slide long — the decision tree

```python
total_frames = max(1, int(np.ceil(duration / bin_width)))
if total_frames <= target_frames:
    return [fuse_sensors_by_time(sensor_frames, target_frames=target_frames, fps=fps)]
```

The first decision: is this sample shorter than (or equal to) one
window? If so, **pad short**. We bin the sample's points into
`target_frames` slots; any slots beyond the sample's actual duration
remain empty. The output is one window, possibly with empty trailing
frames.

For C1 gestures (~3 s = ~81 native frames) at `target_frames = 54`,
this branch usually doesn't apply — but for samples shorter than
2 seconds (or near-empty after clutter suppression) it's the safe path.

```python
full_grid = _full_uniform_grid(all_pts, t_min, total_frames, bin_width)
windows = []
for start in range(0, total_frames - target_frames + 1, stride_frames):
    windows.append(full_grid[start:start + target_frames])
```

Otherwise, **slide long**. We first bin the *whole* sample onto a
uniform grid of length `total_frames` (could be 200, 400, 1000 — as
long as the sample is). Then we slice out 54-frame windows starting at
positions 0, stride, 2·stride, … . Each window becomes a separate
training example.

This handles C2 activities elegantly. A 12-second walking sample
(~324 frames) at stride = 27 (1 second) yields windows starting at
frames [0, 27, 54, 81, …]. The same person walking yields ~10
overlapping windows, each 2 seconds long, each labelled "walking".

The stride is configurable. With `window_stride_seconds = 1.0`
(default), windows overlap by 50%. Reducing the stride increases the
number of training examples per long sample (but they become more
correlated); increasing it reduces correlation but loses examples.

Important: **we don't oversample within a sample**. The C2 walking
sample becomes ~10 distinct windows, all labelled "walking", and they
all live in the *same fold* of the subject-independent split (because
they came from the same subject). So sliding doesn't leak data
between train and test — it just gives us more examples of "walking
from this subject".

## 5.5 `fuse_sensors_by_time` — the binning math

```python
parts = [
    np.concatenate(frames, axis=0)
    for frames in sensor_frames.values()
    if frames and any(f.shape[0] for f in frames)
]
if not parts:
    return [np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]
all_pts = np.concatenate(parts, axis=0)
```

Step 1: pool everything. We concatenate every sensor's every non-empty
frame into one giant `(M, 5)` array. After this we no longer track
which sensor produced which point — the BEV will treat them all the
same.

```python
t_min = all_pts[:, 0].min()
bin_width = 1.0 / fps
bins = [np.zeros((0, 5), dtype=np.float64) for _ in range(target_frames)]

rel_t = all_pts[:, 0] - t_min
bin_idx = np.floor(rel_t / bin_width).astype(np.int64)
in_range = (bin_idx >= 0) & (bin_idx < target_frames)
pts = all_pts[in_range]
bin_idx = bin_idx[in_range]
```

Step 2: bin assignment. We compute each point's bin index by
`floor((t - t_min) * fps)`. This is just integer arithmetic on
relative time. We then drop anything outside `[0, target_frames)` —
which only happens if the sample is longer than the window
(`fuse_sensors_by_time` is the "single window" path; for multi-window
processing we use `_full_uniform_grid` instead).

```python
if pts.shape[0]:
    order = np.argsort(bin_idx, kind="stable")
    pts = pts[order]
    bin_idx = bin_idx[order]
    edges = np.searchsorted(bin_idx, np.arange(target_frames + 1))
    for i in range(target_frames):
        bins[i] = pts[edges[i]:edges[i + 1]]
return bins
```

Step 3: slice into per-frame arrays. This is the interesting bit.

The naive way would be a Python loop:
```python
for bi in range(target_frames):
    bins[bi] = pts[bin_idx == bi]
```
Each `pts[bin_idx == bi]` is a fresh boolean-mask scan over the entire
array — O(M) per bin, O(M·F) total. For M = 1500 points and F = 54
bins that's 81 000 ops, which is fine but wasteful.

The `argsort + searchsorted` trick is O((M log M) + F):

1. Sort the points by bin index. After this, all points in bin 0 are
   contiguous, then all in bin 1, etc.
2. `np.searchsorted(sorted_bin_idx, np.arange(F+1))` finds the *edges*
   between bins in one vectorised call. `edges[i]` is the position
   where bin `i` starts (and `edges[i+1]` is where it ends).
3. Slicing `pts[edges[i]:edges[i+1]]` is an O(1) view, no copy.

For 1500 points and 54 bins this is ~10× faster than the naive scan.
For longer samples (C2 walking at ~9000 points across 324 frames) the
speedup matters even more, especially because the dataloader hits this
function many thousands of times during a training epoch.

The `kind="stable"` argument to `argsort` preserves the original order
of points within the same bin. Not strictly necessary for correctness
(BEV doesn't care about within-frame order), but it's good practice
and free.

## 5.6 `split_into_windows` — handling multi-window samples

```python
def split_into_windows(sensor_frames, *, target_frames, fps, stride_frames=None):
    if stride_frames is None:
        stride_frames = target_frames  # non-overlapping by default
    ...
    if total_frames <= target_frames:
        return [fuse_sensors_by_time(...)]
    full_grid = _full_uniform_grid(all_pts, t_min, total_frames, bin_width)
    windows = []
    for start in range(0, total_frames - target_frames + 1, stride_frames):
        windows.append(full_grid[start:start + target_frames])
```

The structure mirrors `fuse_sensors_by_time` but with two differences:

1. The grid spans the *whole* sample (`total_frames` slots), not just
   one window's worth.
2. We slice fixed-length windows out of the long grid at strided
   positions.

`_full_uniform_grid` is a helper that does the same argsort/searchsorted
binning as `fuse_sensors_by_time` but with a runtime-determined number
of bins. Keeping it factored out avoids duplicating the binning logic.

### The tail-window subtlety

```python
last_start = total_frames - target_frames
if last_start > 0 and (windows == [] or windows[-1] is not full_grid[last_start:last_start + target_frames]):
    tail = full_grid[last_start:last_start + target_frames]
    if not windows or not _windows_equal(windows[-1], tail):
        windows.append(tail)
```

Suppose `total_frames = 100`, `target_frames = 54`, `stride_frames = 27`.
The strided positions are `0, 27, 54, …` up to but not exceeding
`total_frames - target_frames = 46`. So we'd get windows starting at
0, 27, 46? No — we'd get windows starting at 0 and 27, then the next
position would be 54 which is > 46, so we stop.

That means the last window starts at frame 27 and ends at frame 81 —
**we'd lose the last 19 frames of the sample**. Probably fine for
periodic activities (one more step cycle isn't critical) but wasteful.

The tail-window block adds one more window starting at exactly
`total_frames - target_frames` so the last window ends at the last
frame. We deduplicate (don't add a tail that matches the last strided
window) to avoid double-counting in the common case where the strides
do reach the end.

This is a minor refinement but it adds up: across thousands of long
C2 samples we recover thousands of extra training windows that would
otherwise be dropped.

## 5.7 What this looks like for each campaign

The decision tree from §5.4 produces different behaviour for each
campaign. With `target_frames = 54` (2 s) and `fps = 27`:

- **C1 gestures (~3 s, ~81 native frames).** The sample is longer than
  one window. With stride 27 (1 s), we get 2 windows: frames [0–53]
  and a tail window at the end. So one C1 gesture gives 2 training
  examples, not 1.
  - Wait — looking at our smoke test output: `train items: 5` for one
    subject × six classes × one rep, and we set `repetition_filter =
    ('01',)`. Let me reconcile: 1 subject × 6 classes × 1 rep × ~2
    windows = ~12, but we saw 5. The smoke test trims subjects with
    `args.max_subjects_per_fold`, so 2 train subjects × 6 classes ×
    1 rep × something. The point: the exact count depends on which
    samples actually had enough duration after clutter, not on the
    nominal 3 s.

- **C2 activities (3–18 s).** Most are several windows. A 12 s walking
  sample gives ~10 overlapping windows.

- **C3 sentiment (50–180 s).** Excluded by default. If we ever included
  it, one sample would give 50–100+ windows — which is itself a reason
  to keep them excluded: they'd dominate the training distribution.

This is the windowing's main contribution: each campaign produces
roughly the right *number of training examples* for the pipeline,
without us having to special-case anything.

## 5.8 Sanity check on real data

Run this on a short C1 sample and a long C2 sample to see both branches
of the decision tree fire:

```python
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.transforms import transform_sample
from rfbc.data.clutter import suppress_clutter
from rfbc.data.windowing import split_into_windows
from rfbc.config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG

def windowing_demo(sample):
    raw = load_sample_sensors(sample, cfg.selected_sensor_indices())
    world = transform_sample(raw)
    clean = suppress_clutter(world, eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    stride_frames = max(1, int(round(cfg.window_stride_seconds * cfg.fps)))
    windows = split_into_windows(clean, target_frames=cfg.target_frames,
                                 fps=cfg.fps, stride_frames=stride_frames)
    print(f"{sample}:  {len(windows)} window(s)")
    for i, w in enumerate(windows):
        n_pts = sum(f.shape[0] for f in w)
        nonempty = sum(1 for f in w if f.shape[0] > 0)
        print(f"  window {i}: {n_pts} total points across {nonempty}/{len(w)} non-empty frames")

windowing_demo(SampleId("C1", "U01", "M01", "01"))   # ~3 s gesture
windowing_demo(SampleId("C2", "U01", "M01", "01"))   # walking, longer
```

Expected output (numbers will vary):

- C1 sample: 1–2 windows, 1000-ish total points each, most frames non-empty.
- C2 sample: 5–10 windows, similar per-window point counts. Most or
  all frames non-empty in each window.

If a C1 sample produces 0 windows, something upstream killed all the
points (DBSCAN found no cluster). If a C2 sample produces a
single window, the recording was unusually short.

## 5.9 Self-check questions

1. Why do CNNs require fixed input shapes, and why didn't we go with an
   RNN or Transformer instead?
2. Why is the choice of `target_frames = 54` not arbitrary? Name three
   considerations behind it.
3. The radars all run at "approximately 27 Hz". Why does this matter,
   and what does the windowing code do about it?
4. Walk through the `floor((t - t_min) * fps)` computation. What
   physical quantity does each step represent?
5. Explain the `argsort + searchsorted` slicing trick. Why is it faster
   than the naive `pts[bin_idx == i]` loop, and by roughly how much?
6. Describe the pad-short / slide-long decision tree. When does each
   branch trigger, and how does it differ in output structure?
7. Why is the tail-window block in `split_into_windows` necessary? What
   is being recovered, and what would be lost without it?
8. The default `window_stride_seconds = 1.0` produces 50% overlap
   between windows. What's the trade-off — what does smaller vs larger
   stride do to the training distribution?

---

# Module 6 — BEV construction: sparse points → dense tensor

This is the pivot module. Up to now we've been doing physics: the data
is a list of detections in metres, with timestamps and densities. The
CNN doesn't speak metres. It speaks tensors of fixed shape with
floating-point channel values that look like an image.

`bev.py` is the bridge. After this module the data is *image-shaped*
and every CNN technique from the literature applies. We walk one file —
`rfbc/data/bev.py` — and the BEV-related parameters in `config.py`.

## 6.1 What "BEV" means and why it's the right representation

**BEV** stands for *bird's-eye view*. The convention comes from
autonomous driving: take a 3D point cloud (LiDAR, radar) of the
environment around a car, project it onto the ground plane, bin into a
2D grid, and feed that grid to a CNN. The same approach is used
constantly in 3D object detection (PointPillars, BEVFusion, CenterPoint
all rely on it).

For RF-Behavior, BEV is the natural choice for three reasons:

1. **The recording-area floor is the relevant reference frame.** The
   activity space is a 6 m × 2 m rectangle on the floor. The person
   stands or moves on that floor. A bird's-eye projection preserves
   horizontal motion (where in the room is the person, which way are
   they facing) which is exactly what gestures and activities encode.

2. **It maps cleanly to the sensors' geometry.** Eight ground radars
   in a horizontal ring all look across the floor. Their detections
   are naturally distributed in the x-y plane. Projecting to BEV
   loses very little information about *what each sensor saw*;
   projecting to, say, a side view (x-z plane) would emphasise some
   sensors and underweight others.

3. **It matches the WiFi-CSI HAR precedent.** CSI papers turn the
   amplitude-vs-frequency-vs-time signal into a 2D spectrogram. Our
   BEV-with-temporal-channels has the same tensor shape conceptually,
   which makes results directly comparable. (More on this in §6.8.)

The 2D projection isn't free — we lose vertical structure, and a
careless implementation collapses "head" and "feet" into the same
pixel. The z-band trick in §6.4 mitigates this.

## 6.2 The `(T, Z, F, H, W)` shape — why each axis exists

The output tensor has five conceptual axes which we then flatten into
three for the CNN. From outermost to innermost in `bev.py`'s internal
representation:

- **T (time)** — `target_frames = 54`. One slice per frame in the
  windowed time grid. This is where temporal information lives.
- **Z (z-band)** — `num_z_bands = 3`. Three vertical slabs through
  the recording area, picked to roughly correspond to "low" (legs),
  "middle" (torso), "high" (head/upper limbs). See §6.4.
- **F (feature)** — `feat_per_cell = 3`. Three per-cell channel
  features: point count, mean density, max density. See §6.3.
- **H (vertical pixel index)** — `grid_size = 64`. The y-axis of the
  BEV grid (north-south in the room).
- **W (horizontal pixel index)** — `grid_size = 64`. The x-axis of
  the BEV grid (east-west in the room).

Internal layout `(T, Z, F, H, W)` is just a NumPy convenience while we
build the tensor: it makes the inner loops clean. At the end we flatten
the leading three axes into a single channel dimension:

```python
return out.reshape(cfg.target_frames * Z * F, H, W)
```

So the CNN sees a tensor of shape `(C, H, W) = (486, 64, 64)` where
`C = 54 × 3 × 3`. Conv2d kernels treat every channel as semantically
equivalent — they don't know that channel 0 is "frame 0, low-z, count"
and channel 487 is "frame 0, low-z, mean density". They just learn
weights per-channel. This works because the *spatial* structure
(within H × W) carries the body's geometric information, and the
*channel* structure carries the time × z-band × feature distinctions
that the kernels learn to combine.

### Why flatten T into channels rather than treat it as a fourth axis?

A natural alternative is `(C, T, H, W)` — a 3D-conv-style tensor with
explicit time axis. We don't, for two reasons:

1. **Compression tooling.** `torch.nn.utils.prune` and
   `torch.ao.quantization` work on `Conv2d` cleanly. `Conv3d` is
   supported but with rougher edges and fewer pre-built quantization
   recipes. Module 2's argument applies here.
2. **WiFi-CSI HAR convention.** Their spectrograms use time as one of
   the *spatial* dimensions of a 2D tensor (frequency × time), which
   the CNN then treats as image-shaped. Putting T into channels is
   slightly different but in the same spirit, and lets us reuse 2D
   ConvNet architectures verbatim.

If we ever want to switch to 3D convs as an ablation, the change is
local to `points_to_bev`'s `reshape` call.

## 6.3 The three per-cell features — why count, mean density, max density

For each `(time, z-band, BEV cell)`, we compute three numbers:

- **`feat 0`: point count** — how many detections fell in that cell.
  Tells the CNN "where the body is at all". A cell with count 0 saw
  nothing; a cell with count 5 saw a lot.
- **`feat 1`: mean density** — average confidence across the cell's
  detections. Tells the CNN "how confident were the detections in
  this region". Useful for separating strong returns (probably a
  body) from weak ones (probably edge clutter that survived DBSCAN).
- **`feat 2`: max density** — the peak density value seen in the
  cell. Captures the *strongest* return regardless of how many other
  points fell in the same cell. A single very high-density detection
  in a sparse cell still shows up in this channel.

Why these three specifically?

The choice has both empirical and theoretical motivation. In point-
cloud BEV literature (PointPillars, VoxelNet) the standard cell
features are *count*, *mean intensity*, *max intensity*, sometimes
plus *mean position offset*. We reuse the standard set, mapping
"intensity" → "density" because that's what RF-Behavior provides.

The three features are *complementary*: count is binary-like
(present vs absent), mean smooths over detections, max is robust to
isolated low-density points. A CNN that only got count would lose
all confidence information; one that only got mean would be fooled
by a single high-density outlier; one that only got max would
ignore "many medium-density detections in the same cell". Three
features together cover the cases.

We deliberately *don't* include things like position offset within
the cell, or velocity (which we don't have), or RGB-style colour
encoding. Each extra feature is a 2× cost in channels. 486 channels
is already a lot for a "deliberately simple" baseline; adding more
without empirical evidence they help is a bad trade.

## 6.4 Z-bands — a partial recovery of vertical structure

If we projected all z values onto a single 2D BEV slice, we'd lose
the height information entirely. That's bad for HAR specifically:
"hand at chest height" and "hand at head height" produce identical
BEV slices but they're different gestures.

Instead, we split the vertical dimension into a small number of
*z-bands*:

```python
z_bands: tuple[float, ...] = (-0.5, 0.5, 1.2, 2.2)  # 3 bands defined by 4 edges
```

The four edges define three bands:

- **Band 0**: `-0.5 m to 0.5 m` — feet, lower legs
- **Band 1**: `0.5 m to 1.2 m` — torso, hips, hands at hip height
- **Band 2**: `1.2 m to 2.2 m` — chest, head, hands raised

Each band gets its own H × W slice. Detections at z = 0.3 contribute
to band 0; at z = 1.8 to band 2. The CNN sees these as separate
channels of the same image, so it can learn things like "lateral-raise
gesture has high density in band 2 (raised hands) and band 1 (torso)
simultaneously" or "ascending stairs has progressively higher density
in band 0 (lifted leg) frame-over-frame".

The choice of 3 bands is a deliberate trade-off:

- 1 band (no vertical info): too aggressive a collapse.
- 3 bands (current): captures the major body regions without exploding
  the channel count.
- 8 bands: unnecessary for this dataset's sparsity. With ~2.5 detections
  per frame per sensor, very few cells would ever be populated in any
  one band.

The edge values (-0.5, 0.5, 1.2, 2.2 m) are **not** at uniform spacing.
0.5 m and 1.2 m are anatomically motivated: 1.2 m is roughly the upper
torso of an adult standing on the floor (the radars themselves are at
~0.9 m, so the band boundary at 1.2 m is just above radar height), and
0.5 m is roughly the hip height threshold. The -0.5 m floor admits
points slightly below the floor (DBSCAN-surviving multipath occasionally
ends up there because of the world-frame transform's small angle
errors); 2.2 m caps at typical arms-overhead height plus margin.

These edges live in `config.py` and can be tuned. If we ever moved to
a dataset of taller subjects (basketball players, say) we'd raise the
top edge.

### Why not voxelise — proper 3D grid?

A natural alternative is a full 3D `(H, W, D)` voxel grid. We don't
because:

1. It's many more cells. 64 × 64 × 16 = 65 K cells per (time, feature)
   slice vs 64 × 64 × 3 = 12 K. Mostly empty.
2. 3D convs make compression tooling rougher (Module 2 again).
3. Z-bands give us *most of the win* of voxelisation at a fraction of
   the cost: human anatomy has roughly three macro-bands worth of
   structural variation, not 16.

## 6.5 The grid extent — `±4 m`, 64 × 64 cells, 12.5 cm cell size

```python
grid_size: int = 64
grid_extent_m: float = 4.0
```

The BEV grid covers `(-4 m, +4 m)` in both x and y. Cell size is
`(2 × 4 m) / 64 = 0.125 m = 12.5 cm`.

Three justifications:

**Why ±4 m extent?** The recording area is 2 m (x) × 6 m (y), but the *origin*
is at the room's centre (Module 3, Table 2). So the area itself spans
roughly `(-3 m, +3 m)` in x and `(-1 m, +1 m)` in y, plus the
furthest-out radars at ±2.9 m. ±4 m gives margin for points that drift
slightly outside the rectangle during clutter survival or that come
from a person at the edge of the area, while not wasting cells on
empty space far from the action.

**Why 64 × 64 cells?** Two reasons. (1) It matches the grid size in
the cited WiFi-CSI HAR papers (Varga & Cao use 64×64 spectrograms),
which keeps comparisons direct. (2) 64 is a power of 2, which makes
downsampling clean — three rounds of `MaxPool2d(2)` brings you to 8×8
without remainders. Most CNN architectures assume the input dimensions
are divisible by powers of 2.

**Why 12.5 cm cell size?** Falls out of the above. Worth checking
whether this is too coarse: an adult body is ~50 cm wide, so it
occupies ~4 × 4 cells in BEV. That's enough resolution to see the
torso vs the limbs but not enough to see, say, finger movements. For
hand-gesture classification (C1) we're banking on the wrist's *spatial
trajectory* being distinctive, not the fingers. If supervisor feedback
suggests we need finer resolution, halving the cell size to 6.25 cm
means going to 128×128 grid (4× more cells) and ~2 GB BEV cache. We
can; we shouldn't unless an experiment says we need to.

## 6.6 `points_to_bev` — line by line

```python
def points_to_bev(frames, cfg):
    if len(frames) != cfg.target_frames:
        raise ValueError(...)
```

The contract: exactly `target_frames` frames, no more, no less. The
windowing module guarantees this. If something upstream breaks the
contract we want a loud failure.

```python
H = W = cfg.grid_size
Z = cfg.num_z_bands
F = cfg.feat_per_cell

extent = cfg.grid_extent_m
cell = (2.0 * extent) / cfg.grid_size

z_edges = np.asarray(cfg.z_bands, dtype=np.float64)

out = np.zeros((cfg.target_frames, Z, F, H, W), dtype=np.float32)
```

Setup. We pre-allocate the full output tensor in float32 (saves memory
versus float64, fine for CNN training). Empty cells stay zero.

```python
for t, frame in enumerate(frames):
    if frame.shape[0] == 0:
        continue
    x = frame[:, 1]
    y = frame[:, 2]
    z = frame[:, 3]
    d = frame[:, 4]
```

Outer loop over time. Empty frames contribute nothing — `out[t, ...]`
stays zero, which is correct (the CNN sees an all-zero slice for that
moment in time, learns "nothing happened then"). We unpack columns by
name (x, y, z, density) for readability.

### Spatial discretisation

```python
ix = np.floor((x + extent) / cell).astype(np.int64)
iy = np.floor((y + extent) / cell).astype(np.int64)
in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
if not in_bounds.any():
    continue
ix, iy, z, d = ix[in_bounds], iy[in_bounds], z[in_bounds], d[in_bounds]
```

The bin formula: `(x + extent) / cell` shifts the world origin to
`(extent, extent)` (so all coordinates are non-negative) and divides
by cell width. `np.floor` then truncates to integer cell index.

Worked example: a point at `world x = 0.3 m`, `extent = 4.0`,
`cell = 0.125`. `(0.3 + 4.0) / 0.125 = 34.4`, `floor = 34`. So that
point lands in column 34 of the grid (out of 0–63). Column 32 would be
the centre, so column 34 is just to the right of centre — checks out
with x = 0.3 m being slightly positive.

The `in_bounds` mask drops points outside `(-extent, +extent)`.
After clutter the cluster is normally well within ±3 m, so this clip
is rarely doing anything, but it's defensive: a stray surviving
multipath ghost at x = 5 m would otherwise crash with an out-of-range
index.

### Z-band assignment

```python
zi = np.digitize(z, z_edges) - 1
for band in range(Z):
    sel = zi == band
    if not sel.any():
        continue
    ix_b, iy_b, d_b = ix[sel], iy[sel], d[sel]
```

`np.digitize(z, z_edges)` returns the index of the first edge
*greater than* z, for each point. The `- 1` shifts to give the band
index that contains z: e.g. z = 0.3 falls between edges 0.5 and 1.2,
so `digitize` returns 1, and `band = 0`. Points below the lowest edge
get band -1, points above the highest edge get band Z. The `for band
in range(Z)` loop ignores both — only points in valid bands [0, Z)
contribute.

We loop over bands rather than vectorising across them because each
band scatter writes into a different output sub-array. The 3 iterations
are cheap.

### The scatter operations

```python
np.add.at(out[t, band, 0], (iy_b, ix_b), 1.0)
np.add.at(out[t, band, 1], (iy_b, ix_b), d_b)
np.maximum.at(out[t, band, 2], (iy_b, ix_b), d_b)
```

This is the most subtle part of the file. `np.add.at` and
`np.maximum.at` are the *unbuffered* versions of `+=` and `np.maximum`.

Why "unbuffered" matters: if you write
```python
out[t, band, 0][iy_b, ix_b] += 1.0
```
and `iy_b, ix_b` contains repeated indices (multiple points in the
same cell), the naive form does *not* increment the cell once per
point. NumPy applies the `+=` once per *unique* index — the last
write wins, the others get clobbered. So if 3 points land in cell
(10, 5), you get count = 1, not 3.

`np.add.at(arr, idx, vals)` is the correct variant: it processes each
index in turn, applying the operation. With repeated indices it does
the right thing — the final cell count is exactly the number of points
that hit it.

Same logic for `np.maximum.at`. Naive `out[..., iy_b, ix_b] = np.maximum(out[..., iy_b, ix_b], d_b)` is broken with repeats; `np.maximum.at` correctly takes the max over all values that hit each cell.

These are standard NumPy idioms for "scatter with reduction" — the
same patterns appear in voxel-grid code, image binning, sparse-tensor
construction. If you've seen `scatter_add_` in PyTorch, it's the same
operation.

### Finalising the mean

```python
for band in range(Z):
    counts = out[t, band, 0]
    sums = out[t, band, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, 0.0)
    out[t, band, 1] = mean
```

After the scatter, `out[t, band, 1]` holds the *sum* of densities, not
the mean. We compute mean = sum / count, with a safe handling for
empty cells (count = 0 → mean = 0). The `np.errstate` suppresses the
"divide by zero" warning that the broadcast division would otherwise
produce; we know empty cells get overwritten by the `np.where` branch
anyway, so the warning is just noise.

A tidier version of this would compute the mean inline as we scatter,
but `np.add.at` doesn't combine with mean naturally — hence the
two-pass approach (sum then divide).

### Final reshape

```python
return out.reshape(cfg.target_frames * Z * F, H, W)
```

Flattens the leading three axes. The order is `target_frames` slowest,
then `Z`, then `F` — so channel 0 is `(t=0, band=0, feat=0)`, channel
1 is `(t=0, band=0, feat=1)`, …, channel `T*Z*F - 1` is
`(t=T-1, band=Z-1, feat=F-1)`. This ordering doesn't matter for the
CNN's learning ability, but it does matter for visualisation and
debugging — see Module 8.

## 6.7 What the resulting tensor looks like for a real sample

For the C1/U01/M01/01 sample we used as a running example:

- ~1200 points fed into BEV (after clutter)
- 54 frames, 3 z-bands, 3 features → 486 channels
- 64 × 64 spatial → 4096 cells per channel
- Total cells: 486 × 4096 ≈ 2 million per sample
- **Non-empty cells: ~2700** (about 0.13 %)

The tensor is *extremely sparse*. Most channels and most cells are
zero. This is exactly what we expected from the sparsity discussion in
Module 1, §1.6. CNNs handle this fine — convolutions of all-zero
neighbourhoods produce all-zero outputs, and the BatchNorm/learnable
biases pick up the signal in the populated cells. We could in
principle save memory with sparse-tensor representations, but for a
deliberately simple baseline the memory cost is tolerable
(~8 MB per sample as float32, much less when zlib-compressed in the
.npz cache).

## 6.8 Comparison to a WiFi-CSI HAR spectrogram

For your supervisor: the most useful framing of the BEV tensor is as
"a generalised spectrogram". A 2D CSI spectrogram has axes
`(frequency_subcarriers, time)`; ours has axes
`(spatial_y, spatial_x)` *per channel*, with the channels carrying
`(time, z-band, feature)` information.

| Property | CSI spectrogram | RF-Behavior BEV |
|---|---|---|
| Tensor shape | (C, F, T) | (C, H, W) |
| Spatial axes | freq subcarriers, time | room x, room y |
| Channels | typically 1–3 (amp/phase) | 486 (54 × 3 × 3) |
| Sparsity | dense | very sparse |
| Cell semantics | RF amplitude/phase | point density / count |

The CNN architectures from CSI HAR (small ResNet-style nets, EfficientNet
variants, simple stacks of 3×3 convs) all transfer. The compression
findings transfer too: pruning sensitivities, quantization quirks,
class-wise effects. The fact that *our* spatial axes encode physical
room geometry rather than spectrogram frequency is an interesting
domain difference but doesn't change the structural argument.

This is why your eventual results table will sit cleanly next to
Varga & Cao 2025's: same tensor shape, same convolution machinery,
same compression operators, different domain.

## 6.9 Sanity check on real data

Run this to construct a BEV tensor end-to-end and inspect its
structure:

```python
import numpy as np
from rfbc.data.loader import SampleId, load_sample_sensors
from rfbc.data.transforms import transform_sample
from rfbc.data.clutter import suppress_clutter
from rfbc.data.windowing import split_into_windows
from rfbc.data.bev import points_to_bev
from rfbc.config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG
sample = SampleId("C1", "U01", "M01", "01")

raw = load_sample_sensors(sample, cfg.selected_sensor_indices())
world = transform_sample(raw)
clean = suppress_clutter(world, eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
stride = max(1, int(round(cfg.window_stride_seconds * cfg.fps)))
windows = split_into_windows(clean, target_frames=cfg.target_frames, fps=cfg.fps, stride_frames=stride)

bev = points_to_bev(windows[0], cfg)
print(f"BEV shape: {bev.shape}")        # expect (486, 64, 64)
print(f"BEV dtype: {bev.dtype}")         # expect float32
print(f"Total cells:    {bev.size}")
print(f"Non-empty cells: {(bev > 0).sum()} ({100 * (bev > 0).mean():.3f}%)")

# Look at one specific channel — say t=10, z-band=1 (torso), feature=0 (count)
chan = 10 * cfg.num_z_bands * cfg.feat_per_cell + 1 * cfg.feat_per_cell + 0
slice_ = bev[chan]
print(f"\nChannel {chan} (frame 10, z-band 1, count):")
print(f"  non-zero cells: {(slice_ > 0).sum()}")
print(f"  max value: {slice_.max()}")
print(f"  positions of non-zero cells (iy, ix):")
ys, xs = np.where(slice_ > 0)
for y, x in zip(ys[:6], xs[:6]):
    world_x = (x + 0.5) * (2*cfg.grid_extent_m/cfg.grid_size) - cfg.grid_extent_m
    world_y = (y + 0.5) * (2*cfg.grid_extent_m/cfg.grid_size) - cfg.grid_extent_m
    print(f"    cell ({y:>2}, {x:>2}) -> world (~{world_x:+.2f}, {world_y:+.2f})  count={slice_[y, x]}")
```

You should see the non-zero cells cluster around the world origin (the
person's location), all within ~1 m of (0, 0). If they don't, the
clutter step or transform is misbehaving.

## 6.10 Self-check questions

1. What does each axis in the conceptual `(T, Z, F, H, W)` layout
   represent? Why do we flatten the leading three into channels for
   the CNN?
2. Name the three per-cell features and explain in one sentence each
   why none of them is redundant given the others.
3. Why z-bands instead of full 3D voxels? Why z-bands instead of full
   2D collapse?
4. Derive the cell-size figure (12.5 cm) from `grid_size` and
   `grid_extent_m`. What's the trade-off in halving the cell size?
5. Walk through the `(x + extent) / cell` discretisation formula.
   What does it return for a point at world `x = -2.0`?
6. What's the difference between `np.add.at(arr, idx, val)` and
   `arr[idx] += val`? Why does that difference matter for the BEV
   scatter?
7. Estimate the sparsity (non-empty cells / total cells) of a typical
   BEV tensor. What's the implication for memory and compute?
8. State the structural similarity between a BEV tensor and a CSI
   spectrogram. Why does this matter for your project specifically?

---

# Module 7 — Subject-independent splits as methodology

The next two modules are about *how the data reaches the model*, not
about preprocessing per se. Module 7 covers the train/val/test split
and why it's a methodology decision rather than a plumbing decision.
Module 8 covers the PyTorch glue.

Files for this module:

- `rfbc/data/splits.py`
- `scripts/build_split.py`
- `code/splits/splits.json` (the locked artifact, version-controlled)

## 7.1 What subject-independent actually means, and why it matters

A *subject-dependent* split puts samples from the same subject in
both the training set and the test set. Subject U05 contributes
some training examples (say reps 01–04 of gesture M01) and some
test examples (reps 05–08 of the same gesture). The model is
*allowed to see* each subject during training.

A *subject-independent* split puts every subject in exactly one of
{train, val, test}. If U05 is in training, none of U05's samples
appear in validation or test. The model **never sees** U05 during
training.

The difference is huge. Humans don't all move the same way:

- Their bodies have different sizes, so the spatial extent of a
  "lateral-raise" gesture varies across people.
- Their movement patterns have idiosyncrasies — some people start
  gestures from the shoulder, some from the elbow, some are
  fast-and-jerky, some slow-and-smooth.
- Their walking gaits, arm swings, posture during stationary tasks
  differ.

A subject-dependent classifier can exploit these idiosyncrasies to
recognise *who the subject is*, and then use that knowledge as a
shortcut for the gesture/activity label. "If the centre of mass is
this tall and the arms swing this fast, it's U05; and U05 only ever
labels lateral-raise as M01, so predict M01" — pure subject
memorisation. This works perfectly on the test set if the test set
contains other U05 samples, even though the model has learned almost
nothing about what M01 actually *is*.

In real deployment, a HAR system has to recognise gestures from
**new people who weren't in the training data**. So the deployment-
relevant question is exactly the one a subject-independent split
asks: "if I train on these subjects and test on those subjects, what
accuracy do I get?"

### What Varga & Cao 2025 measured

The Varga & Cao paper that the project context cites measured the
gap between subject-dependent and subject-independent accuracy on a
WiFi-CSI HAR benchmark. The headline finding: their CNN scored
~95 % subject-dependent and dropped to ~70 % subject-independent on
the same architecture and data, just by changing the split. That
25-point drop is the magnitude of "subject memorisation" present in
the subject-dependent number — most of the apparent accuracy was the
model exploiting per-subject shortcuts, not learning the task.

The dataset paper (Zuo et al. 2025) does **not** specify which split
type its 99 % baseline used. With 25 subjects and 99 % accuracy on
the Tesla model, it almost certainly was subject-dependent —
subject-independent results on point-cloud HAR are typically in the
70–85 % range. This is mentioned in `project_context_week3.md` and
matters for setting expectations: our baseline will not hit 99 %, and
that's not a bug.

For *this* project specifically the subject-independent split is
non-negotiable. The whole point of a class-wise compression study is
to measure how compression affects per-class accuracy in *the
generalisation regime* — the regime where the model is doing real
work, not memorising training subjects. Doing the study on a subject-
dependent baseline would be measuring how compression affects
memorisation, which is uninteresting and not what any of the cited
disparate-impact papers measure.

## 7.2 Why the C1 ∩ C2 intersection, not the union

```python
def make_split(campaigns, radar_root, train_frac=0.7, val_frac=0.15, seed=42):
    discovered = discover_subjects(campaigns, radar_root=radar_root)
    sets = [set(v) for v in discovered.values() if v]
    intersection = sorted(set.intersection(*sets))
```

We build the split over the **intersection** of subject sets across
campaigns, not the union. If a subject appears in C1 but not C2, they
are excluded entirely.

Why? Because of the macro/micro analysis. If we want to compare per-
class compression effects on micro-movements (C1 single-hand
gestures) vs macro-movements (C2 activities), we need *the same
subjects* contributing to both groups. Otherwise the comparison is
confounded — any difference between micro and macro accuracy could
be a difference between the C1-subject pool and the C2-subject pool,
not a difference between micro and macro intrinsically.

The intersection size is **23 subjects** for C1 ∩ C2 (we verified
this in §5 of module 1). The two outliers — U13 and U30 appear in
C2 but not C1; U36 and U42 appear in C1 but not C2 — are simply
dropped. Better 23 subjects with consistent membership than 27 with
confounded membership.

For studies that only use one campaign (e.g. a future "pure-C2
compression" ablation), one would call `make_split` with
`campaigns=("C2",)` and recover the full 25-subject pool. The
intersection logic is general — it handles any subset of campaigns
you pass.

## 7.3 The 70/15/15 split — why these fractions, why these counts

```python
n = len(shuffled)
n_train = int(round(train_frac * n))   # 70%
n_val   = int(round(val_frac * n))     # 15%
if n_train + n_val >= n:
    n_val = max(0, n - n_train - 1)
```

At 23 subjects, 70/15/15 gives 16/3/4. (`round(0.7 * 23) = 16`,
`round(0.15 * 23) = 3`, remainder 4.) The smallest fold is val with
3 subjects.

The fraction choice is informed by two considerations:

**Training fraction:** 70 % is a common middle ground. 80–90 % gives
more training data but a tinier test fold; 60 % gives a more
generous test fold but undertrained models. At 23 subjects we can't
afford too aggressive a split toward the test side, because every
test-fold subject is a coarse-grained statistic — losing one due to
a corrupted file would matter. 70 % keeps train big enough to fit a
deliberately-simple CNN.

**Val vs test:** Equal-ish (15 each) is fine. Val is used for
early-stopping and learning-rate decisions during training; test is
the final once-only number. Both need to be representative, neither
needs to be huge.

**The guard against val swallowing test:** Edge case — if rounding
gives `n_train + n_val == n`, we'd have an empty test fold. The
guard `n_val = max(0, n - n_train - 1)` ensures test has at least
one subject. With 23 subjects this branch doesn't fire, but the code
is defensive for smaller campaign subsets.

## 7.4 The "lock the split, never regenerate" rule

```python
def main() -> None:
    if SPLITS_PATH.exists() and not args.force:
        raise SystemExit(
            f"splits.json already exists at {SPLITS_PATH}. "
            "Re-running would change the split mid-experiment. "
            "Pass --force only if you really mean it."
        )
```

This is the most important methodological safeguard in the project.

**Why the rule exists.** Imagine running an experiment, getting a
class-wise accuracy of 73.4 % on M07, then later re-running
`build_split.py` (different seed by accident, or fresh project
clone), getting a different split, retraining, and getting 70.1 %
on M07. The number changed because the test set changed — but the
*paper* claims "we measured 73.4 %", which now no longer reproduces
on the artifact in the repo. This is the kind of silent failure that
destroys scientific credibility.

**The safeguard.** `scripts/build_split.py` refuses to overwrite an
existing `splits.json`. If you really want to change it (e.g. you
realised the seed produced a bad split), you have to pass `--force`
explicitly. That single keystroke acknowledges "yes, I am
deliberately changing experimental conditions, I understand all
prior results are now invalidated".

**Version control.** `splits.json` is committed to git. Anyone who
clones the repo and runs the experiment gets the *same* split, by
construction. This is one of the small but important reasons we
chose JSON over pickle or torch tensors: JSON is human-readable, a
git diff on it shows exactly which subjects moved which way, and
it's trivially portable across operating systems and Python
versions.

```json
{
  "campaigns": ["C1", "C2"],
  "subjects": {
    "train": ["U29", "U28", "U04", "U25", "U07", "U22", "U32", "U21", "U44", "U10", "U14", "U03", "U35", "U19", "U17", "U41"],
    "val":   ["U06", "U33", "U11"],
    "test":  ["U12", "U01", "U05", "U38"]
  },
  "seed": 42
}
```

Read that file and you can immediately tell which subjects are
which. If a supervisor asks "which subjects are in your test fold?"
the answer is one `cat` away, not a function call away.

## 7.5 The `Split` dataclass — immutability matters

```python
@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    campaigns: tuple[str, ...]
    seed: int
```

`frozen=True` makes the dataclass immutable: once a `Split` exists,
its fields can't be modified. Why does that matter?

In an early version, the fields might have been `list[str]`, mutable.
Some careless code could then do
```python
split.train.append("U99")
```
and silently change the split mid-run. With `frozen=True` and the
fields typed as `tuple`, Python raises an error at the attempt — and
catches it at the boundary instead of producing a wrong train/val/test
assignment.

The same immutability is why `Split` is hashable (frozen dataclasses
with hashable fields are hashable by default), which means it can be
used as a cache key or stored in a set if we ever need that.

## 7.6 The pipeline: build once, load thereafter

```
scripts/build_split.py   ----writes----> splits/splits.json
                                              |
rfbc.data.splits.load_split  <----reads---/
                                              |
                                          (loaded by dataset.py,
                                           smoke_test.py, all
                                           downstream training code)
```

This pattern — generate-once-then-load-everywhere — appears in many
ML codebases and has a name: *artifact-based reproducibility*. The
artifact (`splits.json`) is the contract; producers and consumers
are decoupled. We can refactor `make_split` freely as long as the
JSON schema stays the same.

`load_split` is the inverse of `save_split` — reads the JSON,
reconstructs a `Split` instance with tuples in the right places. It
also raises a clear error if the file doesn't exist, pointing the
user at `build_split.py`. We don't auto-generate on load, because
auto-generation would undermine the lock-the-split rule.

## 7.7 What this looks like in practice — verifying the split

Run this to inspect the split that's actually in use:

```python
from rfbc.data.splits import load_split
s = load_split()
print(f"campaigns: {s.campaigns}")
print(f"seed:      {s.seed}")
print(f"train ({len(s.train)}): {list(s.train)}")
print(f"val   ({len(s.val)}): {list(s.val)}")
print(f"test  ({len(s.test)}): {list(s.test)}")
assert set(s.train).isdisjoint(s.val), "train ∩ val should be empty!"
assert set(s.train).isdisjoint(s.test), "train ∩ test should be empty!"
assert set(s.val).isdisjoint(s.test), "val ∩ test should be empty!"
print("\nFolds are disjoint. Subject independence holds.")
```

The three assertions are the formal statement of subject
independence: no subject appears in more than one fold. If any of
these ever fails, the split file is broken and downstream results
are invalid.

## 7.8 Self-check questions

1. State the precise difference between a subject-dependent and a
   subject-independent split.
2. Why does the same model score 25 percentage points higher on a
   subject-dependent split than on a subject-independent one (per
   Varga & Cao 2025)? Explain in terms of what the model is
   actually learning.
3. Why do we use C1 ∩ C2 (intersection) and not C1 ∪ C2 (union)?
4. Why is `splits.json` in JSON format, version-controlled, and the
   `build_split.py` script refuses to overwrite it without `--force`?
5. Why is the `Split` dataclass `frozen=True`?
6. Walk through the 70/15/15 fractions applied to 23 subjects.
   Where does the `n_val = max(0, n - n_train - 1)` guard come into
   play?
7. The Varga & Cao paper found a ~25-point gap between subject-
   dependent and subject-independent accuracy. What does this tell
   you to expect about our baseline accuracy vs the dataset paper's
   99 % number?

---

# Module 8 — PyTorch Dataset, caching, and the training loop

The final module ties the preprocessing into PyTorch. After this,
batches of `(486, 64, 64)` tensors flow out of the dataset, through a
DataLoader, into a model, and the project is "doing machine learning"
in the conventional sense. We walk three files:

- `rfbc/data/dataset.py`
- `rfbc/models/stub_cnn.py`
- `scripts/smoke_test.py`

## 8.1 What a PyTorch `Dataset` is and what we make ours do

A `torch.utils.data.Dataset` is an abstraction that says: "I have N
items, I know how to give you item `i`". You implement two methods:

- `__len__` — total number of items
- `__getitem__(i)` — return `(input_tensor, label)` for index `i`

PyTorch's `DataLoader` then handles batching, shuffling, parallel
loading via worker processes, GPU prefetching, and the rest of the
boilerplate. Our job is just to define what "item `i`" means and how
to materialise it.

The natural unit of "item" for HAR is **a window**, not a sample.
Recall from Module 5 that one C2 walking sample produces ~10
overlapping windows; each window is one CNN input. So our dataset
indexes by `(sample, window_idx)`, not by sample alone.

```python
@dataclass(frozen=True)
class WindowIndex:
    sample: SampleId
    window_idx: int
    label: int
```

`label` is an integer class id assigned at dataset construction time
by looking up the class folder name in a `label_map`. We re-label
because the disk's `M01–M21` strings aren't directly usable as
CrossEntropyLoss targets — those need contiguous integers `[0, K)`.

## 8.2 Index construction — fast scan, no preprocessing

```python
for campaign in split.campaigns:
    for subject in subjects:
        subj_dir = self.radar_root / campaign / subject
        if not subj_dir.is_dir():
            continue
        for class_dir in sorted(subj_dir.iterdir()):
            ...
            for rep_dir in sorted(class_dir.iterdir()):
                ...
                sample = SampleId(...)
                self.index.append(WindowIndex(sample=sample, window_idx=0, label=label))
```

`__init__` walks the disk **without** running any preprocessing.
It enumerates `(campaign, subject, class, rep)` directories and
appends one placeholder `WindowIndex(window_idx=0)` per sample.

The trick: at this point we don't know how many windows each sample
will produce (depends on its duration after clutter suppression).
We could call `_compute_windows` per sample during `__init__` to
find out, but that's wasteful — preprocessing all 8000+ samples up
front would take minutes and consume memory whether or not we end
up training on them.

So we lie a little. We register one entry per sample initially. If a
sample turns out to have more than one window when first accessed,
we lazily expand the index (see §8.4). For C1 (single-window) the
lie is no lie at all; for C2 (multi-window) the index grows on first
epoch.

The benefit is fast startup. A `SubjectIndependentRadarDataset` with
all subjects and all classes is ready in milliseconds — the disk
walk is the only cost.

## 8.3 The cache-or-compute pattern

```python
def _load_or_compute_windows(self, sample: SampleId) -> list[np.ndarray]:
    if self.cfg.use_cache:
        cache_path = self._cache_path(sample)
        if cache_path.exists():
            with np.load(cache_path) as f:
                return [f[k] for k in sorted(f.files, key=lambda s: int(s.split("_")[1]))]
    windows = self._compute_windows(sample)
    if self.cfg.use_cache:
        cache_path = self._cache_path(sample)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{f"win_{i}": w for i, w in enumerate(windows)},
        )
    return windows
```

The first time a sample is accessed, the entire preprocessing
pipeline runs: load `.pkl`s, transform, suppress clutter, window,
BEV. That's the expensive part — anywhere from 100 ms to a few
seconds depending on sample size.

The result is written to `cache_path` as a compressed `.npz` file.
The cache path encodes everything: `code/cache/<config_hash>/<campaign>/<subject>/<class>/<repetition>.npz`.

The second time the same sample is accessed (later epoch, or
different run), the cache hits. `np.load` reads the compressed `.npz`
in milliseconds. Total speedup: roughly 100× on a cache-hot dataset.

This pattern matters enormously for the eventual compression sweep.
Once the cache is warm, you can train dozens of differently-pruned
models in the time it would otherwise take to preprocess the data
twice.

### Config-hash cache invalidation

```python
def _config_hash(cfg: PipelineConfig) -> str:
    keys = ("sensor_set", "dbscan_eps", "dbscan_min_samples", "fps",
            "window_seconds", "window_stride_seconds", "target_frames",
            "grid_size", "grid_extent_m", "z_bands", "feat_per_cell")
    payload = {k: getattr(cfg, k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
```

The cache is keyed by a hash of the pipeline-affecting config
fields. So `code/cache/a3b9c2f10d/...` is the cache for one
specific configuration; if you change `window_seconds` from 2 to 3
the hash changes to something like `code/cache/f0e1d2c3b4/...` and
the old cached tensors are *ignored* (still on disk, but a different
directory). The new config will rebuild the cache on first access.

This is what makes ablations safe. Want to try `grid_size=128`? Just
change the config, run the smoke test, and the new BEVs are
generated and cached separately. The old 64-grid cache stays intact
and is available if you flip back. No risk of stale cache
contaminating a result.

What's deliberately **not** in the hash:

- `use_cache` itself — turning caching off shouldn't invalidate the
  cache.
- `cache_dir` — moving the cache directory shouldn't invalidate it
  either.
- `campaigns` — which campaigns to load is a dataset-construction
  choice, not a preprocessing choice.

If you add a new config field that *does* affect preprocessing,
remember to add its name to the `keys` tuple. Otherwise the cache
will lie about being current.

## 8.4 The two-pass index expansion

```python
def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    entry = self.index[idx]
    windows = self._load_or_compute_windows(entry.sample)
    if entry.window_idx >= len(windows):
        if not self._expanded:
            self._expand_index()
            return self.__getitem__(idx)
        raise IndexError(...)
    tensor = torch.from_numpy(windows[entry.window_idx]).float()
    return tensor, entry.label
```

The placeholder `window_idx=0` entries from §8.2 hide the fact that
some samples have more windows. When `__getitem__` is called on a
multi-window sample, it loads all windows, sees that `window_idx=0`
maps to a valid window, returns it — fine for the first window of
each sample.

But the *other* windows of those samples never have their own
indices and are never sampled. To fix this, the first time
`__getitem__` finds a sample whose loaded window count exceeds its
registered count, it triggers `_expand_index`:

```python
def _expand_index(self) -> None:
    new_index: list[WindowIndex] = []
    for entry in self.index:
        if entry.window_idx != 0:
            new_index.append(entry)
            continue
        windows = self._load_or_compute_windows(entry.sample)
        for w in range(len(windows)):
            new_index.append(WindowIndex(
                sample=entry.sample,
                window_idx=w,
                label=entry.label,
            ))
    self.index = new_index
    self._expanded = True
```

`_expand_index` walks every sample (caching them all in the
process), and replaces each `(sample, 0)` entry with one entry per
window. After this the index reflects the true number of items, and
`__len__` is correct.

This two-pass design is a deliberate trade between startup time
and correctness. The first pass is fast (disk walk only). The
second is exhaustive (every sample preprocessed) but cached on
disk, so it only runs once across the project's lifetime.

A subtle point: when `_expand_index` runs from inside `__getitem__`,
it modifies `self.index` *while* the parent DataLoader is iterating
through indices it cached at the start of the epoch. The iteration
itself doesn't break because the DataLoader holds onto its own index
list, but the *next* epoch will pick up the expanded index and see
the correct count. There's no race condition because everything
runs in the main thread until you set `num_workers > 0`.

## 8.5 `_compute_windows` — the pipeline orchestration

```python
def _compute_windows(self, sample: SampleId) -> list[np.ndarray]:
    sensor_indices = self.cfg.selected_sensor_indices()
    sensor_frames = load_sample_sensors(
        sample, sensor_indices, root=self.radar_root, skip_missing=True,
    )
    sensor_frames = transform_sample(sensor_frames)
    sensor_frames = suppress_clutter(
        sensor_frames,
        eps=self.cfg.dbscan_eps,
        min_samples=self.cfg.dbscan_min_samples,
    )
    stride = max(1, int(round(self.cfg.window_stride_seconds * self.cfg.fps)))
    time_windows = split_into_windows(
        sensor_frames,
        target_frames=self.cfg.target_frames,
        fps=self.cfg.fps,
        stride_frames=stride,
    )
    return [points_to_bev(w, self.cfg) for w in time_windows]
```

This is the entire pipeline, top to bottom, in one function. Every
stage we walked in Modules 3–6 fires in sequence:

1. `load_sample_sensors` (Module 3)
2. `transform_sample` (Module 3)
3. `suppress_clutter` (Module 4)
4. `split_into_windows` (Module 5)
5. `points_to_bev` per window (Module 6)

The result is a list of BEV tensors, one per window. This is the
unit that gets cached, the unit that becomes a DataLoader item.

Reading this function and being able to recite "load, transform,
clutter, window, BEV" is a good signal that the whole pipeline
makes sense to you.

## 8.6 The stub CNN — just enough model to verify plumbing

```python
class StubCNN(nn.Module):
    def __init__(self, in_channels, num_classes, base_width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_width * 4, num_classes)
```

Three Conv–BN–ReLU blocks with `MaxPool2d(2)` between them, then a
global pool and a linear classifier. This is a stripped-down version
of a 1990s LeNet architecture — the simplest CNN that exhibits all
the structure (conv layers, BatchNorm, activations, pooling, final
classifier) that the real baseline will share.

It is **not** the project baseline. It exists for two reasons:

1. To verify that `(486, 64, 64)` tensors flow through a CNN
   correctly (no shape mismatches, no NaNs, no OOM).
2. To exercise the parts of PyTorch that the compression tools
   target — Conv2d, BatchNorm2d, Linear are *exactly* the modules
   that `torch.nn.utils.prune` and `torch.ao.quantization` know how
   to handle. If pruning works here, it'll work on the real model.

The real baseline (week 4 work) will be deeper and wider, but it
will have the same component structure. The week 4 model swap is a
one-file replacement: replace `StubCNN` with a `BaselineCNN` and
nothing else in the pipeline changes.

### Architectural details worth noting

- **`bias=False` in Conv2d**: when followed by BatchNorm2d, the bias
  is redundant (BN's own affine bias absorbs it). Setting
  `bias=False` saves parameters with no loss of expressiveness. This
  is a standard convention.
- **`AdaptiveAvgPool2d(1)`** instead of a fixed final pool size:
  collapses any spatial extent to 1×1, so the linear classifier
  size is invariant to input resolution. If we ever change
  `grid_size` from 64 to 128, the model still works.
- **No dropout**: the stub model is too small to overfit dramatically
  on a 1-epoch smoke test. The real baseline may add dropout once
  proper training cycles begin.

## 8.7 `smoke_test.py` — line by line

The full test is short — ~100 lines — and exercises every component
we've built. Let's trace the key parts.

```python
split = load_split()
trimmed = type(split)(
    train=split.train[: args.max_subjects_per_fold],
    val=split.val[: max(1, args.max_subjects_per_fold)],
    test=split.test[: max(1, args.max_subjects_per_fold)],
    campaigns=split.campaigns,
    seed=split.seed,
)
```

Load the locked split, then trim to a tiny fraction of subjects.
For a smoke test we don't want to wait for all 23 subjects to
preprocess; 2 per fold is enough to confirm correctness. The trim
is conservative — `max(1, ...)` ensures val and test each have at
least one subject, even if `max_subjects_per_fold=0`.

```python
train_ds = SubjectIndependentRadarDataset(
    trimmed, fold="train", cfg=cfg,
    class_filter=tuple(args.classes),
    repetition_filter=tuple(args.repetitions),
)
val_ds = SubjectIndependentRadarDataset(trimmed, fold="val", cfg=cfg, ...)
```

Two datasets, one per fold. `class_filter` and `repetition_filter`
restrict to a tiny subset — by default M01, M02, M03 with rep 01
only. That's enough samples for a smoke test, few enough to finish
in seconds when the cache is cold.

```python
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                         shuffle=True, num_workers=args.num_workers, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, drop_last=False)
```

Standard PyTorch DataLoader setup. `shuffle=True` for training,
`shuffle=False` for validation. `drop_last=False` keeps partial
batches at the end (matters because the smoke test has few items).
`num_workers=0` runs everything in the main process — fine for the
smoke test; for full training you'd raise this.

```python
x, y = next(iter(train_loader))
print(f"sample batch: x={tuple(x.shape)}, dtype={x.dtype}, y={tuple(y.shape)}")
```

The single most important line. We peek at one batch *before*
training. If the shape isn't `(batch, 486, 64, 64)` or the dtype
isn't float32, we've broken something upstream and the rest of
the test is meaningless.

```python
in_channels = x.shape[1]
num_classes = max(train_ds.num_classes, val_ds.num_classes)
model = StubCNN(in_channels=in_channels, num_classes=num_classes).to(args.device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
```

Build the model with `in_channels` derived from the actual batch
(486 in default config), Adam optimiser (sensible default for new
projects), CrossEntropyLoss (multi-class classification).

```python
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
```

The training loop. Every line here is textbook PyTorch:

- `model.train()` enables training mode (matters for BatchNorm
  stats and dropout).
- `xb.to(device, non_blocking=True)` moves data to GPU
  asynchronously when possible.
- `optim.zero_grad()` clears accumulated gradients from the
  previous step.
- `loss.backward()` runs autograd through the model.
- `optim.step()` updates weights.
- `running` accumulates loss × batch size; dividing by
  `len(train_ds)` at the end gives the mean per-item loss.

Nothing exotic; nothing project-specific. This loop is the same as
in every PyTorch tutorial. The point of putting it in the smoke
test is to verify the *whole stack* runs end-to-end without errors.

```python
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
```

Per-class validation accuracy. `model.eval()` switches BatchNorm
to running-stats mode; `torch.no_grad()` disables autograd to save
memory and time. We tally correct and total *per class*, which is
the most important diagnostic for this project — single-number
accuracy hides exactly the class-wise patterns we care about.

The per-class output, even on a tiny smoke test, is foreshadowing
the real evaluation: when we get to compression sweeps, every
result will be per-class, and this loop is the prototype.

## 8.8 What the smoke test verifies — checklist

When `python -m scripts.smoke_test` finishes with "Smoke test
complete." printed, you have confirmed:

1. Disk paths resolve correctly.
2. `.pkl` files can be loaded.
3. `transform_sample` produces world-frame points.
4. `suppress_clutter` survives DBSCAN on real data.
5. `split_into_windows` produces correctly-sized windows.
6. `points_to_bev` produces tensors of the expected shape.
7. The on-disk cache writes and reads correctly.
8. `SubjectIndependentRadarDataset.__getitem__` returns valid
   tensors of the right dtype.
9. `DataLoader` can batch them.
10. `StubCNN` accepts the batch and produces logits.
11. `CrossEntropyLoss` accepts the logits and computes a loss.
12. `optim.step()` updates weights.
13. Validation runs without errors.
14. Per-class accuracy is reported.

That's the entire stack. Any error during a smoke run pinpoints
exactly which of these steps broke, because the test runs them in
order.

## 8.9 What's left for week 4 and beyond

The smoke test verifies plumbing. What it doesn't do, and what comes
next:

- Train a real model on a real-sized dataset.
- Choose the actual CNN architecture (not the stub).
- Pick optimiser + learning rate schedule + epochs by validation
  performance.
- Implement the pruning sweep (`torch.nn.utils.prune` over different
  sparsity levels).
- Implement the quantization sweep (`torch.ao.quantization` over
  different bit-widths).
- Implement Hooker-style CIE metrics and per-class accuracy
  evaluations under each compression setting.
- Write up the macro/micro grouping analysis.

All of these are isolated changes on top of the existing pipeline.
The preprocessing, dataset, cache, and training loop don't need to
change — they're done.

## 8.10 Self-check questions

1. Why does the dataset index by `(sample, window_idx)` rather than
   by sample alone?
2. Walk through the cache-or-compute pattern. What happens on the
   first access to a sample? On the second access?
3. Explain config-hash cache invalidation. Why is `use_cache` itself
   excluded from the hash, but `grid_size` included?
4. Why is the index "two-pass" — initial placeholders + later
   expansion? What does this trade off?
5. Name three reasons `StubCNN` is intentionally tiny.
6. In `__init__` of `StubCNN`, `Conv2d` is constructed with
   `bias=False`. Why?
7. Walk through one full step of the training loop. Match each line
   to a PyTorch operation (forward, backward, optimiser, etc.).
8. The smoke test reports per-class accuracy on val. Why is this
   already worth tracking on a 1-epoch run with a stub model, even
   though the numbers themselves are meaningless?

---

# Wrap-up

This is the end of the eight-module walkthrough. You now have a
file-by-file, decision-by-decision understanding of the preprocessing
pipeline that supports the rest of the project.

The thing that all eight modules have in common is that **every code
choice is grounded in either a property of the RF-Behavior dataset or
a methodological commitment of the project**. There is no "we did it
this way because it's how it's usually done" in the codebase — every
default in `config.py`, every loop in `bev.py`, every condition in
`splits.py` traces back to something concrete.

That property is what makes the project defensible at supervisor
check-ins. When someone asks "why did you do X?", the answer is
always either "because the dataset has property Y" or "because our
research question requires Z" — not "because that seemed reasonable".

The next module, conceptually, is one we haven't written yet:
**week 4 work**. Choose the real baseline CNN, train it properly on
the full subject pool, hit 70–80 % subject-independent accuracy,
then layer compression on top. The pipeline you understand now will
support all of that without modification.

