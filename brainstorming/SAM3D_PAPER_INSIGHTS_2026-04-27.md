# SAM 3D Paper Insights - 2026-04-27

Source: arXiv 2511.16624 — "SAM 3D: 3Dfy Anything in Images"
PDF stored at: sam3d.pdf

These are insights from reading the paper that are directly relevant to how we
should advance the metric scale recovery work.

---

## Architecture Recap (for reference)

| Paper name | Codebase name | Params | Output |
|---|---|---|---|
| Geometry Model | `ss_generator` | 1.2B, Mixture-of-Transformers | coarse shape O ∈ R⁶⁴ + layout (R, t, **s**) |
| Texture & Refinement | `slat_generator` | 600M, sparse latent flow transformer | refined geometry + texture |

Input encoding: DINOv2 on 4 token streams — cropped object + binary mask, full
image + full binary mask. Optional conditioning on a MoGe or LiDAR pointmap.

---

## Insight 1: The SS Generator Already Predicts Explicit Scale s ∈ R³ — We Are Not Using It

**What the paper says:**

> "The Geometry Model models the conditional distribution p(O, R, t, s | I, M),
> where O ∈ R⁶⁴ is coarse shape, R ∈ R⁶ the 6D rotation, t ∈ R³ the translation,
> and s ∈ R³ the scale."

The SS generator was trained to explicitly output a 3D scale vector s as part of
its flow-matching target, alongside rotation and translation. This is not a latent
side-effect — it is a first-class output of Stage 1.

**Current approach and the gap:**

Our MetricScaleHead takes `mean_pool(SS_latent [B, N, 8]) → [B, 8]` as its shape
signal. This is an opaque average over the entire SS representation. If the SS
generator explicitly outputs (R, t, s) as part of its denoising trajectory, then `s`
should be readable directly — and would be a far stronger, cleaner metric signal
than a pooled latent.

**Action item:**

Investigate the SS generator's output interface in the codebase. Check whether the
predicted `(R, t, s)` values are accessible after `sample_sparse_structure()` and
whether `s` can be fed directly into MetricScaleHead as a replacement for or
supplement to the mean-pooled latent. This could fundamentally improve the
MetricScaleHead architecture.

The scale s here is in normalized camera coordinates (relative to the pointmap),
so combining it with `log(pointmap_scale)` to get absolute metric WHD remains
the correct conversion step.

---

## Insight 2: Full-Image Context Is Missing from MetricScaleHead — and the Paper Says It Is Critical for Scale

**What the paper says:**

SAM 3D encodes both the cropped object and the full image:

> "Full image: We encode the full image I and its full image binary mask, providing
> global scene context and recognition cues absent from the cropped view."

Scale relative to scene context is one of the strongest pictorial cues for absolute
object size. An annotated bottle that fills half the frame is a different metric
size than one occupying a corner. The paper explicitly designs around this by
providing the model with both views.

**Current approach and the gap:**

MetricScaleHead currently receives:
- Mean-pooled SS latent `[B, 8]` — from the cropped object path
- `log(pointmap_scale)` `[B, 1]` — global from MoGe
- `pointmap_shift_z` `[B, 1]` — global from MoGe

There is no full-image feature. The cropped object path tells the model what
the shape looks like but not how large the object is relative to the scene.

**Action item:**

Add a DINOv2 full-image embedding (or a pooled MoGe global feature) as an
additional input to MetricScaleHead. The full-image tokens are already computed
by the pipeline for conditioning — they can be pooled and concatenated to the
MetricScaleHead input without additional forward passes. This is architecturally
well-motivated by the paper and expected to improve absolute scale prediction.

---

## Insight 3: DPO Alignment Biases the Model Toward Aesthetics, Not Metric Accuracy

**What the paper says:**

Post-training stages include SFT on Art-3DO (professional artist meshes, evaluated
for symmetry, closure, no floaters) followed by DPO on human preference pairs
(D+/D−). Annotators rate shape quality by aesthetic rubric:

> "DPO training... using D+/D− pairs from Stage 2 of our data engine. We found
> this off-policy data was effective at eliminating undesirable model outputs."

Undesirable outputs include floaters, bottomless meshes, missing symmetry — not
dimensionally incorrect aspect ratios.

**Implication for metric work:**

The SLAT features and canonical mesh reflect human-preferred shape quality, not
metric ground truth. This explains:

- The canonical aspect ratio error we measured (mean 0.05–0.15 per axis) —
  the model produces "nice" symmetric shapes, not dimensionally faithful ones
- Why SLAT conditioning on the metric token may have limited returns without also
  addressing the aesthetic bias in the SLAT denoiser
- Why mean-pooled SLAT features are a somewhat noisy basis for metric readout

**Longer-term implication:**

Genuine metric accuracy would require metric loss to be part of the SFT/DPO
objectives during Stage 1/2 training, not just at fine-tuning time. That is a
much larger undertaking (requires labelled real images with ground truth WHD and
re-running the full post-training pipeline), but the paper makes clear that
is where precision would come from. Our current approach (adding a metric head on
top of a frozen aesthetics-aligned model) is the right first step and is consistent
with what the paper shows is feasible with post-training, but it has a ceiling
set by the aesthetic bias baked into SLAT.

---

## Insight 4: Training Data Already Has Metric Scale in Stage 1 — via Pointmap-Relative Annotation

**What the paper says:**

Stage 3 of the data engine has annotators label object pose by:

> "manipulating the 3D object's translation, rotation, and scale relative to a
> point cloud. We find that point clouds provide enough structure to enable
> consistent shape placement and orientation."

This means the SS generator's `s` output was trained with implicit metric
scale supervision (objects placed correctly relative to a metric-scaled point
cloud). The scale information is already in the model — just expressed in
normalized camera coordinates, not absolute metres.

**Implication:**

The SS generator's explicit `s` is not a random latent feature — it was
directly supervised against real-world scaled point clouds. Accessing it
directly (Insight 1) is therefore even more motivated: it is the most
direct path to absolute scale information already present in the model,
and converting it to metres via the MoGe pointmap scale should be a
relatively clean operation.

---

## Insight 5: Best-of-N Sampling Could Improve Metric Consistency at Inference — for Free

**What the paper says:**

The data engine uses N=8 candidates, filtered first by a model, then by human
annotators (a form of best-of-N search). The paper shows near-linear Elo
improvement as N increases (Section A.7).

**Application to metric inference:**

At inference time, sampling K SLAT outputs and selecting the one whose canonical
bounding box aspect ratio best matches the predicted WHD aspect ratio requires no
additional training. The selection criterion could be:

```
best = argmin_k  ||[bx_k/by_k, bx_k/bz_k] - [W_pred/H_pred, W_pred/D_pred]||
```

This is purely a post-processing step on top of the existing pipeline. It would
not improve the metric prediction itself, but would improve geometric consistency
between the predicted shape and the predicted dimensions. Worth experimenting with
once the metric head is trained and validated.

---

## Priority Order

| Insight | Implementation effort | Expected impact |
|---|---|---|
| Read `s` directly from SS generator output | Low (investigation first) | High — may replace mean-pooled latent |
| Add full-image DINOv2 to MetricScaleHead inputs | Medium | Medium-high — scale-from-context is a strong cue |
| Best-of-N metric-consistent sampling at inference | Low | Low-medium — free, no training needed |
| Metric objective in DPO/SFT stages (Stage 1/2) | Very high | High — the real fix, long-term |

**Immediate next investigation:** check whether `sample_sparse_structure()` exposes
the predicted `(R, t, s)` values or only the coarse voxel output O.

---

## What the Paper Does Not Address (Our Contribution)

The paper has no evaluation of absolute metric accuracy in metres — all shape
metrics are geometry quality (Chamfer, F1, vIoU) and all layout metrics are
pose accuracy (ADD-S, 3D IoU, rotation error). The question "does the predicted
object have the correct real-world dimensions?" is entirely absent from the
paper's evaluation.

This is the gap our work fills. Converting the implicit pointmap-relative scale
in Stage 1 into explicit, supervised, absolute WHD prediction in metres is a
novel contribution on top of this foundation model.
