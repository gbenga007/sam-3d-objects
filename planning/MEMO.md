# SAM 3D Objects — Project Memo
## Physical Accuracy: Metric Scale Recovery & Geometric Fidelity

---

## The Problem

SAM 3D Objects generates plausible 3D reconstructions but has two fundamental limitations
for physical accuracy:

1. **Non-metric output.** The model operates entirely in canonical space — a normalized
   unit cube `[-0.5, 0.5]³`. A 5cm mug and a 5m car produce latent codes of similar scale.
   There is no mechanism that grounds the output to real-world physical units.

2. **Generative prior overrides conditioning.** The model is a flow matching generative
   model trained to produce plausible 3D shapes. It does not have a reconstruction loss
   that forces it to match the conditioning input. Fine details absent from the training
   prior (a 2mm crack, an unusual handle shape) will be smoothed out or hallucinated.

These are separate problems requiring separate solutions.

---

## Key Architectural Insights (from codebase analysis)

### The Scale Break
MoGe (Microsoft) produces **metric depth in meters**. That metric information is
immediately normalized into `pointmap_scale` and `pointmap_shift` statistics
(`inference_pipeline_pointmap.py:306-311`). These statistics are computed but never
passed to either the SS or SLAT generators. Both generators only receive `["image"]`
as conditioning (`inference_pipeline.py:77`). Scale is recovered post-hoc by the pose
decoder, but only in scale-shift-invariant (SSI) space — a dimensionless ratio, not meters.

**The metric information exists in the pipeline. It is simply not used.**

### The Conditioning Architecture
The SLAT generator uses cross-attention to consume conditioning signals. The condition
embedder (`EmbedderFuser`) accepts a sequence of tokens of shape `[batch, seq_len, 768]`
and concatenates them. Adding a new conditioning signal is therefore a matter of:
(a) producing a token of the right shape, and (b) adding it to the sequence.
No surgery on the transformer architecture is required.

### The SS Latent
The SS (Sparse Structure) generator outputs `shape_latent` of shape `[batch, 4096, 8]`
— 4096 spatial positions each with an 8-dimensional feature vector. This encodes 3D
object shape in canonical space. It captures proportions and topology but NOT absolute
metric scale (because it was trained without metric supervision).

**Important:** The SS latent alone cannot predict metric scale. It needs a metric anchor —
which comes from MoGe's `pointmap_scale` and `pointmap_shift`.

---

## The Solution Architecture (Phase 1 — Metric Scale)

The boss's proposal: connect the scale from the first stage (SS geometry model) into
the second stage (SLAT texture/refinement model) as a conditioning signal, and add a
metric scale decoder in the second stage. Fine-tune only the new components on metric data.

### Why this is the right approach
- **Minimal intervention.** The generative prior is preserved. Only new lightweight
  components are trained. This avoids catastrophic forgetting.
- **Correct information flow.** The SS latent has the best geometric representation of
  the object. The SLAT stage has the most complete information (image + geometry + scale).
  Placing the metric decoder in SLAT uses maximum context.
- **Tractable fine-tuning.** Freezing the backbone means we only need a dataset that
  provides metric dimensions — not a dataset that supports full 3D generation training.

### Component 1: MetricScaleHead
Predicts log(metric_scale) from the SS latent + MoGe statistics.

```
pooled SS latent [batch, 8]          ← shape structure (proportions)
+ log(pointmap_scale) [batch, 1]     ← metric anchor from MoGe
+ pointmap_shift_z    [batch, 1]     ← scene depth context
→ MLP (10 → 64 → 32 → 1)
→ log(metric_scale)                  ← a single scalar: "how big is this object"
```

The SS latent contributes object proportions; the pointmap statistics contribute the
metric anchor. Neither alone is sufficient.

### Component 2: ScaleTokenProjector
Bridges from the scalar scale value to SLAT's conditioning space.

```
log(metric_scale) [batch, 1]
→ nn.Linear(1, 768)
→ scale token [batch, 1, 768]        ← one new token appended to DINO tokens
```

SLAT's cross-attention operates on dim=768 tokens. The scalar must be projected.
This is a learned projection — during fine-tuning, the model learns how to use
the scale signal in the context of 3D generation.

### Component 3: MetricScaleDecoder
Predicts physical dimensions from the refined SLAT latent.

```
pooled SLAT latent + scale token context
→ MLP
→ [width, height, depth] in meters   ← per-axis metric prediction
```

This is the final output that directly answers "how big is this object in physical units."
Supervised with ground truth dimensions from Objectron and NOCS REAL275 during fine-tuning.

---

## Dataset Strategy

### Primary: Objectron (Google)
- 4M images, 9 categories, metric 3D bounding boxes from AR tracking
- Real-world captures, good image quality, large scale
- License: C-UDA 1.0 (permits ML training)

### Secondary: NOCS REAL275
- 3,200 frames, 6 categories (incl. bottle, mug — everyday objects)
- Explicit metric size in camera coordinates + masks already provided
- Smaller but very clean annotations

### Avoided: Pix3D (incorrect scale), CO3D (no metric annotations), ShapeNet (synthetic, non-commercial)

### Validation: WildRGB-D (CVPR 2024)
- 8,500 objects, 46 categories — more diverse than training set
- Use only for out-of-distribution evaluation, not training

### Key preprocessing step
MoGe must be run on every training image to extract `pointmap_scale` and `pointmap_shift`.
These are the metric anchor that the ScaleHead depends on. A correlation check between
MoGe's recovered scale and ground truth metric scale should be done before training
to validate the assumption.

---

## Fine-Tuning Strategy

**Frozen:** All existing SS model, SLAT model, condition embedders, all decoders.
**Trained:** MetricScaleHead, ScaleTokenProjector, MetricScaleDecoder, and the new
cross-attention K/V projection in SLAT for the scale token.

**Loss:**
- Smooth L1 on `log(predicted_scale)` vs `log(GT_scale)` — log-space reduces scale
  sensitivity (a 10% error on a small object is treated the same as on a large one)
- Smooth L1 on per-axis `[width, height, depth]` predictions

**Why smooth L1 and not MSE?** More robust to outliers in the dataset (occasional wrong
scale annotations won't dominate training).

---

## Phase 2 — Geometric Fidelity (Future)

Metric scale recovery is solvable at inference time with the above architecture.
Geometric fidelity (reproducing a 2mm crack, exact surface texture) is a harder problem
because it requires changing the generative model itself, not just adding decoders.

The core issue: the flow matching loss only supervises the velocity field. There is no
loss term that penalizes deviation from the conditioning image. The model learns a prior
that generates plausible objects, not the specific input object.

**Likely approaches to explore:**
- Perceptual loss: render output at training time → compare to input image
- Depth consistency loss: MoGe(rendered output) should match MoGe(input)
- Higher-resolution latent representations for finer geometric detail
- Dataset: requires high-resolution 3D scans of real objects (OmniObject3D, custom captures)

This is a training change requiring retraining or heavy fine-tuning of the generative
backbone — significantly more compute and data than Phase 1.

---

## Milestones & Progress

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1 — Datasets downloaded and preprocessed | **Complete** | NOCS-Real275 via OmniNOCS; Objectron optional/pending |
| M2 — Architecture implemented, pipeline runs | **Complete** | MetricScaleHead (13-dim), MetricScaleDecoder, SLAT injection |
| M3 — Scale head overfits on 10 samples | **Complete** | Cached overfit 2.61% MAPE |
| M4 — Fine-tuning converges, MAPE < 15% on val | **Complete** | baseline_v2: **2.997%** MAPE (ep200, scene-heldout) |
| M4b — SLAT-conditioned training | **Complete** | v3: **1.74% MAPE** ep5; v2: 1.93% ep2 |
| M4c — SS decoder + ratio loss | **In progress** | ss_ratio_v1 ep6: **1.23% MAPE** (new best, beats 1.51% baseline by 28%); ep8/10 in progress |
| M4d — Mixed dataset (Objectron + ARKitScenes) | **In progress** | mixed_v1 ep2/6 best; ep3 regressed (Obj 35→90%), ep2 is paper checkpoint |
| M5 — Metric depth eval pipeline (HAMMER/iBims/DIODE) | **In progress** | Pipeline working 2026-05-27; HAMMER eval running (~454 samples). MoGe baseline adapter written. |
| M6 — OOD evaluation on WildRGB-D | Not started | |
| M7 — Phase 2 geometric fidelity begins | Not started | |

---

## ⚠️ Checkpoint Bug — Discovered 2026-05-21

**Problem:** Any training run with `--unfreeze-ss-decoder` fine-tunes the SS decoder, but the
checkpoint save code did not include the SS decoder state dict. The MetricScaleHead is trained
on fine-tuned SS features; at inference the original SS decoder is loaded → distribution mismatch.

**Impact:**
- `nocs_sceneholdout_ss_ratio_v1_best.pt`: **training-eval MAPE 1.23%** is NOT reproducible at
  inference. Full-pipeline notebook eval shows **38.29% MAPE**. SS decoder weights permanently lost.
- `mixed_v1` epoch-1 checkpoint will also be broken (process loaded old code, can't be hot-patched).
  Epochs 2–6 will be correct after process restart with fixed code.

**Fix applied:** `finetune_metric_scale.py` — `save_metric_checkpoint` and `load_metric_checkpoint`
now accept `ss_decoder=` param. All 4 save call sites and 1 load call site updated to pass
`pipeline.models.get("ss_decoder") if args.unfreeze_ss_decoder else None`.

**For the paper:** The honest headline number from a properly-saved checkpoint is TBD.
`slat_conditioned_v3_best.pt` (no SS decoder fine-tuning) is valid: 5.78% full-pipeline MAPE.
Need to retrain ss_ratio_v1 with fixed code to get the true end-to-end number.

---

## Current Best Result (updated 2026-05-27)

**NOCS-only best (broken):** `nocs_sceneholdout_ss_ratio_v1_best.pt` — epoch 6/10, **1.23% MAPE (training)**
→ 38.29% full-pipeline MAPE due to checkpoint bug (SS decoder not saved). **Unusable for paper.**

**NOCS-only best (valid):** `nocs_sceneholdout_slat_conditioned_v3_best.pt` — epoch 5/5, **5.78% MAPE full-pipeline**

**Mixed-data paper checkpoint:** `mixed_v1_best.pt` — epoch 2/6
- NOCS: 4.72%, Objectron: 35.38%, ARKitScenes: 40.18%, overall: 34.89%

| Run | Epochs | Best MAPE | Per-category highlights |
|---|---|---|---|
| baseline_v2 (frozen SLAT) | 200 | 0.64% | Long training; no SLAT injection |
| slat_conditioned_v2 | 2 | 1.93% | First stable SLAT injection |
| slat_conditioned_v3 | 5 | **5.78% (full-pipeline)** | 1.74% training eval; valid for paper |
| **mixed_v1** (paper) | ep2 best | **34.89% overall (full)** | NOCS 4.72%, Obj 35.38%, ARKit 40.18% |

**Active: mixed_v1 training** — epoch 4/6 running on training pod (A100 24GB, separate from this eval pod).

**Paper writing started 2026-05-19** — 4 of 6 sections drafted at `gbenga007/eccv-paper-vigir`.
Target venue: MUSTCV workshop at ECCV 2026. Abstract pending mixed_v1 final epoch results.

---

## Metric Depth Benchmark Eval (started 2026-05-27)

**Strategy:** Use MoGe pointmap + our MetricScaleDecoder scale prediction.
- MoGe provides affine-invariant pointmap (correct local shape, unknown scale+shift)
- Our decoder predicts `s_iso = max(W,H,D)` in metres
- We rescale MoGe pointmap by `s_iso/moge_extent` (object-centroid-preserving)
- Evaluated on: HAMMER (454 samples), iBims-1 (759), DIODE (2558)

**Key finding (smoke test):** On HAMMER large objects (40%+ of image = furniture-scale):
- `depth_metric` rel~0.96 (bad — our scale head predicts ~0.16m for furniture)
- `depth_scale_invariant` rel~0.10 (good — MoGe shape quality)
- `local_points` rel~0.16 (good — local geometry quality)

**Status:**
- HAMMER full eval running: `artifacts/eval_depth/HAMMER_sam3d_v3_best.jsonl` (~90min remaining)
- MoGe baseline adapter: `scripts/moge_baseline.py` (written, not yet run)
- Mixed_v1_best eval: queued after HAMMER completes
- iBims-1, DIODE: queued after HAMMER completes

---

## Open Questions

1. ~~**MoGe correlation check**~~ — **RESOLVED.** Pearson r=0.708 log-log on 1,177 NOCS instances.
   MoGe v1 is affine-invariant (not metric); `pointmap_scale` is the pipeline's alignment factor,
   which correlates with metric extent. MoGe-2 (arXiv 2507.02546) adds explicit metric prediction.

2. **Scale token impact on generation quality:** Not yet evaluated. Mixed_v1 epoch-1 will give
   the first signal (per-source MAPE); qualitative figures will be needed before submission.

3. ~~**Per-axis vs isotropic**~~ — **RESOLVED.** Per-axis decoder works at 1.23% MAPE.
   Inference uses `max(W,H,D)` for isotropic mesh scale (canonical bbox invariant).

4. **Dataset for Phase 2 (geometric fidelity):** Still unresolved. Deferred until paper submitted.
