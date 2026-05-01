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
| M4b — SLAT-conditioned training | **In progress** | slat_conditioned_v1 running; ep1 eval in ~6-9h |
| M5 — OOD evaluation on WildRGB-D | Not started | |
| M6 — Phase 2 geometric fidelity begins | Not started | |

---

## Current Active Run (2026-05-01)

**nocs_sceneholdout_slat_conditioned_v1** — SLAT cross-attention conditioned on metric scale token.

| Setting | Value |
|---|---|
| Dataset | NOCS-Real275 via OmniNOCS, 16,118 train / 3,228 heldout (cached eval) |
| Warm-start | baseline_v2_best.pt (2.997% MAPE) |
| Trainable | MetricScaleHead + MetricScaleDecoder (lr=1e-4) + SLAT cross_attn×24 + norm2×24 (~100M, lr=1e-5) |
| Stage1 steps | 4 (SS, no_grad) |
| Stage2 steps | 1 (SLAT, with_grad — memory limit; 2+ steps OOM'd or produced nan) |
| Epochs | 10, eval every 1 |
| Estimated time | ~67-90h total; first epoch eval ~6-9h from 2026-05-01 00:00 |
| Script | `scripts/train_slat_conditioned_v1.sh` |
| Logs | `/tmp/slat_conditioned_v1.log` |
| Checkpoint | `artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v1*.pt` |

Key bugs fixed before this run:

1. **Gradient flow** (commit 64363e4): `sample_slat` had internal `torch.no_grad()` blocking all cross-attn gradients.
2. **Checkpoint mismatch** (commit e42d85f): baseline_v2 has 10-dim MetricScaleHead; current is 13-dim. Smart partial load zeroes new ss_scale columns.
3. **bfloat16 attention overflow** (commit e42d85f): MetricScaleHead output magnitude unconstrained → nan softmax in SLAT cross_attn for most samples. Fix: `F.layer_norm` on scale_token before SLAT injection; decoder receives raw token.
4. **Gradient clipping** (commit e42d85f): `clip_grad_norm_(max_norm=1.0)` + nan/inf skip guards added.

Eval note: `--eval-feature-cache` uses static cached SLAT features — SLAT conditioning benefit is invisible in these numbers. True comparison requires a live heldout eval after training.

---

## Open Questions

1. **MoGe correlation check:** Does `pointmap_scale` from MoGe actually correlate well
   with ground truth metric scale on Objectron/NOCS? If not, the scale head's metric
   anchor is weak and the architecture assumption needs revisiting.

2. **Scale token impact on generation quality:** Adding a new conditioning token to SLAT
   may affect the quality of the 3D shape. Needs evaluation — does the generated geometry
   degrade when the scale token is added?

3. **Per-axis vs isotropic scale:** The MetricScaleDecoder predicts [w, h, d] separately.
   If training data is insufficient, predicting a single isotropic scale first and
   expanding to per-axis later may be more stable.

4. **Dataset for Phase 2:** No dataset identified yet for fine-grained geometric detail
   supervision. This is the main blocker for Phase 2.
