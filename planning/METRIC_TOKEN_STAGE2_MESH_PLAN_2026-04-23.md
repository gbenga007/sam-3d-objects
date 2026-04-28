# Metric Token Stage-2 Mesh Plan - 2026-04-23

## Goal

Advance from metric-dimension readout to metric-aware generation:

- use the existing metric scale token as a conditioning token for stage-2 SLAT
- keep predicting metric dimensions `[width, height, depth]`
- keep mesh prediction in canonical space
- improve metric-size prediction so the canonical mesh can be paired with an
  accurate metric scale estimate
- improve mesh generation so decoded geometry is more consistent with the input
  object while remaining canonical

This is the correct next prototype phase, but it should be treated as a
conditioning and dataset expansion effort, not yet as proof of exact 1-to-1
mesh reconstruction.

## Clarified Target Formulation

The intended target for the next phase is:

```text
canonical mesh prediction
+ separate accurate metric-size prediction
```

not:

```text
direct metric-space mesh prediction
```

This makes sense and is a cleaner first goal.

Why this is a good decomposition:

- stage-2 already naturally predicts canonical object geometry
- metric supervision is currently strongest for dimensions, not for direct
  metric mesh vertices
- a canonical mesh plus accurate scale estimate is enough to recover metric box
  dimensions and can later be converted to metric geometry if desired

So the first success criterion should be:

- better canonical mesh fidelity
- plus better metric dimension / size prediction

rather than forcing the mesh decoder itself to emit final metric-scaled vertices
immediately

## Current Starting Point

Already implemented:

- `MetricScaleHead` produces a `768`-dim token from:
  - pooled SS latent `[B, 4096, 8] -> [B, 8]`
  - `log(pointmap_scale)` `[B, 1]`
  - `pointmap_shift_z` `[B, 1]`
- `_ScaleAugmentedEmbedderProxy` can append that token to stage-2 condition
  embeddings without changing the SLAT generator interface
- `MetricScaleDecoder` predicts metric dimensions from:
  - stage-2 latent features
  - the metric token

What is still missing is a clean stage-2 training path where the generator itself
learns to use that token for mesh generation quality, not only for downstream
dimension readout.

## Honest Assessment

This phase is feasible now from an engineering standpoint.

The risk is not "can we inject the token?" The risk is scientific:

- current metric results validate in-domain metric readout on NOCS-Real275
- current mixed-domain results do not yet validate robust cross-domain metric
  reasoning
- current supervision is enough for dimension prediction, but not enough to
  support a strong claim that mesh outputs become "exactly like" the input

So the next phase should be framed as:

- metric-aware stage-2 conditioning prototype
- plus dataset/supervision upgrades for mesh fidelity

## Architecture Plan

### A. Keep the Current Metric Token

Retain the current token definition for the first prototype:

```text
token = MLP([mean(shape_latent), log(pointmap_scale), shift_z]) -> [B, 1, 768]
```

Reason:

- it already works with the metric decoder
- it matches stage-2 condition width (`768`)
- it lets us isolate whether stage-2 conditioning helps before redesigning the
  token itself

### B. Inject the Token Into Stage-2 During Training

Use `_ScaleAugmentedEmbedderProxy` so SLAT sees:

```text
[existing stage-2 condition tokens] + [metric token]
```

Required work:

- make the stage-2 training path always append the metric token when enabled
- expose a clean flag/config for:
  - disabled
  - inference-only injection
  - training + inference injection

### C. Keep Auxiliary Metric Supervision

Retain the dimension head as an auxiliary objective:

- predict `[width, height, depth]`
- compute metric losses during training/eval
- optionally derive a single object scale summary from those dimensions for
  reporting or downstream mesh rescaling

Reason:

- this keeps pressure on the token to remain metrically informative
- it provides a direct measurable signal even when mesh quality is ambiguous

### D. Start Conservative on Generator Training

First prototype should avoid a large full-model unfreeze.

Recommended order:

1. train only the metric head / decoder with stage-2 token injection active
2. then unfreeze the smallest stage-2 components needed to react to the token
3. only later consider broader stage-2 finetuning

The immediate question is whether SLAT generation measurably changes when the
metric token is present.

## Dataset Requirements

For this phase, the dataset must support both metric supervision and geometry
supervision.

Minimum fields:

- RGB image
- instance mask / object identity
- metric dimensions in real units
- target 3D shape representation that can supervise mesh quality
- category and split metadata

Desirable fields:

- calibrated camera information
- object pose / canonical orientation
- multiple views or stronger geometric targets

## Dataset Strategy

### 1. Keep OmniNOCS for Metric Supervision

Use OmniNOCS for:

- metric dimensions
- cross-domain images
- mixed-source evaluation

But do not treat OmniNOCS alone as sufficient mesh-fidelity supervision.

### 2. Add a Mesh-Supervised Dataset

We need a dataset where target geometry is known more faithfully.

Candidate directions:

- synthetic rendered object datasets with exact source meshes
- ShapeNet-based rendered object datasets with masks and camera metadata
- Objaverse / 3D-FUTURE style assets if alignment, licensing, and rendering are
  manageable
- real scan datasets only if camera/object alignment is strong enough for
  conditioning studies

Practical recommendation:

- use synthetic mesh-supervised data first for stage-2 token-conditioning
- then use OmniNOCS as the real-image metric supervision track

### 3. Separate Claims by Dataset Type

The paper/experiment story should likely become:

- synthetic mesh-supervised experiments:
  - does metric token improve metrically faithful geometry generation?
- real-image OmniNOCS experiments:
  - does metric token improve dimension prediction and mesh scale consistency?

That is much cleaner than over-claiming exact reconstruction from the current
real-image data alone.

## Evaluation Plan

This phase needs both metric and geometry metrics.

### Metric Evaluation

- mean absolute error in cm
- MAPE
- per-axis cm / percent error
- per-category and per-source breakdown
- optional scalar size summary error if we choose to expose one

### Geometry Evaluation

- Chamfer distance
- F-score at one or more thresholds
- IoU / volumetric overlap if available
- mesh dimension consistency vs ground truth

### Conditioning/Fidelity Evaluation

We need to answer whether the token improves instance specificity, not only
category plausibility.

Suggested diagnostics:

- no metric token vs metric token
- mesh dimension error before/after token injection
- qualitative input / predicted mesh / target mesh panels
- failure cases where dimensions improve but shape identity does not
- canonical-mesh quality and metric-size quality should be reported separately

## First Experiment Matrix

### E1. Stage-2 Conditioning Ablation

Same data, same stage-2 pipeline:

- baseline: no metric token
- metric token injected
- metric token injected + auxiliary WHD loss

Primary questions:

- does the mesh scale become more consistent?
- do dimension errors improve?

### E2. Metric Token Source Ablation

Compare:

- SS latent only
- MoGe stats only
- SS latent + MoGe stats

This is necessary to understand whether the token is truly combining shape and
metric prior information.

### E3. Synthetic-to-Real Strategy

- train stage-2 token conditioning on a mesh-supervised synthetic dataset
- evaluate geometry there
- test whether the same tokenized design remains useful on OmniNOCS metric tasks

## Immediate Implementation Tasks

1. Add a documented config/flag for stage-2 metric token injection.
2. Make stage-2 training/inference use the metric token consistently when enabled.
3. Preserve the WHD auxiliary decoder and losses.
4. Identify the first mesh-supervised dataset candidate and document its fields.
5. Define a minimum evaluation bundle:
   - cm error
   - MAPE
   - Chamfer
   - qualitative panels

## Recommendation

Proceed with this phase now, but keep the claim narrow:

- "metric-aware stage-2 conditioning for improved geometry scale consistency"

Do not yet frame it as:

- "exact metric-faithful mesh reconstruction from the input"

That stronger claim needs new data, stronger supervision, and cleaner ablations.
