# SAM 3D Objects - Metric Accuracy TODO

Goal: make SAM 3D Objects physically accurate by recovering metric scale
(real-world units), then later address near 1-to-1 geometric fidelity with the
input object.

Last updated: 2026-04-22

---

## Current Status

Metric-scale recovery has moved from architecture sketch to a working prototype.
The frozen SAM3D/MoGe feature cache workflow is implemented, and cached metric
heads train successfully on OmniNOCS NOCS-Real275.

Best completed runs:

| Split | Train / Held-out | Held-out MAPE | Baseline MAPE | Notes |
|---|---:|---:|---:|---|
| Image-grouped | 2002 / 500 | 2.17% | 12.90% | Held-out cm error: W=0.43, H=0.33, D=0.25 |
| Scene-heldout | 12882 / 3228 cached | 3.23% | 10.90% | Saved final checkpoint cm error: W=0.69, H=0.54, D=0.44 |

The scene-heldout best checkpoint by held-out mean absolute percentage error is
epoch 175 at 3.09% MAPE, but only the final epoch-200 checkpoint was saved. Final
epoch 200 improved median error but worsened mean error.

Artifacts:

```text
/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt
/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_metrics.jsonl
/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cm_backfill.jsonl
/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500.pt

/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_metrics.jsonl
/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cm_backfill.jsonl
/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6.pt
```

Current implemented design:

```text
planning/CURRENT_METRIC_READOUT_DESIGN_2026-04-22.md
planning/MIXED_OMNINOCS_TRAINING_STATUS_2026-04-22.md
planning/OMNINOCS_RGB_DOWNLOAD_STATUS_2026-04-22.md
```

Durable reproducibility artifacts:

```text
artifacts/metric_scale/checkpoints/
artifacts/metric_scale/manifests/
artifacts/metric_scale/metrics/
```

---

## Phase 1 - Dataset Preparation

### Completed

- [x] Downloaded OmniNOCS NOCS-Real275 annotations.
- [x] Downloaded NOCS-Real275 source RGB test frames to `/mnt/dest/OmniNOCS/real_test/`.
- [x] Confirmed NOCS masks are available through OmniNOCS annotations.
- [x] Implemented/used `OmniNOCSReal275Dataset` for metric-dimension training records.
- [x] Added `OmniNOCSObjectDataset` for common release-style metadata across
      NOCS-Real275, Objectron, ARKitScenes, and Hypersim when source RGB roots
      are available.
- [x] Ran MoGe-2 correlation check on 1,177 object instances from 200 NOCS-Real275 frames.
- [x] Validated metric-anchor assumption: overall log-log Pearson `r=0.708`.
- [x] Diagnosed bottle/cup MoGe failure modes:
  - bottle transparency is a real issue and improves when segmented pixels are painted opaque;
  - cup remains structurally harder because open-top geometry confuses depth extent.

### Remaining

- [ ] Confirm whether ARKitScenes and Hypersim OmniNOCS extraction finished cleanly.
- [ ] Locate or download source RGB roots for Objectron, ARKitScenes, and Hypersim.
- [ ] Decide whether to include Objectron in the next training phase or keep the current
      NOCS-first path until inference integration is complete.
- [ ] If using Objectron, download source RGB and adapt `scripts/preprocess_objectron.py`
      to the OmniNOCS annotation layout.
- [ ] Add a durable dataset/cache manifest so `/tmp` artifacts can be reproduced or moved
      without relying only on notes.
- [ ] Add optional transparent-object preprocessing experiment for bottle/cup using masks.

---

## Phase 2 - Architecture Implementation

### Completed

- [x] Added `MetricScaleHead` in `sam3d_objects/model/backbone/scale_head.py`.
- [x] Updated `MetricScaleHead` to output a 768-dim scale token directly from pooled SS
      latent plus MoGe pointmap stats, avoiding the earlier scalar bottleneck.
- [x] Added `_ScaleAugmentedEmbedderProxy` for appending a scale token to condition
      embeddings without changing generator architecture.
- [x] Added `MetricScaleDecoder` in
      `sam3d_objects/model/backbone/metric_scale_decoder.py`.
- [x] Confirmed `pointmap_scale` and `pointmap_shift` are available from the existing
      pointmap preprocessing path.
- [x] Prototyped metric prediction from frozen SS/SLAT features in
      `sam3d_objects/training/finetune_metric_scale.py`.
- [x] Added metric-head checkpoint loading to `InferencePipelinePointMap`.
- [x] Added inference outputs for predicted metric dimensions in meters and centimeters.

### Remaining

- [ ] If scale-aware generation is still desired, align the scale-token dimension
      with the live SLAT condition tokens and partially unfreeze or adapt SLAT
      cross-attention so generation itself learns to use the scale token.

---

## Phase 3 - Fine-Tuning Setup

### Completed

- [x] Implemented frozen-backbone fine-tuning script:
      `sam3d_objects/training/finetune_metric_scale.py`.
- [x] Freezes existing SAM3D pipeline models during metric-head training.
- [x] Trains `MetricScaleHead` and `MetricScaleDecoder` with Smooth L1 loss in
      log-dimension space.
- [x] Added cached latent workflow:
  - `--cache-latents`
  - `--save-feature-cache`
  - `--load-feature-cache`
  - `--cache-only`
- [x] Added grouped splits: `record`, `image`, and `scene`.
- [x] Added deterministic split controls: `--seed`, `--shuffle-split`.
- [x] Added per-axis, per-category, and JSONL metrics.
- [x] Added centimeter-scale absolute error metrics.
- [x] Added optional Weights & Biases logging for train/eval/baseline metrics.
- [x] Added category mean-size held-out baseline.
- [x] Added best-checkpoint saving by held-out mean absolute percentage error.
- [x] Validated 10-sample cached overfit to 2.61% MAPE.
- [x] Validated 80/20 cached held-out run around 14.2% MAPE near epoch 150.
- [x] Completed image-grouped 2002/500 run: 2.17% held-out MAPE.
- [x] Completed scene-heldout 12890/3228 run: best 3.09% held-out MAPE at epoch 175.

### Remaining

- [ ] Add resume-from-checkpoint support for metric-head training.
- [ ] Add a cache metadata validator: script version, dataset roots, split groups,
- [x] Added cache/artifact manifests with dataset roots, split details, category counts,
      scene counts, feature tensor schema, artifact sizes, and checkpoint hashes.
- [ ] Add category-balanced or stratified grouped split modes.
- [ ] Add leave-one-category-out diagnostics, especially for bottle/camera coverage gaps.
- [ ] Add training/eval tests for cached feature loading and grouped split behavior.

---

## Phase 4 - Evaluation

### Completed

- [x] Report mean and median absolute percentage error.
- [x] Report per-axis dimensional MAPE.
- [x] Report per-category MAPE.
- [x] Add absolute centimeter error metrics to cached evaluation output.
- [x] Backfilled centimeter-error metrics for the existing saved final checkpoints/caches.
- [x] Compare against category mean-size baseline.
- [x] Run image-grouped held-out evaluation.
- [x] Run full-scene held-out evaluation.

### Remaining

- [ ] Recover or retrain the scene-heldout epoch-175 best checkpoint for cm-error reporting.
- [ ] Compare against the original pipeline's non-metric/SSI scale behavior.
- [ ] Add prediction-vs-ground-truth scatter plots for each axis and category.
- [ ] Visualize predicted vs ground-truth 3D bounding boxes over input images.
- [ ] Run a held-out split that includes all six NOCS categories under stricter grouping.
- [ ] Run cross-scene folds instead of a single scene-6 holdout.
- [ ] Evaluate on WildRGB-D or another out-of-distribution metric dataset.
- [ ] Audit bottle/cup performance with and without opaque-mask painting.

---

## Phase 5 - Productization / Integration

- [x] Decided artifact location for trained metric-head checkpoints outside `/tmp`.
- [ ] Add documented commands for cache creation, training, evaluation, and inference.
- [ ] Add a small reproducible smoke-test cache for CI or local sanity checks.
- [ ] Add `.gitignore` rules or artifact policy for large feature caches and checkpoints.
- [ ] Decide whether Docker/CI additions should be committed with this work.
- [ ] Clean up and commit the current training-script changes and planning docs.

---

## Phase 6 - Geometric Fidelity (Longer Term)

This phase addresses the second goal: 1-to-1 reconstruction of fine geometric detail
(surface texture, cracks, exact shape). It requires training changes to the generative
model, not just metric readout heads.

- [ ] Literature review: reconstruction losses for flow matching / score-based models.
- [ ] Design: add perceptual loss term by rendering output and comparing to input image.
- [ ] Design: add depth consistency loss using MoGe depth of rendered output vs input depth.
- [ ] Identify a dataset with fine-grained geometric ground truth, such as high-res
      3D scans of real objects.
- [ ] Implement training with combined flow matching and reconstruction losses.
- [ ] Evaluate Chamfer distance and F-score on held-out scans.

---

## Milestones

| Milestone | Status | Notes |
|---|---|---|
| M1 | Mostly complete | NOCS/OmniNOCS path is usable; Objectron remains optional/pending |
| M2 | Prototype complete | Metric heads and cached training path run end-to-end |
| M3 | Complete | 10-sample cached overfit reached 2.61% MAPE |
| M4 | Prototype complete | Scene-heldout best is 3.09% MAPE, well below 15% target |
| M5 | Not started | OOD evaluation still pending |
| M6 | Not started | Geometric fidelity work remains future phase |
