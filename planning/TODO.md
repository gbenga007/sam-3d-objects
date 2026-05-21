# SAM 3D Objects - Metric Accuracy TODO

Goal: make SAM 3D Objects physically accurate by recovering metric scale
(real-world units), then later address near 1-to-1 geometric fidelity with the
input object.

Last updated: 2026-05-12

---

## Current Status

Last updated: 2026-05-21

**mixed_v1 training active** — epoch 1/6, step ~36K/75,654 (48%), loss=0.098, 34h in.
- GPU: A10 24GB (migrated from A100-80GB on 2026-05-19)
- ETA epoch-1 checkpoint: ~29h from last check
- Log: `/tmp/mixed_v1.log` (mirrored hourly to `artifacts/metric_scale/logs/mixed_v1.log`)
- Pre-training baselines: NOCS 5.01%, Objectron 87.0%, ARKitScenes 80.2%, overall 71.4%
- Loss descending cleanly: 0.977→0.115 (step 25K)→0.102 (step 33K)→0.098 (step 36K)

**ss_ratio_v1 COMPLETED** — killed at ep10 mid-epoch (2026-05-18). ep6 = 1.23% MAPE is the final best.
Best checkpoint: `artifacts/metric_scale/checkpoints/nocs_sceneholdout_ss_ratio_v1_best.pt`
ep6 per-category: bottle=0.67%, can=0.67%, camera=0.68%, cup=1.21%, bowl=1.99%, laptop=1.86%.

**Paper writing started (2026-05-19)** — Introduction, Related Work, Method, Experiments sections
drafted and pushed to `git@github.com:gbenga007/eccv-paper-vigir.git`. Abstract pending epoch-1 results.
Target venue: MUSTCV workshop at ECCV 2026.

Completed runs:

| Run | Split | Train / Held-out | Held-out MAPE | Notes |
|---|---|---:|---:|---|
| frozen-SLAT image-grouped | Image-grouped | 2002 / 500 | 2.17% | W=0.43, H=0.33, D=0.25 cm |
| frozen-SLAT scene-heldout 768-dim | Scene-heldout | 12882 / 3228 | 3.09% | Best ep175 |
| **nocs_sceneholdout_1024dim_baseline_v2** | Scene-heldout | 12882 / 3228 | **2.997%** | ep200, 13-dim head |
| nocs_sceneholdout_slat_conditioned_v1 | Record-train | 16054 / 64 | NaN ep1 | Killed; slat-lr=1e-5 too high |
| **nocs_sceneholdout_slat_conditioned_v2** | Record-train | 16054 / 64 | **1.93%** | ep2 best; ep3 crashed |
| **nocs_sceneholdout_slat_conditioned_v3** | Record-train | 16054 / 64 | **1.74%** | ep5 best; ep4 transient spike (5.77%) |

Active runs:

| Run | Status | Log | Notes |
|---|---|---|---|
| **nocs_sceneholdout_ss_ratio_v1** | Killed at ep10 (~34%) 2026-05-18 | `artifacts/metric_scale/logs/ss_ratio_v1.log` | ep6 best=1.23%; ep7-9 regressed to 1.46-1.69%; killed to free Ceph + GPU for mixed_v1 |
| **mixed_v1** | Launched 2026-05-18 (PID 2833164, third try) | `/tmp/mixed_v1.log` → `artifacts/metric_scale/logs/mixed_v1.log` (hourly cron) | NOCS+Obj+ARKit, 76K records, 1:1:1 balanced; warm-start ss_ratio_v1_best.pt; pipeline-skip handler added; ~16 days |

ss_ratio_v1 epoch trajectory:

| Epoch | MAPE | Mean cm |
|---|---|---|
| baseline | 1.51% | 0.22 |
| 1 | 2.06% | 0.27 |
| 2 | 1.54% | 0.20 |
| 3 | 2.02% | 0.26 |
| 4 | 1.59% | 0.22 |
| 5 | 1.30% | 0.19 |
| **6** | **1.23% ← new best** | 0.20 |
| 7 | 1.46% | 0.21 |
| 8 | in progress | — |

ep6 per-category: bottle=0.67%, can=0.67%, camera=0.68%, cup=1.21%, bowl=1.99%, laptop=1.86%.
ETA to completion: ~35 hours (~1.5 days from 2026-05-17 02:17).

Log (v3, completed): `/tmp/slat_conditioned_v3.log`
Wandb: `https://wandb.ai/reformed-tulip/sam3d-metric-scale`

Key findings:
- `max(canonical_mesh_bbox) = 1.000 ± 0.002`: hard invariant. Isotropic mesh scale = `max(W_pred, H_pred, D_pred)`.
  See: `planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md`
- Mesh aspect ratio (H, D) comes from SLAT reconstruction, not the metric head — this is the
  next problem to address (depth reprojection loss). See: `/mnt/source/.claude/memory/project_aspect_ratio.md`
- OmniNOCS train/test metadata files are identical — true held-out eval requires
  `--heldout-samples 64 --split-group record`.

Durable artifacts:

```text
artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt  ← v3 best (1.74%)
artifacts/metric_scale/checkpoints/nocs_sceneholdout_ss_ratio_v1_best.pt          ← current best (1.30%, ep5)
artifacts/metric_scale/checkpoints/nocs_sceneholdout_ss_ratio_v1.pt               ← rolling resume ckpt
artifacts/metric_scale/metrics/
artifacts/metric_scale/manifests/
artifacts/metric_scale/logs/ss_ratio_v1.log          ← training log (backed up every 5min from /tmp)
artifacts/metric_scale/eval_slat_conditioned_v2/     ← 64-sample eval, isotropic scaling, 3.0% mean MAPE
artifacts/metric_scale/eval_v2_peraxis/              ← 64-sample eval, per-axis scaling
artifacts/metric_scale/eval_v3_mesh_bbox/            ← v3_best: iso 11.42%, peraxis 5.54%, gap 5.88pp; 192 PLYs
```

Planning docs:

```text
planning/CURRENT_METRIC_READOUT_DESIGN_2026-04-22.md
planning/MIXED_OMNINOCS_TRAINING_STATUS_2026-04-22.md
planning/SLAT_CONDITIONED_METRIC_TRAINING_2026-04-27.md      ← architecture + training command
planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md        ← bbox invariant finding
planning/METRIC_TOKEN_STAGE2_MESH_PLAN_2026-04-23.md
planning/ANISOTROPIC_RESCALE_EXPERIMENT_2026-05-11.md        ← confirms SS is source of aspect ratio error
planning/SS_ARCHITECTURE_AND_ASPECT_RATIO_FIX_2026-05-11.md  ← SS architecture + ratio loss design + impl
planning/PAPER_PLAN_METRIC_SCALE_2026-05-17.md               ← Paper framing, experiments, ablations, baselines
```

Aspect ratio eval tool (run on GPU when not training):

```bash
python scripts/mesh_scale_eval.py \
  --checkpoint artifacts/metric_scale/checkpoints/<ckpt>.pt \
  --n-samples 64 --stage1-steps 4 --stage2-steps 1 \
  --output artifacts/metric_scale/eval_<run>/mesh_scale_eval.jsonl \
  --output-dir artifacts/metric_scale/eval_<run>/plys
```

Compares isotropic vs per-axis mesh bbox errors vs GT. Gap = cost of wrong voxel aspect ratio.
Saves 3 PLYs per sample (canonical/iso/peraxis). v3_best baseline: iso 11.42%, peraxis 5.54%, gap +5.88pp.

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
- [x] **Phase 3a**: Mixed dataset training — NOCS + Objectron + ARKitScenes
      - Warm-start from ss_ratio_v1_best.pt (1.23% MAPE, ep6) ✓
      - Sampling: Option C — `--max-records-per-source 30000 --balanced-sampling` ✓
      - Per-source heldout: 64 NOCS + 200 Objectron + 200 ARKitScenes ✓
      - 6 epochs, ~63h/epoch → ~16 days total
      - **Launched 2026-05-18** as `mixed_v1` (PID 2815224, log `artifacts/metric_scale/logs/mixed_v1.log`)
- [x] Add `--balanced-sampling` flag (WeightedRandomSampler, weight=1/source_count) to `finetune_metric_scale.py`
- [x] Add `--heldout-per-source` flag + per-source heldout in `make_train_eval_subsets`
- [x] Resolve Objectron RGB density: wrote `scripts/download_objectron_selective.py`, streamed
      96,728 train + 26,846 test frames from `gs://objectron/videos/.../video.MOV` via PyAV
- [x] Write `scripts/train_mixed_v1.sh` launcher
- [ ] Monitor mixed_v1 epoch evals; checkpoint best per-source MAPE
- [ ] **Phase 3b** (follow-up, decide after mixed_v1 finishes): Add Hypersim at 0.25× weight if
      broader category coverage needed. Risk: synthetic domain gap in MoGe/DINOv2 features.
- [x] Enable flash_attn backend — switched in `scripts/train_mixed_v1.sh` (2026-05-20);
      verified fp32 cross-attn upcast is independent of ATTN_BACKEND (it wraps forward(), not sdpa)
- [ ] Add a durable dataset/cache manifest so `/tmp` artifacts can be reproduced or moved
      without relying only on notes.
- [ ] Add optional transparent-object preprocessing experiment for bottle/cup using masks.

---

## Phase 7 - Paper (MUSTCV @ ECCV 2026)

Repo: `git@github.com:gbenga007/eccv-paper-vigir.git`

### Completed
- [x] Identify venue: MUSTCV workshop at ECCV 2026
- [x] Draft Introduction (5 paragraphs + contributions)
- [x] Draft Related Work (4 paragraphs; all key bib entries filled)
- [x] Draft Method (6 subsections with equations)
- [x] Draft Experiments skeleton (tables with real NOCS numbers; placeholders for mixed results)

### Pending (in priority order)
- [ ] Fill abstract — waiting for epoch-1 mixed_v1 per-source MAPE (~29h)
- [ ] Compute category-mean prior per-category MAPE (no training; just dataset stats)
- [ ] Implement depth-bridge baseline (MoGe-2 → masked pointcloud bbox; inference only)
- [ ] Run ablation training runs (3–4 × ~60h each):
      A1: w/o SS scale features; A2: w/o pointmap stats; A4: w/o AR loss; A5: w/o CFG dropout
- [ ] Architecture diagram (frozen backbone greyed out; new components highlighted)
- [ ] Qualitative figure (input / GT / category-mean / ours — 4 instances)
- [ ] Scatter plot: predicted vs GT object size, coloured by source
- [ ] Mesh aspect-ratio eval: rerun `scripts/mesh_scale_eval.py` on ss_ratio_v1_best.pt
- [ ] Draft Conclusion
- [ ] WildRGB-D OOD evaluation (stretch; needed for main-conference level)

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

### Completed (continued)

- [x] Updated `MetricScaleHead` output dim 768 → 1024 to match SLAT `cond_channels`.
- [x] Updated `MetricScaleDecoder` scale_token_dim 768 → 1024.
- [x] Implemented `collect_slat_cross_attn_params()`: unfreezes cross_attn + norm2
      in all 24 SLAT blocks (~100M params) while keeping self-attn, FFN, adaLN frozen.
- [x] Added `--unfreeze-slat-cross-attn`, `--slat-lr` (default 1e-5), `--eval-feature-cache`
      flags to `finetune_metric_scale.py`.
- [x] Wired scale token injection into SLAT live training path (no_grad removed from
      SLAT when unfreeze_cross_attn=True; two-group AdamW optimizer).
- [x] Validated incompatibility guard: `--unfreeze-slat-cross-attn` blocks `--cache-latents`.
- [x] Ran canonical mesh bbox diagnostic (25 NOCS-Real275 samples, 6 categories):
      confirmed `max(canonical_bbox) = 1.000 ± 0.002` hard invariant.
      Inference formula: `s = max(W_pred, H_pred, D_pred)`.
      See: `planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md`
- [x] Launched SLAT-conditioned training: 20 epochs, 12882/3228 scene-heldout,
      wandb run `nocs_sceneholdout_slat_conditioned_1024dim`.
      Architecture doc: `planning/SLAT_CONDITIONED_METRIC_TRAINING_2026-04-27.md`

### Remaining

- [x] Implemented `collect_slat_cross_attn_params()`, gradient flow, with_grad fix.
- [x] Fixed gradient flow bug: `sample_slat` had internal `torch.no_grad()` blocking
      all SLAT cross-attn gradients (commit 64363e4).
- [x] Added SS scale features (3-dim log-SSI) + `metric_modality_embed` to MetricScaleHead
      (commit 5777894). Input dim: 10 → 13.
- [x] Fixed checkpoint shape mismatch (10→13 dim): `_adapt_scale_head_state_dict`
      (commit e42d85f).
- [x] Fixed bfloat16 attention overflow: `F.layer_norm` on scale_token before SLAT
      injection only; decoder receives unnormalized token (commit e42d85f).
- [x] Added gradient clipping (`clip_grad_norm_ max_norm=1.0`) and nan/inf guards
      to both training paths (commit e42d85f).
- [x] Launched full-dataset SLAT-conditioned training (16,118 samples, 10 epochs,
      eval every epoch). Script: `scripts/train_slat_conditioned_v1.sh`.
- [x] Check epoch 1 eval MAPE — v2 ep1: 2.23%, ep2: 1.93% (beats baseline_v2's 2.997%).
- [x] Compare final MAPE vs baseline_v2: v2 best is 1.93% vs 2.997% — improvement confirmed.
- [x] Run live heldout eval with trained cross-attn (64-sample, record-group held-out).
- [ ] Ablate: metric token injection without cross-attn unfreeze (frozen SLAT + injection).
- [ ] Address mesh aspect ratio: confirmed Stage 1 voxel topology is the source.
      Fix (v1, implemented 2026-05-11): soft-variance aspect ratio loss on SS decoder output.
        - `compute_ss_aspect_ratio_loss`: rank-sorted log-normalised GT dims vs soft voxel std-dev
        - `--ss-ratio-loss-weight` (default 0.0), `--unfreeze-ss-decoder`
        - Decoder-only: SS backbone frozen; SS decoder gets second forward on detached shape_latent
        - Training target: GT [W,H,D] sorted descending, log-normalised to zero mean (scale-free)
      Fix (v2, future): `--unfreeze-ss-cross-attn` — propagate loss through backbone cross-attn
        to PointPatchEmbed (requires modifying sample_sparse_structure with_grad support).
      Overfit test (done): gradient flow confirmed, loss 0.0015→0.000481 over 5 epochs.
      Full training run launched: ss_ratio_v1 (epoch 1/10 active).
      Baseline mesh bbox eval (v3_best): iso 11.42%, peraxis 5.54%, gap +5.88pp.
        - laptop worst at iso (19.12%→5.92%), smallest axis worst (18%→6.6%)
        - can anomalous: per-axis WORSE than iso (9.33% vs 7.89%)
      Next: re-run mesh_scale_eval on ss_ratio_v1_best.pt after epoch 1 to measure gap reduction.
      Architecture + loss design: `planning/SS_ARCHITECTURE_AND_ASPECT_RATIO_FIX_2026-05-11.md`
- [ ] Choose and document the first mesh-supervised dataset for stage-2 fidelity
      experiments (Hypersim metric meshes are a leading candidate).

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

- [x] Add resume-from-checkpoint support for metric-head training (`--resume-from`).
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
- [x] Add documented training script: `scripts/train_slat_conditioned_v1.sh`.
- [ ] Add documented commands for cache creation, evaluation, and inference.
- [ ] Add a small reproducible smoke-test cache for CI or local sanity checks.
- [ ] Add `.gitignore` rules or artifact policy for large feature caches and checkpoints.
- [ ] Decide whether Docker/CI additions should be committed with this work.

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

See also:

```text
planning/METRIC_TOKEN_STAGE2_MESH_PLAN_2026-04-23.md
```

---

## Milestones

| Milestone | Status | Notes |
|---|---|---|
| M1 | Mostly complete | NOCS/OmniNOCS path is usable; Objectron remains optional/pending |
| M2 | Prototype complete | Metric heads and cached training path run end-to-end |
| M3 | Complete | 10-sample cached overfit reached 2.61% MAPE |
| M4 | Complete | Scene-heldout best is 3.09% MAPE; SLAT-conditioned v2 best is 1.93% MAPE |
| M4b | **Complete** | v3: **1.74% MAPE** ep5; aspect ratio problem identified as next step |
| M4c | **In progress** | ss_ratio_v1 ep6: **1.23% MAPE** (new best); ep8/10 in progress |
| M4d | **In progress** | Phase 3a mixed training (mixed_v1 launched 2026-05-18): NOCS+Obj+ARKit, 1:1:1 balanced, ~16 days |
| M5 | Not started | OOD evaluation still pending |
| M6 | Not started | Geometric fidelity / aspect ratio work — depth reprojection loss next |
