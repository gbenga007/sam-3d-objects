# Mixed OmniNOCS Training Status - 2026-04-22

## Summary

Mixed OmniNOCS training is now supported at the dataset API level, but only for
sources whose source RGB frames are available locally.

Implemented:

```text
sam3d_objects/data/dataset/metric/omninocs.py
OmniNOCSObjectDataset
```

The loader understands the common OmniNOCS release metadata/mask format for:

```text
nocs_real275
objectron
arkitscenes
hypersim
```

It emits the same object-instance record shape used by the metric training path:

```text
RGBA image
metric_dims [width, height, depth] in meters
category
source
image_name
object_id
mask_pixels
```

## What Is Available Locally

OmniNOCS metadata and instance masks are present for:

```text
/mnt/dest/OmniNOCS/omninocs_release_nocs_real275
/mnt/dest/OmniNOCS/omninocs_release_objectron
/mnt/dest/OmniNOCS/omninocs_release_ARKitScenes
/mnt/dest/OmniNOCS/omninocs_release_hypersim
```

Source RGB is currently available for NOCS-Real275:

```text
/mnt/dest/OmniNOCS/real_test
```

The Objectron, ARKitScenes, and Hypersim OmniNOCS release folders appear to
contain annotations/masks/NOCS maps, not source RGB frames. Those sources require
separate RGB roots before they can be cached/trained.

## Training Script Support

The training script now supports:

```bash
--dataset omninocs-mixed
--omninocs-sources nocs_real275 objectron arkitscenes hypersim
--omninocs-root /mnt/dest/OmniNOCS
--rgb-root /mnt/dest/OmniNOCS/real_test
--objectron-rgb-root ...
--arkitscenes-rgb-root ...
--hypersim-rgb-root ...
--max-records-per-source N
--skip-missing-rgb / --no-skip-missing-rgb
```

Records whose RGB source frame is missing are skipped by default. This prevents
annotation-only sources from silently breaking training, while allowing mixed
training as soon as RGB roots are supplied.

## Smoke Test

The new loader was smoke-tested with the available NOCS RGB root:

```text
sources=['nocs_real275']
records=5
first image shape=(480, 640, 4)
first source=nocs_real275
```

## Remaining Work Before True Mixed Training

1. Locate or download source RGB frames for Objectron.
2. Locate or download source RGB frames for ARKitScenes.
3. Locate or download source RGB frames for Hypersim.
4. Run small cache-only smoke tests per source.
5. Add source-balanced sampling if per-source caps are not sufficient.
6. Cache a bounded mixed training set and report metrics by both `source` and
   `category`.

## Suggested First Mixed Run

Start with NOCS + Objectron after Objectron RGB is available:

```bash
env LIDRA_SKIP_INIT=1 ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --dataset omninocs-mixed \
  --omninocs-sources nocs_real275 objectron \
  --rgb-root /mnt/dest/OmniNOCS/real_test \
  --objectron-rgb-root /path/to/objectron/rgb \
  --overfit-samples 2000 \
  --heldout-samples 500 \
  --max-records-per-source 2500 \
  --split-group image \
  --shuffle-split \
  --seed 42 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --save-feature-cache /tmp/metric_scale_omninocs_mixed_nocs_objectron_cache.pt \
  --manifest-output artifacts/metric_scale/manifests/metric_scale_omninocs_mixed_nocs_objectron_manifest.json \
  --cache-only
```

## 2026-04-23 Update

Mixed OmniNOCS loading is now working across all four sources with local RGB:

```text
total object instances: 1,217,988
  hypersim: 1,025,834
  arkitscenes: 169,678
  nocs_real275: 16,118
  objectron: 6,358
```

Unique images currently visible to the loader:

```text
hypersim: 49,917
arkitscenes: 47,015
nocs_real275: 2,754
objectron: 5,295
```

The earlier 400-example mixed run was intentionally bounded with:

```text
--max-records-per-source 100
```

For the next larger-scale experiment, the practical all-source balanced cap is
set by Objectron, which currently has 6,358 object instances. The next run will
therefore use all Objectron records and match the other three sources to that
cap instead of letting Hypersim dominate the training pool.

Planned larger balanced run:

```bash
env LIDRA_SKIP_INIT=1 ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --dataset omninocs-mixed \
  --omninocs-root /mnt/dest/OmniNOCS \
  --omninocs-sources nocs_real275 objectron arkitscenes hypersim \
  --rgb-root /mnt/dest/OmniNOCS/real_test \
  --objectron-rgb-root /mnt/dest/OmniNOCS/omni3d_rgb \
  --arkitscenes-rgb-root /mnt/dest/OmniNOCS/omni3d_rgb \
  --hypersim-rgb-root /mnt/dest/OmniNOCS/omni3d_rgb/hypersim \
  --cache-latents \
  --max-records-per-source 6358 \
  --overfit-samples 23000 \
  --heldout-samples 2400 \
  --shuffle-split \
  --split-group image \
  --seed 42 \
  --epochs 20 \
  --batch-size 512 \
  --eval-every 5 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --wandb \
  --wandb-mode offline \
  --wandb-project sam3d-metric-scale \
  --wandb-run-name mixed_omninocs_balanced_6358_per_source_bs512 \
  --metrics-output artifacts/metric_scale/metrics/mixed_omninocs_balanced_6358_per_source_bs512_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/mixed_omninocs_balanced_6358_per_source_bs512_manifest.json \
  --save-feature-cache /tmp/mixed_omninocs_balanced_6358_per_source_bs512_cache.pt \
  --output artifacts/metric_scale/checkpoints/mixed_omninocs_balanced_6358_per_source_bs512.pt
```

Notes:

- `batch-size` only affects the metric-head optimization phase. The expensive
  part of this workflow is still latent caching, which currently processes
  images one at a time through the frozen SAM 3D pipeline.
- `split-group image` is preferred over `record` for mixed experiments because
  it avoids placing different objects from the same frame into both train and
  held-out splits.
- W&B is currently configured for `offline` mode in this environment because no
  login credentials are present. The run will still emit structured W&B logs
  locally and can be synced later if desired.
