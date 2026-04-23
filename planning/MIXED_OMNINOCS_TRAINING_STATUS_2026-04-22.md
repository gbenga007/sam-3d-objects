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
