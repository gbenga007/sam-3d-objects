# Metric-Scale Reproducibility Report - 2026-04-22

This report records the durable artifacts created from the existing metric-scale
prototype runs.

## Artifact Policy

Tracked in this directory:

- metric-head checkpoints;
- manifest JSON files;
- metric JSONL summaries;
- short reports.

Not tracked:

- feature caches, because they are multi-GB tensors.

## Preserved Checkpoints

```text
artifacts/metric_scale/checkpoints/metric_scale_omninocs_imagegroup_train2000_holdout500.pt
artifacts/metric_scale/checkpoints/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200.pt
```

## External Feature Caches

```text
/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt
/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
```

The image-grouped cache is about 2 GB. The scene-heldout cache is about 13 GB.
Their paths, sizes, split counts, scene/category counts, and tensor schema are
recorded in the manifests.

## Manifests

```text
artifacts/metric_scale/manifests/metric_scale_omninocs_imagegroup_train2000_holdout500_manifest.json
artifacts/metric_scale/manifests/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200_manifest.json
```

The manifests include SHA-256 hashes for the lightweight checkpoints and metrics
files. They do not hash feature caches by default because hashing 2-13 GB caches
is slow and unnecessary for routine local validation.

## Reproduced Metrics

### Image-Grouped 2002 / 500

Checkpoint:

```text
artifacts/metric_scale/checkpoints/metric_scale_omninocs_imagegroup_train2000_holdout500.pt
```

Held-out result:

```text
mean_abs_pct=2.17
median_abs_pct=1.53
axis_mean_abs_pct=[2.33, 2.24, 1.93]
mean_abs_cm=0.34
axis_mean_abs_cm=[0.43, 0.33, 0.25]
```

Category baseline:

```text
mean_abs_pct=12.90
mean_abs_cm=1.58
```

### Scene-Heldout Scene 6

Checkpoint:

```text
artifacts/metric_scale/checkpoints/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200.pt
```

Cached split:

```text
train=12882
heldout=3228
```

Held-out result:

```text
mean_abs_pct=3.23
median_abs_pct=1.45
axis_mean_abs_pct=[3.04, 2.86, 3.79]
mean_abs_cm=0.56
axis_mean_abs_cm=[0.69, 0.54, 0.44]
```

Category baseline:

```text
mean_abs_pct=10.90
mean_abs_cm=1.54
```

Note: the planning dry split counted 12890 training records before cache-time
mask filtering. The persisted feature cache contains 12882 train features.

## Eval Commands

Image-grouped:

```bash
env LIDRA_SKIP_INIT=1 \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --load-feature-cache /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt \
  --load-checkpoint artifacts/metric_scale/checkpoints/metric_scale_omninocs_imagegroup_train2000_holdout500.pt \
  --eval-only \
  --device cuda \
  --metrics-output artifacts/metric_scale/metrics/metric_scale_omninocs_imagegroup_train2000_holdout500_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/metric_scale_omninocs_imagegroup_train2000_holdout500_manifest.json
```

Scene-heldout:

```bash
env LIDRA_SKIP_INIT=1 \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --load-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt \
  --load-checkpoint artifacts/metric_scale/checkpoints/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200.pt \
  --eval-only \
  --device cuda \
  --metrics-output artifacts/metric_scale/metrics/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/metric_scale_omninocs_sceneholdout_train12890_scene6_final_epoch200_manifest.json
```

Manifest validation example:

```bash
env LIDRA_SKIP_INIT=1 \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --load-feature-cache /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt \
  --validate-manifest artifacts/metric_scale/manifests/metric_scale_omninocs_imagegroup_train2000_holdout500_manifest.json \
  --load-checkpoint artifacts/metric_scale/checkpoints/metric_scale_omninocs_imagegroup_train2000_holdout500.pt \
  --eval-only \
  --device cuda
```
