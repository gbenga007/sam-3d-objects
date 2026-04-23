# Metric-Scale Artifacts

This directory is for durable, reproducible metric-scale experiment outputs.

Track in git:

- lightweight metric-head checkpoints (`*.pt`, currently about 524 KB each);
- manifest JSON files;
- metric JSONL summaries;
- short reports and command notes.

Do not track in git:

- frozen SAM3D/MoGe feature caches (`*_cache.pt`, often 2-13 GB);
- full model checkpoints from upstream SAM3D;
- generated meshes or large rendered media.

Feature caches should remain in external storage such as `/tmp`, `/mnt/dest`, or
object storage, with their paths, sizes, splits, and tensor schema recorded in a
manifest.

Suggested layout:

```text
artifacts/metric_scale/
  checkpoints/
  manifests/
  metrics/
  reports/
```
