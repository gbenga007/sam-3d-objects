# SLAT-Conditioned Training Changes
## 2026-05-02

This note records the changes made to make `train_slat_conditioned_v1` train
with live SLAT evaluation instead of a cached eval file, and to document the
current training mechanics.

## Files Changed

```text
sam3d_objects/training/finetune_metric_scale.py
scripts/train_slat_conditioned_v1.sh
planning/SLAT_CONDITIONED_TRAINING_MECHANICS_2026-05-02.md
planning/SLAT_CONDITIONED_TRAINING_CHANGES_2026-05-02.md
```

## Training Script Changes

File:

```text
scripts/train_slat_conditioned_v1.sh
```

Changes:

```text
removed --eval-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
added --heldout-samples 64
kept --eval-every 1
kept --unfreeze-slat-cross-attn
kept --p-uncond-scale-token 0.1
kept --log-step-every 1
updated comments to describe live held-out eval instead of cached frozen-SLAT eval
updated dataset paths to /mnt/source/datasets_sam3d/OmniNOCS
```

Reason:

```text
The eval cache file was missing, and cached eval would not measure the updated
SLAT cross-attention weights anyway. Live held-out eval now runs the current
pipeline and reflects the actual model being trained.
```

## Fine-Tuning Code Changes

File:

```text
sam3d_objects/training/finetune_metric_scale.py
```

### Removed live eval cache argument

Removed the separate live-training eval cache shortcut:

```text
--eval-feature-cache
```

The explicit cached training/eval modes remain available through:

```text
--cache-latents
--load-feature-cache
```

Reason:

```text
Live training should validate against the current pipeline, not stale cached
SLAT features.
```

### Added live held-out evaluation

Added:

```python
evaluate_live_dataset(...)
```

Behavior:

```text
runs held-out images through pointmap, SS, and SLAT
injects the current metric scale token
uses current SLAT cross-attention weights
runs under torch.no_grad()
sets scale-token dropout to 0.0 during eval
temporarily disables SLAT activation checkpointing for eval only
restores checkpointing and train/eval modes afterward
returns the same dimension-error metric schema used by cached eval
```

Metrics are written with split:

```text
live_heldout
```

W&B metrics use prefix:

```text
live_heldout_eval
```

Best checkpoint selection now uses live held-out:

```text
mean_abs_pct
```

### Fixed train/held-out splitting

Changed `make_train_eval_subsets(...)` so:

```text
--overfit-samples 0 --heldout-samples N
```

now trains on:

```text
len(dataset) - N
```

and reserves:

```text
N held-out examples
```

Previously, `--overfit-samples 0` consumed the full dataset and left no held-out
examples for live eval.

### Kept cached modes intact

The cached branch still uses cached features for:

```text
--cache-latents
--load-feature-cache
--eval-only
```

Only the separate `--eval-feature-cache` live-training shortcut was removed.

## Existing Stability Features Preserved

The current working tree already contains the TRELLIS-inspired stability work,
and these were preserved:

```text
AdaptiveGradClipper
gradient-level finite checks
scale-token dropout via --p-uncond-scale-token
per-step W&B logging via --log-step-every
SLAT block activation checkpointing during training
first-backward SLAT gradient presence check
rolling recovery checkpoints via --checkpoint-every
```

## Documentation Added

Added:

```text
planning/SLAT_CONDITIONED_TRAINING_MECHANICS_2026-05-02.md
```

It documents:

```text
what pipeline is trained
what each stage consumes and emits
what modules are updated
what modules remain frozen
what is predicted
what loss is used
how live eval works
why this is not full TRELLIS flow-matching training
```

Added this file:

```text
planning/SLAT_CONDITIONED_TRAINING_CHANGES_2026-05-02.md
```

to record the implementation changes.

## Verification Run

The following checks passed:

```bash
/root/.local/bin/micromamba run -n sam3d-objects python -m py_compile \
  sam3d_objects/training/finetune_metric_scale.py

/root/.local/bin/micromamba run -n sam3d-objects python \
  sam3d_objects/training/finetune_metric_scale.py --help

bash -n scripts/train_slat_conditioned_v1.sh
```

Also verified:

```text
--eval-feature-cache is no longer present in the CLI
no references remain to /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
GPU preflight: NVIDIA A100-SXM4-80GB, no active compute processes
pipeline config exists: checkpoints/hf/pipeline.yaml
warm-start checkpoint exists: artifacts/metric_scale/checkpoints/nocs_sceneholdout_1024dim_baseline_v2_best.pt
```

## Operational Result

The current launch is ready to run live training:

```bash
bash scripts/train_slat_conditioned_v1.sh
```

Expected behavior:

```text
train MetricScaleHead + MetricScaleDecoder at lr=1e-4
train SLAT cross_attn + norm2 at lr=1e-5
reserve 64 held-out examples
run live held-out eval each epoch
save best checkpoint by live held-out mean_abs_pct
save rolling recovery checkpoint each epoch
```
