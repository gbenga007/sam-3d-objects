# Metric Scale Experiments - 2026-04-21

This note records the metric-scale head experiments run against
`sam3d_objects/training/finetune_metric_scale.py`.

## Goal

Validate whether the proposed metric-scale path is learning the intended target:
physical object dimensions in meters from frozen SAM3D features plus MoGe metric
pointmap statistics.

The trainable pieces are:

- `MetricScaleHead`: pools SS shape latent and combines it with MoGe
  `pointmap_scale` and `pointmap_shift`.
- `MetricScaleDecoder`: mean-pools SLAT features and predicts log-space
  `[width, height, depth]`.

The SAM3D pipeline is frozen. The loss is Smooth L1 in log-dimension space.

## Environment

Commands were run from:

```bash
/mnt/source/sam-3d-objects
```

Environment:

```bash
/root/.local/bin/micromamba run -n sam3d-objects ...
```

Attention overrides were required because the locally built `flash-attn` wheel
is not usable on this host due to a glibc mismatch:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa
```

GPU observed by the pipeline:

```text
NVIDIA A100-SXM4-80GB
```

## Code Changes

`sam3d_objects/training/finetune_metric_scale.py` now supports:

- `--cache-latents`: run the frozen SAM3D/MoGe feature extraction once, then
  train only the metric heads on cached tensors.
- `--heldout-samples`: reserve examples after the train split for cached eval.
- `--seed` and `--shuffle-split`: reproducible deterministic random split.
- `--eval-every`: report train and held-out cached metrics during training.

Cached tensors per object:

- SS shape latent.
- MoGe pointmap scale.
- MoGe pointmap shift.
- SLAT features.
- SLAT batch indices.
- Ground-truth metric dimensions.
- UID/category/image metadata for traceability.

The cached path is fast because it removes repeated frozen SAM3D sampling from
every epoch. After caching, each epoch is only the lightweight MLP scale head and
decoder.

## Experiment 1: Non-Cached 10-Sample Overfit

Purpose: smoke-test the full pipeline and verify gradients flow.

Command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --overfit-samples 10 \
  --max-records 10 \
  --epochs 20 \
  --batch-size 1 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --output /tmp/metric_scale_overfit_10.pt
```

Result:

```text
epoch 1 final loss:  ~1.63
epoch 20 final loss: ~0.0746
checkpoint: /tmp/metric_scale_overfit_10.pt
```

Same-set eval from that checkpoint:

```text
mean_smooth_l1_log_loss=0.073801
mean_abs_pct_error=32.00%
median_abs_pct_error=21.53%
axis_mean_abs_pct_error=[37.9, 32.01, 26.09]
```

Interpretation:

The wiring worked and the target was learnable, but this was not yet a clean
memorization test. The run repeatedly resampled frozen SAM3D features and used
only 20 epochs.

## Experiment 2: Cached 10-Sample True Overfit

Purpose: remove sampling noise and prove the metric heads can memorize the
target when SAM3D features are fixed.

Command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --overfit-samples 10 \
  --max-records 10 \
  --epochs 300 \
  --batch-size 10 \
  --lr 0.001 \
  --weight-decay 0 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --eval-every 25 \
  --output /tmp/metric_scale_overfit_10_cached.pt
```

Result:

```text
initial training loss: ~1.52
epoch 250 eval: loss=0.000889, mean_abs_pct=3.56%, median_abs_pct=2.93%
epoch 275 eval: loss=0.000684, mean_abs_pct=3.06%, median_abs_pct=2.51%
epoch 300 eval: loss=0.000524, mean_abs_pct=2.61%, median_abs_pct=2.04%
checkpoint: /tmp/metric_scale_overfit_10_cached.pt
```

Interpretation:

This is a successful true overfit. The trainable metric heads can fit the metric
dimension labels from fixed SAM3D/MoGe features. This validates the learning
target, gradient path, and basic label scale.

## Experiment 3: Cached 80 Train / 20 Held-Out Object Split

Purpose: test whether the same metric heads learn signal that transfers beyond
the exact cached training objects.

Important limitation: this is an instance-level held-out split, not a strict
scene-level split. It uses deterministic random object indices from the first
120 records. Future tests should add grouped scene/image splits.

Command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --max-records 120 \
  --overfit-samples 80 \
  --heldout-samples 20 \
  --shuffle-split \
  --seed 42 \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --eval-every 25 \
  --output /tmp/metric_scale_cached_train80_holdout20.pt
```

Selected metrics:

```text
epoch 100:
  train_eval   loss=0.013475, mean_abs_pct=13.98%, median_abs_pct=12.01%
  heldout_eval loss=0.022285, mean_abs_pct=18.62%, median_abs_pct=14.41%

epoch 125:
  train_eval   loss=0.009645, mean_abs_pct=11.01%, median_abs_pct=8.35%
  heldout_eval loss=0.020439, mean_abs_pct=16.57%, median_abs_pct=14.00%

epoch 150:
  train_eval   loss=0.008101, mean_abs_pct=9.55%, median_abs_pct=7.46%
  heldout_eval loss=0.017020, mean_abs_pct=14.21%, median_abs_pct=12.11%

epoch 175:
  train_eval   loss=0.006287, mean_abs_pct=8.73%, median_abs_pct=7.25%
  heldout_eval loss=0.017271, mean_abs_pct=14.48%, median_abs_pct=12.37%

epoch 200:
  train_eval   loss=0.005869, mean_abs_pct=8.62%, median_abs_pct=7.14%
  heldout_eval loss=0.018027, mean_abs_pct=15.09%, median_abs_pct=11.69%
```

Final checkpoint:

```text
/tmp/metric_scale_cached_train80_holdout20.pt
```

Interpretation:

The held-out object error improves from the earlier 10-sample non-cached result
and reaches roughly 14% mean absolute percentage error around epoch 150. After
that, train error continues improving while held-out mean error starts to drift
up. That is the first sign of ordinary overfitting in this metric head setup.

This is a useful result: the model is not merely memorizing labels in the 80/20
setting, but the current head/data setup is still limited. The next higher-value
step is to make the split stricter and less noisy.

## Intuition

The cached 10-sample run answers: "Can this architecture learn the intended
quantity at all?" Yes.

The cached 80/20 run answers: "Is there useful signal in these frozen features
for unseen objects?" Some, yes. The held-out error is materially lower than the
non-cached 10-sample eval and tracks training progress until about epoch 150.

The remaining gap likely comes from a mix of:

- small training set;
- simple mean-pooling of SLAT features;
- instance-level split leakage/variance;
- object-category size priors dominating some categories;
- only one-step SS/SLAT sampling for speed;
- no explicit category embedding or image/frame grouping logic.

## Next Steps

1. Add grouped split modes:
   - by `image_name`;
   - by scene;
   - optionally by category for leave-one-category-out diagnostics.
2. Save the cached feature tensors to disk so repeated head experiments do not
   need to rerun the frozen SAM3D pipeline.
3. Add per-category metrics and per-axis metrics to `evaluate_cached_features`.
4. Try larger cached train sets, for example 500 train / 100 held-out.
5. Use epoch 150 as the current early-stopping reference for this 80/20 setup.
6. Compare the current mean-pooled decoder against a small attention-pooling
   decoder over SLAT features.
