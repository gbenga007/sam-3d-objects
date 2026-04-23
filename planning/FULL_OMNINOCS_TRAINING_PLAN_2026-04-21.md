# Full OmniNOCS Metric-Scale Training Plan - 2026-04-21

## Decision

Move from tiny overfit experiments to real OmniNOCS training, but do it through
a persistent cached-feature workflow rather than a long uncached training job.

The previous experiments justify this:

- 10-sample cached overfit reached roughly 2.6% mean absolute percentage error.
- 80 train / 20 held-out cached split reached roughly 14.2% held-out mean
  absolute percentage error near epoch 150.

That means the metric heads can learn the target and there is some signal that
transfers to held-out objects. The next experiment should test this on a larger
and cleaner split.

## Guardrails

1. Use grouped validation splits.
   - Object-level random splits are useful for debugging but can leak frame or
     scene context.
   - The next training run should support grouping by `image_name` or scene.

2. Persist frozen SAM3D/MoGe features.
   - The expensive part is:
     `image -> MoGe pointmap -> SS sampling -> SLAT sampling`.
   - This should be computed once and saved.
   - Head training should then load cached tensors and iterate quickly.

3. Track meaningful metrics.
   - Train and validation Smooth L1 loss in log-dimension space.
   - Mean and median absolute percentage error.
   - Per-axis error.
   - Per-category error.
   - A simple category mean-size baseline for context.

4. Keep the first full run bounded.
   - Start with `stage1_steps=1` and `stage2_steps=1`.
   - Use deterministic splits.
   - Use early-stopping intuition from the 80/20 experiment: validation may peak
     before the final epoch.

## Implementation Tasks

1. Extend `finetune_metric_scale.py`.
   - Add split grouping by record order, `image_name`, or scene.
   - Add save/load of cached feature tensors.
   - Add a cache-only mode so feature extraction can be run separately.
   - Add train-only-from-cache mode.
   - Add per-axis/per-category metrics.
   - Add category mean-size baseline metrics.

2. Run a bounded full-training cache.
   - Use as many OmniNOCS records as feasible for the current machine/session.
   - Prefer grouped validation by scene or image.
   - Save cache files under `/tmp` first to avoid polluting the repo.

3. Train from cache.
   - Initial settings:

```bash
--epochs 200
--batch-size 16
--lr 0.001
--weight-decay 0
--stage1-steps 1
--stage2-steps 1
```

4. Record results.
   - Add exact commands, cache paths, checkpoint paths, metrics, and
     interpretation back into `planning/`.

## Initial Command Shape

Feature cache:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --max-records N \
  --overfit-samples TRAIN_N \
  --heldout-samples VAL_N \
  --split-group scene \
  --shuffle-split \
  --seed 42 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --save-feature-cache /tmp/metric_scale_cache.pt \
  --cache-only
```

Training from cache:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --load-feature-cache /tmp/metric_scale_cache.pt \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0 \
  --device cuda \
  --eval-every 25 \
  --output /tmp/metric_scale_full.pt
```

## Success Criteria

Minimum useful outcome:

- Feature cache can be saved and reloaded.
- Training from cache reproduces train/validation metrics.
- Validation error improves over category mean-size baseline.

Strong outcome:

- Scene- or image-grouped validation mean absolute percentage error is below
  15-20% with stable per-category behavior.

Failure outcome:

- Validation does not beat the category baseline.
- Per-category errors reveal the head is mostly learning category priors rather
  than image/geometry-specific scale.

## Executed Run: Image-Grouped 500 / 121 Cache

This was the first bounded "full training path" run. It did not cache all 16,118
OmniNOCS train objects because feature extraction is still the expensive step.
Instead, it sampled across the full metadata with deterministic image grouping:
whole `image_name` groups were kept together, shuffled with seed 42, then enough
groups were selected to reach the requested train/held-out counts.

Dataset distribution available:

```text
frames: 2754
objects: 16118
object_scenes:
  scene_1: 1728
  scene_2: 2435
  scene_3: 2964
  scene_4: 3045
  scene_5: 2718
  scene_6: 3228
```

Feature-cache command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --overfit-samples 500 \
  --heldout-samples 120 \
  --split-group image \
  --shuffle-split \
  --seed 42 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --save-feature-cache /tmp/metric_scale_omninocs_imagegroup_train500_holdout120_cache.pt \
  --cache-only
```

Observed split/cache:

```text
Split group=image train=500 heldout=121
Saved feature cache to /tmp/metric_scale_omninocs_imagegroup_train500_holdout120_cache.pt
```

Training command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --load-feature-cache /tmp/metric_scale_omninocs_imagegroup_train500_holdout120_cache.pt \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0 \
  --device cuda \
  --eval-every 25 \
  --output /tmp/metric_scale_omninocs_imagegroup_train500_holdout120.pt
```

Category-mean held-out baseline:

```text
mean_abs_pct=12.85%
median_abs_pct=8.58%
axis_mean_abs_pct=[17.06, 12.35, 9.15]

by_category:
  bottle=15.55%
  bowl=7.21%
  camera=23.45%
  can=12.39%
  cup=7.13%
  laptop=10.85%
```

Final epoch 200 metrics:

```text
train_eval:
  loss=0.000673
  mean_abs_pct=2.81%
  median_abs_pct=2.28%
  axis_mean_abs_pct=[3.05, 2.61, 2.78]

train_eval_by_category:
  bottle=3.03%
  bowl=3.27%
  camera=3.35%
  can=1.84%
  cup=2.31%
  laptop=3.18%

heldout_eval:
  loss=0.001620
  mean_abs_pct=3.88%
  median_abs_pct=2.75%
  axis_mean_abs_pct=[4.06, 3.75, 3.83]

heldout_eval_by_category:
  bottle=4.61%
  bowl=4.17%
  camera=4.72%
  can=1.95%
  cup=2.84%
  laptop=5.41%
```

Checkpoint:

```text
/tmp/metric_scale_omninocs_imagegroup_train500_holdout120.pt
```

Interpretation:

This is a strong positive result for the metric-scale prototype. On an
image-grouped held-out split, the learned metric head reached 3.88% mean
absolute percentage error, compared with 12.85% from the category mean-size
baseline. That means the model is learning signal beyond category priors.

## Executed Run: Image-Grouped 2000 / 500 Cache

This run scaled the successful bounded training path from roughly 500/120 to
roughly 2000/500 while keeping the same cheap SAM3D inference settings. The
split was still grouped by `image_name`, shuffled with seed 42, so objects from
the same source image were not split between train and held-out.

Preflight split check:

```bash
env LIDRA_SKIP_INIT=1 \
/root/.local/bin/micromamba run -n sam3d-objects \
python -c "from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset; from sam3d_objects.training.finetune_metric_scale import make_train_eval_subsets; d=OmniNOCSReal275Dataset('/mnt/dest/OmniNOCS/omninocs_release_nocs_real275','/mnt/dest/OmniNOCS/real_test', split='train'); tr, ev=make_train_eval_subsets(d, 2000, 500, 42, True, 'image'); print(len(tr), len(ev))"
```

Observed:

```text
Split group=image train=2003 heldout=500
2003 500
```

Feature-cache command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --overfit-samples 2000 \
  --heldout-samples 500 \
  --split-group image \
  --shuffle-split \
  --seed 42 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --save-feature-cache /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt \
  --cache-only
```

Observed cache:

```text
Saved feature cache to /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt
train=2002
heldout=500
```

Training command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --load-feature-cache /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0 \
  --device cuda \
  --eval-every 25 \
  --metrics-output /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_metrics.jsonl \
  --output /tmp/metric_scale_omninocs_imagegroup_train2000_holdout500.pt
```

Category-mean held-out baseline:

```text
mean_abs_pct=12.90%
median_abs_pct=9.79%
axis_mean_abs_pct=[16.87, 12.57, 9.25]

by_category:
  bottle=16.55%
  bowl=9.00%
  camera=20.08%
  can=12.93%
  cup=6.51%
  laptop=10.65%
```

Evaluation schedule:

```text
heldout epochs: 25, 50, 75, 100, 125, 150, 175, 200
```

Best held-out result:

```text
epoch=200
loss=0.001126
mean_abs_pct=2.17%
median_abs_pct=1.53%
axis_mean_abs_pct=[2.33, 2.24, 1.93]

by_category:
  bottle=1.97%
  bowl=2.40%
  camera=1.55%
  can=1.47%
  cup=2.44%
  laptop=3.26%
```

Final train result:

```text
epoch=200
loss=0.000197
mean_abs_pct=1.56%
median_abs_pct=1.31%
axis_mean_abs_pct=[1.65, 1.63, 1.40]

by_category:
  bottle=1.32%
  bowl=2.23%
  camera=1.02%
  can=1.21%
  cup=2.01%
  laptop=1.72%
```

Artifacts:

```text
feature_cache=/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_cache.pt
metrics_jsonl=/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500_metrics.jsonl
checkpoint=/tmp/metric_scale_omninocs_imagegroup_train2000_holdout500.pt
```

Interpretation:

This is a stronger result than the 500/121 run. The held-out image-grouped error
improved from 3.88% to 2.17%, and the best checkpoint was still the final epoch
rather than an early peak. The category baseline stayed near 13%, so the learned
head is clearly using object/image-conditioned features rather than only
memorizing category mean dimensions. The train/held-out gap is modest
(`1.56%` vs `2.17%`), which supports moving to a larger split or a fuller
dataset pass when time/storage permits.

The result is still not a final benchmark:

- The cache uses 2002 train objects and 500 held-out objects, not all 16,118
  OmniNOCS train objects.
- The split is grouped by image, not by full scene.
- Stage-1 and stage-2 sampling used one step each for speed.
- Metrics are now written to JSONL, but this is still a bounded prototype run
  rather than a full dataset pass.

Next action from this point was to run a stricter scene-heldout split. That run
has now completed and is recorded below.

## Executed Run: Scene-Heldout Full-Scale Cache

This was the next escalation after the 2002/500 image-grouped run. It used
almost the whole OmniNOCS train split while making validation stricter: train on
complete scenes 1-5 and hold out complete scene 6.

Dry-run split command:

```bash
env LIDRA_SKIP_INIT=1 \
/root/.local/bin/micromamba run -n sam3d-objects \
python -c "from collections import Counter; from pathlib import Path; from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset; from sam3d_objects.training.finetune_metric_scale import make_train_eval_subsets; d=OmniNOCSReal275Dataset('/mnt/dest/OmniNOCS/omninocs_release_nocs_real275','/mnt/dest/OmniNOCS/real_test', split='train'); c=Counter(Path(r['image_name']).parts[-2] for r in d.records); print('dataset', len(d), sorted(c.items())); tr, ev=make_train_eval_subsets(d, 12000, 3000, 42, True, 'scene'); print('split_lens', len(tr), len(ev)); print('train_scenes', sorted(Counter(Path(d.records[i]['image_name']).parts[-2] for i in tr.indices).items())); print('heldout_scenes', sorted(Counter(Path(d.records[i]['image_name']).parts[-2] for i in ev.indices).items()))"
```

Dry-run result:

```text
dataset=16118
scene_1=1728
scene_2=2435
scene_3=2964
scene_4=3045
scene_5=2718
scene_6=3228

Split group=scene train=12890 heldout=3228
train_scenes: scene_1, scene_2, scene_3, scene_4, scene_5
heldout_scenes: scene_6
```

Feature-cache command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --overfit-samples 12000 \
  --heldout-samples 3000 \
  --split-group scene \
  --shuffle-split \
  --seed 42 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --cache-latents \
  --save-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt \
  --cache-only
```

Training command:

```bash
env ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --load-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0 \
  --device cuda \
  --eval-every 25 \
  --metrics-output /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_metrics.jsonl \
  --output /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6.pt
```

Artifacts:

```text
feature_cache=/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
metrics_jsonl=/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_metrics.jsonl
checkpoint=/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6.pt
```

Observed artifact sizes:

```text
feature_cache=13G
metrics_jsonl=6.4K
checkpoint=524K
```

Category-mean held-out baseline:

```text
mean_abs_pct=10.90%
median_abs_pct=8.67%
axis_mean_abs_pct=[9.49, 10.03, 13.17]

by_category:
  bowl=6.43%
  can=24.18%
  cup=6.95%
  laptop=14.46%
```

Evaluation schedule:

```text
heldout epochs: 25, 50, 75, 100, 125, 150, 175, 200
```

Best held-out result by mean absolute percentage error:

```text
epoch=175
loss=0.001166
mean_abs_pct=3.09%
median_abs_pct=1.80%
axis_mean_abs_pct=[2.69, 2.60, 3.97]

by_category:
  bowl=5.54%
  can=1.64%
  cup=1.60%
  laptop=2.62%
```

Final epoch 200 held-out result:

```text
epoch=200
loss=0.001580
mean_abs_pct=3.23%
median_abs_pct=1.45%
axis_mean_abs_pct=[3.04, 2.86, 3.79]

by_category:
  bowl=4.65%
  can=0.87%
  cup=1.93%
  laptop=5.37%
```

Final train result:

```text
epoch=200
loss=0.000070
mean_abs_pct=0.76%
median_abs_pct=0.52%
axis_mean_abs_pct=[0.88, 0.73, 0.68]

by_category:
  bottle=0.64%
  bowl=0.59%
  camera=0.57%
  can=0.43%
  cup=0.65%
  laptop=1.68%
```

Interpretation:

The stricter scene-heldout run is a strong positive result, but not a final
benchmark. The learned heads beat the scene-6 category baseline by a wide margin
(`3.09%` best held-out MAPE vs `10.90%` baseline), which supports the claim that
the cached SAM3D/MoGe features contain usable metric signal beyond category
mean-size priors.

The best held-out mean error occurred at epoch 175, while epoch 200 improved
median error but worsened mean error. The training script currently saves only
the final checkpoint, so best-checkpoint selection should be added before
treating this as a polished training workflow.

Important caveat: the scene-6 held-out split contains only bowl, can, cup, and
laptop examples. Bottle and camera appear in the training split but not in the
held-out scene. The previous image-grouped 2002/500 run remains the better
all-category held-out result, while the scene-heldout run is the stricter
cross-scene generalization check for the categories that appear in scene 6.

Recommended next actions:

1. Add best-checkpoint saving keyed to held-out mean absolute percentage error.
2. Run cross-scene folds so every scene becomes held-out once.
3. Add category-balanced or leave-one-category-out diagnostics for bottle and camera.
4. Backfill centimeter-scale error metrics for the saved checkpoints and add
   prediction-vs-ground-truth scatter plots.
5. Wire trained metric heads into the inference pipeline so metric dimensions are
   produced outside the training script.
