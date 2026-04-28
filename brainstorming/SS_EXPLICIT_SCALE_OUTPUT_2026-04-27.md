# SS Generator Explicit Scale Output - 2026-04-27

## Summary

The SS generator (Stage 1) is the MM-DiT model described in the SAM3D paper. It
explicitly outputs a supervised scale token that we are not currently using. This
is almost certainly a better input to MetricScaleHead than the mean-pooled shape
latent we use today.

---

## What the SS Generator Actually Outputs

The backbone is `SparseStructureFlowTdfyWrapper` with `latent_mapping` — a dict of
five separate latent streams, each independently predicted by the model:

| Key | Channels | Tokens | Description |
|---|---|---|---|
| `shape` | 8 | 4096 | Voxel shape latent |
| **`scale`** | **3** | **1** | **Object scale — log-space, SSI coords** |
| `6drotation_normalized` | 6 | 1 | 6D rotation |
| `translation` | 3 | 1 | 3D translation |
| `translation_scale` | 1 | 1 | Additional depth scale factor |

Confirmed in `checkpoints/hf/ss_generator.yaml`:

```yaml
latent_mapping:
  scale:
    _target_: ...mm_latent.Latent
    in_channels: 3
    model_channels: 1024
    pos_embedder:
      token_len: 1   # ← single token per object
```

Loss weights during training (`ss_generator.yaml`):

```yaml
loss_weights:
  shape: 0
  scale: 0.1        # ← explicitly supervised
  translation: 1.0
  6drotation_normalized: 0.1
  translation_scale: 0.0
```

Scale was directly trained against real-world pointmap-aligned annotations from the
SAM3D data engine's Stage 3, where annotators placed objects relative to a metric
point cloud. This is not a latent side effect — it is a first-class output.

---

## Where It Lives in the Codebase

After `pipeline.sample_sparse_structure()` returns, `ss_return_dict` contains all
five keys. The pose decoder (`inference_utils.py:pose_decoder`) then processes them:

```python
# inference_utils.py — pose_decoder
if "x_instance_scale" in pose_target_dict:
    pose_target_dict["x_instance_scale"] = torch.exp(
        pose_target_dict["x_instance_scale"]
    )
# Then SSI-to-metric transform using scene_scale (MoGe pointmap scale)
# Returns:
"scale": pose_instance_dict["instance_scale_l2c"].squeeze(0).mean(-1, keepdim=True).expand(1,3)
```

The main pipeline `__call__` method also rescales by `downsample_factor`:
```python
# inference_pipeline.py:511-513
if "scale" in ss_return_dict:
    ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
```

---

## What We Currently Do (and the Gap)

Our training script (`finetune_metric_scale.py`) and feature cache only ever access:

```python
ss_return_dict["shape"]   # [B, 4096, 8] → mean-pooled to [B, 8] → MetricScaleHead
```

The explicit scale token is read and then discarded. The cache saves only `"shape"`
and `"pointmap_scale"`. We never call `pipeline.pose_decoder(ss_return_dict)` in
the training path.

---

## How to Use It

The conversion from raw SS output to approximate metric scale is straightforward:

```python
# After sample_sparse_structure():
ssi_scale  = ss_return_dict["scale"].squeeze(1)          # [B, 3], log-space SSI
metric_s   = torch.exp(ssi_scale) * pointmap_scale       # [B, 3], approx. metres
```

`pointmap_scale` is already computed in our training pipeline — it's the same value
currently passed to MetricScaleHead as `log(pointmap_scale)`.

The resulting `metric_s` is a 3D vector in camera coordinates (metres). Because the
model predicts a scalar scale (uniform, not per-axis WHD), the three values of
`metric_s` are approximately equal and represent a single object size estimate in
metres — close to `max(W_real, H_real, D_real)`, which is the metric scale factor
we empirically confirmed maps canonical → metric (see
`planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md`).

---

## Proposed Architecture Change

Replace the mean-pooled shape latent in MetricScaleHead with the explicit scale
token. Three variants to ablate:

### Variant A — Replace (minimal change)

```python
# Current:
shape = ss_return_dict["shape"].mean(dim=1)   # [B, 8]
scale_token = scale_head(shape, pointmap_scale, pointmap_shift_z)

# Proposed:
ssi_scale = ss_return_dict["scale"].squeeze(1)           # [B, 3]
metric_s  = torch.exp(ssi_scale) * pointmap_scale        # [B, 3]
scale_token = scale_head(metric_s, pointmap_scale, pointmap_shift_z)
```

MetricScaleHead input changes from `[B, 8+1+1]` → `[B, 3+1+1]`. Much lower
dimensional but higher quality — the model was trained to put scale in this token.

### Variant B — Concatenate (no information loss)

```python
shape     = ss_return_dict["shape"].mean(dim=1)          # [B, 8]
ssi_scale = ss_return_dict["scale"].squeeze(1)           # [B, 3]
metric_s  = torch.exp(ssi_scale) * pointmap_scale        # [B, 3]
combined  = torch.cat([shape, metric_s], dim=-1)         # [B, 11]
scale_token = scale_head(combined, pointmap_scale, pointmap_shift_z)
```

### Variant C — Direct prediction (bypass MetricScaleHead entirely)

If the SS generator's scale is already metrically calibrated, the decoder could
read `metric_s` directly without a learned scale head:

```python
# Skip scale_head entirely:
scale_token = linear_projection(metric_s)   # [B, 1, 1024]
log_WHD     = scale_decoder(slat_feats, scale_token)
```

This is the most aggressive simplification. Viable only if `metric_s ≈ WHD` up to
a fixed affine transform, which requires empirical validation.

---

## Experiment Plan

**After the current SLAT-conditioned run completes:**

1. Run a quick diagnostic: log `ss_return_dict["scale"]` values for 50 NOCS-Real275
   samples alongside ground-truth WHD and compute correlation. If
   `corr(exp(ssi_scale) * pointmap_scale, max(W,H,D)) > 0.8`, proceed directly to
   Variant A.

2. Train Variant A (replace pooled shape latent with explicit scale token):
   - Same scene-heldout 12882/3228 split
   - Same 20 epochs, lr=1e-4
   - Frozen SLAT (no --unfreeze-slat-cross-attn) for a clean ablation
   - Compare MAPE to the frozen-SLAT 3.09% baseline and the current SLAT-conditioned run

3. If Variant A improves on baseline, train Variant B to check whether shape latent
   adds anything on top of the explicit scale token.

4. If Variant A is already near ceiling, try Variant C (direct projection, no
   learned scale head) as a simplicity check.

---

## Why This Matters

The mean-pooled shape latent `[B, 8]` contains information about object geometry
averaged across 4096 voxels. It was NOT trained to represent scale — it was trained
to represent shape structure. Any scale signal in it is incidental.

The `"scale"` token `[B, 1, 3]` was trained with an explicit loss (`scale: 0.1`)
to predict the object's size in SSI-normalized camera coordinates. Converting it to
metric via `* pointmap_scale` should give a direct, low-noise estimate of object
size in metres — exactly what MetricScaleHead is trying to recover by learning from
the pooled shape latent.

This change requires no new data, no new model components, and no architecture
redesign. It is purely a better use of information that is already computed and
discarded at every training step.
