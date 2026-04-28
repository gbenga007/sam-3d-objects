# Current Metric-Readout Design - 2026-04-22

This note documents the current implemented design for predicting physical object
dimensions. It is a metric readout head on top of frozen SAM3D/MoGe features, not
yet a scale-conditioned mesh generation system.

## Goal

Predict object dimensions in real-world units:

```text
[width, height, depth] in meters
```

Inference also returns the same dimensions in centimeters.

## Inputs Per Training Example

Training uses one object instance per example.

Each record contains:

```text
RGBA image:      [H, W, 4]
RGB channels:    full source frame
alpha channel:   binary mask for the selected object instance
metric_dims:     [3] = [width, height, depth] in meters
category:        object category label, used only for diagnostics/baselines
image_name:      source frame id, used for grouped splits
object_id:       instance id in the source frame
```

For OmniNOCS NOCS-Real275, a source frame can contain multiple object instances.
The dataset emits separate records for each instance. Those records share the
same RGB frame but have different alpha masks, object ids, categories, and
ground-truth dimensions.

## Forward Path

### 1. MoGe / Pointmap Preprocessing

MoGe produces metric pointmap information. The existing preprocessor normalizes
the pointmap and exposes:

```text
pointmap_scale:  [B, ...]
pointmap_shift:  [B, ...]
```

The metric head reduces these to:

```text
log(pointmap_scale mean): [B, 1]
pointmap_shift_z:         [B, 1]
```

### 2. Frozen SS Stage

The frozen sparse-structure generator produces:

```text
shape_latent: [B, 4096, 8]
```

The current metric head mean-pools over spatial positions:

```text
pooled_shape: [B, 8]
```

### 3. MetricScaleHead

Inputs:

```text
pooled_shape:        [B, 8]
log pointmap scale:  [B, 1]
pointmap shift z:    [B, 1]
```

Concatenated input:

```text
[B, 10]
```

Output:

```text
metric scale token: [B, 1, 768]
```

This token is a learned metric-context representation. It is not currently used
to alter SLAT generation in the validated training path.

### 4. Frozen SLAT Stage

The frozen SLAT generator runs with normal image conditioning and the SS coords.
It produces sparse latent features:

```text
slat_feats:    [num_voxels, 8]
batch_indices: [num_voxels]
```

The metric decoder mean-pools SLAT features per batch item:

```text
pooled_slat: [B, 8]
```

### 5. MetricScaleDecoder

The decoder combines the pooled SLAT descriptor with the metric scale token.

Metric token projection:

```text
metric scale token: [B, 1, 768]
squeeze:            [B, 768]
linear projection:  [B, 16]
```

Decoder input:

```text
pooled_slat:        [B, 8]
projected token:    [B, 16]
concatenated input: [B, 24]
```

Decoder output:

```text
log_dims: [B, 3]
dims_m:   exp(log_dims)       # meters
dims_cm:  dims_m * 100.0      # centimeters
```

## What Is Trained

Trainable:

```text
MetricScaleHead
MetricScaleDecoder
```

Frozen:

```text
MoGe
SS generator
SLAT generator
SAM3D decoders
condition embedders
pose/layout components
```

Loss:

```text
Smooth L1(log(predicted_dims_m), log(ground_truth_dims_m))
```

## What This Design Has Demonstrated

The current evidence supports the claim that this design is adequate for accurate
metric dimension prediction on the tested OmniNOCS NOCS-Real275 splits.

Best completed held-out results:

| Split | Train / Held-out | Held-out MAPE | Category Baseline MAPE | Notes |
|---|---:|---:|---:|---|
| Image-grouped | 2002 / 500 | 2.17% | 12.90% | All six NOCS categories represented |
| Scene-heldout | 12882 / 3228 cached | 3.09% best, 3.23% final | 10.90% | Held-out scene 6 lacks bottle/camera |

The learned model substantially beats category mean-size baselines, which is
evidence that it uses image/geometry-conditioned signal rather than only category
priors.

## Multi-Object Handling

Training is object-instance centric, not whole-scene multi-object prediction.

Important details:

- A frame may contain multiple objects.
- Each object instance becomes its own training record.
- The RGB content is the full frame, so other objects may still be visible.
- The alpha channel/mask selects the target object for SAM3D preprocessing.
- Grouped splits by `image` keep all instances from the same source image on the
  same side of train/held-out.
- Grouped splits by `scene` keep complete NOCS scenes on the same side.

This matches the SAM3D object pipeline, which operates on one masked object at a
time. Multi-object scenes are handled by running the object pipeline once per
object mask, then composing outputs at a higher level.

## What This Design Does Not Yet Do

It does not yet make the generated mesh itself metric-scale-aware.

The current validated path is:

```text
frozen SS + frozen SLAT features + metric readout heads -> metric dimensions
```

For the current mesh output behavior:

```text
stage-2 SLAT mesh output -> canonical / normalized object mesh
```

The decoded mesh lives in the model's canonical cube-like coordinate frame,
roughly centered in a normalized range around `[-0.5, 0.5]`. The exported GLB
is postprocessed and rotated for convention changes, but it is not rescaled
using the predicted metric dimensions.

Optional layout post-optimization may later apply scene/world transforms, but
that is a separate downstream alignment step and not the native stage-2 mesh
prediction.

So the current system should be understood as:

```text
canonical mesh prediction
+ separate metric dimension prediction
```

It is not yet:

```text
metric scale token -> SLAT cross-attention -> scale-conditioned mesh generation
```

There is an inference-time proxy that can append the metric scale token to SLAT
conditioning. In the current pointmap pipeline path, the token is injected before
stage-2 sampling, but the mesh decoder still outputs a canonical mesh and the
predicted metric dimensions are returned as a separate output:

```text
outputs["glb"]                  -> canonical mesh export
outputs["metric_dimensions"]    -> [width, height, depth] in meters
outputs["metric_dimensions_cm"] -> [width, height, depth] in centimeters
```

This means the present design already supports the decomposition:

```text
canonical geometry + separate metric-size estimate
```

That is a sensible target formulation for the next phase as well.

## Next Metric-Phase Work

- Add a durable artifact location for trained metric checkpoints outside `/tmp`.
- Add cache metadata validation for reproducibility.
- Run cross-scene folds so every scene becomes held out once.
- Add category-balanced or leave-one-category-out diagnostics.
- Add prediction-vs-ground-truth plots for each axis and category.
- If scale-aware mesh generation is desired, align the metric token dimension
  with SLAT condition tokens and train the SLAT conditioning path explicitly.
