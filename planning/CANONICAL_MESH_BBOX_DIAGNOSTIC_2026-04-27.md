# Canonical Mesh Bounding Box Diagnostic - 2026-04-27

## Purpose

Empirically determine whether the FlexiCubes canonical mesh output of SAM3D fills
the unit cube uniformly, and how the canonical bounding box relates to ground-truth
metric dimensions. This answers the question:

> **Is predicting [W, H, D] equivalent to predicting a direct per-axis mesh scale?**

---

## Script

```
scripts/check_canonical_mesh_bbox.py
```

Runs the full SAM3D inference pipeline (MoGe → SS → SLAT, 1 step each) on a sample
of NOCS-Real275 objects, extracts the raw FlexiCubes vertex bounding box (before any
post-processing), and compares it against ground-truth metric WHD from OmniNOCS.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | OmniNOCS NOCS-Real275 |
| Annotations root | `/mnt/dest/OmniNOCS/omninocs_release_nocs_real275` |
| RGB root | `/mnt/dest/OmniNOCS/real_test` |
| Split | train |
| N samples | 25 (spread across 6 categories) |
| Stage 1 steps | 1 |
| Stage 2 steps | 1 |
| Device | NVIDIA A100-SXM4-80GB |
| Seed | 42 |
| Raw output | `/tmp/canonical_bbox_check.json` |
| Config | `checkpoints/hf/pipeline.yaml` |

Samples were selected by stratified sampling across categories (≈4–5 per category,
deterministically with seed 42).

---

## Samples

| Category | Image path | Canonical bbox [x, y, z] | max(bbox) | Real [W, H, D] (m) |
|---|---|---|---|---|
| bottle | nocs_real275/test/scene_5/0115 | [0.733, 0.761, 0.994] | 0.9945 | [0.091, 0.092, 0.159] |
| bottle | nocs_real275/test/scene_5/0045 | [0.709, 0.715, 1.000] | 0.9997 | [0.091, 0.092, 0.159] |
| bottle | nocs_real275/test/scene_5/0510 | [0.317, 0.337, 1.000] | 1.0003 | [0.054, 0.053, 0.183] |
| bottle | nocs_real275/test/scene_4/0476 | [0.376, 0.375, 1.000] | 1.0003 | [0.054, 0.053, 0.183] |
| bowl | nocs_real275/test/scene_1/0316 | [0.999, 0.999, 0.333] | 0.9994 | [0.143, 0.139, 0.066] |
| bowl | nocs_real275/test/scene_3/0490 | [1.001, 1.000, 0.563] | 1.0007 | [0.172, 0.171, 0.087] |
| bowl | nocs_real275/test/scene_5/0095 | [0.998, 0.999, 0.459] | 0.9989 | [0.143, 0.139, 0.066] |
| bowl | nocs_real275/test/scene_6/0086 | [0.994, 1.000, 0.461] | 1.0003 | [0.156, 0.157, 0.077] |
| camera | nocs_real275/test/scene_5/0466 | [0.999, 0.571, 0.672] | 0.9986 | [0.087, 0.152, 0.105] |
| camera | nocs_real275/test/scene_2/0430 | [0.904, 0.519, 0.997] | 0.9972 | [0.067, 0.104, 0.113] |
| camera | nocs_real275/test/scene_5/0145 | [0.999, 0.504, 0.652] | 0.9991 | [0.087, 0.152, 0.105] |
| camera | nocs_real275/test/scene_3/0125 | [0.994, 0.570, 0.654] | 0.9942 | [0.087, 0.152, 0.105] |
| can | nocs_real275/test/scene_6/0412 | [0.456, 0.453, 1.000] | 0.9995 | [0.057, 0.057, 0.130] |
| can | nocs_real275/test/scene_4/0040 | [0.434, 0.430, 1.001] | 1.0006 | [0.069, 0.071, 0.156] |
| can | nocs_real275/test/scene_6/0405 | [0.457, 0.433, 0.998] | 0.9984 | [0.057, 0.057, 0.130] |
| can | nocs_real275/test/scene_6/0123 | [0.491, 0.499, 1.000] | 0.9999 | [0.057, 0.057, 0.130] |
| cup | nocs_real275/test/scene_3/0193 | [0.996, 0.748, 0.974] | 0.9964 | [0.114, 0.086, 0.106] |
| cup | nocs_real275/test/scene_1/0217 | [0.999, 0.844, 0.912] | 0.9989 | [0.107, 0.074, 0.101] |
| cup | nocs_real275/test/scene_4/0233 | [0.998, 0.995, 0.739] | 0.9979 | [0.127, 0.095, 0.092] |
| cup | nocs_real275/test/scene_3/0434 | [0.998, 0.998, 0.895] | 0.9978 | [0.114, 0.086, 0.106] |
| laptop | nocs_real275/test/scene_5/0019 | [0.984, 0.999, 0.673] | 0.9992 | [0.418, 0.335, 0.216] |
| laptop | nocs_real275/test/scene_3/0327 | [0.999, 0.904, 0.620] | 0.9988 | [0.383, 0.318, 0.148] |
| laptop | nocs_real275/test/scene_5/0469 | [0.995, 0.928, 0.604] | 0.9953 | [0.418, 0.335, 0.216] |
| laptop | nocs_real275/test/scene_4/0111 | [0.917, 0.999, 0.586] | 0.9988 | [0.383, 0.318, 0.148] |
| laptop | nocs_real275/test/scene_2/0628 | [0.999, 0.943, 0.691] | 0.9987 | [0.307, 0.279, 0.169] |

---

## Per-Category Summary

| Category | n | Canonical bbox mean [x, y, z] | max(bbox) mean | Real mean [W, H, D] (m) |
|---|---|---|---|---|
| bottle | 4 | [0.533, 0.547, 0.999] | 0.9987 | [0.072, 0.073, 0.171] |
| bowl | 4 | [0.998, 1.000, 0.454] | 0.9998 | [0.153, 0.152, 0.074] |
| camera | 4 | [0.974, 0.541, 0.744] | 0.9973 | [0.082, 0.140, 0.107] |
| can | 4 | [0.459, 0.454, 1.000] | 0.9996 | [0.060, 0.061, 0.137] |
| cup | 4 | [0.998, 0.896, 0.880] | 0.9978 | [0.115, 0.085, 0.102] |
| laptop | 5 | [0.979, 0.954, 0.635] | 0.9982 | [0.381, 0.317, 0.179] |

The category-specific patterns confirm the geometry is physically sensible:

- **Bottles/cans**: tall and narrow → Z≈1.0, X,Y≈0.3–0.5
- **Bowls**: wide and flat → X,Y≈1.0, Z≈0.33–0.56
- **Cameras**: widest in X → X≈1.0, Y and Z reflect the body shape
- **Cups**: roughly symmetric, all axes ≈ 0.75–1.0
- **Laptops**: wide and flat (closed) → X,Y≈1.0, Z≈0.6 (overestimates thickness; expected from monocular ambiguity)

---

## Aggregate Statistics

### Canonical bounding box axes (N=25)

| Axis | mean | std | min | max | % within ±0.2 of 1.0 |
|---|---|---|---|---|---|
| X | 0.830 | 0.243 | 0.317 | 1.001 | 68% |
| Y | 0.741 | 0.240 | 0.337 | 1.000 | 48% |
| Z | 0.779 | 0.211 | 0.333 | 1.001 | 48% |

### max(canonical_bbox) — the key invariant

| Statistic | Value |
|---|---|
| mean | 0.9985 |
| std | 0.0018 |
| min | 0.9942 |
| max | 1.0007 |
| % within ±0.01 of 1.0 | **100%** |
| % within ±0.05 of 1.0 | **100%** |

### Canonical centroid (should be near 0)

| Axis | mean | std |
|---|---|---|
| X | −0.000 | 0.001 |
| Y | −0.003 | 0.012 |
| Z | +0.002 | 0.004 |

Mesh is correctly centered at the origin.

### Canonical-to-metric scale ratio (max_bbox / max_real)

| Statistic | Value |
|---|---|
| mean | 6.225 |
| std | 2.058 |
| CV | 0.331 |

The scale from canonical units to real-world metres varies by ~33% across objects.

### Correlation: canonical bbox axis ↔ real WHD

| Pair | Correlation |
|---|---|
| corr(bbox_x, W_real) | 0.484 |
| corr(bbox_y, H_real) | 0.586 |
| corr(bbox_z, D_real) | 0.361 |

Moderate correlation: canonical axes are informative about real dimensions but
do not determine them (absolute scale is image-dependent, not shape-dependent).

### Aspect ratio preservation error |canonical_ar − real_ar|

Where `canonical_ar = [bx, by, bz] / max(bx, by, bz)` and `real_ar = [W, H, D] / max(W, H, D)`.

| Axis | mean | std |
|---|---|---|
| X | 0.088 | 0.143 |
| Y | 0.145 | 0.149 |
| Z | 0.051 | 0.067 |

The canonical mesh does not perfectly reproduce the real object's aspect ratio, especially
in Y (height), where monocular depth is most ambiguous. Mean error is 5–15% of the
largest dimension.

---

## Conclusions

### Finding 1: Hard normalization invariant

**`max(canonical_bbox) = 1.000 ± 0.002` for every object.**

This is a geometric property of FlexiCubes operating in `[-0.5, 0.5]³` (via
`v_pos / res - 0.5` in `utils_cube.py`). The canonical mesh always extends to exactly
1.0 in its widest dimension. The shorter axes encode the object's predicted aspect ratio.

### Finding 2: The canonical-to-metric scale is instance-dependent

The factor `max(canonical_bbox) / max(W, H, D) ≈ 6.2 ± 2.1 (CV=0.33)` varies
substantially across instances. It cannot be baked in as a constant — it must be
predicted from the image. This is exactly what `MetricScaleHead` does via
`log(pointmap_scale)` and the SS latent.

### Finding 3: Predicting [W, H, D] is correct; per-axis mesh scaling is not

Because the canonical mesh encodes the object's aspect ratio (imperfectly, with
mean error 0.05–0.15), applying predicted [W, H, D] as per-axis stretch factors to
the canonical vertices would distort the shape. The right approach is:

```
s = max(W_pred, H_pred, D_pred)          # single metric scale
metric_vertices = canonical_vertices * s   # isotropic scaling
```

This works because `max(canonical_bbox) ≈ 1.0`, so `s` converts canonical units to
metres without introducing aspect-ratio distortion.

### Finding 4: [W, H, D] prediction is strictly better than predicting scalar `s`

Predicting all three dimensions:
- provides 3× richer supervision signal
- enables per-axis error reporting (useful for diagnostics and ablations)
- gives s for free: `s = max(W_pred, H_pred, D_pred)`
- captures any residual aspect-ratio information not present in the shape alone

There is no benefit to replacing [W, H, D] with a single scalar target.

---

## Implications for Current Training

The existing `MetricScaleDecoder` predicting `log([W, H, D])` is the correct target.

At inference time, the metric mesh is recovered as:

```python
# After training:
log_WHD = metric_scale_decoder(slat_feats, scale_token)  # [B, 3]
WHD = torch.exp(log_WHD)                                  # metres
s = WHD.max(dim=-1).values                                # scalar per instance

# Applying to mesh:
metric_vertices = canonical_vertices * s  # scales canonical mesh to real-world size
```

No bounding-box lookup is needed at inference time. The `max()` operation is sufficient
and correct given the `max(canonical_bbox) ≈ 1.0` invariant.
