# SS Architecture, PointPatchEmbed, and the Aspect Ratio Fix
## 2026-05-11

---

## How the SS Generator Works

The SS (Sparse Structure) generator is a **Multi-Output Token (MOT) flow-matching model**
that jointly generates 3D object geometry and pose from a single masked image + depth map.

### Condition Embedder (runs before any denoising)

Three input streams are embedded and concatenated into a single condition sequence:

```
1. DINOv2(image_crop, 518px)       → ~1369 tokens × 1024d   ← visual appearance
2. DINOv2(mask_crop, 518px)        → ~1369 tokens × 1024d   ← mask shape / silhouette
3. PointPatchEmbed(pointmap, 256px) → 1024 tokens × 512d    ← metric 3D depth
                                               ↓ projected to 1024d via EmbedderFuser
                    concat → [~3762 tokens, 1024d] = condition sequence
```

Config: `checkpoints/hf/ss_generator.yaml`
Class: `EmbedderFuser` → `sam3d_objects.model.backbone.dit.embedder.embedder_fuser`

The entire condition sequence is frozen during metric fine-tuning. It is computed once
per sample (not once per denoising step — it doesn't depend on the noisy latent).

### Backbone: SparseStructureFlowModel

24 × `MOTModulatedTransformerCrossBlock` blocks (model_channels=1024, cond_channels=1024,
num_heads=16). Each block:
- **self-attn** on the multi-token latent state
- **cross-attn** to the full condition sequence (all ~3762 tokens)
- **FFN** + adaLN modulation from the timestep embedding

Unlike SLAT, SS uses `use_fp16: false` — it runs in float32 throughout.

### Multi-Token Outputs

SS jointly denoises **five latent streams** via a shared transformer:

| Stream | Tokens | Dims | Meaning |
|---|---|---|---|
| `shape` | 4096 | 8 | 3D occupancy structure — the voxel geometry |
| `scale` | 1 | 3 | log-SSI scale per axis (metric surrogate) |
| `translation` | 1 | 3 | object centre in camera coordinates |
| `6drotation_normalized` | 1 | 6 | object orientation |
| `translation_scale` | 1 | 1 | distance scale factor |

The `scale`, `rotation`, and `translation` streams share transformer parameters with each
other (via `latent_share_transformer`) but not with `shape`. This means pose and scale
information is jointly processed while shape has its own dedicated capacity.

**Supervision weights** (`loss_weights` in yaml):
- `translation`: 1.0 — primary supervised output
- `6drotation_normalized`: 0.1 — lightly supervised
- `scale`: 0.1 — lightly supervised (log-SSI, not metric)
- `shape`: 0 — **no direct reconstruction loss** (only flow-matching velocity)
- `translation_scale`: 0.0 — unsupervised

The `shape` stream has zero direct loss weight, meaning its training signal comes entirely
from the flow-matching objective — it learns to produce plausible-looking voxel topologies,
not metrically accurate ones.

### ShortCut Distillation

SS uses `ShortCut` (not standard Euler flow-matching). ShortCut trains a model to take
large denoising steps by distilling a multi-step Euler trajectory into 2 steps. Key
implications:

- Inference runs with `inference_steps: 2`
- Training involves a self-consistency loss and a CFG consistency target
- `ratio_cfg_samples_in_self_consistency_target: 0.25` — 25% of self-consistency
  targets use CFG-guided samples
- The condition embedder is called inside the ShortCut's inner loop (multiple CFG passes)

For fine-tuning, this means simply setting `--stage1-steps 1` (single Euler step) would
produce a model trained differently from the 2-step ShortCut distillation. Whether this
matters for aspect ratio supervision is an open question — the topology should be
determinable from 1 step.

---

## What PointPatchEmbed Does

`PointPatchEmbed` is the key bridge between MoGe's metric depth and the SS generator.

**Input:** The MoGe pointmap, cropped to the object bounding box — a `(B, 3, H, W)` tensor
of metric (x, y, z) world-space coordinates in metres.

**Processing:**
```
256×256 pointmap (x,y,z in metres)
  ↓ resize to 256×256
  ↓ PointRemapper (linear remap_output: normalises the value range)
  ↓ Linear(3 → 512) per pixel
  ↓ split into 8×8 patches → 32×32 = 1024 windows
  ↓ per-window: prepend CLS token, add positional embed, run intra-patch transformer
  ↓ extract CLS token from each window
→ 1024 tokens × 512d, projected to 1024d by EmbedderFuser
```

**What each token contains:** A summary of the metric 3D structure within one 8×8 pixel
patch of the depth map. The CLS token aggregates the (x,y,z) distribution inside that
patch via intra-window self-attention.

**What the full sequence encodes:**
- The spatial extent of the object in x, y, z (metres) — this IS aspect ratio
- The depth profile of the object surface
- Where MoGe has invalid/uncertain depth (via the `invalid_xyz_token` learned embedding)

**Critically:** The PointPatchEmbed output is a pre-denoising conditioning input, computed
before any SS backbone forward pass. It enters SS via cross-attention at every one of the
24 transformer blocks.

---

## The Key Insight: The Information Is Already There

**The metric depth, including per-axis object extent and aspect ratio, is already in the
SS condition sequence.** It is encoded in the 1024 PointPatchEmbed tokens that SS
cross-attention reads at every block.

The reason SS generates wrong aspect ratios is **not a lack of information** in the
condition. It is a **lack of supervision signal** that would teach the model to use that
information for aspect ratio fidelity.

SS was trained with:
- Flow-matching velocity loss on the `shape` stream → generates *plausible-looking* voxels
- Light supervision on `scale` (log-SSI), `translation`, `rotation` → pose estimation
- **No loss** that penalises the voxel topology for having wrong proportions

The cross-attention learned to read the PointPatchEmbed tokens for "does this image look
like a cup/camera/laptop?" not for "how wide vs tall is this object in real metric units?"

---

## Why the Naive "Inject a New Token" Approach Is Wrong

The earlier plan (inject a metric scale token into SS cross-attention, mirroring SLAT v2/v3)
would have:
1. Created a circular dependency (MetricScaleHead needs SS outputs → can't pre-compute)
2. Added redundant information (PointPatchEmbed already carries metric extent)
3. Tried to fix a supervision problem by changing the architecture

Even with a pre-SS scale token (MoGe-only), we'd be injecting a summary statistic of
information that is already present in richer form (the full 1024-token PointPatchEmbed
sequence). The cross-attention can already read that.

---

## The Right Fix: Direct Aspect Ratio Supervision on SS Output

Add a loss term on the **SS soft occupancy output** that penalises incorrect voxel
bounding-box proportions relative to GT [W, H, D].

### Why soft occupancy, not hard coords

The hard voxel coords are produced by `argwhere(ss > 0)` — a non-differentiable step.
The **soft** output of the SS decoder (before thresholding) is differentiable.

The SS decoder maps `shape_latent [B, 8, 16, 16, 16]` → `ss [B, 1, 64, 64, 64]` (logits).
These logits are used for thresholding. A differentiable proxy for the bounding box extent
can be computed from the soft logit volume.

### Differentiable aspect ratio loss — implemented 2026-05-11

The loss lives in `compute_ss_aspect_ratio_loss` in `finetune_metric_scale.py`.

**Training target:** `log_gt_norm` — GT dims `[W, H, D]` in metres, sorted descending,
log-normalised to zero mean. Example: GT `[0.30, 0.20, 0.10]` →
`log = [-1.20, -1.61, -2.30]` → normalised `[+0.50, +0.09, -0.60]`. Scale-free.

**Prediction:** Per-axis soft standard deviation of the voxel occupancy volume.

```python
probs = sigmoid(ss_logits).squeeze(1)          # [B, 64, 64, 64]

# Marginal distributions along each voxel axis
p_d = probs.sum(dim=[H, W])                    # [B, 64]
p_h = probs.sum(dim=[D, W])
p_w = probs.sum(dim=[D, H])

# Soft centroid and variance per axis
mean_d = Σ(p_d * d_idx) / Σ(p_d)
var_d  = Σ(p_d * (d_idx - mean_d)²) / Σ(p_d)
# (same for h, w)

log_ext = log([√var_w, √var_h, √var_d])       # [B, 3]

# Rank-sort descending — largest voxel spread matches largest GT dim
log_ext_sorted, _ = log_ext.sort(dim=1, descending=True)
log_gt_sorted, _  = log(gt_dims).sort(dim=1, descending=True)

# Normalise to remove global scale (aspect ratio only)
log_ext_norm = log_ext_sorted - mean(log_ext_sorted)
log_gt_norm  = log_gt_sorted  - mean(log_gt_sorted)

ratio_loss = smooth_l1(log_ext_norm, log_gt_norm)
total_loss = metric_dim_loss + lambda_ratio * ratio_loss
```

`lambda_ratio` (`--ss-ratio-loss-weight`) should start at 0.01–0.1.
The rank-sort handles axis permutation ambiguity — no fixed mapping between
voxel XYZ and GT W/H/D is assumed.

### What this requires

- **SS backbone stays frozen** — shape_latent is detached before the decoder call
- **SS decoder is unfrozen** (`--unfreeze-ss-decoder`) — the convolutional 16³→64³
  decoder receives gradients and learns to map frozen backbone latents to
  proportionally accurate occupancy volumes
- The SS decoder is called **separately** (second forward pass on detached
  `shape_latent`) — no change to `sample_sparse_structure` needed

### Gradient path

```
ratio_loss → log_ext_norm → var_w/h/d → probs → sigmoid(ss_logits)
                                                          ↓
                                              SS decoder weights only
```

The SS backbone is not in this path. Whether the decoder can fix aspect ratios
from the frozen latent alone depends on how much aspect-ratio information is
encoded in `shape_latent`. A future `--unfreeze-ss-cross-attn` step would
propagate gradients through the backbone cross-attention to PointPatchEmbed.

### Alternative: Supervise the `scale` stream directly

The SS model outputs a `scale` token `[1, 3]` (log-SSI per axis, loss weight 0.1).
This is already a feature of the existing training. Adding metric GT supervision
to this stream is a lower-effort alternative to the soft occupancy approach,
but it only teaches the pose head — not the shape decoder.

---

## Relationship to Existing Training Infrastructure

**Implemented (2026-05-11):**
- `collect_ss_decoder_params(pipeline)` — unfreezes SS decoder, returns param list
- `compute_ss_aspect_ratio_loss(ss_logits, gt_dims)` — the soft-variance ratio loss
- `--ss-ratio-loss-weight` flag (default 0.0, disabled by default)
- `--unfreeze-ss-decoder` flag — adds SS decoder to optimizer at `args.lr`
- `predict_log_dims` now returns `(log_dims, dropped, ss_ratio_loss)` 3-tuple
- SS decoder called separately inside `predict_log_dims` (no `sample_sparse_structure` changes)

**Future step — `--unfreeze-ss-cross-attn`:**
- Analogous to `--unfreeze-slat-cross-attn` but for SS backbone
- Requires modifying `sample_sparse_structure` to support `with_grad=True`
- Propagates gradients through SS cross-attention → PointPatchEmbed
- More powerful: teaches the backbone to use metric depth for proportional shapes
- SS runs float32 already — no fp32 casting needed (unlike SLAT)

---

## Open Questions

1. **Near-empty volume guard:** If SS outputs nearly-empty occupancy, soft variance
   is ill-defined. May need `if probs.sum() < threshold: skip ratio loss`.

2. **Does the SS backbone latent encode enough aspect-ratio info?** If not, the
   decoder-only approach won't converge. Overfit test on 10 samples will show this.

3. **`scale` stream correlation with GT aspect ratios** — undiagnosed. Could run
   a quick Pearson correlation on the training set.

---

## Summary

| What | Detail |
|---|---|
| SS condition | DINOv2 (image) + DINOv2 (mask) + PointPatchEmbed (metric depth) |
| PointPatchEmbed | 1024 tokens of metric (x,y,z) structure — per-axis extent = aspect ratio |
| Why SS has wrong proportions | No supervision signal on shape proportions; flow-matching loss only |
| Wrong fix | Inject new metric token — circular dependency + redundant with PointPatchEmbed |
| Right fix (v1, implemented) | Soft-variance aspect ratio loss on SS decoder output + `--unfreeze-ss-decoder` |
| Right fix (v2, future) | Propagate loss through SS backbone cross-attn via `--unfreeze-ss-cross-attn` |
| Training target | Rank-sorted log-normalised GT dims `[W, H, D]` — scale-invariant aspect ratio |
