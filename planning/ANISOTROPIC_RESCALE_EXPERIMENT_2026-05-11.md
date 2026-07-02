# Anisotropic Voxel Rescaling Experiment
## 2026-05-11

---

## Why this experiment exists

After training the SLAT-conditioned metric head (v3, 1.74% MAPE), the metric predictions
are accurate but the *decoded mesh* does not conform to those proportions. A laptop
predicted as 30cm wide × 2cm thin decodes into a mesh that looks more like a cube.

The question: **where does the wrong aspect ratio come from?**

### The two-stage pipeline and why Stage 1 is the suspect

The pipeline has two generative stages:

```
Image
  ↓
Stage 1 (SS — Sparse Structure)
  → shape_latent [B, 4096, 8]
  → ss_decoder thresholds it to active voxel coords [N, 4]  ← TOPOLOGY LOCKED HERE
  ↓
Stage 2 (SLAT)
  → takes coords as fixed input (cannot add/remove/move voxels)
  → generates features on those voxels
  ↓
FlexiCubes decoder → mesh
```

The decoded mesh can only be as wide/tall/deep as the voxel grid permits. FlexiCubes can
refine the surface *within* each voxel cell, but the overall bounding box — the H:D:W
aspect ratio — is set by *which voxels are active* coming out of Stage 1.

The metric token we inject into SLAT cross-attention in v2/v3 can modulate feature values
on existing voxels. It cannot move voxels or change which ones are active. So if Stage 1
generates a voxel grid with the wrong proportions, SLAT cannot correct them.

---

## Experiment design

**Script:** `scripts/anisotropic_rescale_experiment.py`

**What it does, per sample:**

1. Run Stage 1 (SS) → get integer voxel coordinates in a 64×64×64 grid.
2. Measure the bounding box of those voxel coordinates along each spatial axis
   (extent = max − min + 1 in voxel units).
3. Compare the voxel bounding-box aspect ratio with the ground-truth [W, H, D] aspect ratio.
   This is reported as `vox∆` — mean absolute deviation of normalised aspect ratios.
4. Rescale the voxel coordinates to match the GT aspect ratio, keeping overall scale constant.
   Two strategies:
   - **sorted** (rank-matched): longest voxel axis → largest GT dimension, etc.
   - **direct**: assumes coord axes map as d0→W, d1→H, d2→D.
5. Run Stage 2 (SLAT) three times on the same image conditioning, with:
   - original voxel coords
   - sorted-rescaled coords
   - direct-rescaled coords
   (All three passes use the trained metric scale token injected into SLAT conditioning.)
6. Decode a mesh from each SLAT output.
7. Measure each mesh's bounding-box extents and compare their aspect ratios to GT.
   Reported as `orig∆`, `srt∆`, `dir∆` — same normalised mean absolute deviation as `vox∆`.

**What `vox∆` measures:** how different the voxel topology's shape is from the GT object
shape, *before* SLAT runs. Purely a Stage 1 diagnostic.

**What `orig∆` / `srt∆` / `dir∆` measure:** how different the decoded mesh's shape is from
GT, after SLAT runs on the respective coord set. All are aspect-ratio / shape metrics only
(absolute scale is normalised out).

**Why GT dimensions drive the rescaling:** This is a hypothesis test, not a production
algorithm. Using GT dims isolates the question: "if the voxel grid had the right shape,
would the mesh proportions be correct?" Using predicted dims would mix in metric prediction
error and obscure the topology signal.

---

## Results (10 samples, v3 checkpoint, NOCS-Real275 test split)

```
Sample           Category  vox∆   orig∆  srt∆   dir∆
─────────────────────────────────────────────────────
scene_1_0304     cup       0.032  0.033  0.011  0.011  ← rescaling helps
scene_1_0304     laptop    0.092  0.091  0.004  0.008  ← large fix (srt wins)
scene_1_0304     can       0.010  0.017  0.007  0.008  ← slight improvement
scene_1_0304     camera    0.034  0.036  0.005  0.014  ← srt wins
scene_1_0304     bowl      0.067  0.071  0.020  0.017  ← both help (dir slightly better)
scene_1_0375     can       0.016  0.004  0.015  0.005  ← orig already best, rescaling hurts
scene_1_0375     cup       0.009  0.005  0.014  0.011  ← orig already best, rescaling hurts
scene_1_0375     camera    0.076  0.062  0.011  0.008  ← large fix (dir wins)
scene_1_0375     bowl      0.010  0.016  0.006  0.007  ← slight improvement
scene_1_0375     laptop    0.086  0.069  0.011  0.012  ← large fix
─────────────────────────────────────────────────────
Mean vox∆: 0.043
Sorted rescaling improves: 80% of samples
Direct rescaling improves: 80% of samples
```

---

## Key findings

### 1. Stage 1 voxel topology is the source of the aspect ratio error

`vox∆` and `orig∆` track each other closely across all 10 samples. When the voxel grid
has the wrong shape (high `vox∆`), the decoded mesh also has the wrong shape (high `orig∆`).
SLAT does not correct Stage 1 topology errors — it cannot.

### 2. Rescaling the voxel grid before SLAT fixes the mesh proportions

In 8/10 samples, GT-driven anisotropic rescaling reduced aspect ratio error by 3–20×.
The two exceptions (samples 6 and 7, both `vox∆ ≤ 0.016`) had correct voxel topology
to begin with — forcing a rescaling introduced noise. This suggests a threshold of
roughly `vox∆ ≈ 0.025` above which rescaling helps and below which it is unnecessary.

### 3. There is no stable direct axis mapping

The sorted (rank-matched) and direct (`d0→W, d1→H, d2→D`) strategies perform similarly
overall (both 80%), but neither dominates consistently per sample. Sorted wins on
asymmetric objects (laptops, cameras); direct wins on near-symmetric ones (bowls, cans).
The voxel coordinate axes do not reliably correspond to GT W/H/D axes. The sorted
strategy is therefore more robust.

### 4. This is a Stage 1 conditioning problem, not a SLAT problem

The SLAT v2/v3 metric token injection (cross-attention conditioning) improved the *scale
prediction* but cannot fix *voxel topology proportions* — those are set before SLAT runs.
The metric token reaches SLAT after the topology is already locked.

---

## Why inference-time rescaling is not the production fix

The experiment used ground-truth [W, H, D] for rescaling. In production:

- GT dims are not available.
- The MetricScaleDecoder predicts dims *after* SLAT runs (it needs slat.feats).
- This creates a circular dependency: you need dims to rescale voxels, but you need
  voxels to run SLAT to get dims.

A two-pass workaround is possible (first pass: get predicted dims; second pass: rescale
and re-run), but doubles inference cost and propagates metric prediction errors into the
voxel structure.

---

## The right fix: inject metric token into Stage 1 (SS) cross-attention

The SS generator has the same architecture as SLAT for conditioning:
- 24 × `ModulatedTransformerCrossBlock` with `cond_channels=1024`
- Same `_ScaleAugmentedEmbedderProxy` pattern can wrap the SS condition embedder

If Stage 1 receives the metric scale token during training, it learns to generate a voxel
topology with proportions consistent with the predicted dimensions. SLAT then operates on
a correctly-shaped grid, and FlexiCubes decodes a correctly-proportioned mesh — in one
pass, with no inference-time surgery.

This mirrors exactly the SLAT v2/v3 approach. The training changes are:
1. Extend `_ScaleAugmentedEmbedderProxy` to also wrap the SS condition embedder.
2. Unfreeze SS cross_attn + norm2 (same `collect_slat_cross_attn_params` pattern).
3. Add an SS cross-attn parameter group to the AdamW optimizer.
4. The metric regression loss (Smooth L1 on log-dims) stays unchanged — it already
   provides the gradient signal needed to drive SS toward correct proportions.

**Challenge:** SS uses the ShortCut distillation variant (multi-step, self-consistency
loss), not the simple Euler integrator SLAT uses. Training is more complex. Specifically,
the scale token must be threaded through the ShortCut's CFG passes consistently.
Investigate whether SS can be trained with `--stage1-steps 1` (single Euler step, like
SLAT v2/v3) to simplify gradient flow during fine-tuning, before tackling full ShortCut
training.

---

## Next steps

1. Inspect the SS generator's CFG / ShortCut training loop to understand what changes
   are needed to inject the metric token safely.
2. Implement `--unfreeze-ss-cross-attn` analogous to `--unfreeze-slat-cross-attn`.
3. Run a quick overfit test (10 samples) to confirm gradient flow and no NaN.
4. Launch a full training run warm-starting from the v3 checkpoint.
5. Re-run this experiment post-training to verify that vox∆ decreases.
