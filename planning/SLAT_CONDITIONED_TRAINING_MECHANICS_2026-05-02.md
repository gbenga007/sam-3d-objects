# SLAT-Conditioned Metric Training Mechanics
## 2026-05-02

This note documents the current `scripts/train_slat_conditioned_v1.sh` training
path and what is actually being optimized.

## Entry Point

Launch:

```bash
bash scripts/train_slat_conditioned_v1.sh
```

Main Python entry point:

```bash
sam3d_objects/training/finetune_metric_scale.py
```

The current launch script uses live training, not cached training. It does not
pass `--cache-latents` or `--load-feature-cache`.

It reserves a small live held-out set:

```bash
--heldout-samples 64
--eval-every 1
```

Held-out eval re-runs the current SS/SLAT pipeline, so the reported metrics
reflect the current SLAT cross-attention weights.

## Per-Sample Inputs and Target

Input:

```text
RGBA image + alpha mask
```

Supervised target:

```text
metric_dims = [width, height, depth] in meters
```

The model predicts:

```text
log(width), log(height), log(depth)
```

Loss:

```python
loss = smooth_l1_loss(pred_log_dims, log(metric_dims))
```

The direct training objective is metric object size regression. There is no
ground-truth SLAT latent or mesh loss in this run.

## Forward Path

### 1. Pointmap / MoGe Cues

The pipeline computes a pointmap and extracts metric-ish image cues:

```text
pointmap
pointmap_scale
pointmap_shift
```

These are used as inputs to the sparse-structure preprocessor and metric scale
head. This path is frozen.

### 2. Sparse Structure Stage

The SS generator runs first and is always frozen.

Important outputs:

```text
shape  - sparse-structure latent used by MetricScaleHead
coords - sparse coordinates used by SLAT
scale  - SS scale token used as an explicit metric cue
```

This stage runs under `torch.no_grad()`. Its outputs are detached before being
used by trainable modules.

### 3. MetricScaleHead

`MetricScaleHead` is trainable.

Inputs:

```text
mean-pooled SS shape latent
SS scale features
pointmap_scale
pointmap_shift
```

Output:

```text
scale_token: [batch, 1, 1024]
```

The scale token has two uses:

```text
1. appended to SLAT conditioning
2. passed directly to MetricScaleDecoder
```

Before SLAT injection, a copy of the token is layer-normalized. The original
token is still passed to the metric decoder, preserving compatibility with
warm-started metric-head checkpoints.

### 4. SLAT Stage

The scale token is appended to the SLAT conditioning sequence beside the normal
image/DINO-style conditioning tokens.

During live training with `--unfreeze-slat-cross-attn`, SLAT sampling is called
with gradients enabled:

```python
pipeline.sample_slat(..., with_grad=True)
```

This is required so gradients can reach the unfrozen SLAT cross-attention
parameters through `cross_attn.to_kv`.

Output:

```text
slat.feats
slat.coords
```

### 5. MetricScaleDecoder

`MetricScaleDecoder` is trainable.

Inputs:

```text
slat.feats
scale_token
batch indices from slat.coords
```

Output:

```text
pred_log_dims = [log_w, log_h, log_d]
```

## Trainable Modules

The optimizer updates:

```text
MetricScaleHead
MetricScaleDecoder
SLAT transformer block cross_attn parameters
SLAT transformer block norm2 parameters
```

SLAT unfreezing is selective. In every SLAT transformer block:

```text
block.cross_attn.*
block.norm2.*
```

are set to `requires_grad=True`.

Everything else remains frozen:

```text
MoGe / pointmap model
SS generator
SLAT self-attention
SLAT MLP / FFN
SLAT other norms and modulation
preprocessors
all other pipeline weights
```

Optimizer groups:

```text
MetricScaleHead + MetricScaleDecoder: lr = 1e-4
SLAT cross_attn + norm2:              lr = 1e-5
```

## Training Stabilization

Current live training uses several TRELLIS-inspired safeguards:

```text
adaptive gradient clipping
gradient-level finite checks
scale-token dropout with p=0.1
SLAT block activation checkpointing during training
recovery checkpoints every epoch
per-step W&B logging for loss, grad norms, NaN skips, OOM skips, and dropout
```

The first backward pass verifies that SLAT cross-attention parameters received
gradients. If they do not, training raises an error instead of silently running
as a metric-head-only fine-tune.

## Live Evaluation

Cached eval via `--eval-feature-cache` has been removed from the live-training
path.

Live eval:

```text
uses held-out dataset items
runs current pointmap, SS, and SLAT stages
injects the current scale token
uses the current SLAT cross-attention weights
disables scale-token dropout
runs under torch.no_grad()
temporarily disables SLAT activation checkpointing for eval only
```

Metrics are written with split name:

```text
live_heldout
```

Best checkpoint selection uses live held-out `mean_abs_pct`.

## What This Is Not

This is not full TRELLIS flow-matching training.

TRELLIS normally trains a denoiser to predict a velocity field at randomly
sampled timesteps against known target SLAT latents.

This run instead backpropagates metric-dimension regression loss through a live
SLAT generation pass. Therefore, SLAT cross-attention is optimized indirectly:
it changes when doing so helps the metric decoder predict object dimensions.

In short:

```text
Goal: improve metric-scale prediction with a scale-conditioned SLAT path.
Not goal: retrain the entire SLAT generator against ground-truth 3D latents.
```
