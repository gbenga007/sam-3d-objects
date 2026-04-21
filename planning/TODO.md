# SAM 3D Objects — Metric Accuracy TODO

Goal: Make SAM 3D Objects physically accurate — recover metric scale (real-world units) and
achieve near 1-to-1 geometric fidelity with the input object.

---

## Phase 1 — Dataset Preparation

### 1.1 Download and validate datasets
- [ ] Download Objectron (Google Cloud Storage: `gs://objectron`)
- [ ] Download NOCS REAL275 (GitHub + supplementary scale zip)
- [ ] Spot-check metric annotations on 10-20 samples from each — confirm bounding box
      dimensions are in meters and match visual estimates

### 1.2 Preprocessing — Objectron
- [ ] Extract single representative frame per object instance from video clips
- [ ] Parse 3D bounding box annotations → extract `[width, height, depth]` in meters
- [ ] Generate segmentation masks (run SAM on extracted frames using bounding box as prompt)
- [ ] Run MoGe on each frame → extract `pointmap_scale` and `pointmap_shift` per instance
- [ ] Save as unified record: `{image, mask, pointmap_scale, pointmap_shift, metric_dims}`

### 1.3 Preprocessing — NOCS REAL275
- [ ] Parse per-instance metric size from supplementary scale files
- [ ] Confirm masks are already provided (they are)
- [ ] Run MoGe on each frame → extract `pointmap_scale` and `pointmap_shift`
- [ ] Save in same unified record format as Objectron

### 1.4 Build unified dataloader
- [ ] Implement `MetricScaleDataset` that loads both sources with consistent format
- [ ] Verify `pointmap_scale`/`pointmap_shift` correlate with ground truth metric scale
      (scatter plot: MoGe scale vs. GT scale) — this validates the metric anchor assumption
- [ ] Define train/val split (hold out 10% per category)

---

## Phase 2 — Architecture Implementation

### 2.1 Expose pointmap statistics through the pipeline
- [ ] In `inference_pipeline_pointmap.py`: confirm `pointmap_scale` and `pointmap_shift`
      are accessible after MoGe runs and before SS generation
- [ ] Pass them forward in `ss_input_dict` so they reach the scale head

### 2.2 Implement `MetricScaleHead`
- [ ] New file: `sam3d_objects/model/backbone/scale_head.py`
- [ ] Input: pooled SS latent `[batch, 8]` + `log(pointmap_scale)` + `pointmap_shift_z`
      → total input dim = 10
- [ ] Architecture: MLP (10 → 64 → 32 → 1), output = `log(metric_scale)` scalar
- [ ] Rationale: SS latent encodes object proportions; pointmap stats provide the metric
      anchor from MoGe. Together they can predict physical scale.

### 2.3 Implement `ScaleTokenProjector`
- [ ] In same file: `sam3d_objects/model/backbone/scale_head.py`
- [ ] Input: `log(metric_scale)` scalar `[batch, 1]`
- [ ] Architecture: `nn.Linear(1, 768)` → conditioning token `[batch, 1, 768]`
- [ ] Rationale: SLAT's cross-attention expects tokens of dim=768. The scalar scale value
      must be projected into that space to be consumed as a conditioning signal.

### 2.4 Wire scale head between SS and SLAT
- [ ] In `inference_pipeline_pointmap.py`: after SS generation, pool `shape_latent`
      via mean over spatial dim → `[batch, 8]`
- [ ] Run through `MetricScaleHead` → `log_scale`
- [ ] Run through `ScaleTokenProjector` → scale token `[batch, 1, 768]`
- [ ] Concatenate scale token with existing DINO condition tokens before SLAT generation

### 2.5 Implement `MetricScaleDecoder`
- [ ] New file: `sam3d_objects/model/backbone/metric_scale_decoder.py`
- [ ] Input: pooled SLAT latent `[batch, 8]` + scale token `[batch, 1, 768]`
      (project scale token down to 8-dim for concatenation, or use separately)
- [ ] Architecture: MLP → `[width, height, depth]` in meters (log-space output)
- [ ] Wire into SLAT stage output in `inference_pipeline_pointmap.py`
- [ ] Rationale: SLAT refines geometry with full image + geometry context. It has more
      information than SS alone, making it the right place for the final metric prediction.

### 2.6 Update config/YAML
- [ ] Add scale token to SLAT condition embedder input mapping in pipeline YAML
- [ ] Ensure `EmbedderFuser` concatenates scale token alongside DINO tokens

---

## Phase 3 — Fine-Tuning Setup

### 3.1 Define loss function
- [ ] Primary: smooth L1 loss on `log(predicted_scale)` vs `log(GT_metric_scale)`
      (log-space reduces sensitivity to large objects vs small objects)
- [ ] Secondary: smooth L1 on `[width, height, depth]` predictions from MetricScaleDecoder
- [ ] No loss on existing components — they stay frozen

### 3.2 Implement fine-tuning script
- [ ] New file: `sam3d_objects/training/finetune_metric_scale.py`
- [ ] Freeze: all existing SS model, SLAT model, condition embedders, decoders
- [ ] Train only: `MetricScaleHead`, `ScaleTokenProjector`, `MetricScaleDecoder`,
      and the new cross-attention K/V projection in SLAT for the scale token
- [ ] Optimizer: AdamW, lr=1e-4, weight decay=1e-4
- [ ] Log: scale prediction error (mean absolute % error) on val set each epoch

### 3.3 Validate training loop
- [ ] Overfit on 10 samples first — confirm loss decreases and predictions are reasonable
- [ ] Run full training on Objectron + NOCS REAL275 combined

---

## Phase 4 — Evaluation

- [ ] Metric: mean absolute percentage error (MAPE) on scale predictions vs GT
- [ ] Metric: per-axis dimensional error in cm (width, height, depth)
- [ ] Baseline comparison: current pipeline scale (SSI space) vs new metric-grounded output
- [ ] Visualize: predicted vs GT bounding box overlaid on input image for qualitative check
- [ ] Test on WildRGB-D (out-of-distribution) to assess generalization

---

## Phase 5 — Geometric Fidelity (Longer Term)

This phase addresses the second goal: 1-to-1 reconstruction of fine geometric detail
(surface texture, cracks, exact shape). Requires training changes to the generative model.

- [ ] Literature review: reconstruction losses for flow matching / score-based models
- [ ] Design: add perceptual loss term (rendered output vs input image) during training
- [ ] Design: add depth consistency loss (MoGe depth of rendered output vs input depth)
- [ ] Dataset: identify a dataset with fine-grained geometric ground truth
      (high-res 3D scans of real objects — e.g., OmniObject3D, or custom captures)
- [ ] Implement training with combined flow matching + reconstruction loss
- [ ] Evaluate: Chamfer distance, F-score on held-out scans

---

## Milestones

| Milestone | Description |
|-----------|-------------|
| M1 | Datasets downloaded, preprocessed, dataloader validated |
| M2 | Architecture implemented and pipeline runs end-to-end without errors |
| M3 | Scale head overfits on 10 samples (sanity check) |
| M4 | Fine-tuning converges on full dataset, MAPE < 15% on val set |
| M5 | Out-of-distribution evaluation on WildRGB-D |
| M6 | Phase 5 reconstruction fidelity work begins |
