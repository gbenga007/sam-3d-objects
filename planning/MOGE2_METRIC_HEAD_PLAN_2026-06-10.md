# MoGe-2 Metric Anchor + W,H,D Head — Current State & Plan
## 2026-06-10

Goal: a publishable (workshop-modest) metric-scale result. Replace the MoGe-v1 (affine-
invariant) input pointmap with a **MoGe-2 metric pointmap**, retrain the MetricScaleHead +
MetricScaleDecoder on the mixed ~10k SFT set, and predict per-axis **W, H, D**.

---

## 1. Current State (what we have established)

### The mechanism (verified in code)
`out["scale"] = ssi_ratio · pointmap_scale`, where `pointmap_scale` is the **scene** scale the
pipeline derives from the INPUT pointmap via `ObjectCentricSSI(use_scene_scale=True)`
(pipeline.yaml). `ssi_ratio` is unit-free. So **the metric scale of the output is set by the
units of the input pointmap.** MoGe-v1 is affine-invariant → non-metric anchor → the head was
forced to learn a category prior (the ceiling: mixed ~22%, ARKit ~40%).
Refs: `pose_target.py` ssi_to_metric; `inference_pipeline_pointmap.py:447-458`;
`img_and_mask_transforms.py:ObjectCentricSSI`.

### Oracle (frozen STOCK model, NO head) — `scripts/gt_pointmap_metric_oracle.py`
Same model, same images, swap only the pointmap (NOCS):

| condition | median | mean |
|---|---|---|
| GT-metric pointmap | **6.7%** (n=100) | 7.9% |
| MoGe-v1 pointmap | **71.5%** (n=100) | 78.2% |

⇒ The "scale break" is a **depth-source problem, not a model problem**, proven end-to-end.
The stock pose decoder already yields metric scale from a metric pointmap; the head was
compensating for MoGe-v1.

### MoGe-2 anchor screen (n=100 NOCS) — `scripts/moge2_vs_gt_pointmap_scale_screen.py`
MoGe-2 `pointmap_scale` vs GT: **median ratio 1.080**, log-std **0.089**, **no size
dependence (r=−0.092)**. Implied error 9.0% raw → **7.2% after a single global scalar**.
KEY: the depth-bridge's catastrophic small-object over-measurement (NOCS-can 19cm→72cm) does
NOT transfer, because `pointmap_scale` is a whole-frame scene statistic that averages out
per-object close-range error. ⇒ Feeding MoGe-2 **as the pointmap** is the robust integration
(better than depth-bridge bbox or "extent as a head feature").

### Implication for the head
With a MoGe-2 metric anchor, recovering isotropic scale is nearly a **calibration constant**
(~1.08). The head earns its keep on **per-axis W,H,D**, which a constant cannot produce
(needs canonical shape × metric anchor). Mesh-bbox eval already shows per-axis 5.54% vs
isotropic 11.42% MAPE — aspect ratio for the *metric output* is handled by per-axis
prediction, NOT by touching the generator. See [[project_pointmap_scale_anchor]],
[[project_translation_scale_oracle]], [[project_aspect_ratio]].

---

## 2. The Plan — MoGe-2 anchor, mixed ~10k, predict W,H,D

### Stage A — MoGe-2 pointmap precompute (data prep)
For each training image in the mixed ~10k set (NOCS + Objectron-with-RGB + ARKit), run
**MoGe-2** (`MoGe2Estimator`, `/mnt/source/MoGe`, self-intrinsics — best config per oracle)
→ metric pointmap [H,W,3]. Cache to disk keyed by uid.
- Reuse the screen's inference path. Self-intrinsics (no GT camera needed at inference).
- ~a few GPU-hours total. One-time.

### Stage B — Inject MoGe-2 pointmap into training
`finetune_metric_scale.py` currently calls `pipeline.compute_pointmap(image)` (→ MoGe-v1).
`compute_pointmap(image, pointmap=...)` already accepts an override (same arg the inference
`run()` uses). Change: load the cached MoGe-2 pointmap per item and pass it in. This makes
BOTH the anchor scalar (`pointmap_scale`) AND the SS conditioning (PointPatchEmbed sees the
MoGe-2 pointmap) metric — no other surgery.

### Stage C — Cache clean latents, then train the head (addresses "noisy latents")
Use the existing `encode_metric_scale_features` to encode each instance ONCE (deterministic
seed) → cache `shape_latent`, `ss_scale_features`, `pointmap_scale/shift` (now MoGe-2).
Train MetricScaleHead + MetricScaleDecoder on the cache for many epochs via
`predict_cached_log_dims`. Generator stays FROZEN. Loss: smooth-L1 on `log [W,H,D]`.
- This is the right fix for run-to-run latent noise: cache once, don't regenerate per epoch.
- **Do NOT use `--unfreeze-ss-decoder`** unless its weights are saved (the
  [[feedback-checkpoint-completeness]] bug). Frozen generator avoids that trap entirely.

### Stage D — Eval + the ablation that makes it a paper
On per-source heldouts (NOCS / Objectron / ARKit), report **per-axis W,H,D MAPE** (+ isotropic).
Baselines to beat (this IS the contribution):
1. **MoGe-v1-anchored head** = `mixed_scratch_10k_best.pt` (22.3% overall) — same recipe, old anchor.
2. **Calibration constant** = stock pose decoder + MoGe-2 pointmap + global/per-category scalar
   (no learned head). The head must beat this on per-axis / cross-source to justify itself.

Expected story: MoGe-2 anchor ≫ MoGe-v1 anchor (esp. ARKit/Objectron cross-source), and the
learned head > calibration on per-axis dims.

### Stage D-ablation — object-centric scale as an extra head feature (optional, 2nd experiment)
NOT in the v1 head. Hold as a clean ablation. Idea: in addition to the existing SCENE
`pointmap_scale` (and the already-object-centric `pointmap_shift`), feed an **object
`pointmap_scale`** = the object's own metric extent measured directly from the masked MoGe-2
metric points (a robust percentile bbox = the depth-bridge recipe; compute from RAW metric
points, NOT the SSI-normalized cropped pointmap; precompute in Stage A).
- WHY not redundant: it's a DIRECT per-object measurement vs. the scene anchor (which only
  yields metric scale via the model's `ssi_ratio`). Its error profile is COMPLEMENTARY —
  accurate room-scale (ARKit ~22.7%), over-measures small objects (~72%), size-dependent —
  vs. the scene anchor's flat ~1.08 bias. So the head can learn to ROUTE: trust direct
  measurement for large objects, scene anchor for small. This is the route-by-scale hybrid in
  [[project_translation_scale_oracle]], in cleaner form.
- Also more explicit than hoping the head extracts object extent from SS shape features
  (present but unsupervised, [[project_aspect_ratio]]).
- CAVEAT: feed as a FEATURE the head can down-weight, never as the anchor (it carries the
  small-object bias). `pointmap_shift` is already object-centric → an "object shift" is redundant.
- Ablation table: scene-anchor-only vs. scene-anchor + object-extent, reported per-source
  (the payoff is expected on ARKit/Objectron). If it helps cross-source → extra paragraph;
  if not → drop, nothing lost.

---

## 2b. Flow-step schedule gap (2026-06-11): 4/1 train cache vs 25/25 inference

**Fact:** the Stage C latent cache is encoded at `--stage1-steps 4 --stage2-steps 1`
(SS=4, SLAT=1 flow steps); stock inference runs **25+25** (pipeline defaults; the
GT-pointmap oracle used 25/25). This is the SAME reduced schedule every prior training
run used (incl. `mixed_scratch_10k`, whose 22.3% heldout number was also evaluated at
4/1) — so the MoGe-v1-anchor baseline comparison stays apples-to-apples, and a 25/25
cache would have been ~10–15× slower (~2 days vs ~4 h).

**Risk:** a 1-step SLAT latent is a single Euler step from noise — a different
distribution than a 25-step sample. The head's most step-sensitive input is
`slat_feats` (MetricScaleDecoder side). NOT step-sensitive: `pointmap_scale/shift`
(preprocessor, no flow at all — the metric anchor is untouched by this gap); mildly
sensitive: pooled SS shape latent (4-step, same as the previous head consumed).

**Deployment-condition eval (QUEUED to run after the Stage C cache finishes):**
re-cache ONLY the 464 heldout records at 25/25 (`--cache-heldout-only`, ~2–3 h GPU):

```
finetune_metric_scale.py  <same dataset/heldout/seed args as Stage C>
  --stage1-steps 25 --stage2-steps 25
  --cache-latents --cache-only --cache-heldout-only
  --save-feature-cache artifacts/metric_scale/feature_caches/moge2_mixed_10k_heldout_25steps.pt
  --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps
```

Same dataset args + seed ⇒ identical heldout split; only the flow-step count differs.
Then eval the trained head on BOTH heldout caches (4/1 and 25/25):
- gap small → deployment number is safe; report eval condition in the paper.
- gap large → either re-cache train at 25/25 (accept ~2 days) or train with
  step-count augmentation; at minimum the paper's eval must use the 25/25 cache.

---

## 3. The flow-matching / random-timestep question (answered)

**Not needed for this plan, and partly a conceptual mismatch.**

- The MetricScaleHead is a **regression head on the final clean latent** — there is no
  flow-matching loss in head training. Per your TRELLIS notes
  (`TRELLIS_TRAINING_INSIGHTS_2026-05-01.md`, summary table): the logit-normal random-timestep
  sampler "only fires in `FlowMatching.compute_loss()`, which we never call; we run
  `generate()` and always land at t=0." So random timesteps are **N/A** to the head.
- **"Not noisy latents"** is achieved by **caching** (encode once, deterministic seed; Stage C),
  NOT by random timesteps. Random timesteps train the velocity field across the trajectory —
  they do not make the *inference* latent cleaner. Different lever.
- Random-timestep / logit-normal flow matching becomes relevant ONLY if you **fine-tune the SS
  generator itself** (the SS-encoder GT-latent path for true voxel aspect ratio,
  [[project_aspect_ratio]]). That is a bigger, riskier project (needs POSE, generator training)
  and is **out of scope** for the modest paper. Aspect ratio for the metric output is already
  covered by per-axis W,H,D (Stage D). Keep the generator frozen.

**Recommendation:** ship the frozen-generator, MoGe-2-anchored, cached-latent head. List
SS-generator latent-supervision (with proper flow-matching random-timestep training) as
**Future Work / Limitation** in the paper.

---

## 4. Scope, risks, deferrals

- **In scope:** Stages A–D above. Frozen generator. Per-axis W,H,D. NOCS+Objectron+ARKit ~10k.
- **De-risk first (cheap):** NOCS-only proof — precompute MoGe-2 for NOCS, short head train,
  confirm (a) beats MoGe-v1 NOCS head, (b) per-axis beats calibration. Few hours; green-lights
  or kills the paper before the full mixed run.
- **Risks:** (1) Objectron local-RGB coverage is thin (~6,358 records) — known limitation;
  weight sources accordingly. (2) Cross-source unverified for the *pointmap path* (screen was
  NOCS-only; no GT depth on disk for ARKit/Objectron) — the mixed training + heldout eval is
  what closes this. (3) MoGe-2 inference adds a precompute step + a runtime dependency at deploy.
- **Explicitly deferred:** SS/SLAT generator fine-tuning; SS-encoder aspect-ratio latent
  supervision; texture/geometric fidelity (Phase 2).

---

## 5. Milestones

| # | Milestone | Output |
|---|---|---|
| A | MoGe-2 pointmap precompute (NOCS first, then mixed) | cached pointmaps on disk |
| B | MoGe-2 pointmap injected into training path | code change in finetune_metric_scale |
| C | NOCS-only de-risk: short head train + cache | per-axis MAPE vs MoGe-v1 head + calibration |
| D | Full mixed ~10k head train (frozen generator) | `mixed_moge2_v1_best.pt` |
| E | Eval table: per-source per-axis W,H,D + ablation | paper Table 1 |
| F | Diagnosis figures (oracle 8.6% vs MoGe-v1 71%; screen ratio 1.08) | paper Fig 1-2 |

Paper shape: **diagnosis** (pointmap is the anchor; MoGe-v1 breaks it) + **fix** (MoGe-2 anchor
+ lightweight head → per-axis metric W,H,D, beats MoGe-v1 anchor and calibration baseline).
