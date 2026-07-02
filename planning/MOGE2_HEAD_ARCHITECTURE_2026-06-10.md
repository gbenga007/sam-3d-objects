# Current Architecture — MoGe-2 Anchored Metric-Scale Head (Frozen Regime)
## 2026-06-10

Canonical architecture for the metric-scale paper. Companion to the staged build plan in
`MOGE2_METRIC_HEAD_PLAN_2026-06-10.md`. **Frozen regime** (NOT SLAT-conditioned): only the two
small heads are trainable; the whole SAM3D stack is frozen, which is what makes the SS and SLAT
latents cacheable.

**One line:** a metric pointmap (MoGe-2) flows through the frozen SAM3D stack to produce a shape
latent + a metric anchor; two small heads — the ONLY trainable parts — turn those into per-axis
W,H,D.

---

## Flow

```
RGB image + object mask                              GT [W,H,D]  (training target)
        │                                                  │
        ▼                                                  │
[FROZEN] MoGe-2  ──►  global metric pointmap [H,W,3]        │   (precomputed, Stage A)
        │                                                  │
        ▼                                                  │
[FROZEN] preprocess (derived LIVE from global pmap):       │
   ├─ crop_around_mask → object-centric pointmap (SSI) ──┐ │
   └─ ObjectCentricSSI → pointmap_scale (scene, METRIC)  │ │
                         pointmap_shift  (object-centric)│ │
        │                                                │ │
        ▼                                                ▼ │
[FROZEN] SS generator  ◄── DINOv2(img)+DINOv2(mask)+PointPatchEmbed(obj pmap)
        │
        ├─►  shape_latent [B,4096,8]  + ss_scale_features [1,3]
        │            │
        │            ▼
        │     [LEARNED] MetricScaleHead
        │       in : pooled shape_latent[B,8] + ss_scale_features
        │            + pointmap_scale + pointmap_shift
        │       out: scale_token [1,1024]
        │                          │
        ▼                          │
[FROZEN] SLAT generator            │   (NO token injection in the frozen regime)
        │                          │
        ▼                          ▼
      slat_feats ───────►  [LEARNED] MetricScaleDecoder
                             in : slat_feats + scale_token
                             out: predicted log[W,H,D]
                                       │
                                       ▼
                          smooth-L1( pred , GT log[W,H,D] )
                                       │
                          backprop ──► head + decoder ONLY
```

---

## Frozen vs. learned

| | Component |
|---|---|
| **Learned** (~300K params) | MetricScaleHead (~200K), MetricScaleDecoder (~100K) |
| **Frozen** | MoGe-2; all preprocessing/normalization; SS generator + condition embedders (DINOv2 ×2, PointPatchEmbed); SLAT generator + its cross-attention; SS decoder; SLAT decoders; pose decoder |

---

## Pre-training encode + cache  (answers "are SS & SLAT latents cached ahead of time?")

**Yes.** Before head training, a one-time encode pass (`encode_metric_scale_features`,
`finetune_metric_scale.py:483`) runs the FROZEN stack once per training sample and writes the
latents to disk, keyed by uid:

```
per sample (once, ahead of training):
   MoGe-2 pointmap (Stage A) → preprocess → SS generator  → shape_latent, ss_scale_features
                                          → SLAT generator → slat_feats
   + pointmap_scale, pointmap_shift
   ───────────────────────────────────────────────►  cache[uid]
```

Cached fields: `shape_latent`, `ss_scale_features`, `pointmap_scale`, `pointmap_shift`,
`slat_feats`. With MoGe-2 injected (Stage B), the cached `pointmap_scale` IS the MoGe-2 metric
anchor and `shape_latent` is SS conditioned on the MoGe-2 pointmap — the whole MoGe-2 benefit is
baked into the cache.

**Why this is valid:** SS + SLAT are frozen, so a sample's latents never change during head
training. (If SLAT were trained — the SLAT-conditioned regime — the cache would go stale every
step and SLAT would have to run live; that regime is explicitly NOT used here.)

**Then training:** each epoch runs ONLY the two learned heads on the cache (`predict_cached_log_dims`,
`:530`) + smooth-L1 loss. The expensive generator forwards happen once; the cheap heads train for
many epochs. Fixed encode seed ⇒ a consistent (non-noisy) latent per sample across epochs.

---

## Head / decoder I/O

**MetricScaleHead** — `scale_head(shape_latent, ss_scale_features, pointmap_scale, pointmap_shift)`
- pooled `shape_latent [B,4096,8] → [B,8]` (object proportions)
- `ss_scale_features [1,3]` (SS model's own log-SSI `scale` stream)
- `pointmap_scale` (scene metric anchor, MoGe-2), `pointmap_shift` (object-centric; incl. distance z)
- → `scale_token [1,1024]`
- NOTE: pose-decoder scale/translation are intentionally EXCLUDED (placement, not size; the
  translation_scale oracle confirmed they don't help — [[project_translation_scale_oracle]]).

**MetricScaleDecoder** — `predict_metric_dimensions(slat_feats, scale_token, slat_coords)`
- → `log[W,H,D]` (per-axis metres). Aspect ratio comes from `slat_feats` (shape); overall scale
  from the token. Per-axis is where the learned head beats a calibration constant.

**Loss:** smooth-L1 on `log[W,H,D]` vs GT. Backprop into head + decoder only.

---

See [[project_pointmap_scale_anchor]] for why a metric pointmap is the anchor (out["scale"] ∝
input pointmap units; oracle GT-pmap 8.6% vs MoGe-v1 71%). Planned 2nd-experiment ablation
(object-centric `pointmap_scale` as an extra head feature) is in the plan doc, not in this v1 arch.
