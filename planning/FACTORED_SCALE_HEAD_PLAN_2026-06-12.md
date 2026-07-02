# Factored Scale Head — MoGe-2-style decoupled iso/proportions training (2026-06-12)

## Why (one paragraph)

Mixed cached training of the absolute-WHD head is architecturally capped: 1000-epoch run
(`mixed_moge2_v2_long`) plateaued at **35.4% @ ep255 then overfit** (48.8% by ep1000), while
the identical regime on NOCS-only reached **3.48%**. One 13-input MLP regressing absolute
sizes from 4 cm cans to 2 m beds cannot serve three scale regimes — the same failure MoGe-2
documents for entangled scale prediction (Mogev2.pdf §3.2 / Tab. 4): wide-range scale targets
destabilize training and interfere with relative geometry. Their fix — decouple scale into its
own branch with exclusive, log-space, stopgrad-protected supervision — is the fix here too.

## Evidence base (all 2026-06-11, heldout-375, Test 1 + decomposition)

- Pose decoder `out["scale"]` × canonical voxel extents, NO learned head, isotropic median
  error: GT pointmap **6.7%** (NOCS), live MoGe-2 **16.7 / 75.7 / 19.6%** (NOCS/Obj/ARKit).
- Error decomposition (MoGe-2 anchor, rank-sorted axes, per-axis MAPE mean):
  | source | raw | iso-corrected only | gt-proportions only |
  |---|---|---|---|
  | NOCS | 19.1% | **7.7%** | 17.1% |
  | Objectron | 100.2% | **14.4%** | 92.3% |
  | ARKit | 33.3% | **22.5%** | 27.8% |
  ⇒ nearly all cross-source error is the ONE isotropic factor; proportions are decent
  (ARKit's 22.5% residual = the old aspect-ratio problem, attack later, separately).
- log(gt_iso) vs log(pred_iso): NOCS slope 1.07 / R² 0.87 (GT-pmap: 1.10 / 0.95 ≈ identity);
  Objectron 0.39 / 0.29; ARKit 0.53 / 0.34 — systematic, conditionable bias (learnable).

## The loss (user's idea, MoGe-2 §3.2 form)

MoGe-2: `L_s = ‖log(ŝ) − stopgrad(log(s*))‖²`, s* = optimal alignment scale of the model's own
relative prediction to metric GT; stopgrad on the target protects the relative branch.

Ours:

```
canon_ext        = voxel extents of SS coords (max ≈ 1, FlexiCubes invariant)   # stopgrad
pred_iso_scale   = max(pred_WHD) / stopgrad(max(canon_ext))
L_s              = ‖ log(pred_iso_scale) − stopgrad(log(s*)) ‖²                 # exclusive
L_prop           = smooth_l1( log(pred_WHD / max(pred_WHD)),
                              log(gt_WHD  / max(gt_WHD)) )                      # scale-free
L                = L_s + λ_prop · L_prop
```

Two target choices for `s*`:
1. **PRIMARY — box-derived**: `s* = gt_iso / max(canon_ext)`. Exact MoGe-2 analog (alignment of
   the relative shape to metric GT), noiseless, available for every OmniNOCS record.
2. **SECONDARY — depth-derived (user's original)**: `s* = iso of pose_decoder out["scale"] run
   with GT metric pointmap` (~7% noise). Uses GT depth, not box labels ⇒ (a) consistency
   regularizer between the WHD head and the pose/mesh pathway, (b) the DATA-EXPANSION path:
   any RGB-D dataset without 3D boxes becomes scale-supervision data (MoGe-2's own trick).
   Locally GT depth exists only for NOCS (real_test Kinect) today.

## Architecture

Keep MetricScaleHead/MetricScaleDecoder inputs; change the decoder OUTPUT + parameterization:

- **iso branch**: predicts `log_iso_correction` (a residual to the anchor estimate
  `anchor_iso = max(canon_ext ⊙ pose_scale)`), so
  `pred_iso = anchor_iso · exp(log_iso_correction)`. Inputs: existing 13 features +
  NEW anchor features: `log(anchor_iso)`, per-axis `log(pose_scale)` (3), `canon_ext` (3).
- **proportions branch**: predicts normalized `log proportions` (3, max-axis pinned to 0) from
  the shape-side features (pooled shape latent, pooled slat feats, canon_ext).
- `pred_WHD = pred_iso × softmax-free normalized proportions` (exp of pinned log-props).
- Ablations: (A1) residual vs absolute iso; (A2) scalar iso correction vs per-axis correction;
  (A3) λ_prop sweep; (A4) + depth-derived consistency target on the NOCS subset.

Axis convention: supervise proportions in the GT (W,H,D) order as today (the cached regime's
existing convention — the decoder already learns axis order); rank-sorting only for analysis.

## Data plumbing

- Cached latents: `feature_caches/moge2_mixed_10k.pt` (train 8,482 / heldout 375, 4/1 steps).
- Anchor table (pose_scale + voxel extents, live MoGe-2, SS 25 steps — deployment condition):
  - heldout 375: `metrics/pose_scale_heldout_moge2.jsonl` (Test 1, done)
  - train 8,482: `metrics/pose_scale_train_moge2.jsonl` — extraction RUNNING (launched
    2026-06-11, ~10 h, resumable by uid; script `scripts/pose_scale_heldout_compare.py`
    pointed at `metrics/train_8482_meta.json`)
  - join key: `uid`. Samples missing from the anchor table (pipeline skips): drop (count them).
- NOTE the deliberate step asymmetry: latent features at 4/1 (cache) but anchors at 25 steps
  (matches deployment + Test 1 numbers). Anchors are independent INPUTS, so no train/eval skew
  as long as train and heldout anchors use the same steps (they do). The 4/1-vs-25/25 latent
  gap measured earlier (+4.6pp) still applies to the latent features only; re-evaluate on the
  25/25 heldout cache as before.
- GT-pmap pose-decoder targets for NOCS train subset (ablation A4): one more extraction pass,
  `--condition gt` over `train_8482_meta.json` (auto-skips non-NOCS), ~3,267 × ~4 s ≈ 3.5 h.

## Training & evaluation

- Regime: cached (head+decoder only, frozen generator, checkpoint inference-valid). bs 32,
  lr 1e-4, wd 1e-4, **≥1000 epochs** (NOCS-only showed 200 is far from convergence), seed 0,
  eval-every 5. Runs cost minutes-to-hours; iterate freely.
- Eval: heldout-375, per-source mean/median per-axis MAPE + iso MAPE separately (the factored
  form makes iso-vs-proportion attribution natural). Also eval on the 25/25 heldout cache.
- **Baselines to beat:**
  | baseline | overall mean | NOCS / Obj / ARKit |
  |---|---|---|
  | cached absolute head, converged (`mixed_moge2_v2_long` ep255) | 35.4% | 19.6 / 31.4 / 46.7 |
  | live MoGe-v1 + injection (`mixed_scratch_10k`) | 22.3% | 7.4 / 17.7 / 34.3 |
  | oracle floor (perfect iso correction, predicted proportions) | ~15% | 7.7 / 14.4 / 22.5 |
- Success: clearly under 22.3% overall with per-source wins on Objectron+ARKit; NOCS in the
  3–8% band. (3% OVERALL on mixed is not attainable — GT-anchor oracle is ~7% on NOCS alone.)

## Order of work

1. (running) finish train-split anchor extraction; verify join coverage vs cache uids.
2. Implement factored decoder + L_s/L_prop in the cached path of finetune_metric_scale.py
   (new flags, default off — don't disturb existing runs' reproducibility).
3. v1 run: residual iso (scalar) + proportions, box-derived s*, 1000 ep → compare table above.
4. Ablations A1–A3; A4 after the GT-pmap NOCS extraction.
5. If factored-cached beats 22.3%: consider the live MoGe-2 + injection run for the final
   paper number (3–4 days A100, recipe = mixed_scratch_10k + --moge2-pointmap-dir).

## Paper framing (decided 2026-06-12)

**v1b (factored) is the paper's method; v1a is its key ablation.** The contribution is the
analysis chain, with the head as its consequence:

1. **Hook / diagnostic finding**: the frozen pose decoder already recovers metric scale when
   the input pointmap is metric (`out["scale"] ∝ pointmap units`; GT-pmap 6.7% vs MoGe-v1
   71.5%, same frozen weights). The metric-scale gap in image-to-3D generators is largely a
   DEPTH-SOURCE problem, not a model problem.
2. **Error decomposition**: cross-source error is almost entirely isotropic (Objectron
   100%→14% with iso fixed); the generator's canonical proportions are already good.
3. **Failure analysis**: entangled absolute WHD regression plateaus-then-overfits across mixed
   scale regimes (35.4% converged vs 3.5% single-regime) — independently mirrors MoGe-2's
   published motivation for decoupled scale (cite §3.2 / Tab. 4 prominently; we ADAPT their
   principle, we don't invent it).
4. **Method**: residual-on-anchor iso branch + scale-free proportions branch; gradient
   isolation falls out of the parameterization (max of pinned log-dims IS the iso output) —
   no stop-grad surgery.
5. **Differentiators from MoGe-2**: principle transferred to OBJECT metric dimensions in a
   frozen generator (relative factor = the generator's canonical-shape invariant); cross-source
   bias analysis of MoGe-2-as-anchor (1.16/1.76/0.95) that NOCS-only screens miss; and the
   label-free s* extension (scale supervision from RGB-D / Objectron sparse ARKit clouds, no
   3D box labels) as the novel-ish flourish.

**Ablation table**: v1a vs v1b = "same features, same losses, shared vs decoupled gradients" —
isolates the claimed contribution. Hedge: if v1a matches v1b, reframe as "anchor features +
consistency suffice; decoupling adds robustness" — the diagnostic chain (1–3) remains the
contribution either way.

## Status log

- 2026-06-12: smoke PASSED for both decoders (3 ep, NOCS-only cache, partial anchors).
  Factored starts at 39% from ep1 (residual starts at the anchor) vs 273% for v1a.
- 2026-06-12: downloads launched — Objectron geometry.pbdata (2,273 videos, ~18 GB,
  `datasets_sam3d/Objectron_geometry/`) and ARKitScenes lowres_depth+intrinsics for the 904
  needed frames (586 videos, extract-and-delete, `datasets_sam3d/ARKitScenes_depth/`;
  timestamp matching verified 18/18 on first videos). Scripts:
  `scripts/download_objectron_geometry.py`, `scripts/download_arkitscenes_depth.py`.
- 2026-06-12 (evening): **A/B COMPLETE.** v1a anchor (shared grads) best 27.23 mean / 15.70
  median @ ep340 (NOCS 7.4 / Obj 26.4 / ARKit 36.7) BEATS v1b factored 28.25 / 15.87 @ ep220
  (8.3 / 28.3 / 36.9); absolute-head baseline 35.4 / 21.3. ⇒ the gain is the ANCHOR FEATURES,
  not gradient decoupling; v1a also overfits less (ep1000: 32.2 vs 37.9). Paper takes the
  pre-registered hedge: v1a = method of record, v1b = the decoupling ablation, diagnostic
  chain = the contribution. NOCS sits at the oracle floor (7.4 vs 7.7). Remaining gap to the
  ~15% floor: Objectron size-dependent iso correction (26.4 vs 14.4) + pathological ARKit
  categories (stove 334%, sink 104% — inspect GT). Next: mask-size + median-depth iso
  features; per-axis correction (A2); Objectron sparse-cloud target (data on disk);
  weight-decay tuning; live MoGe-2 + injection run.
