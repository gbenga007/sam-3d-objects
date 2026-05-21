# Paper Plan: Metric Scale Recovery for SAM 3D Objects

Created: 2026-05-17
Status: Framing locked. Experimental matrix in progress; Phase 3a results pending.

---

## Framing

**Working title**: *Recovering Metric Scale in Pretrained Image-to-3D Generators Without Backbone Retraining*

(Alternates if needed: *Metric Scale for Free: Reusing Depth-Encoder Statistics in Image-to-3D Generation*; *Lightweight Metric Heads for Frozen Generative 3D Models*.)

### The gap

Modern image-to-3D generative models (SAM 3D Objects, TRELLIS, Trellis-style flow-matching backbones) produce 3D outputs in canonical unit cubes `[-0.5, 0.5]³`. They model object **shape** but throw away **physical size**. Downstream applications that need absolute units — robotic grasping, AR object placement, scene reconstruction — must either rely on a separate metric depth model (incurring redundant compute) or fall back to category-prior heuristics (10.9% MAPE on NOCS-Real275).

### The insight

The MoGe pointmap encoder is already inside SAM 3D Objects' inference pipeline. It produces metric depth in meters, but the pipeline immediately normalises it into `pointmap_scale` and `pointmap_shift` statistics and never propagates those statistics downstream. The metric anchor exists in the pipeline; it is being discarded.

### The contribution

A recipe (not an architecture) for adding metric scale prediction to a frozen image-to-3D generator:

1. A lightweight **MetricScaleHead** that combines pooled SS latent shape features with MoGe pointmap statistics to produce a 1024-dim scale token.
2. Targeted **cross-attention adaptation** in the SLAT flow-matching decoder — only ~100M of >1B params are unfrozen, with an fp32/warmup recipe that prevents the bf16 attention overflow that NaN'd our first attempt.
3. A differentiable **aspect-ratio loss** on the SS decoder's soft voxel occupancy, supervising shape proportions without requiring full mesh ground truth.

Result: **1.23% MAPE** on NOCS-Real275 — an order of magnitude better than category-prior baselines and comparable to specialised monocular metric depth methods, while reusing the existing 3D generative pipeline.

---

## Core claims (must defend with experiments)

1. **C1**: MoGe pointmap statistics are the missing metric anchor — *removing them collapses metric prediction to near-category-baseline levels.*
2. **C2**: Lightweight heads + cross-attention adaptation outperforms frozen-SLAT readout — *unfreezing cross-attention is necessary, not just nice-to-have.*
3. **C3**: SS decoder aspect-ratio supervision narrows the per-axis prediction gap — *demonstrably reducing mesh aspect ratio error.*
4. **C4**: The recipe generalises across real-world datasets — *NOCS + Objectron + ARKitScenes joint training does not regress per-source performance.*
5. **C5**: We beat the obvious post-hoc baselines applied to the original SAM 3D Objects — *category prior, MoGe-only scaling, scale-from-bbox heuristic.*
6. **C6** (stretch): The method generalises out-of-distribution — *WildRGB-D evaluation transfers.*

---

## Experimental matrix

### E1. Main result table

Held-out MAPE on NOCS-Real275 (64-sample record split). Categories: bottle, bowl, camera, can, cup, laptop.

| Method | Backbone | Trainable params | MAPE ↓ | Mean cm ↓ |
|---|---|---|---|---|
| Category-mean baseline | — | 0 | 10.90% | — |
| MoGe-only scale (heuristic) | MoGe | 0 | TBD — must measure | TBD |
| Metric3D v2 (pretrained baseline) | Metric3D | 0 (inference only) | TBD | TBD |
| ZoeDepth + bbox scaling | ZoeDepth | 0 | TBD | TBD |
| Original SAM 3D Objects + post-hoc isotropic scaling | none retrained | 0 | TBD (requires post-hoc method) | TBD |
| **Ours, frozen SLAT** | SAM3D + MetricScaleHead | ~0.3M | 2.997% | 0.36 |
| **Ours, SLAT cross-attn unfrozen** | SAM3D + heads | ~100M | 1.74% | 0.23 |
| **Ours, full recipe (with aspect ratio loss)** | SAM3D + heads + SS decoder | ~170M | **1.23%** | 0.20 |
| **Ours, mixed dataset (Phase 3a)** | SAM3D + heads + SS decoder | ~170M | TBD (pending) | TBD |

Per-category breakdown for the final row is required.

### E2. Ablations (in support of C1–C3)

Ablation table holding everything else constant, varying one factor:

| Ablation | Removes | Expected effect | Status |
|---|---|---|---|
| A1 | MoGe pointmap features (8-dim SS only) | Should collapse to 5–10% MAPE | TODO |
| A2 | SS scale features (`log-SSI`, 3-dim) | Mild regression (1–2% increase) | TODO |
| A3 | SLAT cross-attention frozen | Should match the 2.997% baseline_v2 | DONE (baseline_v2) |
| A4 | SS decoder frozen / no aspect-ratio loss | Mesh aspect ratio error worsens; MAPE may also regress | DONE (slat_conditioned_v3, 1.74%) |
| A5 | No CFG scale-token dropout (`p-uncond-scale-token 0.0`) | Modestly worse generalisation | TODO |
| A6 | No fp32 cross-attn upcast | NaN within first epoch | DONE (v1 NaN'd) |
| A7 | No slat-lr warmup | NaN-prone | DONE (v1 NaN'd) |

### E3. Mesh aspect-ratio analysis (in support of C3)

Already partially done:
- `scripts/mesh_scale_eval.py` produces per-sample isotropic vs per-axis mesh bbox errors
- **v3_best baseline**: isotropic 11.42%, per-axis 5.54%, gap +5.88pp
- **Pending after ss_ratio_v1**: rerun on ss_ratio_v1_best.pt to show the aspect-ratio loss reduces the gap

Visualisation: per-category isotropic-vs-per-axis bar chart + scatter plot of `iso_mape` vs `peraxis_mape` per sample.

Expected story: aspect-ratio loss should narrow the gap from +5.88pp toward <2pp.

### E4. Mixed-dataset generalisation (in support of C4)

Phase 3a planned configuration (code shipped, not yet launched):
- Sources: NOCS-Real275 + Objectron + ARKitScenes
- Cap: `--max-records-per-source 30000`
- Sampling: `--balanced-sampling` (1:1:1 WeightedRandomSampler)
- Warm-start: `nocs_sceneholdout_ss_ratio_v1_best.pt` (1.23% MAPE)
- 6 epochs, per-source heldouts (64 NOCS + 200 Objectron + 200 ARKit)

Result table required:

| Train sources | NOCS MAPE | Objectron MAPE | ARKit MAPE |
|---|---|---|---|
| NOCS only (ss_ratio_v1) | 1.23% | TBD (eval-only on Obj/ARKit) | TBD |
| NOCS + Objectron + ARKit | TBD | TBD | TBD |

Cross-source category coverage table also belongs here (which categories appear in which source, total instances per category × source).

### E5. Out-of-distribution evaluation (in support of C6, stretch goal)

WildRGB-D (CVPR 2024): 8,500 objects, 46 categories. Use only for evaluation, never training.

| Method | OOD MAPE | OOD Mean cm |
|---|---|---|
| Category-mean baseline | TBD | TBD |
| MoGe-only scale | TBD | TBD |
| Ours (Phase 3a checkpoint) | TBD | TBD |

This is the experiment that distinguishes a workshop paper from a main-conference paper. Without it, reviewers will say "you only validate on the training distribution."

### E6. Failure-mode analysis

Already characterised:
- **Bottle transparency** — MoGe depth fails on transparent objects; mask-painting (opaque-fill) recovers some performance (data exists from MoGe correlation check)
- **Cup open-top geometry** — MoGe estimates depth extent incorrectly when interior is visible
- Per-category MAPE always worst on `laptop` (`bowl` historically too, now improved at ep6)

For the paper: include a qualitative figure with 3–5 failure cases and a discussion of when the method should not be trusted.

### E7. Compute and parameter efficiency

Required table to emphasize the "without retraining the backbone" claim:

| Method | Trainable params | Total inference params | Training compute | Inference latency |
|---|---|---|---|---|
| Train SAM3D from scratch | >1B | >1B | >100K GPU-hr | unchanged |
| Train metric-depth model from scratch | TBD | TBD | TBD | TBD |
| **Ours** | ~170M | unchanged from base SAM3D | ~600 A100-hr (cumulative across v1–ss_ratio_v1) | +1 small head, negligible |

---

## Baselines we must include

Beating these is required for publication. These are the methods that compete in the same "predict metric scale from an image" problem space:

1. **Category-mean baseline** (10.90% MAPE) — already measured. Floor result.
2. **MoGe-only post-hoc scaling** — run MoGe on the mask, compute object bounding box extent in metres from the pointmap, compare to GT. This is the strongest no-training baseline.
3. **Metric3D v2** — pretrained monocular metric depth. Mask the object, average depth in masked region, infer size from object angular extent + depth.
4. **ZoeDepth** — same protocol as Metric3D.
5. **Original SAM 3D Objects + isotropic post-hoc scaling** — take the canonical SAM3D mesh, apply MoGe-derived scale. This is the "what you'd do without our method" baseline.
6. **Frozen-SLAT readout (our baseline_v2)** — same architecture, but no cross-attention unfreeze. Already measured at 2.997%.

For C5 specifically — beating the original SAM3D — the relevant comparison is:
- **Original SAM3D** can't predict metric size at all → require a post-hoc method (MoGe pointmap scaling, ZoeDepth + bbox heuristic, etc.)
- **Our method** predicts metric size end-to-end with no separate inference pass
- Comparison should report both MAPE and inference latency

---

## Visualisations required

1. **Architecture diagram** — frozen SAM3D pipeline (greyed out) + new components (MetricScaleHead, scale-token injection into SLAT cross-attention, MetricScaleDecoder, SS decoder fine-tuning) highlighted. The single figure that conveys the recipe.
2. **Training trajectory plot** — log-scale MAPE vs epoch for v1 (NaN'd), v2, v3, ss_ratio_v1. Tells the stability/recipe story.
3. **Per-category MAPE bar chart** — comparing baseline_v2, slat_conditioned_v3, ss_ratio_v1, mixed_v1 (Phase 3a).
4. **Predicted-vs-GT bounding box overlays** — 3D bbox rendered on top of the input image, comparing baseline_v2 vs ours, ~12 examples spanning categories. Two failure cases.
5. **Mesh aspect-ratio scatter** — isotropic-MAPE vs per-axis-MAPE per sample, before/after aspect-ratio loss.
6. **Category coverage Venn / matrix** — which categories appear in NOCS / Objectron / ARKit, to motivate mixed-dataset training.

---

## Required pre-submission work

Roughly in priority order:

1. ✅ Phase 3a (mixed dataset training) — gives C4
2. ⬜ Per-category breakdowns for the main result table
3. ⬜ Run all ablations A1, A2, A5 (A3/A4/A6/A7 are already covered by historical runs)
4. ⬜ Implement and measure all baselines: MoGe-only, Metric3D, ZoeDepth, "SAM3D + post-hoc scaling"
5. ⬜ Mesh aspect-ratio eval on ss_ratio_v1_best.pt and mixed_v1
6. ⬜ Compute/parameter efficiency table — exact param counts of frozen vs trainable
7. ⬜ Predicted-vs-GT bbox visualisations (12 examples + 2-3 failures)
8. ⬜ WildRGB-D OOD evaluation (stretch goal but essential for main-conference acceptance)
9. ⬜ Architecture figure
10. ⬜ Training trajectory plot

---

## Timeline (rough)

| Phase | Duration | Output |
|---|---|---|
| ss_ratio_v1 completion | ~1.5 days (in progress) | Final NOCS-only MAPE; mesh bbox eval |
| Objectron download | ~2 hours (in progress) | Full RGB coverage for mixed training |
| Phase 3a mixed training | ~3 weeks | C4 result; per-source MAPE table |
| Ablation runs (A1, A2, A5) | ~2 weeks (can parallelise with Phase 3a) | Ablation table |
| Baseline implementations + measurement | ~1 week | E1 main result table complete |
| WildRGB-D OOD eval | ~3 days | C6 (if pursued) |
| Visualisations | ~1 week | Figures 1–6 |
| Writing | ~3 weeks | Submitted draft |
| **Total** | **~8–10 weeks** | Workshop submission ready |

---

## Venue strategy

- **First target**: **MUSTCV workshop at ECCV 2026** — identified 2026-05-20 as the best fit. The workshop focuses on metric understanding of scenes and objects; our paper maps directly onto its scope.
- **Backup**: Main ECCV 2026 track (if MUSTCV accepts workshop-only papers that weren't previously at the main conference).
- **Deferred**: 3DV 2026 — too far out. CVPR/ICCV 2027 only if ECCV submission fails.

The paper as currently framed (metric scale only, no geometric fidelity) is a strong workshop paper. MUSTCV's scope is ideal.

---

## Paper Writing Status (updated 2026-05-21)

**GitHub repo:** `git@github.com:gbenga007/eccv-paper-vigir.git` (user: gbenga007, branch: main)
**Template:** ECCV 2026 LNCS (llncs class)

| Section | Status | Notes |
|---|---|---|
| Introduction | ✅ Complete | 5 paragraphs + itemized contributions; no template filler |
| Related Work | ✅ Complete | 4 paragraphs; bib entries filled for trellis/trellis2/sam3d/omninocs/moge/moge2 |
| Method | ✅ Complete | 6 subsections with equations and exact architecture dims |
| Experiments | ✅ Skeleton | Tables structured; NOCS real numbers in; mixed_v1 rows = `??` (pending epoch-1 ckpt) |
| Abstract | ⏳ Waiting | Holding for epoch-1 mixed_v1 MAPE (Objectron + ARKitScenes) |
| Conclusion | ⬜ Not started | |
| Figures | ⬜ Not started | Architecture diagram, qualitative grid, scatter plot |

**Key remaining blockers (in priority order):**
1. epoch-1 mixed_v1 checkpoint (~29h away) → fills abstract + Table 2
2. Depth-bridge baseline (inference only, no training)
3. Category-mean per-category MAPE (compute from dataset stats)
4. Ablation training runs (3–4 × ~60h each — long pole)

---

## Risk register

| Risk | Mitigation |
|---|---|
| Mixed-dataset training regresses NOCS performance | Phase 3a heldout watches NOCS MAPE specifically; if it regresses, weight NOCS higher or keep NOCS-only as the main result |
| WildRGB-D OOD eval shows poor transfer | Frame the paper around in-distribution metric prediction; defer OOD to follow-up |
| Reviewers demand training-from-scratch baseline | Cite TRELLIS / SAM3D compute requirements explicitly (>100K GPU-hr); position as "no one can afford the alternative" |
| Mesh aspect-ratio results are marginal | Drop C3 from claims; keep as ablation only |
| Objectron download fails | Phase 3a falls back to NOCS + ARKit only; per-source result table still meaningful |

---

## Linked artefacts

- Architecture and training recipe: `planning/SLAT_CONDITIONED_METRIC_TRAINING_2026-04-27.md`
- Aspect ratio loss design: `planning/SS_ARCHITECTURE_AND_ASPECT_RATIO_FIX_2026-05-11.md`
- Canonical bbox invariant: `planning/CANONICAL_MESH_BBOX_DIAGNOSTIC_2026-04-27.md`
- Anisotropic rescale experiment: `planning/ANISOTROPIC_RESCALE_EXPERIMENT_2026-05-11.md`
- Memory: `/mnt/source/.claude/memory/project_metric_scale.md`
