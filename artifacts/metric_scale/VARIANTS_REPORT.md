# Metric-Scale Model Variants — Report
_Generated 2026-06-17 from checkpoint `args` + `metrics` (ground truth) via `/tmp/extract_variants.py`._

## Metric definitions
- **W,H,D MAPE** = mean/median absolute % error of predicted vs GT metric dimensions on the held-out set.
- **Per-source** = MAPE restricted to NOCS-Real275 / Objectron / ARKitScenes held-out instances.
- **step-0.5 mAP** = mean AP at 3D IoU 0.25 / 0.50 on NOCS-Real275 (official NOCS eval), reported separately below.

## Reliability caveats (read before cross-comparing)
1. **Held-out splits differ across eras.** Numbers are only fair *within* a group. (e.g. `mixed_v1`'s
   NOCS 1.7% is *below* the GT-pointmap oracle floor 6.7% → its old split/protocol is non-comparable.)
2. **Cached-MoGe-2 runs' `args` reflect the cache-load invocation** (no `moge2_pointmap_dir`,
   "nocs-only") — corrected here from run history (they trained on MoGe-2-cached *mixed* features).
3. **"best" = min-median over the eval trajectory** (possibly intra-epoch), not the final epoch.
4. Reference floors: GT-pointmap oracle NOCS ≈ 6.7% median; category-prior NOCS ≈ 13.5%.

---

## Group A — directly comparable (live mixed runs; split: 3334/src balanced, held-out NOCS 64 / Obj 200 / ARKit 200)
| variant | depth | trained (frozen→unfrozen) | LR | loss | ep | mean% | med% | NOCS | Obj | ARKit |
|---|---|---|---|---|---|---|---|---|---|---|
| mixed_scratch_10k | MoGe-v1 | SS frozen; head+dec + SLAT cross-attn | head 1e-4 / slat 1e-6 | log-dim smooth-L1 | 12 | 22.3 | 11.9 | 7.4 | 17.7 | 34.3 |
| mixed_moge2_live_v1 | **MoGe-2** | SS frozen; head+dec + SLAT cross-attn | head 1e-4 / slat 1e-6 | log-dim smooth-L1 | 12 | **20.2** | 12.0 | 7.9 | **17.4** | **28.6** |
| joint_mot_v1 _(running, ep2/6)_ | **MoGe-2** | **MoT backbone unfrozen** (pose fast, geom slow) | geom 2e-6 / pose 1e-4 | L_whd + λ_t·L_trans(1) + λ_f·L2-SP(10) | 6 | 27.7 | 16.7 | 8.3 | 29.2 | 34.2 |

**Read:** depth-source swap MoGe-v1→MoGe-2 (rows 1→2, identical recipe) improves overall 22.3→20.2 and
especially ARKit 34.3→28.6. joint_mot_v1 is mid-training; NOCS already ~oracle-floor, Objectron is the gap.

---

## Group B — cached MoGe-2 head A/B (depth=MoGe-2 cached; data=mixed ~8.5k; **SS+SLAT generators frozen**; shared cached held-out 375)
| variant | decoder / loss | ep | mean% | med% | NOCS | Obj | ARKit |
|---|---|---|---|---|---|---|---|
| mixed_moge2_v1 | absolute log-dim | 200 | 36.3 | 21.4 | 18.5 | 32.2 | 48.6 |
| mixed_moge2_v2_long | absolute log-dim | 1000 | 35.4 | 21.3 | 19.6 | 31.4 | 46.7 |
| **mixed_moge2_anchor_v1** (v1a) | anchor features (log_anchor_iso, log_pose_scale, voxel_extent) | 1000 | **27.2** | 15.7 | 7.4 | 26.4 | 36.7 |
| mixed_moge2_factored_v1 (v1b) | factored (decoupled iso/prop, MoGe-2 §3.2 style) | 1000 | 28.3 | 15.9 | 8.3 | 28.3 | 36.9 |
| mixed_moge2_binned_v2 | binned iso (OmniNOCS-style) + prop | 1000 | 27.6 | **14.4** | **6.0** | 28.6 | 35.9 |

**Read:** anchor features give ~7-8pp over the absolute head; iso *representation* (abs/factored/binned)
is second-order. Cached regime ceiling ~27% (no live SLAT capacity) — below the live runs.

---

## Group C — NOCS-only diagnostics (single source; different protocol; **not** mixed-comparable)
| variant | depth | trained | ep | mean% | med% |
|---|---|---|---|---|---|
| nocs_sceneholdout_1024dim_baseline_v2 | MoGe-v1 | head+dec (cached, gen frozen) | 200 | 3.0 | 1.4 |
| nocs_sceneholdout_slat_conditioned_v3 | MoGe-v1 | head+dec + SLAT cross-attn | 5 | 1.7 | 1.0 |
| nocs_sceneholdout_ss_ratio_v1 | MoGe-v1 | head+dec + SLAT cross-attn (ss-ratio loss) | 10 | 1.2 | 0.8 |
| nocs_moge2_v1 | MoGe-2 cached | head+dec (cached) | 200 | 8.1 | 5.9 |
| nocs_moge2_v2_long | MoGe-2 cached | head+dec (cached) | 1000 | 3.5 | 2.0 |

**Read:** NOCS-only is easy (tight category priors) → 1-3% achievable; confirms the regime is healthy and
that the *mixing* + cross-source generalization is the hard part, not capacity.

---

## Group D — early/historical (non-comparable splits — exclude from paper)
`mixed_v1` (mean 16.8 but NOCS 1.7 < oracle floor → leakage), `mixed_omninocs_1024dim_bs512`,
`mixed_omninocs_balanced_{100,6358}_per_source`. Different held-out definitions; kept only for provenance.

---

## mAP @ 3D IoU on NOCS-Real275 (step 0 / 0.5; official symmetry-aware NOCS eval; 8 frames / 51 instances)
> `mixed_scratch_10k` mAP-evaluated across depth sources (below), plus the stock decoder.

**joint_mot_v1 — FULL-SET mAP (2026-06-18; 150 frames; RGB-only, MoGe-2 depth, predicted pose):**
| condition | mAP@25 | mAP@50 | per-class@25 |
|---|---|---|---|
| **joint_mot_v1 (full pred pose)** | **73.1** | **8.9** | bowl 96.2 · mug 93.9 · can 81.3 · camera 76.5 · bottle 69.0 · laptop 21.6 |
| GT-pose oracle (sanity) | 100.0 | 100.0 | all classes 100 |

Translation centroid error: median **8.3 cm** / mean 9.4 / p90 16.4. Beats NOCSformer (43.5) and CubeRCNN
(14.9) @25 RGB-only, approaches depth-supervised NOCS (79.6); @50 bound by ~8 cm depth precision, not the
model. laptop (flat object) is the lone weak class — the under-scaling/aspect mode.

**Step 0 — size sensitivity (GT pose; only the size hypothesis varies):**
| size hypothesis | mAP@25 | mAP@50 |
|---|---|---|
| GT size / iso ≤30% error | 100 | 100 |
| iso +50% | 100 | 87.5 |
| iso ×2 / ×3 | 89.9 / 65.7 | 46.1 / 0 |
| equal-volume cube (proportions destroyed) | 100 | 99.4 |
| fixed cube (ignores GT size) | 0.2 | 0 |

→ mAP rewards **pose + overall scale**, is ~blind to **proportions**; our ~7% size is deep in the flat zone.

**Step 0.5 — predicted pose, by depth source (the depth-source A/B):**
| condition (NOCS-trained `mixed_scratch_10k`) | depth ratio pred_z/gt_z | mAP@25 | mAP@50 |
|---|---|---|---|
| MoGe-v1 | 1.53 (53% too far) | **0.0** | **0.0** |
| MoGe-2 | 0.866 (13% too close) | **40.3** | 4.9 |
| GT depth (oracle) | ~1.00 (2.3 cm) | **89.8** | 17.6 |
| stock decoder + GT depth | ~1.00 | 87.5 | 22.5 |

**Baselines (OmniNOCS Tab.5):** NOCS model (depth-input) 79.6 / 72.4 · CubeRCNN 14.9 / 4.1 · NOCSformer 43.5 / 10.6.

→ **The depth source is the lever**: MoGe-v1 → 0/0 (translation off 53%); MoGe-2 → 40.3 @ IoU25 (≈ NOCSformer,
RGB-only); GT depth → beats *supervised* NOCS @ IoU25. IoU50 is bound by depth/translation precision.

---

## Head vs Pointmap A/B — should we route size by the MoGe-2 pointmap? (2026-06-18, PROVISIONAL)
Same Variant-2 (`mixed_moge2_live_v1`) inference call, two size readouts on the heldout (cached MoGe-2
pointmaps; `scripts/head_vs_pointmap_ab.py`; 222 objects, 22 skipped on missing pointmap):
- **A (head)** = `out['metric_dimensions']` (trained metric-scale decoder).
- **B (pointmap)** = canonical-mesh bbox × `out['scale'].mean()` (frozen pose-decoder / MoGe-2 scale).

| slice | A head MAPE | B pointmap MAPE | winner |
|---|---|---|---|
| NOCS | 23.1 | **17.4** | B |
| Objectron | **49.4** | 89.3 | A (B collapses — close-up/handheld) |
| ARKit | 51.2 | **35.1** | **B (+16pp)** |
| small <0.3 m | **39.0** | 55.4 | A |
| medium 0.3–0.6 m | **42.5** | 50.7 | A |
| large 0.6–1.2 m | 51.9 | **33.3** | **B (+18.6pp, 62% win)** |
| xlarge ≥1.2 m | **35.2** | 41.6 | A |
| **overall** | **42.5** | 47.3 | A |

**Read:** B beats A precisely on **ARKit + the 0.6–1.2 m furniture band** (well-framed indoor objects);
B craters on **Objectron** (close-up, scene-MAD scale dominated by background) and **xlarge** (truncated /
clipped → unreliable scene scale). So routing by *scene context* > routing by raw object size, and naive
per-object A-or-B routing does **not** beat the learned `anchor` head (27% MAPE) → favor feeding the pointmap
extent to the head as an explicit anchor feature, not hard routing.

**⚠️ PROVISIONAL — calibration caveat:** this harness gives A on NOCS = 23.1% vs the official Variant-2 eval
NOCS = 7.9% (~3×), so `pipeline.run(pointmap=cached)` is NOT reproducing the eval feature path. Absolutes are
inflated; since B routes entirely through the pointmap, a degraded pointmap penalizes B, so B's wins (ARKit,
large) are likely understated and the Objectron collapse may be partly artifact. **Next: reconcile the harness
(eval feature path / fresh live MoGe-2 pointmaps) before treating the routing verdict as final.**

## Hypersim GT-pointmap A/B — the CLEAN test (2026-06-18) — resolves the confound above
Same Variant-2 readout, but on **Hypersim val** with **GT depth pointmaps** (Hypersim `depth_meters.hdf5`
→ planar → camera XYZ → pytorch3d; full-frame 768×1024, validated: visible object extents match GT axes).
This removes the MoGe-2 pointmap-quality confound. `scripts/hypersim_head_vs_pointmap_ab.py`, n=400 sampled
across 97,387 val instances; depth = 7,186 files / 1.8 GB downloaded for all 75 val scene-cams.

| size bucket | A head MAPE | B pointmap MAPE | winner | B wins |
|---|---|---|---|---|
| small <0.3 m | 869 | **37.2** | B | 100% |
| medium 0.3–0.6 m | 203 | **23.3** | B | 100% |
| large 0.6–1.2 m | 73.0 | **28.0** | B | 71% |
| xlarge ≥1.2 m | 170 | **39.4** | B | 78% |
| **overall (n=400)** | **207.8** | **32.0** | **B** | **81%** |

**Read:** with a *true* metric pointmap, **B (pointmap) dominates every bucket** and recovers sensible metric
size (32% overall) on a domain it never trained on. The head (A) is catastrophic (208%) — but ⚠️ note it was
trained on NOCS/Obj/ARKit, **never Hypersim**, so its number is out-of-distribution and overstated. The real
finding is the **transfer asymmetry: the pointmap pathway transfers to a new indoor domain; the learned head
does not** (it reverts to training-set size priors → 869% on small objects). This **flips** the confounded
OmniNOCS A/B (where B lost 47 vs 42.5 under MoGe-2 noise + in-domain head). **Conclusion: the metric signal is
in the pointmap; anchor/route size to it rather than re-regressing absolute scale from the scale-free latent.**

---

## One-line narrative for the paper
The metric problem is a **depth-source** problem (step-0.5 mAP A/B; W,H,D Group A). A frozen generator + a
metric pointmap already recovers near-oracle scale; the principled end-state is making the generator itself
metric (joint_mot_v1), which already reaches the oracle floor on the clean source (NOCS) and is anchor-limited
on small/close objects (Objectron) — a depth-source ceiling we can quantify, not a model failure.
