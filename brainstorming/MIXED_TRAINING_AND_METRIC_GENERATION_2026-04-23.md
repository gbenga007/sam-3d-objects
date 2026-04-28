# Mixed OmniNOCS Training and Scale-Conditioned Mesh Generation - 2026-04-23

Brainstorm covering two adjacent next steps:

1. Scaling metric-dim prediction to the full OmniNOCS mix (NOCS-Real275,
   Objectron, ARKitScenes, Hypersim) and validating it properly.
2. Turning the metric scale token into an actual conditioning signal for SLAT
   mesh generation, given that this step needs a mesh dataset rather than a
   NOCS-style bounding-box dataset.

These are two different problems despite looking like "the next two steps".

---

## Part 1 - What mixed large-scale training actually buys

The plumbing already exists: `OmniNOCSObjectDataset` handles all four sources,
the training script takes `--dataset omninocs-mixed` and per-source RGB roots,
and feature caching still works. The interesting question is what bigger data
gets us beyond raw size.

### What scales with data

- **Coverage breadth.** NOCS's 6 tabletop categories expand into Objectron's 9
  handheld categories, ARKitScenes's room-scale indoor furniture, and
  Hypersim's synthetic diversity. The size distribution goes from roughly
  5 cm - 30 cm tabletop items to 2 cm - 3 m. That distribution jump is the
  real test of metric generalization.
- **Cross-source category check.** Camera and laptop appear in both NOCS and
  Objectron. If per-source MAPE on the same category diverges a lot, that is a
  calibration signal (domain shift in MoGe's `pointmap_scale`, not head
  capacity).
- **MoGe stress test for free.** Hypersim is synthetic, so MoGe's depth
  behaviour will differ from real frames. It becomes a cheap sim-to-real
  diagnostic without leaving the dataset.

### What data will not fix

- **The bottle/cup transparency failure mode.** That is a MoGe property, not a
  data-quantity property. More bottles in the training set just teaches the
  head "trust MoGe less when it looks like a bottle" - it does not raise the
  ceiling.
- **Categories with low MoGe r.** Per the overnight correlation check, some
  categories have near-zero r between MoGe scale and GT scale. Those remain
  variance-bound; the head can only use shape priors for them.

### Validations worth running before anything else

1. **Per-source held-out.** Train on three sources, hold out one, for each of
   the four sources in turn. Cheap and it is the real OOD check.
2. **Size-distribution MAPE.** Bin the held-out set by `log(GT volume)` into
   deciles and report MAPE per bin. Dim prediction often looks great on
   average and terrible at the tails; a single scalar MAPE hides this.
3. **Leave-one-category-out across sources.** For example hold out all
   `camera` records from all sources. Tests whether the head has learned a
   category-agnostic geometry-to-scale mapping or is leaning on shape priors
   per category.
4. **WildRGB-D OOD.** Only meaningful after the three above. Before them, a
   failure on WildRGB-D could be in-distribution tail variance, not true OOD.

### Tradeoff for Part 1

Scaling the data before doing the bottle/cup opaque-paint preprocessing fix
probably wastes compute. Those categories cap per-source numbers today, and
the fix is nearly free because we already have the instance masks. Ablation
should come first; scaling should come second.

---

## Part 2 - Scale-conditioned mesh generation

A 3D bounding box constrains extent; it says nothing about where mass goes
inside that extent. To train a generator that respects metric scale, we need
`(image, metric-scaled mesh)` pairs, not `(image, bbox)` pairs.

That is why mesh datasets matter here and NOCS-style ones do not suffice.

But we do not have to jump straight to "download ABO and retrain SLAT". There
is a ladder from cheap to expensive, and the cheap rungs may answer most of
the question on their own.

### Rung 1 - Token-dim alignment and inject with no SLAT retrain

Current blocker at inference is trivial: the trained scale token is 768-d but
live SLAT condition tokens are 1024-d, so `_ScaleAugmentedEmbedderProxy`
silently skips injection.

Concrete change:

- Retrain `MetricScaleHead` with `ctx_channels=1024`. Everything else is
  identical. Reuse the existing cached features by re-running only the head.
- Turn on `_ScaleAugmentedEmbedderProxy` injection at inference.
- Do not retrain SLAT. Frozen SLAT cross-attn will at least glance at the new
  token position.

Measure:

- Mesh quality, qualitatively.
- Dim MAPE computed from the generated-mesh bbox versus GT.

Interpretation:

- If MAPE improves at all, even slightly, frozen cross-attn is extracting
  something from the injected token. That is real evidence before any
  unfreezing work.
- If MAPE is unchanged, the token is not reaching anything useful through
  frozen attention, and Rung 2/3 become necessary.

### Rung 2 - Differentiable-render bbox loss (still no mesh dataset)

The SLAT decoder is differentiable. The generated mesh's axis-aligned bbox is
differentiable: it is just `min`/`max` on vertex coordinates. So on any dataset
where we have GT metric dims - i.e. the entire OmniNOCS mix - we can backprop
a dim-consistency loss:

```text
L_dim = SmoothL1( log(bbox(generated_mesh)),
                  log(GT_dims) )
```

Training recipe:

- Align scale token to 1024-d per Rung 1.
- Partially unfreeze SLAT cross-attention K/V projections for the scale token
  position only.
- Loss: flow-matching loss (frozen for non-scale tokens) + `L_dim`.
- No new dataset required.

What this does and does not do:

- Correctly constrains extent, so generated meshes become metric.
- Does not constrain internal geometry. The generator will still produce its
  prior-shaped mesh inside a correctly sized box.
- This is the first time the *generator* itself is scale-trained. It is a real
  capability jump over Rung 1.

### Rung 3 - Proper mesh-dataset fine-tune

This is where a mesh dataset is actually required, and it is the right place
to spend that effort only after Rungs 1-2 either fail or plateau.

Candidates ranked by fit:

- **ABO (Amazon Berkeley Objects).** ~8k products, professional scans, metric
  dims measured in millimetres, multi-view renders plus real photos. Clean
  metric scale, image-pair-ready, license permits research. Probably the best
  single fit.
- **OmniObject3D.** 6k scans, 190 categories, all metrically scaled. More
  categorical diversity than ABO but fewer image pairs per mesh.
- **GSO (Google Scanned Objects).** 1k household objects, pristine metric
  scans plus turntable images. Small but essentially noise-free. Best as a
  calibration/eval set rather than primary training.
- **NOCS CAMERA / REAL CAD meshes.** The ShapeNet meshes behind NOCS are
  available, and NOCS-REAL275 has real-world metric scale metadata. This lets
  us bootstrap without leaving the dataset we already have. Narrow category
  set, but zero new plumbing.

### How the dim predictor transfers into the generator

What we learned in Part 1 feeds Part 2 in four concrete ways:

1. **Warm-start the scale token.** Initialize the scale-conditioned
   generator's head from the trained `MetricScaleHead` (with the
   `ctx_channels=1024` retrain from Rung 1). It already encodes
   `(image features + pointmap stats) -> real-world scale`; the generator
   just needs to learn to *use* it in generation rather than only at readout.
2. **Preserve MoGe as a metric anchor.** Pipeline-wide. `pointmap_scale` and
   `pointmap_shift` remain the most important signal into the token; the
   generator learns to consume the token, not replace the anchor.
3. **Keep the readout head as auxiliary supervision.** During generator
   fine-tuning, keep training the metric-readout head on the full
   NOCS+Objectron+ARKitScenes+Hypersim mix. It is a cheap regularizer - it
   forces the SLAT features the generator emits to still produce correct dims
   when read out. This is essentially a consistency constraint between
   "generated shape" and "predicted dim".
4. **Per-category failure maps.** The Part 1 study tells us where MoGe lies.
   When training the generator, downweight or reweight-up those categories
   accordingly. Do not let bottle/cup drag down the flow-matching objective.

---

## Recommended sequencing

1. **Bottle/cup opaque-paint ablation on NOCS.** Cheap. Either raises the
   ceiling or confirms the remaining failure is structural.
2. **Scaled mixed training + the three validations above** (per-source,
   size-binned, leave-one-category). This is the real Part 1 deliverable and
   the first defensible benchmark.
3. **Rung 1: dim-aligned token inject, no SLAT retrain.** Small code change,
   short train. Measures whether the free path works before any unfreezing.
4. **Rung 2: diff-render bbox loss on the same OmniNOCS mix.** First time the
   generator is scale-trained. Still no new dataset required.
5. **Rung 3: ABO or GSO fine-tune.** Only after Rungs 1-2 have either failed
   or plateaued. That is when a metric mesh dataset actually buys something.

Big tradeoff: Rungs 3-4 are much cheaper than Rung 5 and could answer 80% of
the question on their own. The usual failure mode is to skip to Rung 5 and
spend weeks on dataset wrangling before realizing the simpler path would have
worked. Worth proving Rungs 1-2 insufficient before committing to a mesh
dataset pipeline.

---

## Open questions worth resolving before Rung 3

- Is the bbox loss from Rung 2 enough to also regularize mesh *shape*, not
  just extent? Intuition says no; worth a small experiment on NOCS where we
  also have GT meshes (via ShapeNet CAD) to measure shape MAPE alongside dim
  MAPE.
- How does MoGe handle Hypersim's synthetic frames? If the synthetic-to-real
  gap in `pointmap_scale` is large, Hypersim may need to be down-weighted in
  the mix or excluded from the primary dim head training, even if it is
  useful for diversity tests.
- For Rung 3, do we want one mesh dataset or a mix? ABO plus GSO together
  give a bigger set with consistent metric scale; OmniObject3D alone gives
  more category coverage. This is a deliberate choice, not a default.
