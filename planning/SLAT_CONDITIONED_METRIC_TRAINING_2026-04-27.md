# SLAT-Conditioned Metric Scale Training - 2026-04-27

## Goal

Advance beyond frozen-SLAT metric readout to a pipeline where the SLAT
generator itself is conditioned on the metric scale token. The target
formulation is:

```
canonical mesh prediction  +  separate accurate metric-size prediction
```

The SLAT generator learns to produce geometry that is consistent with the
metric token. A `MetricScaleDecoder` then reads the metric dimensions from
the scale-aware SLAT latent.

---

## Architecture

### Conditioning flow

```
MoGe pointmap (scale, shift_z)
    +                           ──► MetricScaleHead ──► scale_token [B, 1, 1024]
SS latent (mean-pooled [B, 8])                                 │
                                                               ▼
                                               _ScaleAugmentedEmbedderProxy
                                               appends token to DINOv2 sequence
                                               → condition [B, L+1, 1024]
                                                               │
                                                               ▼
                                               SLatFlowModel (stage 2)
                                               24 × ModulatedSparseTransformerCrossBlock
                                               ├─ self_attn   (frozen)
                                               ├─ cross_attn  ← TRAINABLE
                                               └─ FFN + adaLN (frozen)
                                                               │
                                                               ▼
                                               scale-aware SLAT feats
                                                               │
                                                               ▼
                                               MetricScaleDecoder
                                               → [width, height, depth] in metres
```

### Why 1024-dim token

`cond_channels: 1024` in `slat_generator.yaml` — the cross-attention K/V
projection `to_kv: Linear(1024 → 2048)` accepts context of exactly 1024-dim.
The token is appended directly to the DINOv2 condition sequence with no
dimension bridging.

---

## SLatFlowModel Structure

From `slat_generator.yaml` and `structured_latent_flow.py`:

| Component | Detail |
|---|---|
| Backbone class | `SLatFlowModelTdfyWrapper` → `SLatFlowModel` |
| Transformer blocks | 24 × `ModulatedSparseTransformerCrossBlock` |
| model_channels | 1024 |
| cond_channels | 1024 |
| num_heads | 16 |
| in/out_channels | 8 (SLAT latent dim) |
| dtype | float16 |
| U-Net IO | `io_block_channels: [128]`, `num_io_res_blocks: 2`, `patch_size: 2` |

Each of the 24 blocks contains:

```
norm1    LayerNorm(1024, affine=False)
adaLN    Linear(1024 → 6144)  mod: shift/scale/gate for self-attn + FFN
self_attn
  to_qkv Linear(1024 → 3072)
  to_out  Linear(1024 → 1024)
norm2    LayerNorm(1024, affine=True)   ← learned weight + bias
cross_attn
  to_q   Linear(1024 → 1024)           ← queries from voxel features
  to_kv  Linear(1024 → 2048)           ← K/V from condition (incl. scale token)
  to_out Linear(1024 → 1024)
norm3    LayerNorm(1024, affine=False)
FFN      SparseLinear(1024 → 4096 → 1024)
```

---

## Trainable Parameters

| Component | Params | Note |
|---|---|---|
| `cross_attn.to_q` × 24 | ~25M | queries from voxels |
| `cross_attn.to_kv` × 24 | ~50M | **maps scale token to K/V** |
| `cross_attn.to_out` × 24 | ~25M | output projection |
| `norm2` × 24 | ~50K | affine LayerNorm before cross-attn |
| `MetricScaleHead` | ~200K | newly initialised |
| `MetricScaleDecoder` | ~100K | newly initialised |
| **Total** | **~100M** | rest of SLAT frozen |

Frozen: self-attn, adaLN, FFN, input/output U-Net blocks, condition_embedder
(DINOv2), SS generator, MoGe.

---

## Gradient Flow

```
Loss (Smooth L1, log-dim space)
  ↓
MetricScaleDecoder(slat_feats, scale_token)  ← no detach on either
  ↓ (via pooled slat_feats → SLAT blocks)
SLatFlowModel.blocks[*].cross_attn.to_out
  ↓
cross_attn.to_kv([DINOv2_tokens; scale_token])
  ↓ (gradient to scale_token directly)
MetricScaleHead
```

Two gradient paths to `MetricScaleHead`:
1. **Direct**: `decoder.scale_proj(scale_token) → loss`
2. **Via SLAT**: `decoder(slat_feats) → cross_attn.to_kv(scale_token) → loss`

---

## Why the Existing Cache Cannot Be Used

`encode_metric_scale_features` runs SLAT without scale token injection and
saves the resulting `slat_feats`. Those pre-baked features are frozen and
come from before any SLAT cross-attention training.

For SLAT to actually learn from the scale token, SLAT must be re-run every
training step with the *current* scale token injected. This means:
- No `--cache-latents` / `--load-feature-cache` when `--unfreeze-slat-cross-attn`
- Every step: MoGe → SS (no_grad) → MetricScaleHead → inject → SLAT (grad) → decoder
- `--stage2-steps 1` is essential to keep SLAT a single forward pass

Evaluation can still use a pre-baked eval cache (`--eval-feature-cache`) as a
cheap proxy metric. That cache measures readout quality from frozen SLAT features
and is a conservative lower bound.

---

## Training Configuration

## Compute Budget

Forward-only pipeline (caching) runs at ~0.5s/example on A100-SXM4-80GB.
Live training (forward + backward through 24 SLAT blocks) is ~1.5–2× slower,
so ~0.75–1.0s/example.

| Config | Steps | Estimated time |
|---|---|---|
| 200 epochs × 12882 | 2.58M | ~22 days — not feasible |
| **20 epochs × 12882 (Option A)** | **258K** | **~2.2 days** |
| 200 epochs × 2000 (Option B) | 400K | ~3.5 days |

**Option A chosen**: 20 epochs, full scene-heldout split. Same train/eval data as
the 3.09% MAPE frozen-SLAT baseline, so any improvement is directly attributable
to SLAT conditioning rather than data differences.

---

## Training Command

```bash
env LIDRA_SKIP_INIT=1 ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
/root/.local/bin/micromamba run -n sam3d-objects \
python sam3d_objects/training/finetune_metric_scale.py \
  --annotations-root /mnt/dest/OmniNOCS/omninocs_release_nocs_real275 \
  --rgb-root /mnt/dest/OmniNOCS/real_test \
  --split train \
  --split-group scene \
  --overfit-samples 12882 \
  --heldout-samples 3228 \
  --seed 42 \
  --epochs 20 \
  --batch-size 1 \
  --lr 1e-4 \
  --stage1-steps 1 \
  --stage2-steps 1 \
  --device cuda \
  --inject-scale-token-into-slat \
  --unfreeze-slat-cross-attn \
  --slat-lr 1e-5 \
  --eval-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt \
  --eval-every 5 \
  --wandb \
  --wandb-mode online \
  --wandb-project sam3d-metric-scale \
  --wandb-entity reformed-tulip \
  --wandb-run-name nocs_sceneholdout_slat_conditioned_1024dim \
  --metrics-output artifacts/metric_scale/metrics/nocs_sceneholdout_slat_conditioned_1024dim_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/nocs_sceneholdout_slat_conditioned_1024dim_manifest.json \
  --output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_1024dim.pt
```

Key flags:
- `--epochs 20`: ~2.2 days at 0.75s/step on A100-80GB
- `--eval-every 5`: eval at epochs 5, 10, 15, 20 using the pre-baked frozen-SLAT cache
- `--unfreeze-slat-cross-attn`: unfreezes `cross_attn` + `norm2` in all 24 SLAT blocks (~100M params)
- `--inject-scale-token-into-slat`: appends scale token to SLAT condition before each denoiser step
- `--slat-lr 1e-5`: 10× lower LR for SLAT cross-attn to limit drift of DINOv2 conditioning
- `--eval-feature-cache`: pre-baked frozen-SLAT features used for cheap proxy eval
- `--stage2-steps 1`: single-step SLAT makes backprop through 24 blocks tractable

---

## CFG Subtlety (Inference)

`SLatFlowModelTdfyWrapper` has `force_zeros_cond: true`. During inference, the
unconditional CFG pass calls `condition_embedder(...)` and then zeros the
output. The `_ScaleAugmentedEmbedderProxy` wraps `condition_embedder`, so its
output (DINOv2 + scale token) is zeroed together. This is correct — the
unconditional pass should not see any conditioning. No action needed.

---

## Experiment Matrix

| Run | SLAT frozen? | inject token? | Purpose |
|---|---|---|---|
| Previous (scene-heldout, 768-dim) | Yes | No | Baseline metric readout |
| Previous (scene-heldout, 1024-dim) | Yes | No | Dimension bump ablation |
| **This run** | **cross-attn unfrozen** | **Yes** | **Scale-aware SLAT generation** |

---

## Next Steps After This Run

1. Compare MAPE against the frozen-SLAT baselines.
2. Ablate: metric token injection without cross-attn unfreeze (pure injection into frozen model).
3. Add mesh-supervised dataset (ShapeNet / Objaverse) to evaluate geometry quality improvement.
4. Visualise predicted vs. ground-truth bounding boxes.
5. Consider gradually unfreezing adaLN_modulation if cross-attn alone is insufficient.
