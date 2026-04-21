# Metric Scale Recovery — Architecture Sketch

Full pipeline from input image to metric dimensions output.
New trainable components are marked. Everything else is frozen during Phase 1 fine-tuning.

```
INPUT
─────
  RGB Image + Mask
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
  ┌─────────────┐                              ┌─────────────┐
  │    MoGe     │  (frozen, always runs)       │    DINO     │  (frozen)
  │ depth model │                              │   image     │
  └──────┬──────┘                              │  features   │
         │                                     └──────┬──────┘
   metric pointmap                                    │
   [H, W, 3]                                   image tokens
         │                                     [B, N, 768]
         │                                            │
         ▼                                            │
  ┌─────────────────┐                                 │
  │ SSI Normalizer  │  (existing)                     │
  └────────┬────────┘                                 │
           │                                          │
   pointmap_scale [B,3]                               │
   pointmap_shift [B,3]                               │
           │                                          │
           │           ╔══════════════════════════════╪═══════╗
           │           ║   STAGE 1 — SS Generator     │       ║
           │           ║   (frozen)                   │       ║
           │           ║                              ▼       ║
           │           ║              ┌───────────────────┐   ║
           │           ║              │  Condition Embed  │   ║
           │           ║              │  (DINO tokens)    │   ║
           │           ║              └────────┬──────────┘   ║
           │           ║                       │               ║
           │           ║              ┌────────▼──────────┐   ║
           │           ║              │  Flow Matching    │   ║
           │           ║              │  Transformer      │   ║
           │           ║              │  (MOT / DiT)      │   ║
           │           ║              └────────┬──────────┘   ║
           │           ║                       │               ║
           │           ║              shape_latent [B,4096,8]  ║
           │           ║                       │               ║
           │           ║              ┌────────▼──────────┐   ║
           │           ║              │   SS Decoder      │   ║
           │           ║              └────────┬──────────┘   ║
           │           ║                       │               ║
           │           ║                voxel coords          ║
           │           ╚═══════════════════════╪═══════════════╝
           │                                   │
           │              shape_latent ─────── │ ────────┐
           │              (detached)           │         │
           │                                   │         ▼
           │                                   │  ┌─────────────────────┐
           │                                   │  │   MetricScaleHead   │  ◄─ TRAINABLE (new)
           │                                   │  │                     │
           └───────────────────────────────────┼──►  pool(latent) [B,8] │
                                               │  │  + log(ps)    [B,1] │
               pointmap_scale ─────────────────┘  │  + shift_z    [B,1] │
               pointmap_shift                      │  ──────────────     │
                                                   │  MLP 10→64→128→768  │
                                                   └──────────┬──────────┘
                                                              │
                                                   scale_token [B, 1, 768]
                                                              │
                           ╔═══════════════════════════╗     │
                           ║  STAGE 2 — SLAT Generator ║     │
                           ║  (frozen)                 ║     │
                           ║                           ║     │
                           ║  ┌─────────────────────┐  ║     │
                           ║  │  Condition Embedder  │  ║     │
                           ║  │  (DINO tokens)       │  ║     │
                           ║  │  + scale_token ◄─────╫──╫─────┘
                           ║  │  [B, N+1, 768]       │  ║  ◄── injected via proxy
                           ║  └──────────┬───────────┘  ║
                           ║             │               ║
                           ║  ┌──────────▼───────────┐  ║
                           ║  │  Sparse Flow Match.  │  ║
                           ║  │  Transformer (SLAT)  │  ║
                           ║  │  cross-attn on cond  │  ║
                           ║  └──────────┬───────────┘  ║
                           ║             │               ║
                           ║       SLAT latent           ║
                           ║    SparseTensor [V, 8]      ║
                           ╚═════════════╪═══════════════╝
                                         │
                    ┌────────────────────┤
                    │                    │
                    ▼                    ▼
           ┌──────────────┐   ┌──────────────────────────┐
           │ SLAT Decoder │   │   MetricScaleDecoder     │  ◄─ TRAINABLE (new)
           │  (frozen)    │   │                          │
           └──────┬───────┘   │  pool(SLAT feats) [B,8]  │
                  │           │  + scale_proj(token)[B,16]│
                  │           │  ─────────────────────    │
                  │           │  MLP 24→128→64→3         │
                  │           └─────────────┬────────────┘
                  │                         │
                  ▼                         ▼
         Gaussian Splat             log([w, h, d])
         or Mesh                    → exp()
                                    [width, height, depth]
                                    in metres
```

## Legend

```
  ╔══╗  frozen (no gradient during fine-tuning)
  │  │  existing component
  ◄─┘  new trainable component
  ───►  gradient flows here during fine-tuning
```

## New Components

| Component | File | Parameters |
|-----------|------|------------|
| `MetricScaleHead` | `sam3d_objects/model/backbone/scale_head.py` | MLP: 10 → 64 → 128 → 768 |
| `MetricScaleDecoder` | `sam3d_objects/model/backbone/metric_scale_decoder.py` | MLP: 24 → 128 → 64 → 3 |
| `_ScaleAugmentedEmbedderProxy` | `sam3d_objects/model/backbone/scale_head.py` | No parameters — proxy wrapper |

## Gradient Flow During Fine-Tuning

```
GT metric dims
    └─► Smooth L1 loss on [w, h, d]
            └─► MetricScaleDecoder weights          (trained)
            └─► scale_token (direct input to decoder)
                    └─► MetricScaleHead weights     (trained)
```

Gradients through the SLAT generation process are blocked (frozen + `no_grad`).
SLAT receives the scale token as conditioning but does not yet learn to use it —
that requires a subsequent phase with SLAT's cross-attention weights partially unfrozen.

## Phase 2 Extension (future)

To make SLAT actively use the scale token, partially unfreeze SLAT's
cross-attention K/V projection weights for the new token position and
run a second fine-tuning phase. This allows the generation process itself
to become scale-aware, not just the decoder readout.
