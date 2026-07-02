# TRELLIS Training Practices Adopted - 2026-05-02

Implementation of the actionable items from
`planning/TRELLIS_TRAINING_INSIGHTS_2026-05-01.md`.

Source studied: `/mnt/source/TRELLIS/`
- `trellis/trainers/basic.py` (gradient clip, NaN skip, EMA, FP16 modes)
- `trellis/trainers/flow_matching/mixins/classifier_free_guidance.py` (p_uncond=0.1)
- `trellis/utils/grad_clip_utils.py` (AdaptiveGradClipper)

---

## What was already in place

Pre-existing in `sam3d_objects/training/finetune_metric_scale.py`
(verified before this change):

- `AdaptiveGradClipper` with 95th-percentile rolling buffer (lines 46-77).
- Gradient checkpointing flipped on for all 24 SLAT blocks when
  `--unfreeze-slat-cross-attn` is active.
- NaN guards on the reduced loss and on the clipped grad-norm in both
  the cached and live training paths.
- Two-group AdamW (heads at `lr=1e-4`, SLAT cross-attn at `slat-lr=1e-5`).

---

## What changed in this commit

### 1. CFG-style scale-token dropout (`--p-uncond-scale-token`)

`predict_log_dims` now takes `p_uncond_scale_token: float = 0.0`. With
that probability the SLAT-injected scale token is replaced with zeros
*before* `_ScaleAugmentedEmbedderProxy` appends it to the DINOv2
condition sequence. The `MetricScaleDecoder` still receives the
unzeroed token, so the metric regression signal is preserved on every
step.

The function returns `(log_dims, scale_token_dropped)` so the caller
can log how often dropout fired.

Why: TRELLIS uses `p_uncond=0.1` to stop the denoiser from depending
on conditioning. With our cross-attention unfrozen and the scale token
present on every step, SLAT will otherwise overfit to expecting the
metric token at inference — and degrade gracefully when the token is
absent (frozen-baseline inference) or noisy.

### 2. Step-level wandb logging (`--log-step-every`)

Live training loop emits a step record after each successful
`optimizer.step()`:

| Key | Meaning |
|---|---|
| `train_step/loss` | reduced SmoothL1 for this step |
| `train_step/grad_norm` | total pre-clip gradient L2 norm |
| `train_step/heads_grad_norm` | pre-clip norm restricted to MetricScaleHead+Decoder params |
| `train_step/slat_grad_norm` | pre-clip norm restricted to SLAT cross_attn+norm2 params |
| `train_step/clip_threshold` | current `AdaptiveGradClipper._max_norm` |
| `train_step/clip_buffer_filled` | 0 until 1000 steps, then 1 (threshold has self-calibrated) |
| `train_step/nan_skipped` | running count of NaN-skipped steps |
| `train_step/oom_skipped` | running count of OOM-skipped steps |
| `train_step/scale_token_dropped` | how many micro-batch examples in this step had the token zeroed |
| `train_step/epoch`, `train_step/global_step` | bookkeeping |

Why: epoch-level loss averages hide the single bad batch that destabilises
~100M-param fine-tuning. Per-group grad norms show when one parameter
group (heads vs cross-attn) starts dominating or exploding while the
other looks fine.

### 3. Per-parameter gradient finiteness check

Before computing per-group norms, the loop now does

```python
nan_in_grads = any(
    p.grad is not None and not torch.isfinite(p.grad).all()
    for p in all_params
)
```

and skips the optimizer step (zeroing gradients) if any param has a
non-finite gradient. Mirrors `basic.py:394-408` in TRELLIS. The
previous code relied on `clip_grad_norm_` returning a non-finite total
norm, which is essentially the same signal but goes through a global
sum that can theoretically lose information.

### 4. `AdaptiveGradClipper.log()`

Returns `{"max_norm": float, "buffer_filled": bool}` so the
training loop can surface the current adaptive threshold to wandb.

### 5. Wandb step-counter unification

All `wandb_log` calls in the live training path now use
auto-incrementing step (no explicit `step=` argument). Step-level and
epoch-level metrics share the same monotonic step counter, with
`epoch` and `global_step` recorded as payload keys. This avoids the
conflict between low-valued `step=epoch+1` and high-valued
`step=global_step` that would otherwise drop log lines.

The cached training path also drops its `step=epoch+1` for consistency
(it never logs at step granularity, so behavior is unchanged in
practice — only the x-axis interpretation differs).

### 6. Script update

`scripts/train_slat_conditioned_v1.sh` adds:

```
  --p-uncond-scale-token 0.1 \
  --log-step-every 1 \
```

---

## Deliberately not adopted

| Practice | Why we skipped |
|---|---|
| Logit-normal `t` sampler (mean=1, std=1) | We never call `FlowMatching.compute_loss()`. Our training calls `pipeline.sample_slat()` (the inference path). With `--stage2-steps 1` cross-attn always sees `t=0`. |
| Time-binned loss logging (10 bins) | Same reason: only one bin (`t=0`) is ever populated. |
| EMA at rate 0.9999 | Half-life ~7000 steps. The TRELLIS doc itself notes EMA "lags significantly" during fine-tuning. Our run is short and the trainable subset is small; EMA would mostly track the warm-start. Add later if a longer run is planned. |
| `inflat_all` FP16 master params | The bf16 attention overflow that motivated this in TRELLIS is already fixed in our path by `F.layer_norm` on the scale token (commit e42d85f). Reintroduce only if instability returns. |
| `batch_split` gradient accumulation | Single-GPU + batch_size=1 means it would just be plain accumulation. Lower priority than the dropout/instrumentation changes. |

---

## Files touched

- `sam3d_objects/training/finetune_metric_scale.py`
  - `AdaptiveGradClipper.log()` added.
  - `predict_log_dims` signature: new `p_uncond_scale_token`; return type
    `tuple[torch.Tensor, bool]`.
  - Live training loop: per-group grad norms, explicit per-param NaN
    skip, NaN-skip counter, step-level wandb logging.
  - CLI: `--p-uncond-scale-token` (default `0.0`),
    `--log-step-every` (default `1`).
  - All `wandb_log` calls in the live path drop explicit `step=`;
    `evaluate_and_record` likewise.
- `scripts/train_slat_conditioned_v1.sh`
  - Adds `--p-uncond-scale-token 0.1 --log-step-every 1`.

---

## Validation checklist before launch

1. Eval feature cache present at
   `/tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt`.
   (Currently missing per memory note 2026-05-02 — must be rebuilt.)
2. Warm-start checkpoint present at
   `artifacts/metric_scale/checkpoints/nocs_sceneholdout_1024dim_baseline_v2_best.pt`
   (still on disk, 685KB).
3. After launch, confirm in wandb:
   - `train_step/grad_norm` populated within the first 10 steps.
   - `train_step/scale_token_dropped` shows ~10% non-zero rate.
   - `train_step/clip_buffer_filled` flips to 1 around step 1000 and
     `train_step/clip_threshold` settles below 1.0.
   - `train_step/nan_skipped` stays at 0 in normal operation.
