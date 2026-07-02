# Training Insights from TRELLIS — Flow and SLAT Models
## 2026-05-01

Source: `/mnt/source/TRELLIS/`
Primary files studied:
- `trellis/trainers/flow_matching/flow_matching.py`
- `trellis/trainers/basic.py`
- `trellis/trainers/flow_matching/mixins/classifier_free_guidance.py`
- `trellis/utils/grad_clip_utils.py`
- `trellis/utils/elastic_utils.py`
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json`

These are principles from a working, large-scale SLAT flow model training. Adapt where applicable.

---

## 1. Flow Matching: Train with One Timestep, Infer with Many

**What they do:**
Each training example sees a single randomly sampled timestep `t ∈ [0, 1]`. The denoiser predicts the velocity field at that single `t`. Inference uses 50 fixed Euler steps.

**Loss target — velocity prediction (not noise):**
```python
# flow_matching.py:100-104, 169-171
def get_v(self, x_0, noise, t):
    return (1 - sigma_min) * noise - x_0   # target velocity

target = get_v(x_0, noise, t)
loss = F.mse_loss(pred, target)            # single MSE, no time-weighting
```

**Why this matters for us:**
`--stage2-steps 1` in our training is correct and matches their approach exactly. Do not try to backprop through multiple denoising steps — they never do. One step per training example is sufficient for the flow model to learn the velocity field.

**Timestep schedule — logit-normal, not uniform:**
```python
# flow_matching.py:136-141
t = torch.sigmoid(torch.randn(batch_size) * std + mean)
# SLAT config: mean=1.0, std=1.0 (biased toward middle of [0,1])
```
Logit-normal biases training toward the middle timesteps where the velocity field carries the most signal. Uniform sampling over-samples `t≈0` and `t≈1` which are easy (near noise / near data). For fine-tuning, this is especially relevant — the middle timesteps are where conditioning has the most leverage.

**Timestep scaling before the model:**
```python
# flow_matching.py:167
pred = denoiser(x_t, t * 1000, cond)   # t scaled to [0, 1000]
```
The raw `[0, 1]` timestep is scaled by 1000 before entering the model. The model's time embedding is calibrated for this range.

**sigma_min = 1e-5:**
A small noise floor is added to prevent `x_t` from collapsing to exactly `x_0` at `t=0`. This stabilises the loss landscape at early timesteps.

---

## 2. Gradient Management

### 2a. Adaptive Gradient Clipping

Fixed `clip_grad_norm_(max_norm=1.0)` is a blunt instrument — it clips the same amount regardless of whether the model is in a stable or unstable regime. TRELLIS uses an adaptive clipper that learns the right threshold from the training history.

**Implementation** (`grad_clip_utils.py:7-80`):
```python
class AdaptiveGradClipper:
    # Tracks a rolling buffer of the last 1000 gradient norms
    # Sets clip threshold = 95th percentile of that buffer
    # Hard cap: min(percentile_threshold, max_norm)  [max_norm=1.0]
    # Only updates threshold once buffer is full (1000 steps)
    # Skips update if grad_norm is non-finite

    def __call__(self, parameters):
        grad_norm = clip_grad_norm_(parameters, max_norm=self._max_norm)
        if torch.isfinite(grad_norm):
            self._grad_norm[self._buffer_ptr] = grad_norm
            # ... update buffer ...
            if buffer_full:
                self._max_norm = min(np.percentile(self._grad_norm, 95), self.max_norm)
        return grad_norm
```

**Why better than fixed clipping:**
Sparse models (SLAT operates on variable-length sparse tensors) have highly variable gradient norms depending on sparsity. A fixed threshold will either clip too aggressively on sparse batches or not enough on dense ones. The adaptive clipper naturally adjusts to the model's actual gradient distribution.

**Recommendation for our training:** Replace `clip_grad_norm_(max_norm=1.0)` with `AdaptiveGradClipper(max_norm=1.0, clip_percentile=95, buffer_size=1000)`. The first 1000 steps use the hard cap; after that it self-calibrates.

### 2b. NaN Guard — Skip the Optimizer Step

```python
# basic.py:394-408
if not any(not p.grad.isfinite().all() for p in self.model_params):
    self.optimizer.step()
else:
    print('Warning: NaN detected in gradients. Skipping update.')
```

This is cleaner than the nan/inf detection we currently do at the loss level. Check the *gradients* (not the loss) and skip the optimizer step entirely if any param has a non-finite gradient. This prevents corrupted parameter updates from propagating.

We currently have `nan/inf skip guards` in our training loop but they operate at the loss level. Moving the check to gradient level is more robust — a nan loss will produce nan gradients, but the reverse isn't always true (e.g. gradient explosion via clipping overflow).

---

## 3. FP16 / Mixed Precision — `inflat_all` Mode

For large sparse models, standard `torch.autocast` (AMP) can silently underflow gradients in bfloat16 / float16. TRELLIS uses a manual FP16 scheme called `inflat_all`:

1. Model params stored in fp16 for memory efficiency.
2. A separate copy of all params is maintained in fp32 as "master params".
3. During backward: loss is scaled by `2^log_scale` before `.backward()`, then unscaled before the optimizer step.
4. `log_scale` grows by `fp16_scale_growth=0.001` per step on success, decrements by 1 on NaN.

```python
# basic.py:365-402
scaled_l = l * (2 ** self.log_scale)
scaled_l.backward()
model_grads_to_master_grads(model_params, master_params)
master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))   # unscale
optimizer.step()
master_params_to_model_params(model_params, master_params)
self.log_scale += self.fp16_scale_growth                   # grow on success
# on NaN: self.log_scale -= 1                              # shrink on failure
```

**For our training:** Our training currently runs in bfloat16 (SLAT is `dtype: float16`). The bfloat16 attention overflow we fixed (commit e42d85f) with `F.layer_norm` on the scale token is a symptom of the same underlying issue. If training becomes unstable again, `inflat_all` is the principled fix.

---

## 4. EMA — Rate 0.9999 for Large Models

```python
# basic.py:305-307
for master_param, ema_param in zip(master_params, ema_params):
    ema_param.mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)
# ema_rate = 0.9999
```

At 0.9999, the EMA half-life is ~7000 steps. This is very conservative — it takes thousands of steps for any weight change to be reflected in the EMA model. For large sparse models this is intentional: the EMA model is smoother and more stable for evaluation and inference.

**What this means for fine-tuning:** When fine-tuning only a subset of parameters (e.g., our SLAT cross-attention), a high EMA rate means the EMA model lags the training model significantly. During fine-tuning, run evaluation against the *training* model checkpoints (not EMA), since EMA will dilute the fine-tuning signal for thousands of steps.

---

## Note on Cross-Attention Position in the Flow Matching Loop

Our cross-attention layers ARE inside the flow matching Euler solver — they're called once per inference step in each of the 24 `ModulatedSparseTransformerCrossBlock` blocks. The SLAT generator is a `FlowMatching` model (`reversed_timestamp=False`, `time_scale=1000`).

With `--stage2-steps 1`, `t_seq = linspace(0, 1, 2) = [0.0, 1.0]`. The Euler solver evaluates velocity at t=0 (one step from noise to data). The cross-attention therefore **always sees a timestep embedding of 0** during training — the pure-noise end of the denoising trajectory.

Implication: if the scale token is most informative at mid-range timesteps (as TRELLIS's logit-normal sampler suggests), training exclusively at t=0 leaves signal on the table. Increasing `--stage2-steps` would expose the cross-attention to later timesteps but multiplies memory cost proportionally — already constrained on 40GB with checkpointing enabled.

---

## 5. Classifier-Free Guidance Training

```python
# classifier_free_guidance.py:40-44
mask = list(np.random.rand(B) < self.p_uncond)   # p_uncond = 0.1
cond = torch.where(mask, neg_cond, cond)          # replace 10% with null cond
```

10% of training examples use the null/negative conditioning (zero tensor for image conditions). This is the standard CFG training trick — it teaches the model to generate unconditionally, which enables guidance at inference.

**For our scale token:** When fine-tuning SLAT cross-attention with the metric scale token, consider applying the same dropout — 10% of examples see a zeroed scale token. This prevents the SLAT model from becoming dependent on the scale token and retains the ability to run without it at inference.

The unconditional pass already zeroes the full condition sequence (`force_zeros_cond: true` in our SLAT config), so the scale token is zeroed together with DINOv2 tokens. This is correct. The training dropout on just the scale token is an additional regularisation step worth experimenting with.

---

## 6. Elastic Activation Checkpointing — The 40GB GPU Solution

TRELLIS's most immediately useful technique for our memory constraint.

**The problem:** Storing all intermediate activations during a SLAT forward pass for backward computation takes more memory than the 40GB card has. Hence `--stage2-steps 1` barely fits on 80GB and OOMs on 40GB.

**Their solution** (`elastic_utils.py`):

`LinearMemoryController` dynamically decides what fraction of a module's activations to recompute (vs. store) during the backward pass. It works by fitting a linear model of `memory = k * input_size * mem_ratio + b` from observed stats:

```python
class LinearMemoryController:
    target_ratio = 0.75     # aim for 75% of GPU memory used
    max_mem_ratio_start = 0.5  # start: recompute half of activations
    update_every = 500         # refit linear model every 500 steps
    # every 500 steps: max_mem_ratio += 0.1 (up to 1.0)
    # so over 5000 steps it gradually reduces checkpointing as model learns expected sizes
```

Modules annotated with `ElasticModuleMixin` report their input size and actual mem_ratio used; the controller tells them what ratio to target next step. Internally the module uses `torch.utils.checkpoint.checkpoint()` on blocks proportional to `1 - mem_ratio`.

**For our 40GB situation:** The SLAT model's `SLatFlowModelTdfyWrapper` would need to implement `ElasticModuleMixin`. The core cost is the 24 transformer block activations. Checkpointing those blocks (recompute on backward, don't store activations) would roughly halve the memory footprint at ~1.5-2× compute cost.

A simpler first step: manually apply `torch.utils.checkpoint.checkpoint()` to each of the 24 `ModulatedSparseTransformerCrossBlock` forward calls. This avoids the full elastic infrastructure while achieving most of the memory saving.

---

## 7. Batch Splitting with Gradient Accumulation

```python
# basic.py:354-368
for i, mb_data in enumerate(data_list):      # batch_split=4 sub-batches
    sync_context = model.no_sync() if i != last else nullcontext()
    with sync_context:
        loss = training_losses(**mb_data)
        l = loss['loss'] / len(data_list)    # divide by split count
        l.backward()
# optimizer.step() once, after all splits
```

**Key detail:** DDP gradient sync (`all_reduce`) is suppressed on all but the last batch split. This avoids 4× the communication overhead of a normal DDP step. Only the final split triggers the sync.

For single-GPU training (our case) this is just gradient accumulation, but the pattern is worth adopting: accumulate over multiple mini-batches before stepping, especially if we want to simulate larger effective batch sizes on the 40GB card.

---

## 8. Time-Binned Loss Monitoring

```python
# flow_matching.py:175-182
time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
for i in range(10):
    if (time_bin == i).sum() != 0:
        terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}
```

Track loss per-timestep-bin during training. This reveals which parts of the diffusion trajectory are well-learned vs. still noisy. For fine-tuning with a new conditioning signal (our scale token), you'd expect improvement at mid-range `t` bins first (where the signal is strongest), and slower improvement at `t≈0` (near-noise) and `t≈1` (near-data).

**Actionable:** Add 10 wandb metrics `loss/bin_0` through `loss/bin_9` to our finetune script. This is cheap to compute (one `np.digitize` per batch) and gives a much richer picture of training progress than a single scalar loss.

---

## 9. Freezing Strategy for Fine-Tuning

TRELLIS doesn't do selective unfreezing — they train the full denoiser end-to-end. But their mechanism for identifying trainable parameters is clean:

```python
# basic.py:98-100
self.model_params = sum(
    [[p for p in model.parameters() if p.requires_grad]
     for model in self.models.values()], []
)
```

Everything is driven by `requires_grad`. Freeze what you don't want updated by calling `.requires_grad_(False)` on those parameters before building `model_params`. The optimizer only sees params with `requires_grad=True`.

**For our two-group AdamW:** Our current approach (two parameter groups with different learning rates) is the right pattern. One group for MetricScaleHead+Decoder (lr=1e-4), one for SLAT cross-attn (lr=1e-5). TRELLIS uses a single lr=1e-4 for the full denoiser, but we can't do that without risking catastrophic forgetting.

---

## Summary: Priority Actions for Our Training

| Priority | Action | Status | Why |
|---|---|---|---|
| High | Replace fixed grad clip with `AdaptiveGradClipper` | **Done 2026-05-01** | Sparse tensors have variable grad norms; fixed 1.0 is too blunt |
| High | Switch NaN guard from loss-level to gradient-level | **Done 2026-05-01** | Cleaner; catches gradient overflow that a finite loss can miss |
| Medium | Apply `torch.utils.checkpoint` to SLAT transformer blocks | **Done 2026-05-01** | Required to fit 40GB GPU; flips `block.use_checkpoint=True` on all 24 blocks |
| High | Add time-binned loss logging (10 bins) | Not applicable | With stage2-steps=1, cross-attn always sees t=0 (one bin only — nothing to split) |
| Medium | Add 10% scale token dropout during CFG training | Not yet | Prevents SLAT from becoming too dependent on the metric token |
| Low | Switch to logit-normal timestep sampling (mean=1, std=1) | Not applicable | training_time_sampler_fn only fires in FlowMatching.compute_loss(), which we never call; we run generate() and always land at t=0 |
| Low | Consider EMA at rate 0.9999 for final model export | Not yet | Smooth inference model; evaluate against non-EMA during fine-tuning |
