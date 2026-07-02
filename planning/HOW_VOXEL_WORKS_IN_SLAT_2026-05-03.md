# How Stage-1 Voxels Work In SLAT

This note documents how the stage-1 sparse-structure (SS) voxel prediction is used by SLAT in the current `sam3d-objects` pipeline and training code.

## Short Answer

Stage 1's predicted sparse voxel support is used by SLAT as the sparse coordinate lattice. It is not passed as a dense voxel tensor, and it is not conditioning in the same sense as image tokens or the metric scale token.

The SS stage decides where latent features should exist. SLAT then predicts an 8-channel structured latent feature vector at each of those occupied coordinates.

## Code Path

### 1. SS predicts occupied voxel coordinates

Implementation: `sam3d_objects/pipeline/inference_pipeline.py::sample_sparse_structure`

The frozen SS generator samples a latent, the SS decoder turns that into an occupancy grid, and the code extracts occupied coordinates:

```python
ss = ss_decoder(
    shape_latent.permute(0, 2, 1)
    .contiguous()
    .view(shape_latent.shape[0], 8, 16, 16, 16)
)
coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
```

The resulting `coords` tensor has shape `[num_voxels, 4]`, with columns:

```text
[batch_index, x, y, z]
```

Those coordinates may then be pruned and/or downsampled:

```python
coords, downsample_factor = downsample_sparse_structure(coords)
return_dict["coords"] = coords
```

### 2. Training passes those coordinates into SLAT

Implementation: `sam3d_objects/training/finetune_metric_scale.py::predict_log_dims`

The training path runs SS under `torch.no_grad()`, then passes the predicted coordinates into SLAT:

```python
with torch.no_grad():
    ss_return_dict = pipeline.sample_sparse_structure(...)

slat = pipeline.sample_slat(
    slat_input_dict,
    ss_return_dict["coords"],
    inference_steps=stage2_steps,
    use_distillation=False,
    with_grad=unfreeze_cross_attn,
)
```

So the stage-1 occupancy support is frozen/discrete for the training step. Gradients do not flow back into SS or through the coordinate selection.

### 3. SLAT generates features at those coordinates

Implementation: `sam3d_objects/pipeline/inference_pipeline.py::sample_slat`

SLAT receives the coordinates and creates a latent shape based on the number of occupied voxels:

```python
latent_shape = (image.shape[0],) + (coords.shape[0], 8)
condition_args += (coords.cpu().numpy(),)
slat = slat_generator(latent_shape, DEVICE, *condition_args, **condition_kwargs)
```

The output features are wrapped into a sparse tensor using the same coordinates:

```python
slat = sp.SparseTensor(
    coords=coords,
    feats=slat[0],
).to(DEVICE)
```

This means SLAT predicts an 8-dimensional latent feature at every stage-1 occupied voxel.

### 4. The SLAT model uses coords internally

Implementation: `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py::SLatFlowModelTdfyWrapper.forward`

The SLAT wrapper pulls the coordinates from the condition args and constructs a sparse tensor:

```python
coords = torch.tensor(coords).to(x.device)
x = sp.SparseTensor(
    feats=x[0],
    coords=coords,
)
```

Inside the SLAT model, those coordinates are used for sparse positional encoding:

```python
h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
```

They also define the sparse tensor layout used by sparse attention and sparse processing.

## What Counts As Conditioning?

There are two different mechanisms here:

1. **SS voxel coordinates**
   - Define the sparse support/topology.
   - Tell SLAT where latent tokens exist.
   - Used for sparse positional encoding and sparse attention layout.
   - Discrete and non-differentiable in the current training path.

2. **Image tokens and metric scale token**
   - Enter SLAT through the condition embedder and cross-attention.
   - These are the semantic/conditioning tokens.
   - With `--unfreeze-slat-cross-attn`, gradients can update SLAT cross-attention and flow back through the metric scale token path.

So the stage-1 voxels are "conditioning" only in the structural sense. They condition the support of the SLAT latent, not the cross-attention context.

## Current Training Implication

In the current launched training setup:

- SS is frozen.
- SS is run under `torch.no_grad()`.
- `ss_return_dict["coords"]` determines the SLAT sparse support.
- SLAT predicts `slat.feats` at those coords.
- The metric decoder consumes `slat.feats`, the metric scale token, and `slat.coords[:, 0]`.
- Cross-attention weights can be updated when `--unfreeze-slat-cross-attn` is enabled.
- Gradients do not update the stage-1 voxel predictor or the discrete coordinate extraction.

