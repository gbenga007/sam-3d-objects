# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 3D Objects is a Meta foundation model for reconstructing full 3D shape geometry, texture, and spatial layout from single images. It is paired with SAM 3D Body for human mesh recovery and alignment.

## Environment Setup

Requires Linux 64-bit with NVIDIA GPU (32GB+ VRAM, e.g. A100/H100/H200) and CUDA 12.1.

```bash
# Create environment
mamba env create -f environments/default.yml

# Set CUDA-aware pip indices before installing
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121 https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121"

# Install package with extras (p3d must be installed after base to avoid pytorch version conflict)
pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'

# Apply required Hydra 1.3.2 patch (fixes upstream issue #2863)
./patching/hydra
```

Model checkpoints are downloaded from HuggingFace and require access approval (geographically restricted).

## Running Inference

```bash
# Quick demo — outputs splat.ply
python demo.py
```

Via Python API:
```python
from notebook.inference import Inference, load_image, load_single_mask

inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
image = load_image(...)
mask = load_single_mask(...)
output = inference(image, mask, seed=42)
output["gs"].save_ply("splat.ply")   # Gaussian splat
output["mesh"]                        # Triangle mesh with texture
```

Notebooks in `notebook/` cover single-object, multi-object, and SAM 3D Body alignment workflows.

## Testing

```bash
pytest
```

Dev dependencies are in `requirements.dev.txt`. No test suite exists in the repo currently — validation is done through notebooks.

## Architecture

The system is configuration-driven via Hydra YAML. The top-level config (`checkpoints/hf/pipeline.yaml`) composes all model components.

### Inference Pipeline (`sam3d_objects/pipeline/`)

Two pipeline variants:
- `InferencePipeline` — base pipeline
- `InferencePipelinePointMap` — uses PointMap embeddings for better spatial grounding (preferred for real-world scenes)

Each pipeline chains:
1. **Depth model** (`depth_models/`) — MoGe monocular depth estimation from Microsoft
2. **Shape (SS) generator** — predicts 3D shape structure via flow matching
3. **Layout (SLAT) generator** — predicts spatial arrangement
4. **Decoders** — convert SS/SLAT representations to Gaussian splats or meshes
5. **Pose decoder** — estimates object pose from image
6. **Post-optimization** (`layout_post_optimization_utils.py`) — ICP alignment and occlusion handling

### Model Backbone (`sam3d_objects/model/backbone/`)

- `dit/` — Diffusion Transformer (DiT) base architecture
- `tdfy_dit/` — DiT variant with octree-based sparse 3D representations (used for SLAT)
- `generator/` — wraps backbones with flow matching and classifier-free guidance (CFG)
- `layers/` — specialized transformer layers (Llama3-style attention)

### Public API (`notebook/inference.py`)

The `Inference` class is the safe entry point. It loads Hydra configs with a strict security model: a whitelist of allowed modules (`sam3d_objects`, `torch`, `torchvision`, `moge`) and a blacklist of dangerous builtins to prevent arbitrary code execution from config files.

### Outputs

- `output["gs"]` — `GaussianModel` (3DGS format), exportable via `.save_ply()`
- `output["mesh"]` — triangle mesh with UV texture
- `output["layout"]` — plane equations for scene layout
- `output["pose"]` — object pose transformation matrix

## Key Design Decisions

- **Segmentation-driven**: operates on masked objects via SAM masks, not raw scenes
- **Flow matching** (not DDPM): generative backbone uses continuous normalizing flows
- **Sparse octree representations** in TDFY-DiT allow memory-efficient 3D processing
- **Hydra configs** define the full model graph; changing pipeline behavior means editing YAML, not Python
- The `patching/hydra` patch is mandatory — without it the pipeline will fail with Hydra compose errors
