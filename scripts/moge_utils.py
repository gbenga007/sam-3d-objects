# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Shared MoGe preprocessing utilities for dataset scripts.

Runs MoGe on an RGB image and returns pointmap_scale and pointmap_shift
using the same SSI normalizer and camera convention that the inference
pipeline uses at runtime — ensuring train/inference consistency.
"""

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI
from sam3d_objects.pipeline.depth_models.moge import MoGe


def load_moge(device: str = "cuda") -> MoGe:
    """
    Load MoGe ViT-L from HuggingFace.

    Requires the model checkpoint to be available. On a machine without the
    full SAM3D environment, run:
        pip install moge
    and ensure HuggingFace access is configured.
    """
    try:
        from moge.model import MoGeModel
    except ImportError:
        raise ImportError("Install MoGe: pip install moge")

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
    return MoGe(model=model, device=device)


def compute_pointmap_stats(
    rgba_image: np.ndarray,
    moge_model: MoGe,
    device: str = "cuda",
) -> dict:
    """
    Run MoGe on an RGBA image and return pointmap_scale and pointmap_shift.

    Uses the object mask (alpha channel) when computing SSI statistics so the
    scale and shift reflect the object's depth, not the background — matching
    the behaviour of ObjectCentricSSI in the inference pipeline.

    Args:
        rgba_image:  [H, W, 4] uint8 — RGBA, alpha = object mask
        moge_model:  loaded MoGe instance
        device:      torch device string

    Returns:
        {
            "pointmap_scale": list[float]  length 3
            "pointmap_shift": list[float]  length 3
        }
        or None if MoGe fails on this image.
    """
    dev = torch.device(device)
    normalizer = ObjectCentricSSI()

    # Prepare RGB tensor [3, H, W] float32 in [0, 1]
    rgb = rgba_image[..., :3].astype(np.float32) / 255.0
    image_t = torch.from_numpy(rgb).permute(2, 0, 1).to(dev)  # [3, H, W]

    # Object mask [1, H, W]
    mask_t = torch.from_numpy((rgba_image[..., 3] > 0).astype(np.float32)).unsqueeze(0).to(dev)

    try:
        with torch.no_grad():
            output = moge_model(image_t)
        pointmaps = output["pointmaps"]  # [H, W, 3] in MoGe/R3 camera space
    except Exception:
        return None

    # Apply camera convention: R3 → PyTorch3D (matches inference pipeline)
    r3_to_p3d_R, _ = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=dev,
    )
    cam_transform = Transform3d().rotate(r3_to_p3d_R).to(dev)
    points = cam_transform.transform_points(pointmaps)  # [H, W, 3]
    points_chw = points.permute(2, 0, 1)                # [3, H, W]

    # Compute SSI scale and shift using the object mask
    result = normalizer.normalize(points_chw, mask_t)

    return {
        "pointmap_scale": result.scale.cpu().tolist(),   # [3]
        "pointmap_shift": result.shift.cpu().tolist(),   # [3]
    }
