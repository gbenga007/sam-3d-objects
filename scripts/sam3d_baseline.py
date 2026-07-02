"""SAM 3D Objects adapter for MoGe-style metric depth evaluation.

This wraps our trained image-to-3D pipeline (`InferencePipelinePointMap` with
`MetricScaleHead`) behind MoGe's `MGEBaselineInterface`, so the same harness
that evaluates UniDepth/Metric3D/DepthPro can also evaluate our model.

Conceptual mismatch:
  - MoGe's baseline interface expects a *dense scene point/depth map*.
  - Our model reconstructs a single object given a mask.

Bridging:
  - We accept a per-image object mask alongside the standard `image, intrinsics`
    input via `infer_with_mask`.
  - The pipeline produces a canonical (unit-cube) mesh plus a metric prediction
    `[W, H, D]` (metres). We isotropically scale the canonical mesh by
    `max(W, H, D)`, then orient/place it in camera frame using the pose
    decoder's `(rotation, translation)`.
  - PyTorch3D rasterises the placed mesh from the camera viewpoint to a depth
    buffer, which we backproject to a point map at the object's pixels.
    Pixels outside the object mask are set to NaN.
  - The result is consumed by MoGe's `compute_metrics` via the `local_points`
    pathway, restricted to the segmentation label.

Not handled in this first pass:
  - `with_layout_postprocess` (ICP refinement of the pose against the GT
    pointmap). Adds latency and depends on the input pointmap being free of
    artefacts; we can add it as an `--enable-icp` flag once base numbers exist.
  - Per-axis (anisotropic) scaling. Isotropic is the production behaviour and
    matches `mesh_scale_eval.py`; per-axis would require ranking matching that
    isn't well-defined without a canonical-frame convention.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

# Sparse-attn backend must be set before importing the pipeline.
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
# notebook/inference.py assumes a conda activation; satisfy the var with the
# active interpreter's prefix so the import works without `conda activate`.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import click
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from moge.test.baseline import MGEBaselineInterface

# Project imports
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.pipeline.layout_post_optimization_utils import denormalize_f

# PyTorch3D
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [w, x, y, z] (the pose-decoder convention used in the project)
    to a 3x3 rotation matrix."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _image_tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    """[3, H, W] float in [0, 1] (MoGe convention) → (H, W, 3) uint8."""
    if image.ndim == 4:
        image = image[0]
    arr = image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return (arr * 255).round().astype(np.uint8)


def _mask_tensor_to_uint8(mask: torch.Tensor) -> np.ndarray:
    """Bool/int mask → (H, W) uint8 {0, 255}."""
    if mask.ndim == 3:
        mask = mask[0]
    return (mask.detach().cpu().numpy().astype(bool) * 255).astype(np.uint8)


def _normalized_K_to_pixel(norm_K: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Project's normalized [0,1] intrinsics → pixel-space 3x3."""
    return denormalize_f(norm_K.detach().cpu().numpy(), H, W)


class Baseline(MGEBaselineInterface):
    """SAM 3D Objects wrapper."""

    def __init__(
        self,
        config_path: str,
        metric_scale_checkpoint: str,
        device: str = "cuda",
        seed: int = 42,
        stage1_steps: Optional[int] = None,
        stage2_steps: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.seed = seed
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps

        # Instantiate the pipeline directly via Hydra so we don't drag in the
        # notebook's viz dependencies (seaborn / matplotlib / etc.).
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.metric_scale_checkpoint_path = metric_scale_checkpoint
        config.workspace_dir = os.path.dirname(config_path)
        self.pipeline: InferencePipelinePointMap = instantiate(config)

    @click.command()
    @click.option("--config", "config_path", type=click.Path(exists=True), required=True,
                  help="Hydra pipeline config (e.g. checkpoints/hf/pipeline.yaml).")
    @click.option("--metric-checkpoint", "metric_scale_checkpoint", type=click.Path(exists=True),
                  required=True, help="Path to MetricScaleHead/Decoder fine-tune checkpoint.")
    @click.option("--device", default="cuda")
    @click.option("--seed", type=int, default=42)
    @click.option("--stage1-steps", type=int, default=None, help="SS inference steps override.")
    @click.option("--stage2-steps", type=int, default=None, help="SLAT inference steps override.")
    @staticmethod
    def load(config_path, metric_scale_checkpoint, device, seed, stage1_steps, stage2_steps):
        return Baseline(config_path, metric_scale_checkpoint, device, seed, stage1_steps, stage2_steps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, image: torch.Tensor, intrinsics: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """No-mask path: full image treated as the object. Not used in our
        evaluation flow (the runner always provides a mask) but kept to satisfy
        the MGEBaselineInterface contract."""
        H, W = image.shape[-2:]
        full_mask = torch.ones(H, W, dtype=torch.bool, device=image.device)
        return self.infer_with_mask(image, intrinsics, full_mask)

    def infer_with_mask(
        self,
        image: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference for the object indicated by `mask`.

        Strategy: use MoGe's full-resolution raw pointmap at the mask pixels
        as the 3D geometry source, then rescale to metric units using our
        MetricScaleDecoder prediction.

        MoGe v1 is affine-invariant (correct shape, unknown scale/shift). Our
        MetricScaleHead+Decoder predicts the object's true metric extent
        (max of W, H, D in metres). We rescale the MoGe pointmap so that the
        extent of the object points matches our metric prediction. This tests
        our core contribution (scale recovery) without relying on the pose
        decoder, which has limited generalization to arbitrary scenes.

        Args:
            image: [3, H, W] float in [0, 1].
            intrinsics: [3, 3] normalized [0, 1] intrinsics (oracle mode) or
                None (use MoGe predicted intrinsics).
            mask: [H, W] bool object mask.

        Returns:
            dict with:
              - `points_metric`: [H, W, 3] float in metres, NaN outside mask.
              - `depth_metric`:  [H, W]    float in metres, NaN outside mask.
              - `mask`:          [H, W] bool, the object mask.
              - `intrinsics`:    [3, 3] normalized intrinsics (MoGe predicted).
        """
        H, W = image.shape[-2:]
        image_np = _image_tensor_to_uint8_rgb(image)
        mask_np = _mask_tensor_to_uint8(mask)

        out = self.pipeline.run(
            image=image_np,
            mask=mask_np,
            seed=self.seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=self.stage1_steps,
            stage2_inference_steps=self.stage2_steps,
            decode_formats=["mesh"],
        )

        # --- Metric scale from our head ----------------------------------
        metric_dims = out["metric_dimensions"][0].to(self.device)  # [3] W,H,D in metres
        s_iso = metric_dims.max().clamp(min=1e-6)

        # --- MoGe full-res pointmap at mask pixels -----------------------
        # `pointmap_full` is the raw MoGe output [H_full, W_full, 3] in
        # affine-invariant scale (correct shape, unknown absolute metric).
        pm_full = out["pointmap_full"].to(self.device).float()    # [H, W, 3]
        mask_b  = mask.to(self.device).bool()                     # [H, W]

        # Gather the 3D points inside the object mask.
        mask_pts = pm_full[mask_b]           # [N, 3], affine-invariant

        # Compute the maximum 3D extent of the object in MoGe's scale.
        if mask_pts.shape[0] < 4:
            # Degenerate mask — return all-NaN
            nan_pts  = torch.full((H, W, 3), float("nan"), device=self.device)
            nan_dep  = torch.full((H, W),    float("nan"), device=self.device)
            K_norm   = (intrinsics or out["intrinsics"]).to(self.device).float()
            if K_norm.ndim == 3:
                K_norm = K_norm[0]
            return {"points_metric": nan_pts, "depth_metric": nan_dep,
                    "mask": mask_b, "intrinsics": K_norm}

        extents    = mask_pts.max(0)[0] - mask_pts.min(0)[0]   # [3]
        moge_scale = extents.max().clamp(min=1e-8)             # scalar

        # Scale factor: map MoGe's raw scale so the max extent = s_iso metres.
        # This is our model's metric contribution: the shape comes from MoGe,
        # the absolute size comes from MetricScaleDecoder.
        scale_factor = s_iso / moge_scale                      # scalar

        # Rescale all points relative to the object centroid so the centroid
        # depth stays unchanged (we don't claim to fix absolute position, only
        # metric scale). This is the "points_scale_invariant" evaluation mode.
        centroid    = mask_pts.mean(0)                          # [3]
        scaled_pts  = (mask_pts - centroid) * scale_factor + centroid  # [N, 3]

        # Build full-image point and depth maps (NaN outside mask).
        points_metric = torch.full((H, W, 3), float("nan"), device=self.device)
        depth_metric  = torch.full((H, W),    float("nan"), device=self.device)
        points_metric[mask_b] = scaled_pts
        depth_metric[mask_b]  = scaled_pts[:, 2]

        # Intrinsics: GT if oracle mode, else MoGe's prediction.
        if intrinsics is not None:
            K_norm = intrinsics.to(self.device).float()
        else:
            K_norm = out["intrinsics"].to(self.device).float()
            if K_norm.ndim == 3:
                K_norm = K_norm[0]

        return {
            "points_metric": points_metric,     # [H, W, 3]
            "depth_metric": depth_metric,       # [H, W]
            "mask": mask_b,                     # [H, W]
            "intrinsics": K_norm,               # [3, 3] normalized
        }

    # ------------------------------------------------------------------
    # Rasterisation helpers
    # ------------------------------------------------------------------

    def _rasterise_mesh_to_depth(
        self,
        verts_cam: torch.Tensor,
        faces: torch.Tensor,
        K_norm: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Render the mesh from the camera origin and return a depth map.

        Convention note: the project's pose decoder produces translations in
        an OpenCV-style camera frame (X right, Y down, Z forward).
        PyTorch3D's PerspectiveCameras with in_ndc=False uses screen-space
        coordinates: X right, Y DOWN, Z forward — identical to OpenCV.
        No flip is required; passing vertices directly works.
        (The earlier diag(1,-1,-1) flip was wrong: negating Z put the mesh
        behind the camera, making zbuf = -1 everywhere → all-NaN depth.)
        """
        K_pixel = _normalized_K_to_pixel(K_norm, H, W)
        cameras = PerspectiveCameras(
            focal_length=torch.tensor([[K_pixel[0, 0], K_pixel[1, 1]]],
                                      device=self.device, dtype=torch.float32),
            principal_point=torch.tensor([[K_pixel[0, 2], K_pixel[1, 2]]],
                                         device=self.device, dtype=torch.float32),
            image_size=torch.tensor([[H, W]], device=self.device, dtype=torch.float32),
            in_ndc=False,
            device=self.device,
        )

        # PyTorch3D PerspectiveCameras(in_ndc=False) projects with negated x and y:
        #   u = -fx * x/z + cx,  v = -fy * y/z + cy
        # OpenCV projects normally: u = fx * x/z + cx.
        # To reconcile, negate x and y before passing to PyTorch3D (keep z positive).
        # Empirically confirmed: identity flip → mirrored hit region; diag(-1,-1,1) → correct.
        flip = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            device=self.device, dtype=torch.float32)
        verts_p3d = verts_cam @ flip.T

        meshes = Meshes(verts=[verts_p3d], faces=[faces])
        rasteriser = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(H, W),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,                 # CPU-fallback safe
                max_faces_per_bin=None,
            ),
        )
        fragments = rasteriser(meshes)
        zbuf = fragments.zbuf[0, ..., 0]    # [H, W]; -1 where no triangle hit
        depth = torch.where(zbuf > 0, zbuf, torch.full_like(zbuf, float("nan")))
        return depth

    @staticmethod
    def _depth_to_points(depth: torch.Tensor, K_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Backproject a (H, W) depth map to (H, W, 3) camera-frame points using
        normalized intrinsics. NaN depths propagate as NaN points."""
        device = depth.device
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u_norm = (xs + 0.5) / W      # pixel-centre → [0, 1] image coords
        v_norm = (ys + 0.5) / H

        fx, fy = K_norm[0, 0], K_norm[1, 1]
        cx, cy = K_norm[0, 2], K_norm[1, 2]
        x = (u_norm - cx) / fx * depth
        y = (v_norm - cy) / fy * depth
        z = depth
        return torch.stack([x, y, z], dim=-1)
