#!/usr/bin/env python3
"""
Anisotropic voxel rescaling experiment.

Tests whether Stage 1 (SS) voxel topology is the source of mesh aspect ratio errors.

For each sample:
  1. Run Stage 1 → get integer voxel coords; measure their bounding-box aspect ratio.
  2. Compare voxel aspect ratio with GT [W, H, D] aspect ratio.
  3. Rescale coords to match GT aspect ratio (two strategies: sorted and direct).
  4. Run Stage 2 (SLAT) on original and on each rescaled coord set.
  5. Decode meshes; compare sorted bounding-box extents before / after rescaling.

Reading the output:
  If sorted-mesh ≈ GT-sorted (normalised to max=1), rescaling voxel topology fixes proportions.
  If orig-mesh already ≈ GT-sorted, the error lives elsewhere (FlexiCubes / SLAT features).

Usage:
    python scripts/anisotropic_rescale_experiment.py \\
        --checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt \\
        --config checkpoints/hf/pipeline.yaml \\
        --annotations-root /mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275 \\
        --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \\
        --n-samples 10 \\
        --stage1-steps 4 --stage2-steps 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from sam3d_objects.data.dataset.metric.nocs import OmniNOCSReal275Dataset
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import (
    MetricScaleHead,
    _ScaleAugmentedEmbedderProxy,
    extract_ss_scale_features,
)
from sam3d_objects.training.finetune_metric_scale import (
    freeze_pipeline,
    load_metric_checkpoint,
)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(config_path: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    config.compile_model = False
    config.device = device
    pipeline = instantiate(config)
    freeze_pipeline(pipeline)
    return pipeline


# ---------------------------------------------------------------------------
# Coordinate rescaling helpers
# ---------------------------------------------------------------------------

def rescale_coords(coords: torch.Tensor, scale_per_axis: torch.Tensor) -> torch.Tensor:
    """
    Anisotropically rescale integer voxel coords around their bounding-box centre.

    coords:          [N, 4]  (batch_idx, d0, d1, d2)
    scale_per_axis:  [3]     multiplier for each spatial axis

    Returns deduplicated integer coords (some may merge after rounding).
    """
    spatial = coords[:, 1:].float()
    lo = spatial.min(0).values
    hi = spatial.max(0).values
    center = (lo + hi) / 2.0

    scaled = center + (spatial - center) * scale_per_axis.to(spatial.device)
    new_spatial = scaled.round().clamp(min=0).int()

    new_coords = torch.cat([coords[:, :1], new_spatial], dim=1)
    return torch.unique(new_coords, dim=0)


def rescale_factors_sorted(vox_extents: np.ndarray, gt_dims: np.ndarray) -> torch.Tensor:
    """
    Rank-matched: largest voxel axis → largest GT dim, etc.
    Preserves max voxel extent so only shape changes, not overall size.
    """
    vox_rank = np.argsort(vox_extents)[::-1]   # indices of axes sorted large→small
    gt_sorted_desc = np.sort(gt_dims)[::-1]
    target = np.zeros(3)
    for rank, ax in enumerate(vox_rank):
        target[ax] = gt_sorted_desc[rank]
    # Preserve overall scale
    target = target / target.max() * vox_extents.max()
    factors = target / np.where(vox_extents > 0, vox_extents, 1.0)
    return torch.tensor(factors, dtype=torch.float32)


def rescale_factors_direct(vox_extents: np.ndarray, gt_dims: np.ndarray) -> torch.Tensor:
    """
    Direct-axis: d0→W, d1→H, d2→D (assumes coord convention matches GT order).
    Preserves max voxel extent.
    """
    target = gt_dims / gt_dims.max() * vox_extents.max()
    factors = target / np.where(vox_extents > 0, vox_extents, 1.0)
    return torch.tensor(factors, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Mesh bounding-box helper
# ---------------------------------------------------------------------------

def mesh_sorted_extents_cm(mesh_result) -> np.ndarray | None:
    """Sorted (ascending) bounding-box extents in cm, or None on failure."""
    if mesh_result is None or not getattr(mesh_result, "success", True):
        return None
    verts = mesh_result.vertices.float().cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    ext = tm.bounding_box.extents
    if ext.max() < 1e-8:
        return None
    return np.sort(ext) * 100.0  # metres → cm, ascending


# ---------------------------------------------------------------------------
# SLAT pass helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def slat_decode(pipeline, slat_in, coords, scale_token, stage2_steps):
    """Run SLAT on given coords (with scale token injected) and decode mesh."""
    with pipeline.device:
        tok = F.layer_norm(scale_token, [scale_token.shape[-1]]).detach()
        orig_bb = orig_ext = slat_bb = None
        if hasattr(pipeline, "_get_slat_backbone"):
            slat_bb = pipeline._get_slat_backbone()
            if slat_bb is not None:
                orig_bb = slat_bb.condition_embedder
                slat_bb.condition_embedder = _ScaleAugmentedEmbedderProxy(orig_bb, tok)
            ce = getattr(pipeline, "condition_embedders", {})
            orig_ext = ce.get("slat_condition_embedder")
            if orig_ext is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = _ScaleAugmentedEmbedderProxy(orig_ext, tok)
        try:
            slat = pipeline.sample_slat(
                slat_in, coords,
                inference_steps=stage2_steps, use_distillation=False, with_grad=False,
            )
        finally:
            if slat_bb is not None and orig_bb is not None:
                slat_bb.condition_embedder = orig_bb
            if orig_ext is not None:
                pipeline.condition_embedders["slat_condition_embedder"] = orig_ext

        decoded = pipeline.decode_slat(slat, formats=["mesh"])
        raw = decoded.get("mesh")
        mesh = (raw[0] if isinstance(raw, (list, tuple)) else raw) if raw is not None else None
        return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--stage1-steps", type=int, default=4)
    parser.add_argument("--stage2-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading pipeline...")
    pipeline = load_pipeline(args.config, args.device)

    scale_head = MetricScaleHead()
    scale_decoder = MetricScaleDecoder()
    slat_backbone = (
        pipeline._get_slat_backbone() if hasattr(pipeline, "_get_slat_backbone") else None
    )
    load_metric_checkpoint(args.checkpoint, scale_head, scale_decoder, slat_backbone)
    scale_head = scale_head.to(args.device).eval()
    scale_decoder = scale_decoder.to(args.device).eval()

    print("Loading dataset...")
    dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split="test",
    )
    n = min(args.n_samples, len(dataset))
    print(f"  Using {n}/{len(dataset)} samples")
    print()

    # Header
    col = 22
    print(f"{'sample':50s} {'gt(cm)':>{col}} {'vox(vx)':>{col}} "
          f"{'orig-mesh(cm)':>{col}} {'sorted-mesh(cm)':>{col}} {'direct-mesh(cm)':>{col}}")
    print("-" * (50 + 5 * (col + 1)))

    agg = {"vox_matches_gt": [], "sorted_improves": [], "direct_improves": []}

    for idx in range(n):
        item = dataset[idx]
        image      = item["image"]
        gt_dims    = np.array(item["metric_dims"], dtype=np.float32)  # [W, H, D] in metres
        uid        = item.get("uid", str(idx))
        category   = item.get("category", "?")

        # ---- Stage 1: SS ----
        with pipeline.device:
            pm     = pipeline.compute_pointmap(image)
            ss_in  = pipeline.preprocess_image(image, pipeline.ss_preprocessor, pointmap=pm["pointmap"])
            ss_out = pipeline.sample_sparse_structure(
                ss_in, inference_steps=args.stage1_steps, use_distillation=False
            )
            slat_in = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

            coords = ss_out["coords"]  # [N, 4]: (batch, d0, d1, d2)
            ss_feats = extract_ss_scale_features(ss_out)
            if ss_feats is not None:
                ss_feats = ss_feats.detach().to(pipeline.device)
            scale_token = scale_head(
                ss_out["shape"].detach(), ss_feats,
                ss_in.get("pointmap_scale"), ss_in.get("pointmap_shift"),
            )

        # Voxel bounding-box extents (in voxel units)
        spatial = coords[:, 1:].float().cpu()
        vox_ext = (spatial.max(0).values - spatial.min(0).values + 1).numpy()

        # Voxel aspect ratio similarity to GT (normalised)
        vox_sorted = np.sort(vox_ext)
        gt_sorted  = np.sort(gt_dims)
        vox_norm   = vox_sorted / vox_sorted.max()
        gt_norm    = gt_sorted  / gt_sorted.max()
        vox_gt_err = np.abs(vox_norm - gt_norm).mean()
        agg["vox_matches_gt"].append(vox_gt_err)

        # ---- Rescale coords ----
        fac_sorted = rescale_factors_sorted(vox_ext, gt_dims)
        fac_direct = rescale_factors_direct(vox_ext, gt_dims)
        coords_s   = rescale_coords(coords, fac_sorted)
        coords_d   = rescale_coords(coords, fac_direct)

        # ---- Three SLAT passes ----
        mesh_orig   = slat_decode(pipeline, slat_in, coords,   scale_token, args.stage2_steps)
        mesh_sorted = slat_decode(pipeline, slat_in, coords_s, scale_token, args.stage2_steps)
        mesh_direct = slat_decode(pipeline, slat_in, coords_d, scale_token, args.stage2_steps)

        ext_orig   = mesh_sorted_extents_cm(mesh_orig)
        ext_sorted = mesh_sorted_extents_cm(mesh_sorted)
        ext_direct = mesh_sorted_extents_cm(mesh_direct)

        def norm_err(ext_cm):
            if ext_cm is None:
                return None
            e = ext_cm / ext_cm.max()
            return np.abs(e - gt_norm).mean()

        e_orig   = norm_err(ext_orig)
        e_sorted = norm_err(ext_sorted)
        e_direct = norm_err(ext_direct)

        if e_orig is not None and e_sorted is not None:
            agg["sorted_improves"].append(e_sorted < e_orig)
        if e_orig is not None and e_direct is not None:
            agg["direct_improves"].append(e_direct < e_orig)

        def fmt(arr, suffix=""):
            if arr is None:
                return "FAIL".rjust(col)
            s = " ".join(f"{v:5.1f}" for v in arr)
            return f"({s}{suffix})".rjust(col)

        vox_str = "(" + " ".join(f"{v:5.1f}" for v in vox_sorted) + ")".rjust(1)

        err_tag = f"  vox∆={vox_gt_err:.3f}"
        if e_orig is not None:
            err_tag += f"  orig∆={e_orig:.3f}"
        if e_sorted is not None:
            err_tag += f"  srt∆={e_sorted:.3f}"
        if e_direct is not None:
            err_tag += f"  dir∆={e_direct:.3f}"

        label = f"{uid[:30]} ({category[:8]})"
        print(f"{label:50s} {fmt(gt_sorted)} {vox_str:>{col}} "
              f"{fmt(ext_orig)} {fmt(ext_sorted)} {fmt(ext_direct)}")
        print(f"  {err_tag}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print(f"  Mean voxel-vs-GT normalised aspect ratio error: "
          f"{np.mean(agg['vox_matches_gt']):.4f}")
    if agg["sorted_improves"]:
        frac = np.mean(agg["sorted_improves"])
        print(f"  Sorted rescaling improves aspect ratio: {frac*100:.0f}% of samples")
    if agg["direct_improves"]:
        frac = np.mean(agg["direct_improves"])
        print(f"  Direct rescaling improves aspect ratio: {frac*100:.0f}% of samples")
    print()
    print("Columns (all sorted ascending within each triple):")
    print("  gt(cm)          — GT [W,H,D] in cm")
    print("  vox(vx)         — Stage 1 voxel bbox extents in voxel units")
    print("  orig-mesh(cm)   — decoded mesh extents, no rescaling")
    print("  sorted-mesh(cm) — decoded mesh extents, rank-matched rescaling")
    print("  direct-mesh(cm) — decoded mesh extents, d0→W d1→H d2→D rescaling")
    print()
    print("Diagnosis key:")
    print("  vox∆ > 0.1 → Stage 1 voxel topology has wrong aspect ratio")
    print("  srt∆ < orig∆ → rank-matched rescaling corrects proportions")
    print("  dir∆ < orig∆ → direct-axis mapping is approximately correct")


if __name__ == "__main__":
    main()
