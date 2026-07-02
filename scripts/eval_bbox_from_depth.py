"""Bounding-box comparison: metric-depth estimator → 3D bbox vs our SAM3D predictions.

Strategy
--------
For each (sample, object-mask) in HAMMER / iBims-1:

1. Run a metric depth estimator (MoGe-2 or UniDepthV2) on the full image.
   → full-scene metric 3D pointmap [H, W, 3].

2. Apply the object segmentation mask → object points only [N, 3].

3. Compute axis-aligned 3D bounding box:
   W = range along X, H = range along Y, D = range along Z.

4. GT bbox = same procedure on GT depth (sensor) + GT object mask.
   GT has partial occlusion / sensor holes → we use the same visible-surface
   bbox, not a completion estimate.  Both pred and GT are evaluated on the
   same visible-surface pixels to ensure a fair comparison.

5. Error metrics per dimension:
   - absolute relative error: |pred_dim - gt_dim| / gt_dim
   - for the max-side (most stable): |max(pred WH) - max(gt WH)| / max(gt WH)
   - IoU-like overlap in 3D (axis-aligned)

6. Compare against our SAM3D (MetricScaleDecoder) bbox predictions by loading
   sam3d_baseline results if available.

Output: JSONL, one record per sample. Aggregate script: aggregate_eval_results.py.

Usage
-----
# MoGe-2:
python scripts/eval_bbox_from_depth.py --estimator moge2 --datasets HAMMER iBims-1 \
    --output artifacts/eval_bbox/moge2_bbox.jsonl

# UniDepthV2:
python scripts/eval_bbox_from_depth.py --estimator unidepth2 --datasets HAMMER iBims-1 \
    --output artifacts/eval_bbox/unidepth2_bbox.jsonl

# SAM3D (our model) — uses sam3d_baseline.py with the metric head:
python scripts/eval_bbox_from_depth.py --estimator sam3d \
    --config checkpoints/hf/pipeline.yaml \
    --metric-checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt \
    --datasets HAMMER iBims-1 \
    --output artifacts/eval_bbox/sam3d_v3_bbox.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import numpy as np
import torch
from tqdm import tqdm

# utils3d v1.7 compat patches (needed when MoGe's compute_metrics is imported)
import utils3d.torch as _u3t
from utils3d.torch.maps import pixel_coord_map as _pcm, uv_map as _uvm, depth_map_to_point_map as _d2p
from utils3d.torch.utils import sliding_window as _sw
from utils3d.torch.transforms import intrinsics_from_focal_center as _intr_fc
if not hasattr(_u3t, "image_pixel_center"):
    def _image_pixel_center(width, height, dtype=torch.float32, device=None):
        return _pcm(height, width, dtype=dtype, device=device)
    _u3t.image_pixel_center = _image_pixel_center
if not hasattr(_u3t, "image_uv"):
    def _image_uv(width, height, dtype=torch.float32, device=None):
        return _uvm(height, width, dtype=dtype, device=device)
    _u3t.image_uv = _image_uv
if not hasattr(_u3t, "sliding_window_2d"):
    _u3t.sliding_window_2d = _sw
if not hasattr(_u3t, "depth_to_points"):
    def _depth_to_points(depth, intrinsics=None, **kwargs):
        if intrinsics is not None and hasattr(intrinsics, "device") and intrinsics.device != depth.device:
            intrinsics = intrinsics.to(depth.device)
        return _d2p(depth, intrinsics, **kwargs)
    _u3t.depth_to_points = _depth_to_points
if not hasattr(_u3t, "intrinsics_from_focal_center"):
    _u3t.intrinsics_from_focal_center = _intr_fc

from moge.utils.io import read_image, read_depth, read_segmentation


# ---------------------------------------------------------------------------
# Sample loading (same as eval_metric_depth.py)
# ---------------------------------------------------------------------------

def load_sample(sample_dir: Path):
    rgb = read_image(sample_dir / "image.jpg")                  # (H, W, 3) uint8
    depth, _unit = read_depth(sample_dir / "depth.png")         # (H, W) float, NaN where invalid
    seg, seg_labels = read_segmentation(sample_dir / "segmentation.png")
    K_norm = np.asarray(
        json.loads((sample_dir / "meta.json").read_text())["intrinsics"],
        dtype=np.float32,
    )
    return rgb, depth, seg, seg_labels, K_norm


def backproject_depth_to_points(depth: np.ndarray, K_norm: np.ndarray) -> np.ndarray:
    """(H, W) depth [m] + normalized intrinsics → (H, W, 3) camera-frame points."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u_norm = (xs.astype(np.float32) + 0.5) / W
    v_norm = (ys.astype(np.float32) + 0.5) / H
    fx = float(K_norm[0, 0])
    fy = float(K_norm[1, 1])
    cx = float(K_norm[0, 2])
    cy = float(K_norm[1, 2])
    z = depth
    x = (u_norm - cx) / fx * z
    y = (v_norm - cy) / fy * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 3D bounding box from masked point cloud
# ---------------------------------------------------------------------------

def bbox3d_from_points(points: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Given [H, W, 3] points and boolean [H, W] mask, return
    axis-aligned 3D bounding box dimensions W/H/D (X/Y/Z range).

    Returns None if fewer than 100 valid masked points.
    NaN/inf values in points are filtered out before computing bbox.
    """
    pts = points[mask]  # [N, 3]
    # Filter out any NaN or inf values (e.g. from MoGe mask boundary, depth holes)
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if len(pts) < 100:
        return None
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    dims = (mx - mn).astype(float)
    return {
        "W": float(dims[0]),   # X range
        "H": float(dims[1]),   # Y range
        "D": float(dims[2]),   # Z range (depth)
        "max_whd": float(dims.max()),
        "max_wh": float(dims[:2].max()),
        "n_pts": int(len(pts)),
    }


def bbox_errors(pred: Dict[str, float], gt: Dict[str, float]) -> Dict[str, float]:
    """Compute relative errors and max-dim absolute relative error."""
    out = {}
    for key in ("W", "H", "D", "max_whd", "max_wh"):
        gt_v = gt.get(key, 0.0)
        pred_v = pred.get(key, 0.0)
        if gt_v > 1e-4:
            out[f"rel_{key}"] = abs(pred_v - gt_v) / gt_v
        else:
            out[f"rel_{key}"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# GT bbox from sensor depth
# ---------------------------------------------------------------------------

def gt_bbox(depth: np.ndarray, K_norm: np.ndarray,
            seg_mask: np.ndarray) -> Optional[Dict[str, float]]:
    """Backproject GT depth and compute 3D bbox for the object pixels."""
    valid = np.isfinite(depth) & seg_mask
    if valid.sum() < 100:
        return None
    depth_filled = np.where(valid, depth, 0.0)
    points = backproject_depth_to_points(depth_filled, K_norm)  # [H, W, 3]
    return bbox3d_from_points(points, valid)


# ---------------------------------------------------------------------------
# Estimator adapters
# ---------------------------------------------------------------------------

class MoGe2Estimator:
    """MoGe-2 ViT-L metric point estimator."""

    def __init__(self, device: str = "cuda"):
        # Must use local v2 source since installed moge==1.0.0 only has v1
        import sys as _sys
        if "/mnt/source/MoGe" not in _sys.path:
            _sys.path.insert(0, "/mnt/source/MoGe")
        from moge.model.v2 import MoGeModel
        self.device = device
        print("Loading MoGe-2 ViT-L ...")
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device).eval()
        print("MoGe-2 loaded.")

    @torch.inference_mode()
    def infer_points(self, rgb_hwc_u8: np.ndarray, K_norm: np.ndarray) -> np.ndarray:
        """
        rgb_hwc_u8: (H, W, 3) uint8
        K_norm: (3, 3) normalized intrinsics (not used — MoGe-2 predicts its own)

        Returns: (H, W, 3) metric point map in camera frame [metres].
        NaN where model mask is 0.
        """
        H, W = rgb_hwc_u8.shape[:2]
        img_t = torch.from_numpy(rgb_hwc_u8).permute(2, 0, 1).float().to(self.device) / 255.0
        img_t = img_t.unsqueeze(0)  # [1, 3, H, W]

        out = self.model.infer(img_t, apply_mask=True, force_projection=True)
        # out keys: points [B,H,W,3], depth [B,H,W], intrinsics [B,3,3], mask [B,H,W]
        # apply_mask=True sets out-of-mask points to torch.inf (not NaN)
        pts = out["points"].squeeze(0)  # [H, W, 3] (v2 returns B,H,W,3)
        if pts.ndim == 3 and pts.shape[0] == 3 and pts.shape[-1] != 3:
            pts = pts.permute(1, 2, 0)  # handle [3, H, W] case just in case
        pts_np = pts.cpu().float().numpy()
        # Replace inf (from apply_mask=True) with NaN for downstream handling
        pts_np[~np.isfinite(pts_np)] = np.nan
        return pts_np  # [H, W, 3] metric, NaN outside model mask


class UniDepthV2Estimator:
    """UniDepthV2 ViT-L14 metric depth/point estimator."""

    def __init__(self, device: str = "cuda"):
        import json as _json
        self.device = device
        hub_dir = "/root/.cache/torch/hub/lpiccinelli-eth_UniDepth_main"
        cfg_path = f"{hub_dir}/configs/config_v2_vitl14.json"
        with open(cfg_path) as f:
            config = _json.load(f)
        from unidepth.models import UniDepthV2
        print("Loading UniDepthV2 ViT-L14 ...")
        model = UniDepthV2(config)
        import huggingface_hub
        ckpt_path = huggingface_hub.hf_hub_download(
            repo_id="lpiccinelli/unidepth-v2-vitl14",
            filename="pytorch_model.bin",
        )
        info = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        if info.missing_keys:
            print(f"  missing keys: {info.missing_keys[:5]}")
        self.model = model.to(device).eval()
        print("UniDepthV2 loaded.")

    @torch.inference_mode()
    def infer_points(self, rgb_hwc_u8: np.ndarray, K_norm: np.ndarray) -> np.ndarray:
        """
        Returns: (H, W, 3) metric point map in camera frame [metres].
        """
        H, W = rgb_hwc_u8.shape[:2]
        # UniDepthV2.infer expects uint8 [B, 3, H, W] with normalize=True
        img_t = torch.from_numpy(rgb_hwc_u8).permute(2, 0, 1).unsqueeze(0)
        # infer() handles normalization internally
        out = self.model.infer(img_t.float())  # normalize=True does /255 + imagenet norm

        # out["points"] is [B, 3, H, W] metric XYZ
        pts = out["points"].squeeze(0)  # [3, H, W]
        pts = pts.permute(1, 2, 0)     # → [H, W, 3]
        pts_np = pts.cpu().float().numpy()

        # UniDepthV2 gives full-image coverage, no mask needed
        # Depth = Z channel; flag zero-depth as invalid
        depth_valid = pts_np[..., 2] > 0.01
        pts_np[~depth_valid] = np.nan
        return pts_np  # [H, W, 3]


class Sam3DEstimator:
    """Our model: uses sam3d_baseline.py (MetricScaleDecoder path)."""

    def __init__(self, config: str, metric_checkpoint: str, device: str, seed: int):
        from scripts.sam3d_baseline import Baseline
        self.adapter = Baseline(
            config_path=config,
            metric_scale_checkpoint=metric_checkpoint,
            device=device,
            seed=seed,
        )

    @torch.inference_mode()
    def infer_points(self, rgb_hwc_u8: np.ndarray, K_norm: np.ndarray,
                     object_mask: np.ndarray) -> np.ndarray:
        """
        object_mask: (H, W) bool — the object of interest.

        Returns: (H, W, 3) metric point map, NaN outside object mask.
        """
        img_t = torch.from_numpy(rgb_hwc_u8).permute(2, 0, 1).float() / 255.0
        intr_t = torch.from_numpy(K_norm)
        mask_t = torch.from_numpy(object_mask)

        pred = self.adapter.infer_with_mask(img_t, intr_t, mask_t)
        pts = pred["points_metric"]  # [H, W, 3]
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().float().numpy()
        return pts  # [H, W, 3]


# ---------------------------------------------------------------------------
# Per-dataset evaluation loop
# ---------------------------------------------------------------------------

DATASET_FLAGS = {
    "HAMMER":  {},
    "iBims-1": {},
    "DIODE":   {},
}


def run_dataset(
    estimator,
    estimator_name: str,
    dataset_name: str,
    dataset_root: Path,
    out_f,
    limit: Optional[int],
    seed: int,
):
    rng = np.random.default_rng(seed)
    index = [s for s in (dataset_root / ".index.txt").read_text().splitlines() if s.strip()]
    if limit is not None:
        idx = rng.choice(len(index), min(limit, len(index)), replace=False)
        idx.sort()
        index = [index[i] for i in idx]

    n_ok, n_fail = 0, 0
    for sample_id in tqdm(index, desc=f"{estimator_name}/{dataset_name}"):
        sample_dir = dataset_root / sample_id
        try:
            rgb, depth, seg, seg_labels, K_norm = load_sample(sample_dir)
        except Exception as e:
            n_fail += 1
            _write(out_f, {
                "estimator": estimator_name,
                "dataset": dataset_name,
                "sample_id": sample_id,
                "error": f"load_sample: {e}",
            })
            continue

        # Object mask — label==1 (as set by our converters)
        seg_mask = (seg == 1)
        if seg_mask.sum() < 100:
            # Skip samples with no usable object pixels
            continue

        # GT 3D bbox from sensor depth
        gt_box = gt_bbox(depth, K_norm, seg_mask)
        if gt_box is None:
            continue

        # Predicted point map
        try:
            with torch.inference_mode():
                if isinstance(estimator, Sam3DEstimator):
                    pts_pred = estimator.infer_points(rgb, K_norm, seg_mask)
                else:
                    pts_pred = estimator.infer_points(rgb, K_norm)
        except Exception:
            n_fail += 1
            _write(out_f, {
                "estimator": estimator_name,
                "dataset": dataset_name,
                "sample_id": sample_id,
                "error": traceback.format_exc().splitlines()[-1],
            })
            torch.cuda.empty_cache()
            continue

        # Pred bbox — use same visible pixels where GT depth is valid
        # (fair comparison: both measure what the sensor saw)
        gt_valid_mask = np.isfinite(depth) & seg_mask
        pred_box_visible = bbox3d_from_points(pts_pred, gt_valid_mask)

        # Also measure pred bbox over all object mask pixels (model's full view)
        # bbox3d_from_points handles NaN/inf filtering internally
        pred_box_full = bbox3d_from_points(pts_pred, seg_mask)

        record = {
            "estimator": estimator_name,
            "dataset": dataset_name,
            "sample_id": sample_id,
            "gt_bbox": gt_box,
        }
        if pred_box_visible is not None:
            record["pred_bbox_visible"] = pred_box_visible
            record["errors_visible"] = bbox_errors(pred_box_visible, gt_box)
        if pred_box_full is not None:
            record["pred_bbox_full"] = pred_box_full
            record["errors_full"] = bbox_errors(pred_box_full, gt_box)
        _write(out_f, record)
        n_ok += 1

    print(f"[{estimator_name}/{dataset_name}] {n_ok} ok, {n_fail} fail")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if hasattr(v, "item"):
        try: return v.item()
        except Exception: pass
    return str(v)


def _write(f, record: dict):
    f.write(json.dumps(record, default=_json_default) + "\n")
    f.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimator", required=True,
                    choices=["moge2", "unidepth2", "sam3d"])
    ap.add_argument("--datasets", nargs="+", default=["HAMMER", "iBims-1"])
    ap.add_argument("--datasets-root", type=Path,
                    default=Path("/mnt/source/datasets_sam3d/eval"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # sam3d-specific
    ap.add_argument("--config", type=Path)
    ap.add_argument("--metric-checkpoint", type=Path)
    args = ap.parse_args()

    # Build estimator
    if args.estimator == "moge2":
        estimator = MoGe2Estimator(device=args.device)
    elif args.estimator == "unidepth2":
        estimator = UniDepthV2Estimator(device=args.device)
    elif args.estimator == "sam3d":
        if not args.config or not args.metric_checkpoint:
            ap.error("--estimator sam3d requires --config and --metric-checkpoint")
        estimator = Sam3DEstimator(
            config=str(args.config),
            metric_checkpoint=str(args.metric_checkpoint),
            device=args.device,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(args.estimator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode) as f:
        for ds in args.datasets:
            run_dataset(
                estimator, args.estimator, ds,
                dataset_root=args.datasets_root / ds,
                out_f=f,
                limit=args.limit,
                seed=args.seed,
            )

    print(f"[done] → {args.output}")


if __name__ == "__main__":
    main()
