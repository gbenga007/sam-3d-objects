"""Run a metric-depth baseline over our MoGe-format eval datasets.

This is a thin orchestrator that doesn't go through MoGe's
`eval_baseline.py` because MoGe's `EvalDataLoaderPipeline` imports a
third-party `pipeline` module that isn't in our env. Instead we read each
MoGe sample directory manually and feed `compute_metrics` directly.

Per-baseline contract:
  - If the adapter exposes `infer_with_mask(image, intrinsics, mask)`, we
    use it (our sam3d_baseline). External depth estimators expose plain
    `infer(image, intrinsics)` and the mask only gates the metric eval, not
    the model.
  - The adapter returns one of the MoGe pred shapes: `points_metric`,
    `points_scale_invariant`, `depth_metric`, etc. `compute_metrics` does
    mode dispatch from the keys present.

Output: JSONL, one record per (baseline, dataset, sample).
Each record: {baseline, dataset, sample_id, metrics: {...}}.

The `metrics` dict is whatever modes `compute_metrics` returned (`points_metric`,
`local_points`, etc.), each with `{rel, delta1}`. An aggregator script (TBD)
collapses across samples for the headline table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Pipeline-import env (cheap to set even for baselines that don't need it).
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import numpy as np
import torch
from tqdm import tqdm

# utils3d v1.7 renamed several functions used by MoGe's compute_metrics.
# Patch before MoGe imports them so compute_metrics sees the old API.
import utils3d.torch as _u3t
from utils3d.torch.maps import pixel_coord_map as _pcm, uv_map as _uvm, depth_map_to_point_map as _d2p
from utils3d.torch.utils import sliding_window as _sw
from utils3d.torch.transforms import intrinsics_from_focal_center as _intr_fc
if not hasattr(_u3t, "image_pixel_center"):
    # image_pixel_center(width, height, ...) → pixel_coord_map(height, width, ...)
    def _image_pixel_center(width, height, dtype=torch.float32, device=None):
        return _pcm(height, width, dtype=dtype, device=device)
    _u3t.image_pixel_center = _image_pixel_center
if not hasattr(_u3t, "image_uv"):
    # image_uv(width, height, ...) → uv_map(height, width, ...)
    def _image_uv(width, height, dtype=torch.float32, device=None):
        return _uvm(height, width, dtype=dtype, device=device)
    _u3t.image_uv = _image_uv
if not hasattr(_u3t, "sliding_window_2d"):
    # sliding_window_2d was renamed to sliding_window; signature unchanged
    _u3t.sliding_window_2d = _sw
if not hasattr(_u3t, "depth_to_points"):
    # depth_to_points(depth, intrinsics=...) → depth_map_to_point_map(depth, intrinsics)
    # Wrap to move intrinsics to the same device as depth (MoGe builds intrinsics on CPU,
    # depth is on CUDA → device mismatch without this wrapper).
    def _depth_to_points(depth, intrinsics=None, **kwargs):
        if intrinsics is not None and hasattr(intrinsics, "device") and intrinsics.device != depth.device:
            intrinsics = intrinsics.to(depth.device)
        return _d2p(depth, intrinsics, **kwargs)
    _u3t.depth_to_points = _depth_to_points
if not hasattr(_u3t, "intrinsics_from_focal_center"):
    _u3t.intrinsics_from_focal_center = _intr_fc

from moge.utils.io import read_image, read_depth, read_segmentation
from moge.test.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Sample loading (no MoGe dataloader dependency)
# ---------------------------------------------------------------------------

def load_sample(sample_dir: Path):
    rgb = read_image(sample_dir / "image.jpg")                 # (H, W, 3) uint8
    depth, _unit = read_depth(sample_dir / "depth.png")        # (H, W) float, NaN where invalid; _unit is the physical unit (unused)
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
    fx, fy = float(K_norm[0, 0]), float(K_norm[1, 1])
    cx, cy = float(K_norm[0, 2]), float(K_norm[1, 2])
    z = depth
    x = (u_norm - cx) / fx * z
    y = (v_norm - cy) / fy * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def make_gt_dict(rgb, depth, seg, seg_labels, K_norm, device: str,
                 has_sharp_boundary: bool) -> Dict[str, Any]:
    depth_mask = np.isfinite(depth)
    depth_filled = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=1.0)
    points = backproject_depth_to_points(depth_filled, K_norm)

    gt = {
        "image": torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0,
        "depth": torch.from_numpy(depth_filled).float().to(device),
        "depth_mask": torch.from_numpy(depth_mask).bool().to(device),
        "points": torch.from_numpy(points).float().to(device),
        "intrinsics": torch.from_numpy(K_norm).float().to(device),
        "segmentation_mask": torch.from_numpy(seg.astype(np.int64)).to(device),
        "segmentation_labels": seg_labels or {"object": 1},
        "is_metric": True,
        "has_sharp_boundary": has_sharp_boundary,
    }
    return gt


# ---------------------------------------------------------------------------
# Adapter dispatch
# ---------------------------------------------------------------------------

def build_adapter(name: str, *, config: Optional[str], metric_checkpoint: Optional[str],
                  device: str, seed: int):
    if name == "sam3d":
        from scripts.sam3d_baseline import Baseline as Sam3dBaseline
        if config is None or metric_checkpoint is None:
            raise ValueError("sam3d baseline requires --config and --metric-checkpoint")
        return Sam3dBaseline(
            config_path=config,
            metric_scale_checkpoint=metric_checkpoint,
            device=device,
            seed=seed,
        )
    if name == "moge":
        from scripts.moge_baseline import MoGeBaseline
        return MoGeBaseline(device=device)
    # Stubs for the external baselines we'll wire up later.
    raise NotImplementedError(f"adapter '{name}' is not implemented yet")


def call_adapter(adapter, image_tensor: torch.Tensor,
                 intrinsics: torch.Tensor, object_mask: torch.Tensor,
                 oracle_intrinsics: bool) -> Dict[str, torch.Tensor]:
    """Use `infer_with_mask` if the adapter offers it (our sam3d model); else
    fall back to MoGe's standard `infer(image, intrinsics)` and hand-merge the
    mask in afterwards."""
    intr = intrinsics if oracle_intrinsics else None
    if hasattr(adapter, "infer_with_mask"):
        return adapter.infer_with_mask(image_tensor, intr, object_mask)
    return adapter.infer(image_tensor, intr)


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------

def _json_default(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return str(v)


def write_record(out_f, record: dict):
    out_f.write(json.dumps(record, default=_json_default) + "\n")
    out_f.flush()


# ---------------------------------------------------------------------------
# Per-dataset loop
# ---------------------------------------------------------------------------

DATASET_FLAGS = {
    "HAMMER":  {"has_sharp_boundary": True},
    "iBims-1": {"has_sharp_boundary": True},
    "DIODE":   {"has_sharp_boundary": False},
}


def run_dataset(adapter, baseline_name: str, dataset_name: str,
                dataset_root: Path, out_f, device: str,
                oracle_intrinsics: bool, limit: Optional[int]):
    flags = DATASET_FLAGS.get(dataset_name, {"has_sharp_boundary": False})

    index = [s for s in (dataset_root / ".index.txt").read_text().splitlines() if s.strip()]
    if limit is not None:
        index = index[: limit]

    t0 = time.time()
    n_ok, n_fail = 0, 0
    for sample_id in tqdm(index, desc=f"{baseline_name}/{dataset_name}"):
        sample_dir = dataset_root / sample_id
        try:
            rgb, depth, seg, seg_labels, K_norm = load_sample(sample_dir)
        except Exception as e:
            n_fail += 1
            write_record(out_f, {
                "baseline": baseline_name,
                "dataset": dataset_name,
                "sample_id": sample_id,
                "error": f"load_sample failed: {e}",
            })
            continue

        gt = make_gt_dict(rgb, depth, seg, seg_labels, K_norm, device=device,
                          has_sharp_boundary=flags["has_sharp_boundary"])
        object_mask = (gt["segmentation_mask"] == 1)

        # Restrict depth_mask to object pixels only.
        # Our sam3d adapter returns NaN outside the object mask; leaving the
        # full-image depth_mask causes NaN to propagate through the alignment
        # step in compute_metrics (align_points_xyz_shift / align_depth_scale),
        # yielding NaN for all scalar metrics.  Narrowing the mask to the
        # object ensures only valid pred pixels participate in alignment and
        # metric computation — which is also the correct object-level eval.
        gt["depth_mask"] = gt["depth_mask"] & object_mask

        try:
            with torch.inference_mode():
                pred = call_adapter(
                    adapter, gt["image"], gt["intrinsics"], object_mask,
                    oracle_intrinsics=oracle_intrinsics,
                )
                metrics, _ = compute_metrics(pred, gt, vis=False)
        except Exception:
            n_fail += 1
            write_record(out_f, {
                "baseline": baseline_name,
                "dataset": dataset_name,
                "sample_id": sample_id,
                "error": traceback.format_exc().splitlines()[-1],
            })
            torch.cuda.empty_cache()
            continue

        write_record(out_f, {
            "baseline": baseline_name,
            "dataset": dataset_name,
            "sample_id": sample_id,
            "metrics": metrics,
        })
        n_ok += 1

    elapsed = time.time() - t0
    print(f"[{baseline_name}/{dataset_name}] {n_ok} ok, {n_fail} failed in {elapsed:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        choices=["sam3d", "moge",
                                 "metric3d_v2", "moge2", "unidepth_v2",
                                 "depthpro", "mast3r"])
    parser.add_argument("--datasets", nargs="+", default=["HAMMER", "iBims-1", "DIODE"])
    parser.add_argument("--datasets-root", type=Path,
                        default=Path("/mnt/source/datasets_sam3d/eval"))
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL output path; appended/overwritten depending on --append.")
    parser.add_argument("--append", action="store_true",
                        help="Append records to --output (default overwrites).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Per-dataset sample cap (smoke tests).")
    parser.add_argument("--oracle-intrinsics", action="store_true",
                        help="Pass GT intrinsics to the adapter (used for the "
                             "'Depth (w/ GT Cam)' table column).")

    # sam3d-specific
    parser.add_argument("--config", type=Path,
                        help="Pipeline yaml for sam3d (checkpoints/hf/pipeline.yaml).")
    parser.add_argument("--metric-checkpoint", type=Path,
                        help="Metric head/decoder fine-tune checkpoint (sam3d).")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"[run] baseline={args.baseline}, datasets={args.datasets}, "
          f"oracle_K={args.oracle_intrinsics}")
    adapter = build_adapter(
        args.baseline,
        config=str(args.config) if args.config else None,
        metric_checkpoint=str(args.metric_checkpoint) if args.metric_checkpoint else None,
        device=args.device,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output.open(mode) as out_f:
        for dataset_name in args.datasets:
            run_dataset(
                adapter, args.baseline, dataset_name,
                dataset_root=args.datasets_root / dataset_name,
                out_f=out_f,
                device=args.device,
                oracle_intrinsics=args.oracle_intrinsics,
                limit=args.limit,
            )

    print(f"[done] results written to {args.output}")


if __name__ == "__main__":
    main()
