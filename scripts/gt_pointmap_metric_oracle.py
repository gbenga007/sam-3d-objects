#!/usr/bin/env python
"""
GT-pointmap metric-scale oracle.

Tests the claim surfaced by the depth+intrinsics notebook: the frozen SAM3D pose
decoder already recovers METRIC object scale when fed a metric pointmap. The
mechanism (verified in code) is:

    pointmap_scale = mean(|pointmap - median_z|)          # pose_target.get_scale_and_shift
    out["scale"]   = ssi_ratio (model) * pointmap_scale   # ssi_to_metric, size = s_tilde * tscale * s_scene

So out["scale"] is proportional to the *units of the input pointmap*. With MoGe-v1
(affine-invariant) that anchor is non-metric -> the head/decoder had to compensate.
Feed a GROUND-TRUTH metric pointmap (NOCS Kinect depth + REAL275 intrinsics, exactly
the notebook's construction) and the stock frozen model should produce metric scale
with no MetricScaleHead at all.

Decisive A/B (same frozen model, two pointmap sources):
  * pointmap = GT metric  -> condition 1 (upper bound / oracle)
  * pointmap = None        -> MoGe-v1 runs internally (condition 2, the status quo)

Method per held-out NOCS object (RGBA from the real loader; mask baked in alpha):
  full RGB + per-instance mask -> inference(..., pointmap=gt_metric_pointmap)
  pred_dims = mesh.extents * out["scale"]      # canonical mesh -> metric (no ICP; postprocess off)
  pred_iso  = max(pred_dims);  gt_iso = max(GT metric_dims);  rel = |pred_iso-gt_iso|/gt_iso

Only NOCS is wired up: it is the one source with GT depth + fixed intrinsics on disk
(real_test/<scene>/<frame>_depth.png, 640x480, mm). Objectron/ARKit GT depth is not
local; extend RGB_ROOTS/DEPTH + intrinsics per source to add them.

NOTE: __call__ runs with with_layout_postprocess=False, so out["scale"] is the raw
pose-decoder output (ssi_to_metric * pointmap_scale) with NO ICP refinement -- this
isolates the pointmap->scale mechanism rather than ICP-to-pointcloud alignment.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Mirror notebook/inference.py env setup (it imports heavy viz deps we skip here).
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/conda/envs/sam3d"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
import imageio.v3 as iio
from omegaconf import OmegaConf
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import sam3d_objects  # noqa: F401,E402  (registers targets; do not remove)
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap  # noqa: E402,F401
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402


def load_pipeline(config_path):
    """Minimal equivalent of notebook.inference.Inference.__init__ (no viz deps,
    skips the untrusted-config safety check since we load the repo's own yaml)."""
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"  # disable nvdiffrast
    config.compile_model = False
    config.metric_scale_checkpoint_path = None
    config.workspace_dir = os.path.dirname(config_path)
    return instantiate(config)


def run_inference(pipeline, rgb, mask, seed, pointmap):
    """Equivalent of Inference.__call__: merge mask into alpha, run with postprocess off."""
    mask_u8 = (mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb[..., :3], mask_u8], axis=-1)
    # decode_formats=["mesh"] skips the Gaussian decoder (avoids the optional gsplat
    # dep); out["scale"] is set by the pose decoder before decode, so it's unaffected.
    return pipeline.run(
        rgba, None, seed,
        stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
        with_layout_postprocess=False, use_vertex_color=True,
        stage1_inference_steps=None, pointmap=pointmap,
        decode_formats=["mesh"],
    )

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
NOCS_RGB_ROOT = f"{OMNI}/real_test"
RGB_ROOTS = {"nocs_real275": NOCS_RGB_ROOT}

# REAL275 real-camera intrinsics (640x480), identical to the notebook's hardcoded K.
NOCS_K = np.array(
    [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def build_metric_pointmap(depth_m, K):
    """depth (HxW, metres, nan for invalid) + K -> pytorch3d-convention metric pointmap
    [H,W,3]. Mirrors the notebook: X=(u-cx)Z/fx, Y=(v-cy)Z/fy, pointmap=[-X,-Y,Z]."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pm = np.stack([-X, -Y, Z], axis=-1).astype(np.float32)
    return torch.from_numpy(pm)


def depth_path_for(image_name):
    # image_name is like "nocs_real275/test/scene_1/0066"; depth lives under
    # real_test/scene_1/0066_depth.png — strip everything up to and incl. ".../test/".
    rel = image_name.split("/test/")[-1]
    return Path(NOCS_RGB_ROOT) / f"{rel}_depth.png"


def load_depth_m(image_name):
    p = depth_path_for(image_name)
    if not p.exists():
        return None
    d = iio.imread(p).astype(np.float32) / 1000.0  # mm -> m
    d[d <= 0] = np.nan
    return d


def canonical_extents(out):
    """Per-axis extent of the canonical mesh. Prefers glb (when both formats are
    decoded); falls back to the raw MeshExtractResult in out['mesh'][0] (the path
    used when decode_formats=['mesh'], where glb is None)."""
    glb = out.get("glb", None)
    if glb is not None and hasattr(glb, "extents"):
        return np.asarray(glb.extents, dtype=np.float64)
    m = out.get("mesh", None)
    if not m:
        return None
    v = m[0].vertices
    v = v.detach().cpu().float().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return None
    return (v.max(axis=0) - v.min(axis=0)).astype(np.float64)


def pred_dims_from_output(out):
    """metric per-axis dims = canonical mesh extents * out['scale'] (3-vec).
    Mirrors the notebook (tfm.scale(out['scale']) on canonical verts) and
    mesh_scale_eval (scale applied to raw canonical verts)."""
    extents = canonical_extents(out)
    if extents is None:
        return None
    scale = out["scale"][0].detach().cpu().float().numpy().reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    dims = extents * scale[:3]
    if not np.all(np.isfinite(dims)) or dims.max() <= 0:
        return None
    return dims


def rel_iso(pred_dims, gt_dims):
    pred_iso = float(np.max(pred_dims))
    gt_iso = float(np.max(gt_dims))
    if gt_iso < 1e-4:
        return None, pred_iso, gt_iso
    return abs(pred_iso - gt_iso) / gt_iso, pred_iso, gt_iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--pool-per-source", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inference-seed", type=int, default=42)
    ap.add_argument("--min-mask-pixels", type=int, default=200)
    ap.add_argument("--compare-moge", action="store_true",
                    help="also run pointmap=None (MoGe-v1) on each instance for an A/B")
    ap.add_argument("--output",
                    default="artifacts/metric_scale/metrics/gt_pointmap_metric_oracle.jsonl")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pipeline = load_pipeline(args.config)

    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=["nocs_real275"], rgb_roots=RGB_ROOTS,
        split="train", max_records_per_source=args.pool_per_source,
        skip_missing_rgb=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")

    rels_gt, rels_moge = [], []
    idxs = rng.permutation(len(ds))
    for i in idxs:
        if len(rels_gt) >= args.per_source:
            break
        try:
            item = ds[int(i)]
            if item["mask_pixels"] < args.min_mask_pixels:
                continue
            depth_m = load_depth_m(item["image_name"])
            if depth_m is None:
                continue

            rgba = item["image"]
            rgb = np.ascontiguousarray(rgba[..., :3])
            mask = rgba[..., 3] > 127
            gt_dims = np.asarray(item["metric_dims"], dtype=np.float64)

            # depth is native NOCS res; align K to the (possibly resized) image res
            H, W = mask.shape
            K = NOCS_K.copy()
            if depth_m.shape != (H, W):
                sy, sx = H / depth_m.shape[0], W / depth_m.shape[1]
                K[0, :] *= sx
                K[1, :] *= sy
                import cv2
                depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

            pm = build_metric_pointmap(depth_m, K)

            out = run_inference(pipeline, rgb, mask, args.inference_seed, pm)
            pdims = pred_dims_from_output(out)
            if pdims is None:
                continue
            r, p_iso, g_iso = rel_iso(pdims, gt_dims)
            if r is None:
                continue

            cext = canonical_extents(out)
            sc = out["scale"][0].detach().cpu().float().numpy().reshape(-1)
            rec = dict(source="nocs_real275", uid=item["uid"],
                       category=item["category"], image_name=item["image_name"],
                       gt_iso=g_iso, gt_dims=gt_dims.tolist(),
                       pred_iso_gtpm=p_iso, pred_dims_gtpm=pdims.tolist(),
                       rel_gtpm=r,
                       canon_ext=(cext.tolist() if cext is not None else None),
                       scale_gtpm=sc.tolist())
            rels_gt.append(r)

            if args.compare_moge:
                out_m = run_inference(pipeline, rgb, mask, args.inference_seed, None)
                pdims_m = pred_dims_from_output(out_m)
                if pdims_m is not None:
                    rm, pm_iso, _ = rel_iso(pdims_m, gt_dims)
                    if rm is not None:
                        rec.update(pred_iso_moge=pm_iso, rel_moge=rm)
                        rels_moge.append(rm)

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            tail = f"  moge={rec.get('rel_moge', float('nan')):.3f}" if args.compare_moge else ""
            print(f"[{len(rels_gt):>3}] {item['category']:<8} "
                  f"gt_iso={g_iso:.3f} pred={p_iso:.3f} rel_gtpm={r:.3f}{tail}",
                  flush=True)
        except Exception as e:
            print(f"  [skip] idx={int(i)}: {e}", flush=True)
            continue
    fout.close()

    def summ(name, rels):
        rels = np.array(rels)
        if not len(rels):
            print(f"{name:<18} n=0")
            return
        print(f"{name:<18} n={len(rels):>4}  median={np.median(rels)*100:>6.1f}%  "
              f"mean={np.mean(rels)*100:>6.1f}%")

    print("\n=== NOCS isotropic metric-size error (frozen stock model) ===")
    summ("GT-pointmap", rels_gt)
    if args.compare_moge:
        summ("MoGe-v1 pointmap", rels_moge)
    print("\nReference: trained MetricScaleHead ~7% (NOCS), category-prior M1 13.5%.")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
