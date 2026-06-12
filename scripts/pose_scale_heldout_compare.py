#!/usr/bin/env python
"""
Test 1: pose-decoder scale comparison on the mixed-10k heldout split (375 samples).

Runs the frozen stock pipeline (stage1_only — SS + pose decoder, no SLAT/mesh) on
the exact heldout instances from feature_caches/moge2_mixed_10k.pt, one pointmap
CONDITION per invocation, and records the pose decoder's out["scale"]:

    --condition moge_v1   pointmap=None (LIVE MoGe-v1 inside the pipeline, status quo)
    --condition moge2     LIVE MoGe-2 (Ruicheng/moge-2-vitl from /mnt/source/MoGe,
                          self-intrinsics, OpenCV->PyTorch3D [-X,-Y,Z] — the same
                          estimator + conversion as precompute_moge2_pointmaps.py)
    --condition gt        GT depth + intrinsics pointmap (NOCS-only, 64 samples)

Per-axis pred dims = canonical SS voxel extent * out["scale"]. The voxel extent
(coords/64 - 0.5 span + one voxel) is a mesh-extent proxy good to ~1-2 voxels —
fine for cross-condition comparison; do not compare absolute numbers against the
mesh-based gt_pointmap_metric_oracle.py headline without that caveat.

Same inference seed per instance in every condition, so the SS noise draw is
identical and the comparison isolates the pointmap anchor.

Compare runs:
    python scripts/pose_scale_heldout_compare.py --condition moge_v1
    python scripts/pose_scale_heldout_compare.py --condition moge2
    python scripts/pose_scale_heldout_compare.py --condition gt
Outputs artifacts/metric_scale/metrics/pose_scale_heldout_<condition>.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/conda/envs/sam3d"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/source/MoGe")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import sam3d_objects  # noqa: F401,E402  (registers targets; do not remove)
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
RGB_ROOTS = {
    "nocs_real275": f"{OMNI}/real_test",
    "objectron": OMNI,
    "arkitscenes": OMNI,
}
NOCS_RGB_ROOT = RGB_ROOTS["nocs_real275"]

# REAL275 real-camera intrinsics (640x480) — same as gt_pointmap_metric_oracle.py.
NOCS_K = np.array(
    [[591.0125, 0.0, 322.525], [0.0, 590.16775, 244.11084], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def load_pipeline(config_path):
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.metric_scale_checkpoint_path = None
    config.workspace_dir = os.path.dirname(config_path)
    return instantiate(config)


def opencv_to_pytorch3d(pts: np.ndarray) -> np.ndarray:
    """[H,W,3] OpenCV (x-right,y-down,z-forward) -> PyTorch3D ([-X,-Y,Z]). NaN preserved."""
    pm = pts.astype(np.float32, copy=True)
    pm[..., 0] *= -1.0
    pm[..., 1] *= -1.0
    return pm


def build_metric_pointmap(depth_m, K):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    return torch.from_numpy(np.stack([-X, -Y, Z], axis=-1).astype(np.float32))


def load_gt_pointmap(image_name, mask_shape):
    import imageio.v3 as iio
    rel = image_name.split("/test/")[-1]
    p = Path(NOCS_RGB_ROOT) / f"{rel}_depth.png"
    if not p.exists():
        return None
    depth_m = iio.imread(p).astype(np.float32) / 1000.0
    depth_m[depth_m <= 0] = np.nan
    H, W = mask_shape
    K = NOCS_K.copy()
    if depth_m.shape != (H, W):
        import cv2
        sy, sx = H / depth_m.shape[0], W / depth_m.shape[1]
        K[0, :] *= sx
        K[1, :] *= sy
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
    return build_metric_pointmap(depth_m, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["moge_v1", "moge2", "gt"])
    ap.add_argument("--config", default=str(REPO / "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--heldout-meta",
                    default="artifacts/metric_scale/metrics/heldout_375_meta.json",
                    help="JSON sidecar with the heldout uids (extracted from the "
                         "Stage C cache; avoids the 6.4 GB torch.load per run)")
    ap.add_argument("--inference-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = Path(args.output or
                    f"artifacts/metric_scale/metrics/pose_scale_heldout_{args.condition}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The heldout split is defined by the Stage C cache — uid is the join key.
    with open(args.heldout_meta) as f:
        heldout = json.load(f)
    print(f"heldout entries in cache: {len(heldout)}")
    if args.condition == "gt":
        heldout = [h for h in heldout if h["source"] == "nocs_real275"]
        print(f"gt condition is NOCS-only: {len(heldout)} samples")
    if args.limit:
        heldout = heldout[: args.limit]

    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI,
        sources=["nocs_real275", "objectron", "arkitscenes"],
        rgb_roots=RGB_ROOTS,
        split="train",
        max_records_per_source=3334,  # exact mixed-10k recipe -> same record pool
        skip_missing_rgb=True,
    )
    uid_to_idx = {rec["uid"]: i for i, rec in enumerate(ds.records)}

    est = None
    if args.condition == "moge2":
        from eval_bbox_from_depth import MoGe2Estimator
        est = MoGe2Estimator(device="cuda")
    pipeline = load_pipeline(args.config)

    # Resume: skip uids already in the output file.
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["uid"] for line in f if line.strip()}
        print(f"resuming: {len(done)} uids already done")

    fout = open(out_path, "a")
    rels = []
    per_source = {}
    skipped = 0
    for n, entry in enumerate(heldout):
        uid = entry["uid"]
        if uid in done:
            continue
        try:
            idx = uid_to_idx[uid]
            item = ds[idx]
            rgba = item["image"]
            mask = rgba[..., 3] > 127
            gt_dims = np.asarray(item["metric_dims"], dtype=np.float64)

            if args.condition == "moge_v1":
                pm = None
            elif args.condition == "moge2":
                rgb = np.ascontiguousarray(rgba[..., :3])
                pm = torch.from_numpy(
                    opencv_to_pytorch3d(est.infer_points(rgb, np.eye(3)))
                )
            else:
                pm = load_gt_pointmap(item["image_name"], mask.shape)
                if pm is None:
                    skipped += 1
                    continue

            out = pipeline.run(
                rgba, None, args.inference_seed,
                stage1_only=True, pointmap=pm,
            )
            scale = out["scale"][0].detach().cpu().float().numpy().reshape(-1)
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            voxel = out["voxel"]
            voxel = voxel.detach().cpu().numpy() if torch.is_tensor(voxel) else np.asarray(voxel)
            if voxel.size == 0:
                skipped += 1
                continue
            ext = (voxel.max(axis=0) - voxel.min(axis=0)) + 1.0 / 64.0
            pred_dims = ext * scale[:3]
            gt_iso = float(np.max(gt_dims))
            pred_iso = float(np.max(pred_dims))
            rel = abs(pred_iso - gt_iso) / max(gt_iso, 1e-4)

            rec = dict(
                condition=args.condition, uid=uid, source=item["source"],
                category=item["category"], image_name=item["image_name"],
                gt_dims=gt_dims.tolist(), gt_iso=gt_iso,
                scale=scale.tolist(), voxel_extent=ext.tolist(),
                pred_dims=pred_dims.tolist(), pred_iso=pred_iso, rel_iso=rel,
            )
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            rels.append(rel)
            per_source.setdefault(item["source"], []).append(rel)
            print(f"[{n + 1}/{len(heldout)}] {item['source']:<13} {item['category']:<12} "
                  f"gt_iso={gt_iso:.3f} pred_iso={pred_iso:.3f} rel={rel:.3f}", flush=True)
        except Exception as e:
            skipped += 1
            print(f"  [skip] uid={uid}: {type(e).__name__}: {e}", flush=True)
            continue
    fout.close()

    print(f"\n=== condition={args.condition} (this run: n={len(rels)}, skipped={skipped}) ===")
    if rels:
        a = np.array(rels)
        print(f"overall  median={np.median(a) * 100:6.1f}%  mean={np.mean(a) * 100:6.1f}%")
        for src in sorted(per_source):
            a = np.array(per_source[src])
            print(f"{src:<13} n={len(a):>3}  median={np.median(a) * 100:6.1f}%  "
                  f"mean={np.mean(a) * 100:6.1f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
