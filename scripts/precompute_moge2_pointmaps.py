#!/usr/bin/env python
"""
Stage A — precompute MoGe-2 metric pointmaps for the mixed ~10k training set.

For each UNIQUE FRAME in the training record set (the pointmap is per-frame/global, not
per-object — many object instances share one frame), run MoGe-2 once and save the global
metric pointmap to disk. Training (Stage B) then injects these via
`compute_pointmap(image, pointmap=...)` in place of the live MoGe-v1 call.

WEIGHTS: MoGe-2 `Ruicheng/moge-2-vitl` (verified), via `MoGe2Estimator` in
`eval_bbox_from_depth.py` (`moge.model.v2.MoGeModel`, /mnt/source/MoGe) — the SAME estimator
the pointmap_scale screen used, so these pointmaps are consistent with the 1.08 screen finding.

CONVENTION: MoGe-2 returns points in OpenCV camera frame (x-right, y-down, z-forward, metric).
The pipeline expects a provided pointmap in PyTorch3D convention (as the notebook/oracle used),
so we negate x and y on save → [-X, -Y, Z]. NaN outside the model mask is preserved (the
pipeline's ObjectCentricSSI uses nanmedian and PointPatchEmbed has an invalid-xyz token).

RECORD SET: reconstructs `train_mixed_scratch_10k.sh` exactly (OmniNOCSObjectDataset, sources
nocs_real275/objectron/arkitscenes, max_records_per_source 3334, skip_missing_rgb, split train,
min_mask_pixels 500). Covers both train and per-source heldout frames (both run the pipeline).

Output: artifacts/metric_scale/moge2_pointmaps/<sanitized image_name>.npy (float16 [H,W,3])
        + manifest.json mapping image_name -> file. Resumable (skips existing files).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/source/MoGe")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from eval_bbox_from_depth import MoGe2Estimator  # noqa: E402  (verified moge-2-vitl)
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset  # noqa: E402

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
# Exactly the rgb roots from train_mixed_scratch_10k.sh
RGB_ROOTS = {
    "nocs_real275": f"{OMNI}/real_test",
    "objectron": OMNI,
    "arkitscenes": OMNI,
}


def sanitize(image_name: str) -> str:
    return image_name.replace("/", "__").replace(" ", "_")


def opencv_to_pytorch3d(pts: np.ndarray) -> np.ndarray:
    """[H,W,3] OpenCV (x-right,y-down,z-forward) -> PyTorch3D ([-X,-Y,Z]). NaN preserved."""
    pm = pts.astype(np.float32, copy=True)
    pm[..., 0] *= -1.0
    pm[..., 1] *= -1.0
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["nocs_real275", "objectron", "arkitscenes"])
    ap.add_argument("--max-records-per-source", type=int, default=3334)
    ap.add_argument("--min-mask-pixels", type=int, default=500)
    ap.add_argument("--out-dir", default="artifacts/metric_scale/moge2_pointmaps")
    ap.add_argument("--limit", type=int, default=0, help="cap #frames (smoke); 0 = all")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    # Reconstruct the exact training record set (train_mixed_scratch_10k.sh).
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI,
        sources=args.sources,
        rgb_roots=RGB_ROOTS,
        split="train",
        categories=None,
        min_mask_pixels=args.min_mask_pixels,
        max_records=None,
        max_records_per_source=args.max_records_per_source,
        skip_missing_rgb=True,
    )
    print(f"dataset: {len(ds)} object records across {args.sources}", flush=True)

    # First record index per unique frame (image_name).
    frame_to_idx: dict[str, int] = {}
    frame_src: dict[str, str] = {}
    for idx, rec in enumerate(ds.records):
        name = rec["image_name"]
        if name not in frame_to_idx:
            frame_to_idx[name] = idx
            frame_src[name] = rec["source"]
    frames = list(frame_to_idx.items())
    if args.limit:
        frames = frames[: args.limit]
    print(f"unique frames to precompute: {len(frames)}", flush=True)

    est = MoGe2Estimator(device="cuda")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "model": "Ruicheng/moge-2-vitl",
        "convention": "pytorch3d ([-X,-Y,Z]); metric metres; NaN outside model mask",
        "dtype": args.dtype,
        "sources": args.sources,
        "max_records_per_source": args.max_records_per_source,
        "frames": {},  # image_name -> relative npy path
    }
    if manifest_path.exists():  # resume: keep prior frame map
        try:
            prev = json.load(open(manifest_path))
            manifest["frames"].update(prev.get("frames", {}))
        except Exception:
            pass

    done = skipped = failed = 0
    t0 = time.time()
    for i, (name, idx) in enumerate(frames):
        fname = sanitize(name) + ".npy"
        fpath = out_dir / fname
        if fpath.exists():
            manifest["frames"][name] = fname
            skipped += 1
            continue
        try:
            item = ds[idx]  # RGBA at the exact resolution training uses
            rgb = np.ascontiguousarray(item["image"][..., :3])  # [H,W,3] uint8
            pts = est.infer_points(rgb, np.eye(3))              # OpenCV metric, NaN masked
            pm = opencv_to_pytorch3d(pts).astype(save_dtype)
            # tmp name must end in .npy or np.save appends it and os.replace misses
            tmp = fpath.with_name(fpath.name + ".tmp.npy")
            np.save(tmp, pm)
            os.replace(tmp, fpath)                              # atomic
            manifest["frames"][name] = fname
            done += 1
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)
            failed += 1
            continue
        if (i + 1) % 50 == 0 or i + 1 == len(frames):
            json.dump(manifest, open(manifest_path, "w"))
            rate = (done + skipped) / max(time.time() - t0, 1e-6)
            print(f"[{i+1}/{len(frames)}] done={done} skip={skipped} fail={failed} "
                  f"{rate:.1f} frame/s  last={frame_src[name]}", flush=True)

    json.dump(manifest, open(manifest_path, "w"))
    print(f"\nDONE: wrote {done} new, {skipped} existing, {failed} failed. "
          f"Total mapped frames: {len(manifest['frames'])}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
