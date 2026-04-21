#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Preprocess the NOCS REAL275 dataset into the format expected by NOCSDataset.

Steps per instance:
  1. Identify object instances from mask + meta.txt
  2. Crop instance RGBA from the full frame
  3. Resolve metric_dims [w, h, d] in metres from GT pkl (preferred)
     or mesh bounding box (fallback)
  4. Save RGBA image + metadata record

Download instructions
---------------------
    wget http://download.cs.stanford.edu/orion/nocs/real_test.zip
    wget http://download.cs.stanford.edu/orion/nocs/obj_models.zip
    wget http://download.cs.stanford.edu/orion/nocs/gts.zip
    unzip real_test.zip && unzip obj_models.zip && unzip gts.zip

Usage
-----
    python scripts/preprocess_nocs.py \\
        --raw_root  /data/nocs/raw \\
        --out_root  /data/nocs/preprocessed \\
        --categories mug bottle bowl \\
        --min_mask_pixels 500          # skip tiny / occluded instances

Output
------
    {out_root}/
        metadata.jsonl
        images/
            {uid}.png    RGBA uint8
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam3d_objects.data.dataset.metric.nocs import (
    NOCS_CATEGORIES,
    extract_instance_rgba,
    get_metric_dims_from_gt,
    get_metric_dims_from_mesh,
    iter_nocs_instances,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--categories", nargs="+", default=NOCS_CATEGORIES)
    parser.add_argument("--min_mask_pixels", type=int, default=500,
                        help="Skip instances whose mask has fewer pixels than this")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gt_root = raw_root / "gts" / "real_test"
    obj_models_root = raw_root / "obj_models" / "real_test"

    counts: dict[str, int] = {}
    skipped_mask = 0
    skipped_dims = 0

    with open(out_root / "metadata.jsonl", "w") as meta_f:
        for inst in tqdm(
            iter_nocs_instances(str(raw_root), args.categories),
            desc="instances",
        ):
            rgba = extract_instance_rgba(
                inst["color_path"], inst["mask_path"], inst["instance_id"]
            )
            if rgba is None:
                skipped_mask += 1
                continue

            # Filter tiny instances (heavily occluded or far away)
            mask_pixels = (rgba[..., 3] > 0).sum()
            if mask_pixels < args.min_mask_pixels:
                skipped_mask += 1
                continue

            # Resolve metric dimensions — prefer GT pkl, fall back to mesh
            scene = inst["scene"]
            scene_num = scene.replace("scene_", "")
            gt_pkl = gt_root / f"results_real_test_scene_{scene_num}.pkl"
            metric_dims = get_metric_dims_from_gt(
                gt_pkl,
                scene,
                inst["color_path"].name,
                inst["instance_id"],
            )
            if metric_dims is None:
                metric_dims = get_metric_dims_from_mesh(
                    obj_models_root, inst["category"], inst["category"]
                )
            if metric_dims is None:
                skipped_dims += 1
                continue

            # Sanity check: all dims must be positive
            if np.any(metric_dims <= 0):
                skipped_dims += 1
                continue

            uid = str(uuid.uuid4())[:12]
            img_filename = f"{uid}.png"
            Image.fromarray(rgba).save(images_dir / img_filename)

            record = {
                "uid": uid,
                "image_path": f"images/{img_filename}",
                "metric_dims": metric_dims.tolist(),
                "category": inst["category"],
                "source": "nocs",
                "scene": scene,
                "instance_id": inst["instance_id"],
            }
            meta_f.write(json.dumps(record) + "\n")
            counts[inst["category"]] = counts.get(inst["category"], 0) + 1

    total = sum(counts.values())
    print(f"\nSaved {total} records")
    print(f"  Skipped (bad mask / too small): {skipped_mask}")
    print(f"  Skipped (no metric dims):       {skipped_dims}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:10s}: {n}")


if __name__ == "__main__":
    main()
