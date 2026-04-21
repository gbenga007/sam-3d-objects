#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Preprocess the Objectron dataset into the format expected by ObjectronDataset.

Steps per sequence:
  1. Extract one representative frame from video.MOV
  2. Parse 3D bounding box → metric_dims [w, h, d] in metres
  3. Project 3D bbox corners to 2D → binary object mask
  4. Save RGBA image + metadata record

Usage
-----
# Download raw data first (requires gsutil):
#   gsutil -m cp -r gs://objectron/videos/{category}/ {raw_root}/{category}/
#   (repeat for each category you want)

python scripts/preprocess_objectron.py \\
    --raw_root  /data/objectron/raw \\
    --out_root  /data/objectron/preprocessed \\
    --categories cup bottle camera \\
    --max_per_category 500          # optional cap

Output
------
    {out_root}/
        metadata.jsonl
        images/
            {uid}.png      RGBA uint8
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam3d_objects.data.dataset.metric.objectron import (
    OBJECTRON_CATEGORIES,
    extract_frame_and_annotation,
    iter_objectron_sequences,
    rgba_from_rgb_mask,
)
from moge_utils import load_moge, compute_pointmap_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--categories", nargs="+", default=OBJECTRON_CATEGORIES)
    parser.add_argument("--max_per_category", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.jsonl"

    print("Loading MoGe...")
    moge = load_moge(device=args.device)

    counts: dict[str, int] = {}
    skipped = 0
    skipped_moge = 0

    with open(meta_path, "w") as meta_f:
        for seq_dir, category in tqdm(
            iter_objectron_sequences(args.raw_root, args.categories),
            desc="sequences",
        ):
            if (
                args.max_per_category is not None
                and counts.get(category, 0) >= args.max_per_category
            ):
                continue

            result = extract_frame_and_annotation(seq_dir)
            if result is None:
                skipped += 1
                continue

            rgba = rgba_from_rgb_mask(result["rgb"], result["mask_2d"])

            # Run MoGe to get pointmap stats (metric anchor for scale head)
            moge_stats = compute_pointmap_stats(rgba, moge, device=args.device)
            if moge_stats is None:
                skipped_moge += 1
                continue

            uid = str(uuid.uuid4())[:12]
            img_filename = f"{uid}.png"
            Image.fromarray(rgba).save(images_dir / img_filename)

            record = {
                "uid": uid,
                "image_path": f"images/{img_filename}",
                "metric_dims": result["metric_dims"].tolist(),
                "pointmap_scale": moge_stats["pointmap_scale"],
                "pointmap_shift": moge_stats["pointmap_shift"],
                "category": category,
                "source": "objectron",
                "seq_dir": str(seq_dir),
            }
            meta_f.write(json.dumps(record) + "\n")
            counts[category] = counts.get(category, 0) + 1

    total = sum(counts.values())
    print(f"\nSaved {total} records → {out_root}")
    print(f"  Skipped (bad annotation): {skipped}")
    print(f"  Skipped (MoGe failed):    {skipped_moge}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:15s}: {n}")


if __name__ == "__main__":
    main()
