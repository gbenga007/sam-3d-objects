#!/usr/bin/env python3
"""
Download only the Hypersim preview images referenced by OmniNOCS metadata.

Strategy:
1. Read OmniNOCS Hypersim metadata.
2. Resolve the expected Hypersim preview JPEG path for each referenced frame.
3. Group missing frames by scene/camera trajectory.
4. For each missing scene/camera pair, invoke the official Hypersim downloader
   to fetch the whole preview trajectory:

       --scene ai_022_004 --contains scene_cam_01_final_preview --contains .tonemap.jpg

This intentionally downloads the full preview trajectory for a missing camera,
because the downloader filters by substring and camera-level fetches are much
more efficient than invoking it once per frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def expected_preview_path(rgb_root: Path, image_name: str) -> Path:
    scene, camera, frame = image_name.split("/")[:3]
    frame_idx = frame.replace("frame_", "").replace("frame.", "")
    return (
        rgb_root
        / scene
        / "images"
        / f"scene_{camera}_final_preview"
        / f"frame.{frame_idx}.tonemap.jpg"
    )


def load_missing_cameras(metadata_path: Path, rgb_root: Path) -> Counter[str]:
    frames = json.load(open(metadata_path))
    missing_by_camera: Counter[str] = Counter()
    for frame in frames:
        image_name = frame["image_name"]
        expected = expected_preview_path(rgb_root, image_name)
        if not expected.exists():
            scene, camera, _ = image_name.split("/")[:3]
            missing_by_camera[f"{scene}/{camera}"] += 1
    return missing_by_camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-path",
        default="/mnt/dest/OmniNOCS/omninocs_release_hypersim/hypersim_train_metadata.json",
    )
    parser.add_argument(
        "--rgb-root",
        default="/mnt/dest/OmniNOCS/omni3d_rgb/hypersim",
    )
    parser.add_argument(
        "--downloader",
        default="/mnt/dest/OmniNOCS/ml-hypersim/contrib/99991/download.py",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Optional cap on the number of missing camera trajectories to fetch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the camera download plan without invoking the downloader.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    rgb_root = Path(args.rgb_root)
    downloader = Path(args.downloader)

    missing_by_camera = load_missing_cameras(metadata_path, rgb_root)
    if not missing_by_camera:
        print("No missing Hypersim camera trajectories referenced by OmniNOCS.")
        return

    items = sorted(missing_by_camera.items(), key=lambda kv: (-kv[1], kv[0]))
    if args.max_cameras is not None:
        items = items[: args.max_cameras]

    total_missing = sum(count for _, count in items)
    print(
        f"Targeting {len(items)} missing camera trajectories "
        f"covering {total_missing} missing OmniNOCS-referenced frames."
    )

    for camera_key, missing_count in items:
        scene, camera = camera_key.split("/")
        cmd = [
            sys.executable,
            str(downloader),
            "--scene",
            scene,
            "--contains",
            f"scene_{camera}_final_preview",
            "--contains",
            ".tonemap.jpg",
            "--directory",
            str(rgb_root),
            "--silent",
        ]
        print(f"[{camera_key}] missing_frames={missing_count}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
