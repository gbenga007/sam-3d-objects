# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Objectron dataset parser.

Raw layout expected (after gsutil download):
    {root}/
        {category}/              e.g. cup, bottle, camera, ...
            batch-{i}/
                {j}/
                    video.MOV
                    geometry.pbdata       (or annotation.pbdata)

Preprocessed layout produced by scripts/preprocess_objectron.py:
    {preprocessed_root}/
        metadata.jsonl           one record per line (see _Record fields)
        images/
            {uid}.png            RGBA: RGB image + binary mask in alpha channel

Objectron 3-D bounding box convention
--------------------------------------
Each annotation frame contains:
    object.rotation_world     — 3×3 row-major rotation matrix (object → world)
    object.translation_world  — 3-vector (object centre in world space, metres)
    object.scale              — 3-vector (full extents [w, h, d] in metres)

We expose object.scale directly as metric_dims = [w, h, d].

References
----------
https://github.com/google-research-datasets/Objectron
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Objectron categories available on GCS
OBJECTRON_CATEGORIES = [
    "bike",
    "book",
    "bottle",
    "camera",
    "cereal_box",
    "chair",
    "cup",
    "laptop",
    "shoe",
]


# ---------------------------------------------------------------------------
# Dataset class (loads preprocessed records)
# ---------------------------------------------------------------------------

class ObjectronDataset(Dataset):
    """
    Loads preprocessed Objectron records produced by scripts/preprocess_objectron.py.

    Each item:
        image      np.ndarray  [H, W, 4] uint8 — RGBA (alpha = object mask)
        metric_dims np.ndarray [3] float32      — [width, height, depth] in metres
        category   str
        uid        str
    """

    def __init__(self, preprocessed_root: str, categories: list[str] = None):
        self.root = Path(preprocessed_root)
        meta_path = self.root / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found in {self.root}. "
                "Run scripts/preprocess_objectron.py first."
            )
        self.records = []
        with open(meta_path) as f:
            for line in f:
                rec = json.loads(line)
                if categories is None or rec["category"] in categories:
                    self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img_path = self.root / rec["image_path"]
        image = np.array(Image.open(img_path).convert("RGBA"), dtype=np.uint8)
        return {
            "image": image,                                   # [H, W, 4] uint8
            "metric_dims": np.array(rec["metric_dims"], dtype=np.float32),  # [3]
            "category": rec["category"],
            "uid": rec["uid"],
            "source": "objectron",
        }


# ---------------------------------------------------------------------------
# Frame extraction helpers (used by preprocessing script)
# ---------------------------------------------------------------------------

def iter_objectron_sequences(raw_root: str, categories: list[str] = None) -> Iterator[Path]:
    """Yield paths to individual sequence directories under raw_root."""
    root = Path(raw_root)
    cats = categories or OBJECTRON_CATEGORIES
    for cat in cats:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        for batch_dir in sorted(cat_dir.iterdir()):
            for seq_dir in sorted(batch_dir.iterdir()):
                if (seq_dir / "video.MOV").exists() or any(seq_dir.glob("*.pbdata")):
                    yield seq_dir, cat


def extract_frame_and_annotation(seq_dir: Path) -> dict | None:
    """
    Extract one representative frame and its 3-D bounding box from an
    Objectron sequence directory.

    Returns dict with keys:
        rgb        np.ndarray [H, W, 3] uint8
        mask_2d    np.ndarray [H, W]   bool    — from projected bbox hull
        metric_dims [w, h, d] in metres
        intrinsics  np.ndarray [3, 3]
    or None if parsing fails.

    Requires the `objectron` pip package:
        pip install objectron
    """
    try:
        from objectron.dataset import sequence as obj_seq
        from objectron.dataset.graphics import draw_annotation
    except ImportError:
        raise ImportError(
            "Install the objectron package: pip install objectron"
        )

    import cv2

    video_path = seq_dir / "video.MOV"
    ann_paths = list(seq_dir.glob("*.pbdata"))
    if not video_path.exists() or not ann_paths:
        return None

    try:
        seq = obj_seq.Sequence(str(seq_dir))
        # Pick middle frame for representativeness
        mid_idx = len(seq.frames) // 2
        frame = seq.frames[mid_idx]

        # Extract RGB from video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # Parse 3D bounding box for the primary object
        annotation = frame.annotations[0]
        # object.scale = [width, height, depth] full extents in metres
        metric_dims = np.array([
            annotation.object.scale[0],
            annotation.object.scale[1],
            annotation.object.scale[2],
        ], dtype=np.float32)

        # Project 3D bbox corners to 2D to build an approximate mask
        keypoints_2d = np.array([
            [kp.x * W, kp.y * H] for kp in annotation.keypoints
        ], dtype=np.float32)  # [9, 2] — centre + 8 corners

        corners_2d = keypoints_2d[1:]  # drop centre keypoint → [8, 2]
        mask_2d = _corners_to_mask(corners_2d, H, W)

        # Camera intrinsics from annotation
        intrinsics = np.array([
            [frame.camera.intrinsics[0], 0,                         frame.camera.intrinsics[2]],
            [0,                          frame.camera.intrinsics[1], frame.camera.intrinsics[3]],
            [0,                          0,                          1],
        ], dtype=np.float32)

        return {
            "rgb": rgb,
            "mask_2d": mask_2d,
            "metric_dims": metric_dims,
            "intrinsics": intrinsics,
        }

    except Exception:
        return None


def _corners_to_mask(corners_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    """Fill the convex hull of projected 3D bbox corners."""
    import cv2
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(corners_2d)
        pts = corners_2d[hull.vertices].astype(np.int32)
    except Exception:
        pts = corners_2d.astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def rgba_from_rgb_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Combine [H,W,3] RGB + [H,W] bool mask into [H,W,4] uint8 RGBA."""
    alpha = (mask * 255).astype(np.uint8)[..., None]
    return np.concatenate([rgb, alpha], axis=-1)
