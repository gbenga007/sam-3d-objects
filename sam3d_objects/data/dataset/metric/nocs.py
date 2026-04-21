# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
NOCS REAL275 dataset parser.

Raw layout expected (after download):
    {root}/
        real_test/
            scene_1/ ... scene_6/
                XXXXXX_color.png      RGB image
                XXXXXX_mask.png       per-instance colour-coded mask
                XXXXXX_meta.txt       instance-id → category name mapping
        obj_models/real_test/
            {category}_{instance}/
                model.obj             3D mesh (used only for size extraction)
        gts/real_test/
            results_real_test_scene_N.pkl   per-scene GT dicts

Download instructions
---------------------
    # Images + masks
    wget http://download.cs.stanford.edu/orion/nocs/real_test.zip
    unzip real_test.zip

    # Object models (for metric sizes)
    wget http://download.cs.stanford.edu/orion/nocs/obj_models.zip
    unzip obj_models.zip

    # Ground-truth annotations
    wget http://download.cs.stanford.edu/orion/nocs/gts.zip
    unzip gts.zip

Preprocessed layout produced by scripts/preprocess_nocs.py:
    {preprocessed_root}/
        metadata.jsonl
        images/
            {uid}.png    RGBA

NOCS size convention
--------------------
Sizes in the GT pkl are stored as [x_size, y_size, z_size] in NOCS coordinate
space (a unit cube normalised to [-0.5, 0.5]).  The `abs_scale` field (added in
the supplementary zip) gives real-world extents in metres.  We use abs_scale
directly when available; otherwise we fall back to extracting the mesh bounding
box from obj_models/.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

NOCS_CATEGORIES = ["bottle", "bowl", "camera", "can", "laptop", "mug"]

# Colour-to-instance mapping: NOCS masks encode instance id in the R channel
# (values 0–255 where 0 = background).  Category is recovered from meta.txt.


# ---------------------------------------------------------------------------
# Dataset class (loads preprocessed records)
# ---------------------------------------------------------------------------

class NOCSDataset(Dataset):
    """
    Loads preprocessed NOCS REAL275 records produced by scripts/preprocess_nocs.py.

    Each item:
        image       np.ndarray [H, W, 4] uint8 — RGBA (alpha = object mask)
        metric_dims np.ndarray [3] float32      — [width, height, depth] in metres
        category    str
        uid         str
    """

    def __init__(self, preprocessed_root: str, categories: list[str] = None):
        self.root = Path(preprocessed_root)
        meta_path = self.root / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found in {self.root}. "
                "Run scripts/preprocess_nocs.py first."
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
            "image": image,
            "metric_dims": np.array(rec["metric_dims"], dtype=np.float32),
            "category": rec["category"],
            "uid": rec["uid"],
            "source": "nocs",
        }


# ---------------------------------------------------------------------------
# Raw data parsing helpers (used by preprocessing script)
# ---------------------------------------------------------------------------

def iter_nocs_instances(raw_root: str, categories: list[str] = None) -> Iterator[dict]:
    """
    Yield one dict per object instance across all REAL275 scenes.

    Each yielded dict:
        color_path   Path   — RGB image
        mask_path    Path   — colour-coded mask
        meta_path    Path   — meta.txt for this frame
        instance_id  int    — instance id in the mask
        category     str
        scene        str
    """
    root = Path(raw_root)
    cats = set(categories or NOCS_CATEGORIES)

    real_test = root / "real_test"
    if not real_test.exists():
        raise FileNotFoundError(f"real_test/ not found under {root}")

    for scene_dir in sorted(real_test.iterdir()):
        if not scene_dir.is_dir():
            continue
        # Group files by frame prefix (XXXXXX)
        color_files = {p.stem.replace("_color", ""): p
                       for p in scene_dir.glob("*_color.png")}
        for prefix, color_path in sorted(color_files.items()):
            mask_path = scene_dir / f"{prefix}_mask.png"
            meta_path = scene_dir / f"{prefix}_meta.txt"
            if not mask_path.exists() or not meta_path.exists():
                continue

            # Parse meta.txt: "instance_id category_name\n..."
            instance_map = _parse_meta(meta_path)
            for instance_id, category in instance_map.items():
                if category not in cats:
                    continue
                yield {
                    "color_path": color_path,
                    "mask_path": mask_path,
                    "meta_path": meta_path,
                    "instance_id": instance_id,
                    "category": category,
                    "scene": scene_dir.name,
                }


def extract_instance_rgba(color_path: Path, mask_path: Path, instance_id: int
                           ) -> np.ndarray | None:
    """
    Extract a single instance from a NOCS frame as an RGBA image.

    NOCS masks store instance id in the R channel (0 = background).
    Returns [H, W, 4] uint8 or None if instance has no valid pixels.
    """
    rgb = np.array(Image.open(color_path).convert("RGB"), dtype=np.uint8)
    mask_img = np.array(Image.open(mask_path))

    # R channel encodes instance id
    instance_mask = (mask_img[..., 0] == instance_id).astype(np.uint8) * 255
    if instance_mask.sum() == 0:
        return None

    alpha = instance_mask[..., None]
    return np.concatenate([rgb, alpha], axis=-1)


def get_metric_dims_from_gt(
    gt_pkl_path: Path,
    scene: str,
    color_filename: str,
    instance_id: int,
) -> np.ndarray | None:
    """
    Extract metric dimensions [w, h, d] in metres from a NOCS GT pickle.

    GT pkl structure (per scene):
        results_real_test_scene_N.pkl → list of frame dicts, each containing:
            'gt_scales'  np.ndarray [N_instances, 3]  — real-world sizes in metres
            'gt_class_ids' np.ndarray [N_instances]   — class ids
            'image_path'   str

    Returns [w, h, d] float32 array or None.
    """
    if not gt_pkl_path.exists():
        return None

    with open(gt_pkl_path, "rb") as f:
        gt_data = pickle.load(f)

    for frame_gt in gt_data:
        if color_filename not in str(frame_gt.get("image_path", "")):
            continue
        scales = frame_gt.get("gt_scales", None)
        instance_ids = frame_gt.get("gt_instance_ids", None)
        if scales is None:
            continue
        if instance_ids is not None:
            for i, iid in enumerate(instance_ids):
                if iid == instance_id:
                    return scales[i].astype(np.float32)
        else:
            # Fall back: return first instance's scale
            if len(scales) > 0:
                return scales[0].astype(np.float32)

    return None


def get_metric_dims_from_mesh(obj_models_root: Path, category: str, instance_name: str
                               ) -> np.ndarray | None:
    """
    Compute metric dimensions from the object mesh bounding box as a fallback
    when GT pkl does not contain abs_scale.
    """
    try:
        import trimesh
    except ImportError:
        return None

    mesh_path = obj_models_root / f"{category}_{instance_name}" / "model.obj"
    if not mesh_path.exists():
        # Try without instance suffix
        for candidate in obj_models_root.glob(f"{category}_*/model.obj"):
            mesh_path = candidate
            break
        else:
            return None

    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        extents = mesh.bounding_box.extents.astype(np.float32)  # [w, h, d]
        return extents
    except Exception:
        return None


def _parse_meta(meta_path: Path) -> dict[int, str]:
    """Parse meta.txt → {instance_id: category_name}."""
    result = {}
    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    iid = int(parts[0])
                    cat = parts[1].lower().split("_")[0]  # e.g. "mug_1" → "mug"
                    result[iid] = cat
                except ValueError:
                    continue
    return result
