# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Object-instance loader for OmniNOCS release-style metadata.

OmniNOCS provides consistent per-frame metadata across several sources:
NOCS-Real275, Objectron, ARKitScenes, and Hypersim. The release archives contain
metadata plus per-instance masks/NOCS maps. Source RGB frames are dataset-specific
and must be provided separately through an RGB root.

This loader emits one record per object instance:

    image        np.ndarray [H, W, 4] uint8  RGB + selected-object alpha mask
    metric_dims  np.ndarray [3] float32      [width, height, depth] in meters
    category     str
    source       str
    image_name   str
    object_id    int
    mask_pixels  int
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class OmniNOCSSourceSpec:
    source: str
    release_dir: str
    metadata_prefix: str
    mask_prefix: str | None = None


OMNINOCS_SOURCE_SPECS = {
    "nocs_real275": OmniNOCSSourceSpec(
        source="nocs_real275",
        release_dir="omninocs_release_nocs_real275",
        metadata_prefix="nocs_real275",
        mask_prefix=None,
    ),
    "objectron": OmniNOCSSourceSpec(
        source="objectron",
        release_dir="omninocs_release_objectron",
        metadata_prefix="objectron",
        mask_prefix="objectron",
    ),
    "arkitscenes": OmniNOCSSourceSpec(
        source="arkitscenes",
        release_dir="omninocs_release_ARKitScenes",
        metadata_prefix="ARKitScenes",
        mask_prefix=None,
    ),
    "hypersim": OmniNOCSSourceSpec(
        source="hypersim",
        release_dir="omninocs_release_hypersim",
        metadata_prefix="hypersim",
        mask_prefix="hypersim",
    ),
}


class OmniNOCSObjectDataset(Dataset):
    """
    Load one or more OmniNOCS sources as object-instance records.

    Args:
        omninocs_root: root containing ``omninocs_release_*`` directories.
        sources: source names, e.g. ["nocs_real275", "objectron"].
        rgb_roots: mapping from source name to source RGB root. NOCS uses the
            NOCS ``real_test`` directory; other sources require their own RGB
            extraction/download.
        split: "train", "val", or "test".
        skip_missing_rgb: skip records whose RGB frame cannot be found. This is
            useful when only annotations have been downloaded for some sources.
    """

    def __init__(
        self,
        omninocs_root: str = "/mnt/dest/OmniNOCS",
        sources: list[str] | None = None,
        rgb_roots: dict[str, str | None] | None = None,
        split: str = "train",
        categories: list[str] | None = None,
        min_mask_pixels: int = 500,
        max_records: int | None = None,
        max_records_per_source: int | None = None,
        skip_missing_rgb: bool = True,
    ):
        self.omninocs_root = Path(omninocs_root)
        self.sources = sources or ["nocs_real275"]
        self.rgb_roots = {
            source: Path(root) if root else None
            for source, root in (rgb_roots or {}).items()
        }
        self.split = split
        self.min_mask_pixels = min_mask_pixels
        self.skip_missing_rgb = skip_missing_rgb
        self.skipped_missing_rgb = 0
        self.skipped_missing_mask = 0
        cats = set(categories) if categories is not None else None

        records = []
        for source in self.sources:
            spec = self._source_spec(source)
            release_root = self.omninocs_root / spec.release_dir
            meta_path = release_root / f"{spec.metadata_prefix}_{split}_metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"OmniNOCS metadata not found: {meta_path}")
            with open(meta_path) as f:
                frames = json.load(f)

            source_count = 0
            for frame in frames:
                if (
                    max_records_per_source is not None
                    and source_count >= max_records_per_source
                ):
                    break
                image_name = frame["image_name"]
                rgb_path = self._rgb_path(spec, image_name)
                if rgb_path is None or not rgb_path.exists():
                    self.skipped_missing_rgb += len(frame["objects"])
                    if self.skip_missing_rgb:
                        continue
                    raise FileNotFoundError(
                        f"RGB frame not found for source={source} image_name={image_name!r}. "
                        f"Set the appropriate --{source}-rgb-root or use --skip-missing-rgb."
                    )
                inst_path = self._instance_path(release_root, spec, image_name)
                if not inst_path.exists():
                    self.skipped_missing_mask += len(frame["objects"])
                    if self.skip_missing_rgb:
                        continue
                    raise FileNotFoundError(f"OmniNOCS instance mask not found: {inst_path}")

                for obj in frame["objects"]:
                    if (
                        max_records_per_source is not None
                        and source_count >= max_records_per_source
                    ):
                        break
                    if cats is not None and obj["category"] not in cats:
                        continue
                    records.append(
                        {
                            "source": source,
                            "release_root": str(release_root),
                            "rgb_path": str(rgb_path),
                            "inst_path": str(inst_path),
                            "image_name": image_name,
                            "object_id": int(obj["object_id"]),
                            "category": obj["category"],
                            "metric_dims": obj["size"],
                            "uid": (
                                f"{source}_{image_name.replace('/', '_')}_"
                                f"{int(obj['object_id'])}"
                            ),
                        }
                    )
                    source_count += 1
                    if max_records is not None and len(records) >= max_records:
                        break
                if max_records is not None and len(records) >= max_records:
                    break

        self.records = records
        if not self.records:
            skipped = (
                f"skipped_missing_rgb={self.skipped_missing_rgb}, "
                f"skipped_missing_mask={self.skipped_missing_mask}"
            )
            raise RuntimeError(f"No OmniNOCS records loaded ({skipped}).")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        instances = np.array(Image.open(rec["inst_path"]))
        rgb_image = Image.open(rec["rgb_path"]).convert("RGB")
        if rgb_image.size != (instances.shape[1], instances.shape[0]):
            rgb_image = rgb_image.resize(
                (instances.shape[1], instances.shape[0]), Image.Resampling.BILINEAR
            )
        rgb = np.array(rgb_image, dtype=np.uint8)
        mask = instances == rec["object_id"]
        mask_pixels = int(mask.sum())
        rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
        return {
            "image": rgba,
            "metric_dims": np.array(rec["metric_dims"], dtype=np.float32),
            "category": rec["category"],
            "uid": rec["uid"],
            "source": rec["source"],
            "image_name": rec["image_name"],
            "object_id": rec["object_id"],
            "mask_pixels": mask_pixels,
        }

    @staticmethod
    def _source_spec(source: str) -> OmniNOCSSourceSpec:
        try:
            return OMNINOCS_SOURCE_SPECS[source]
        except KeyError as exc:
            valid = ", ".join(sorted(OMNINOCS_SOURCE_SPECS))
            raise ValueError(f"Unsupported OmniNOCS source {source!r}. Valid: {valid}") from exc

    def _instance_path(
        self,
        release_root: Path,
        spec: OmniNOCSSourceSpec,
        image_name: str,
    ) -> Path:
        relative = Path(f"{image_name}_instances.png")
        if spec.mask_prefix is not None:
            relative = Path(spec.mask_prefix) / relative
        return release_root / relative

    def _rgb_path(self, spec: OmniNOCSSourceSpec, image_name: str) -> Path | None:
        rgb_root = self.rgb_roots.get(spec.source)
        if rgb_root is None:
            return None
        if spec.source == "nocs_real275":
            parts = Path(image_name).parts
            scene = parts[-2]
            frame = parts[-1]
            return rgb_root / scene / f"{frame}_color.png"

        image_rel = Path(image_name)
        candidates = []
        if spec.source == "objectron":
            # Omni3D flattens Objectron paths like
            # book/batch-11/29/frame000190 -> datasets/objectron/train/book_batch_11_29_0000190.jpg
            parts = image_rel.parts
            if len(parts) >= 4:
                category = parts[-4]
                batch = parts[-3].replace("batch-", "")
                sequence = parts[-2]
                frame = parts[-1].replace("frame", "").zfill(7)
                flat = f"{category}_batch_{batch}_{sequence}_{frame}.jpg"
                candidates.extend(
                    [
                        rgb_root / "datasets" / "objectron" / self.split / flat,
                        rgb_root / "datasets" / "objectron" / "train" / flat,
                        rgb_root / "datasets" / "objectron" / "test" / flat,
                        rgb_root / "objectron" / self.split / flat,
                        rgb_root / "objectron" / "train" / flat,
                        rgb_root / "objectron" / "test" / flat,
                    ]
                )
        elif spec.source == "arkitscenes":
            candidates.extend(
                [
                    rgb_root / "datasets" / f"{image_name}.jpg",
                    rgb_root / "datasets" / f"{image_name}.png",
                ]
            )
        elif spec.source == "hypersim":
            if len(image_rel.parts) >= 3:
                scene, camera, frame = image_rel.parts[:3]
                frame_idx = frame.replace("frame_", "").replace("frame.", "")
                candidates.append(
                    rgb_root
                    / scene
                    / "images"
                    / f"scene_{camera}_final_preview"
                    / f"frame.{frame_idx}.tonemap.jpg"
                )
            candidates.extend(
                [
                    rgb_root / image_rel.with_suffix(".tonemap.jpg"),
                    rgb_root / image_rel.with_suffix(".jpg"),
                ]
            )

        candidates.extend(
            [
            rgb_root / image_rel,
            rgb_root / image_rel.with_suffix(".png"),
            rgb_root / image_rel.with_suffix(".jpg"),
            rgb_root / image_rel.with_suffix(".jpeg"),
            rgb_root / f"{image_name}_color.png",
            rgb_root / f"{image_name}.png",
            rgb_root / f"{image_name}.jpg",
            rgb_root / f"{image_name}.jpeg",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
