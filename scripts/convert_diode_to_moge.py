"""Convert DIODE val (indoor only) into MoGe-format eval samples.

DIODE doesn't ship instance segmentation, so we use SAM2-base_plus to auto-mask
objects per image (same recipe as iBims-1). Without semantic priors to flag
floor/wall, we drop very large masks (likely planar) and very small ones
(noise), then keep the top-K by area.

Camera intrinsics are the canonical DIODE "computational pinhole" parameters
from the devkit (intrinsics.txt in diode-dataset/diode-devkit):
    fx = 886.81, fy = 927.06, cx = 512, cy = 384   (in 1024 x 768 pixels)

Input
  Sample triplet per line in --filename-list:
      <rgb.png>  <depth.npy>  <depth_mask.npy>
  (relative to --input-root, indoors/scene_XXXXX/scan_XXXXX/...)

Output (per (image, mask) pair):
    <output_dir>/.index.txt
    <output_dir>/<sample_id>_obj<idx>/
        image.jpg
        depth.png           # MoGe log-encoded
        meta.json           # normalized intrinsics
        segmentation.png    # single-label mask {"object": 1}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from moge.utils.io import write_image, write_depth, write_segmentation
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


SAM2_ROOT = Path("/mnt/source/Software/sam2")
MODEL_CONFIGS = {
    "tiny":      ("configs/sam2.1/sam2.1_hiera_t.yaml",  SAM2_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"),
    "small":     ("configs/sam2.1/sam2.1_hiera_s.yaml",  SAM2_ROOT / "checkpoints/sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", SAM2_ROOT / "checkpoints/sam2.1_hiera_base_plus.pt"),
    "large":     ("configs/sam2.1/sam2.1_hiera_l.yaml",  SAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"),
}

# From diode-dataset/diode-devkit intrinsics.txt
DIODE_FX, DIODE_FY, DIODE_CX, DIODE_CY = 886.81, 927.06, 512.0, 384.0
DIODE_W, DIODE_H = 1024, 768

# Marigold/DIODE-paper-recommended depth gates
DIODE_MIN_DEPTH = 0.6     # metres
DIODE_MAX_DEPTH = 350.0


def diode_intrinsics_normalized() -> np.ndarray:
    return np.array([
        [DIODE_FX / DIODE_W, 0.0,                DIODE_CX / DIODE_W],
        [0.0,                DIODE_FY / DIODE_H, DIODE_CY / DIODE_H],
        [0.0,                0.0,                1.0               ],
    ], dtype=np.float64)


def load_diode_sample(root: Path, rgb_rel: str, depth_rel: str, mask_rel: str):
    rgb = np.array(Image.open(root / rgb_rel).convert("RGB"))           # (H, W, 3) uint8
    depth = np.load(root / depth_rel).astype(np.float32).squeeze()       # (H, W) float
    mask = np.load(root / mask_rel).astype(np.float32).squeeze()         # (H, W) {0, 1}
    valid = (mask > 0) & np.isfinite(depth) & (depth >= DIODE_MIN_DEPTH) & (depth <= DIODE_MAX_DEPTH)
    invalid = ~valid
    return rgb, depth, invalid


def filter_masks(masks, image_area: int, min_area: int, max_area_fraction: float,
                 max_keep: int):
    """Drop too-small (noise) and too-large (likely planar) masks, then keep
    the top-K remaining by area."""
    max_area = int(image_area * max_area_fraction)
    kept = []
    for m in masks:
        seg = m["segmentation"]
        area = int(seg.sum())
        if area < min_area or area > max_area:
            continue
        kept.append((area, m))
    kept.sort(key=lambda x: -x[0])
    return [m for _, m in kept[:max_keep]]


def emit_sample(out_dir: Path, rgb: np.ndarray, depth_m: np.ndarray,
                invalid_mask: np.ndarray, object_mask: np.ndarray,
                intrinsics_norm: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(out_dir / "image.jpg", rgb)

    depth_with_nan = depth_m.copy()
    depth_with_nan[invalid_mask] = np.nan
    write_depth(out_dir / "depth.png", depth_with_nan)

    seg = np.zeros(object_mask.shape, dtype=np.uint8)
    seg[object_mask] = 1
    write_segmentation(out_dir / "segmentation.png", seg, labels={"object": 1})

    (out_dir / "meta.json").write_text(json.dumps({
        "intrinsics": intrinsics_norm.tolist(),
    }, indent=2))


def sample_id_from_rgb_path(rgb_rel: str) -> str:
    """e.g. 'indoors/scene_00021/scan_00189/00021_00189_indoors_200_010.png'
       → '00021_00189_indoors_200_010'"""
    return Path(rgb_rel).stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True,
                   help="DIODE val root (contains indoors/ and outdoor/).")
    p.add_argument("--filename-list", type=Path, required=True,
                   help="Each line: <rgb.png>  <depth.npy>  <depth_mask.npy> "
                        "(paths relative to --input-root). Indoor-only file lives at "
                        "/mnt/source/Software/Marigold/data_split/diode/diode_val_indoor_filename_list.txt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="base_plus")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-mask-area", type=int, default=2000)
    p.add_argument("--max-mask-area-fraction", type=float, default=0.30,
                   help="Drop masks larger than this fraction of the image (likely "
                        "floor/wall, since DIODE has no planar-label hints).")
    p.add_argument("--max-objects-per-image", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N samples (smoke test).")
    args = p.parse_args()

    cfg, ckpt = MODEL_CONFIGS[args.model]
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {ckpt}")
    print(f"[sam2] loading {args.model} on {args.device}")
    sam = build_sam2(cfg, str(ckpt), device=args.device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(sam)

    entries = [line.strip().split() for line in args.filename_list.read_text().splitlines()
               if line.strip()]
    if args.limit is not None:
        entries = entries[: args.limit]
    print(f"[diode] {len(entries)} indoor samples")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    K_norm = diode_intrinsics_normalized()
    image_area = DIODE_W * DIODE_H

    index_entries: list[str] = []
    total_kept = 0
    total_dropped = 0

    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16,
                                                 enabled=args.device == "cuda"):
        for line in entries:
            if len(line) != 3:
                print(f"  [skip] malformed line: {line}")
                continue
            rgb_rel, depth_rel, mask_rel = line

            try:
                rgb, depth, invalid = load_diode_sample(args.input_root, rgb_rel, depth_rel, mask_rel)
            except Exception as e:
                print(f"  [skip] {rgb_rel}: {e}")
                continue

            H, W = rgb.shape[:2]
            assert (H, W) == (DIODE_H, DIODE_W), f"unexpected DIODE resolution: {(H, W)}"

            raw_masks = generator.generate(rgb)
            kept = filter_masks(raw_masks,
                                image_area=image_area,
                                min_area=args.min_mask_area,
                                max_area_fraction=args.max_mask_area_fraction,
                                max_keep=args.max_objects_per_image)
            total_kept += len(kept)
            total_dropped += len(raw_masks) - len(kept)

            sid = sample_id_from_rgb_path(rgb_rel)
            print(f"  {sid}: {len(raw_masks)} raw → {len(kept)} kept")

            for obj_idx, mask in enumerate(kept):
                sample_id = f"{sid}_obj{obj_idx:02d}"
                emit_sample(
                    out_dir=args.output_dir / sample_id,
                    rgb=rgb,
                    depth_m=depth,
                    invalid_mask=invalid,
                    object_mask=mask["segmentation"],
                    intrinsics_norm=K_norm,
                )
                index_entries.append(sample_id)

    (args.output_dir / ".index.txt").write_text("\n".join(index_entries) + "\n")
    print(f"[done] {len(entries)} images → {total_kept} samples "
          f"(dropped {total_dropped} masks failing size filters)")
    print(f"       index written to {args.output_dir / '.index.txt'}")


if __name__ == "__main__":
    main()
