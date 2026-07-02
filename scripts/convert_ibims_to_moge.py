"""Convert raw iBims-1 .mat distribution into MoGe-format evaluation samples.

Each input .mat (one per scene) is exploded into one MoGe sample directory per
SAM2-generated object mask:

    <output_dir>/<scene>_obj<idx>/
        image.jpg
        depth.png         (MoGe log-encoded 16-bit PNG with NaN at invalid pixels)
        meta.json         {"intrinsics": [[fx/W, 0, cx/W], ...]}
        segmentation.png  (single-label mask, label "object" = 1)

An .index.txt listing every sample dir is written at <output_dir>/.index.txt so
the result is directly consumable by moge.test.dataloader.EvalDataLoaderPipeline.

Object masks are produced by SAM2AutomaticMaskGenerator with the following
filtering policy:
  - drop masks whose pixel-area is below --min-mask-area
  - drop masks whose overlap with iBims's mask_floor or mask_wall exceeds
    --planar-overlap-threshold (these are background planar surfaces)
  - keep the top --max-masks-per-scene remaining masks by area
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

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


def load_ibims_mat(mat_path: Path):
    raw = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    d = raw["data"]
    rgb = d.rgb                                         # (H, W, 3) uint8
    depth = d.depth.astype(np.float32)                  # (H, W) float, metres
    # iBims stores calib in column-major (Matlab) layout, so what loads as
    # np.asarray(...) is K.T relative to standard row-major intrinsics convention.
    calib = np.asarray(d.calib, dtype=np.float64).T     # (3, 3) pixel-space intrinsics
    # iBims convention: mask_invalid == 1 means valid (despite the name),
    # mask_transp == 1 means opaque/valid. A pixel is usable only if both are 1
    # and the depth is finite and positive.
    valid = (d.mask_invalid > 0) & (d.mask_transp > 0) & np.isfinite(depth) & (depth > 0)
    invalid = ~valid
    floor = d.mask_floor > 0
    wall = d.mask_wall > 0
    return {
        "scene": str(d.image_name),
        "rgb": rgb,
        "depth": depth,
        "calib": calib,
        "invalid": invalid,
        "planar": floor | wall,
    }


def normalize_intrinsics(K_pixel: np.ndarray, width: int, height: int) -> np.ndarray:
    K = K_pixel.copy().astype(np.float64)
    K[0, :] /= width
    K[1, :] /= height
    return K


def filter_masks(masks, planar_mask: np.ndarray, min_area: int, planar_overlap: float, max_keep: int):
    kept = []
    for m in masks:
        seg = m["segmentation"]
        area = int(seg.sum())
        if area < min_area:
            continue
        overlap = float((seg & planar_mask).sum()) / area
        if overlap > planar_overlap:
            continue
        kept.append((area, m))
    kept.sort(key=lambda x: -x[0])
    return [m for _, m in kept[:max_keep]]


def emit_sample(out_dir: Path, rgb: np.ndarray, depth: np.ndarray,
                invalid_mask: np.ndarray, object_mask: np.ndarray,
                intrinsics_norm: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_image(out_dir / "image.jpg", rgb)

    depth_with_nan = depth.copy()
    depth_with_nan[invalid_mask] = np.nan
    write_depth(out_dir / "depth.png", depth_with_nan)

    seg = np.zeros(object_mask.shape, dtype=np.uint8)
    seg[object_mask] = 1
    write_segmentation(out_dir / "segmentation.png", seg, labels={"object": 1})

    (out_dir / "meta.json").write_text(json.dumps({
        "intrinsics": intrinsics_norm.tolist(),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing iBims .mat files (one per scene).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory in MoGe sample format.")
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="base_plus",
                   help="SAM2 model size. 'large' is best quality, 'base_plus' fits "
                        "alongside ~16GB-resident training on a 24GB A10.")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    p.add_argument("--min-mask-area", type=int, default=2000,
                   help="Drop SAM masks smaller than this many pixels.")
    p.add_argument("--planar-overlap-threshold", type=float, default=0.5,
                   help="Drop SAM masks whose intersection with floor/wall masks "
                        "covers more than this fraction of the mask.")
    p.add_argument("--max-masks-per-scene", type=int, default=8,
                   help="Keep at most this many object masks per scene "
                        "(largest by area first).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N .mat files (for smoke tests).")
    args = p.parse_args()

    cfg, ckpt = MODEL_CONFIGS[args.model]
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {ckpt}")
    print(f"[sam2] loading {args.model} on {args.device}")
    sam = build_sam2(cfg, str(ckpt), device=args.device, apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(sam)

    mat_files = sorted(args.input_dir.glob("*.mat"))
    if args.limit is not None:
        mat_files = mat_files[: args.limit]
    print(f"[ibims] {len(mat_files)} .mat files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_entries = []
    total_kept = 0
    total_dropped = 0

    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        for mat_path in mat_files:
            sample = load_ibims_mat(mat_path)
            scene = sample["scene"]
            rgb = sample["rgb"]
            H, W = rgb.shape[:2]

            raw_masks = generator.generate(rgb)
            kept = filter_masks(raw_masks,
                                planar_mask=sample["planar"],
                                min_area=args.min_mask_area,
                                planar_overlap=args.planar_overlap_threshold,
                                max_keep=args.max_masks_per_scene)
            total_kept += len(kept)
            total_dropped += len(raw_masks) - len(kept)
            print(f"  {scene}: {len(raw_masks)} raw → {len(kept)} kept")

            K_norm = normalize_intrinsics(sample["calib"], W, H)

            for obj_idx, mask in enumerate(kept):
                sample_id = f"{scene}_obj{obj_idx:02d}"
                emit_sample(
                    out_dir=args.output_dir / sample_id,
                    rgb=rgb,
                    depth=sample["depth"],
                    invalid_mask=sample["invalid"],
                    object_mask=mask["segmentation"],
                    intrinsics_norm=K_norm,
                )
                index_entries.append(sample_id)

    (args.output_dir / ".index.txt").write_text("\n".join(index_entries) + "\n")
    print(f"[done] {len(mat_files)} scenes → {total_kept} samples "
          f"(dropped {total_dropped} non-object masks)")
    print(f"       index written to {args.output_dir / '.index.txt'}")


if __name__ == "__main__":
    main()
