"""Convert HAMMER (`_dataset_processed.zip`) into MoGe-format eval samples.

HAMMER provides aligned RGB + GT depth (uint16 mm) + per-pixel instance masks
in the `l515_rgb/` view of each scene, so no SAM is needed — each instance
label becomes one MoGe sample.

Why stream from the zip:
  The processed archive is 184 GB uncompressed (434k files) and won't fit
  alongside iBims-1 / DIODE on the persistent volume. We read individual PNGs
  out of the zip via `zipfile.ZipFile.open()` and emit only the per-object
  samples we need.

Output layout (one dir per (scene, frame, instance_label)):
    <output_dir>/.index.txt
    <output_dir>/<scene>_f<frame>_obj<label>/
        image.jpg
        depth.png           # MoGe log-encoded
        meta.json           # normalized intrinsics
        segmentation.png    # single-label mask {"object": 1}
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from moge.utils.io import write_image, write_depth, write_segmentation


HAMMER_RGB_VIEW = "l515_rgb"   # the only view with aligned RGB + GT + instance


def read_intrinsics_txt(zf: zipfile.ZipFile, scene: str) -> np.ndarray:
    """Read pixel-space 3x3 K matrix from <scene>/<view>/intrinsics.txt."""
    name = f"{scene}/{HAMMER_RGB_VIEW}/intrinsics.txt"
    raw = zf.read(name).decode("utf-8")
    rows = [list(map(float, line.split())) for line in raw.strip().splitlines() if line.strip()]
    K = np.asarray(rows, dtype=np.float64)
    assert K.shape == (3, 3), f"unexpected intrinsics shape: {K.shape}"
    return K


def read_png(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.array(Image.open(io.BytesIO(f.read())))


def list_scenes(zf: zipfile.ZipFile) -> list[str]:
    """Scenes that have the `l515_rgb` view (only ~half of HAMMER does — the
    rest only ship d435 IR stereo, which we can't use for image-to-3D)."""
    scenes_with_rgb = set()
    suffix = f"/{HAMMER_RGB_VIEW}/intrinsics.txt"
    for name in zf.namelist():
        if name.endswith(suffix):
            scenes_with_rgb.add(name[: -len(suffix)])
    return sorted(scenes_with_rgb)


def list_frames(zf: zipfile.ZipFile, scene: str) -> list[int]:
    """Frame indices present in <scene>/<view>/rgb/."""
    prefix = f"{scene}/{HAMMER_RGB_VIEW}/rgb/"
    out = []
    for name in zf.namelist():
        if name.startswith(prefix) and name.endswith(".png"):
            stem = Path(name).stem  # "000000"
            try:
                out.append(int(stem))
            except ValueError:
                continue
    return sorted(out)


def normalize_intrinsics(K_pixel: np.ndarray, width: int, height: int) -> np.ndarray:
    K = K_pixel.copy().astype(np.float64)
    K[0, :] /= width
    K[1, :] /= height
    return K


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", "--input-zip", dest="zip_path", type=Path, required=True,
                   help="Path to _dataset_processed.zip")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory in MoGe sample format.")
    p.add_argument("--frame-stride", type=int, default=100,
                   help="Take every N-th frame from each scene (default 100, ~3 frames/scene).")
    p.add_argument("--min-mask-area", type=int, default=2000,
                   help="Drop instance masks smaller than this many pixels.")
    p.add_argument("--max-objects-per-frame", type=int, default=8,
                   help="Cap on the number of objects per frame (largest first).")
    p.add_argument("--limit-scenes", type=int, default=None,
                   help="Process only the first N scenes (for smoke tests).")
    p.add_argument("--depth-unit-mm", type=float, default=1.0,
                   help="HAMMER stores depth as uint16 millimetres; divide by 1000 "
                        "to get metres. Adjust if a future format uses a different unit.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.zip_path, "r")
    try:
        scenes = list_scenes(zf)
        if args.limit_scenes is not None:
            scenes = scenes[: args.limit_scenes]
        print(f"[hammer] {len(scenes)} scenes")

        index_entries: list[str] = []
        total_kept = 0
        total_dropped = 0

        for scene in scenes:
            K_pixel = read_intrinsics_txt(zf, scene)
            frames = list_frames(zf, scene)
            frames = frames[:: args.frame_stride]
            print(f"  {scene}: {len(frames)} frames at stride {args.frame_stride}")

            for f_idx in frames:
                rgb_name = f"{scene}/{HAMMER_RGB_VIEW}/rgb/{f_idx:06d}.png"
                depth_name = f"{scene}/{HAMMER_RGB_VIEW}/_gt/{f_idx:06d}.png"
                inst_name = f"{scene}/{HAMMER_RGB_VIEW}/_instance/{f_idx:06d}.png"

                try:
                    rgb = read_png(zf, rgb_name)
                    depth_u16 = read_png(zf, depth_name)
                    instance = read_png(zf, inst_name)
                except KeyError as e:
                    print(f"    [skip] missing file: {e}")
                    continue

                H, W = rgb.shape[:2]
                K_norm = normalize_intrinsics(K_pixel, W, H)

                # HAMMER depth: uint16 millimetres, 0 = invalid.
                invalid = (depth_u16 == 0)
                depth_m = depth_u16.astype(np.float32) * (args.depth_unit_mm / 1000.0)

                # Per-instance breakdown
                labels = [int(v) for v in np.unique(instance) if v != 0]
                kept_for_frame = []
                for lab in labels:
                    mask = (instance == lab)
                    area = int(mask.sum())
                    if area < args.min_mask_area:
                        total_dropped += 1
                        continue
                    kept_for_frame.append((area, lab, mask))
                kept_for_frame.sort(key=lambda x: -x[0])
                kept_for_frame = kept_for_frame[: args.max_objects_per_frame]

                for _, lab, mask in kept_for_frame:
                    sample_id = f"{scene}_f{f_idx:06d}_obj{lab:03d}"
                    emit_sample(
                        out_dir=args.output_dir / sample_id,
                        rgb=rgb,
                        depth_m=depth_m,
                        invalid_mask=invalid,
                        object_mask=mask,
                        intrinsics_norm=K_norm,
                    )
                    index_entries.append(sample_id)
                    total_kept += 1
    finally:
        zf.close()

    (args.output_dir / ".index.txt").write_text("\n".join(index_entries) + "\n")
    print(f"[done] {len(scenes)} scenes → {total_kept} samples "
          f"(dropped {total_dropped} below-min-area instances)")
    print(f"       index written to {args.output_dir / '.index.txt'}")


if __name__ == "__main__":
    main()
