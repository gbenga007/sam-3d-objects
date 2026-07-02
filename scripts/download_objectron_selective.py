"""
Selective Objectron RGB frame downloader.

Streams Objectron MOV videos from gs://objectron via public HTTPS, extracts only
the frames referenced in OmniNOCS metadata, and writes JPEGs to the local
OmniNOCS objectron/{train,test}/ directories using the flat naming convention
that OmniNOCSObjectDataset's path resolver expects:

    {class}_batch_{n}_{seq}_{frame:07d}.jpg

Streaming avoids the ~172 GB of full-video downloads — only ~40 GB of JPEG
output remains on disk. Resumable: skips frames whose output file already exists.

Usage:
    LIDRA_SKIP_INIT=1 python scripts/download_objectron_selective.py \
        --workers 8 --quality 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av


def parse_metadata(meta_path: Path) -> dict[tuple, set[int]]:
    """Return {(class, batch, seq): {frame_indices}} from OmniNOCS metadata."""
    with meta_path.open() as f:
        records = json.load(f)
    seqs: dict[tuple, set[int]] = defaultdict(set)
    for rec in records:
        parts = rec["image_name"].split("/")
        if len(parts) < 4:
            continue
        cls = parts[0]
        batch = parts[1].replace("batch-", "")
        seq = parts[2]
        frame_idx = int(parts[3].replace("frame", ""))
        seqs[(cls, batch, seq)].add(frame_idx)
    return seqs


def extract_one_video(
    cls: str,
    batch: str,
    seq: str,
    frame_indices: set[int],
    out_dir: Path,
    quality: int,
) -> tuple[str, str, str, int, int, str]:
    """Stream-extract specific frames from one Objectron video.

    Returns (cls, batch, seq, n_saved, n_skipped, status).
    """
    out_paths = {
        idx: out_dir / f"{cls}_batch_{batch}_{seq}_{idx:07d}.jpg"
        for idx in frame_indices
    }
    needed = {idx: p for idx, p in out_paths.items() if not p.exists()}
    if not needed:
        return (cls, batch, seq, 0, len(out_paths), "all_exist")

    url = (
        f"https://storage.googleapis.com/objectron/videos/"
        f"{cls}/batch-{batch}/{seq}/video.MOV"
    )
    max_idx = max(needed)
    saved = 0
    try:
        container = av.open(url, timeout=60)
        try:
            stream = container.streams.video[0]
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx > max_idx:
                    break
                if frame_idx in needed:
                    img = frame.to_image()
                    img.save(needed[frame_idx], quality=quality)
                    saved += 1
        finally:
            container.close()
        return (cls, batch, seq, saved, len(out_paths) - len(needed), "ok")
    except Exception as exc:  # broad: network, codec, anything
        return (cls, batch, seq, saved, len(out_paths) - len(needed), f"error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=Path("/mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_objectron"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/source/datasets_sam3d/OmniNOCS/objectron"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test", "val"],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print a progress line every N completed videos.",
    )
    args = parser.parse_args()

    for split in args.splits:
        meta = args.meta_root / f"objectron_{split}_metadata.json"
        if not meta.exists():
            print(f"[{split}] metadata not found at {meta}, skipping", flush=True)
            continue

        seqs = parse_metadata(meta)
        total_frames = sum(len(v) for v in seqs.values())
        out_dir = args.out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{split}] {len(seqs)} videos, {total_frames} frames target. "
            f"Output: {out_dir}",
            flush=True,
        )

        start = time.time()
        completed = 0
        ok_videos = 0
        err_videos = 0
        total_saved = 0
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    extract_one_video, cls, batch, seq, frames, out_dir, args.quality
                )
                for (cls, batch, seq), frames in seqs.items()
            ]
            for fut in as_completed(futures):
                cls, batch, seq, saved, skipped, status = fut.result()
                completed += 1
                total_saved += saved
                if status == "ok" or status == "all_exist":
                    ok_videos += 1
                else:
                    err_videos += 1
                    errors.append(f"{cls}/batch-{batch}/{seq}: {status}")
                if completed % args.log_every == 0 or completed == len(futures):
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1e-3)
                    eta = (len(futures) - completed) / max(rate, 1e-3)
                    print(
                        f"[{split}] {completed}/{len(futures)} videos "
                        f"({rate:.2f}/s, ETA {eta/60:.1f}min). "
                        f"saved={total_saved} ok={ok_videos} err={err_videos}",
                        flush=True,
                    )

        print(
            f"[{split}] DONE. ok={ok_videos} err={err_videos} "
            f"frames_saved={total_saved}",
            flush=True,
        )
        if errors[:5]:
            print(f"[{split}] First errors:", flush=True)
            for line in errors[:5]:
                print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
