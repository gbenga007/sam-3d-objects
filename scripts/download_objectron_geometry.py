#!/usr/bin/env python
"""
Download Objectron geometry.pbdata (AR session metadata incl. per-frame SPARSE
metric point clouds) for the videos used by the mixed-10k train/heldout set.

Why: Objectron has no dense depth, but the ARKit sparse point cloud is metric.
Projected into a frame it forms a sparse metric pointmap (NaN elsewhere), and the
pipeline's anchor statistic pointmap_scale = nanmean(|points - median_z|) is
NaN-tolerant — candidate GT-ish anchor target for the iso-scale loss on the
source where MoGe-2's anchor bias is worst (1.76x).

Video list: artifacts/metric_scale/metrics/objectron_videos_needed.json
(2,273 videos, ~8 MB each, ~18 GB total). Resumable: skips existing files.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEEDED = REPO / "artifacts/metric_scale/metrics/objectron_videos_needed.json"
OUT = Path("/mnt/source/datasets_sam3d/Objectron_geometry")
BASE = "https://storage.googleapis.com/objectron/videos"


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(".tmp")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} {url}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return False


def main():
    videos = json.load(open(NEEDED))
    print(f"{len(videos)} videos needed", flush=True)
    done = failed = skipped = 0
    t0 = time.time()
    for i, vid in enumerate(videos):
        dest = OUT / vid / "geometry.pbdata"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if fetch(f"{BASE}/{vid}/geometry.pbdata", dest):
            done += 1
        else:
            failed += 1
            print(f"  FAILED: {vid}", flush=True)
        if (i + 1) % 50 == 0:
            rate = (done + skipped) / max(time.time() - t0, 1e-6)
            print(f"[{i + 1}/{len(videos)}] done={done} skip={skipped} fail={failed} "
                  f"({rate:.1f} vid/s)", flush=True)
    print(f"DONE: downloaded={done} skipped={skipped} failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
