#!/usr/bin/env python
"""
Selectively fetch ARKitScenes LiDAR depth (lowres_depth, 256x192 uint16 mm) +
lowres_wide intrinsics for ONLY the frames in the mixed-10k train/heldout set.

Per video: download lowres_depth.zip (~50-400 MB) + lowres_wide_intrinsics.zip
(~3 MB) to local /tmp, extract only the members whose timestamp is nearest to a
needed frame's timestamp (tolerance 0.05 s), write a .done.json manifest, delete
the zips. Disk peak = one zip; total extracted output is tiny (904 frames).

Needed list: artifacts/metric_scale/metrics/arkitscenes_frames_needed.json
  {"Training/41048190": ["3442.489_00001459", ...], ...}   (586 videos)
Frame name convention: "<timestamp>_<framenum>"; depth member names are
"lowres_depth/<video_id>_<timestamp>.png". Sanity check aborts after 3 videos
with zero matches so a wrong convention can't waste the ~200 GB transfer.

Resumable: videos with .done.json are skipped.
"""
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEEDED = REPO / "artifacts/metric_scale/metrics/arkitscenes_frames_needed.json"
OUT = Path("/mnt/source/datasets_sam3d/ARKitScenes_depth")
BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
TMP = Path("/tmp/arkit_zips")
TOL = 0.05  # seconds


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return True
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} {url.rsplit('/', 1)[-1]}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return False


def member_ts(name: str):
    # ".../<video_id>_<timestamp>.png" or ".pincam"
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    try:
        return float(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def extract_nearest(zip_path: Path, needed_ts: list[float], out_dir: Path) -> dict:
    """Extract the member nearest to each needed timestamp. Returns ts -> member."""
    matches = {}
    with zipfile.ZipFile(zip_path) as zf:
        members = [(member_ts(n), n) for n in zf.namelist() if not n.endswith("/")]
        members = [(t, n) for t, n in members if t is not None]
        if not members:
            return matches
        for ts in needed_ts:
            best_t, best_n = min(members, key=lambda m: abs(m[0] - ts))
            if abs(best_t - ts) <= TOL:
                target = out_dir / Path(best_n).name
                if not target.exists():
                    with zf.open(best_n) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                matches[str(ts)] = Path(best_n).name
    return matches


def main():
    needed = json.load(open(NEEDED))
    TMP.mkdir(parents=True, exist_ok=True)
    print(f"{len(needed)} videos, {sum(len(v) for v in needed.values())} frames needed", flush=True)
    done = skipped = failed = 0
    no_match_streak = 0
    total_matched = total_frames = 0
    t0 = time.time()
    for i, (key, frames) in enumerate(sorted(needed.items())):
        split, vid = key.split("/")
        out_dir = OUT / split / vid
        marker = out_dir / ".done.json"
        if marker.exists():
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        needed_ts = sorted({float(f.split("_", 1)[0]) for f in frames})

        ok = True
        manifest = {"frames_requested": len(needed_ts)}
        for asset in ("lowres_depth", "lowres_wide_intrinsics"):
            zp = TMP / f"{vid}_{asset}.zip"
            if not fetch(f"{BASE}/raw/{split}/{vid}/{asset}.zip", zp):
                ok = False
                break
            matches = extract_nearest(zp, needed_ts, out_dir)
            manifest[asset] = matches
            zp.unlink(missing_ok=True)
        if not ok:
            failed += 1
            print(f"  FAILED download: {key}", flush=True)
            continue

        n_match = len(manifest.get("lowres_depth", {}))
        total_matched += n_match
        total_frames += len(needed_ts)
        if n_match == 0:
            no_match_streak += 1
            print(f"  WARNING zero depth matches for {key} "
                  f"(needed ts e.g. {needed_ts[:2]})", flush=True)
            if no_match_streak >= 3 and done == 0:
                print("ABORT: first 3 videos had zero matches — frame-name "
                      "convention is wrong, fix member_ts()/needed list.", flush=True)
                sys.exit(2)
        else:
            no_match_streak = 0
        json.dump(manifest, open(marker, "w"))
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[{i + 1}/{len(needed)}] done={done} skip={skipped} fail={failed} "
                  f"matched={total_matched}/{total_frames} ({el / 60:.0f} min)", flush=True)
    print(f"DONE: videos done={done} skipped={skipped} failed={failed}; "
          f"depth frames matched={total_matched}/{total_frames}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
