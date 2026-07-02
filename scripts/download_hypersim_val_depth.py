#!/usr/bin/env python3
"""Download Hypersim per-pixel depth (depth_meters.hdf5) for every scene-cam referenced by the
OmniNOCS hypersim VAL split, via the official range-request downloader (depth-only, ~MBs/frame).
Used to build GT pointmaps for the head-vs-pointmap A/B."""
import json, subprocess, sys, os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

META = "/mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_hypersim/hypersim_val_metadata.json"
DL = "/mnt/source/datasets_sam3d/ml-hypersim/contrib/99991/download.py"
OUT = "/mnt/source/datasets_sam3d/OmniNOCS/hypersim_depth"
PY = "/opt/conda/envs/sam3d/bin/python"

pairs = OrderedDict()
for r in json.load(open(META)):
    scene, cam, _ = r["image_name"].split("/")[:3]      # cam like "cam_03"
    pairs[(scene, cam)] = True
pairs = list(pairs)
print(f"{len(pairs)} unique val scene-cams", flush=True)


def fetch(sc):
    scene, cam = sc
    geo = f"scene_{cam}_geometry_hdf5"                   # -> scene_cam_03_geometry_hdf5
    cmd = [PY, DL, "--scene", scene, "--contains", geo, "--contains", ".depth_meters.hdf5",
           "--directory", OUT, "--silent"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        return (scene, cam, "ok")
    except Exception as e:
        return (scene, cam, f"FAIL {type(e).__name__}")


done = 0
with ThreadPoolExecutor(max_workers=5) as ex:
    for scene, cam, st in ex.map(fetch, pairs):
        done += 1
        print(f"[{done}/{len(pairs)}] {scene}/{cam}: {st}", flush=True)

n = subprocess.run(["bash", "-c", f"find {OUT} -name '*.depth_meters.hdf5' | wc -l"],
                   capture_output=True, text=True).stdout.strip()
print(f"DONE — {n} depth files on disk", flush=True)
