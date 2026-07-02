#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Drop GT size-outlier records from an existing feature cache (and its heldout
split) without re-running the MoGe-2 / SS / SLAT precompute pipeline.

Uses the same category-aware log-size filter as the OmniNOCS loader
(filter_size_outliers, tol=log(5) by default), so a cache cleaned here matches
what a fresh cache built with the loader filter would contain.

Usage:
    LIDRA_SKIP_INIT=1 python scripts/clean_cache_size_outliers.py \
        --in  artifacts/metric_scale/feature_caches/moge2_mixed_10k.pt \
        --out artifacts/metric_scale/feature_caches/moge2_mixed_10k_clean.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sam3d_objects.data.dataset.metric.omninocs import (  # noqa: E402
    DEFAULT_SIZE_OUTLIER_LOG_TOL,
    filter_size_outliers,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tol", type=float, default=DEFAULT_SIZE_OUTLIER_LOG_TOL,
                    help="log-size deviation tolerance (default log(5)).")
    args = ap.parse_args()

    print(f"Loading {args.inp} ...")
    cache = torch.load(args.inp, map_location="cpu", weights_only=False)
    for split in ("train", "heldout"):
        recs = cache.get(split, [])
        if not recs:
            continue
        kept, dropped = filter_size_outliers(recs, tol_log=args.tol)
        cache[split] = kept
        print(f"  {split}: {len(recs)} -> {len(kept)} (dropped {sum(dropped.values())})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {args.out} ...")
    torch.save(cache, args.out)
    print("done.")


if __name__ == "__main__":
    main()
