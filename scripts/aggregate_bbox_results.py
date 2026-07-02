"""Aggregate bbox-from-depth evaluation results into a comparison table.

Usage:
    python scripts/aggregate_bbox_results.py artifacts/eval_bbox/*.jsonl

Outputs a markdown table comparing:
  - Visible bbox errors (pred restricted to GT-valid pixels — fair comparison)
  - Full bbox errors (pred over full object mask)
For rel_W, rel_H, rel_D, rel_max_wh, rel_max_whd (lower = better).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_records(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(records: List[dict], error_key: str = "errors_visible"):
    """
    Aggregate records by (estimator, dataset).
    Returns nested dict: {(estimator, dataset): {metric: mean_value}}.
    """
    buckets: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        if "error" in r:
            continue
        key = (r["estimator"], r["dataset"])
        errs = r.get(error_key)
        if errs is None:
            continue
        for metric, val in errs.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                buckets[key][metric].append(float(val))

    results = {}
    for key, metrics in buckets.items():
        results[key] = {
            m: float(np.mean(v)) for m, v in metrics.items() if v
        }
        results[key]["n_samples"] = min(len(v) for v in metrics.values()) if metrics else 0
    return results


def make_table(results: dict, metrics: List[str], caption: str) -> str:
    # Collect all keys and sort
    keys = sorted(results.keys())
    estimators = sorted(set(k[0] for k in keys))
    datasets = sorted(set(k[1] for k in keys))

    lines = [f"\n### {caption}\n"]

    # Header
    header = ["Estimator", "Dataset"] + metrics + ["N"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for est in estimators:
        for ds in datasets:
            key = (est, ds)
            if key not in results:
                continue
            r = results[key]
            vals = []
            for m in metrics:
                v = r.get(m)
                vals.append(f"{v*100:.1f}%" if v is not None else "—")
            n = r.get("n_samples", "?")
            lines.append("| " + " | ".join([est, ds] + vals + [str(n)]) + " |")

    return "\n".join(lines)


def dataset_avg(results: dict, metrics: List[str]) -> dict:
    """Average over datasets per estimator."""
    by_est: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (est, ds), r in results.items():
        for m in metrics:
            v = r.get(m)
            if v is not None:
                by_est[est][m].append(v)
    avg = {}
    for est, m_dict in by_est.items():
        avg[est] = {m: float(np.mean(v)) if v else float("nan") for m, v in m_dict.items()}
        avg[est]["n_datasets"] = len(set(ds for (e, ds) in results if e == est))
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    all_records = []
    for path in args.files:
        recs = load_records(path)
        all_records.extend(recs)
        ok = sum(1 for r in recs if "error" not in r and "errors_visible" in r)
        err = sum(1 for r in recs if "error" in r)
        print(f"{path.name}: {ok} ok, {err} errors")

    if not all_records:
        print("No records found.")
        return

    METRICS = ["rel_W", "rel_H", "rel_D", "rel_max_wh", "rel_max_whd"]

    vis_results = aggregate(all_records, error_key="errors_visible")
    full_results = aggregate(all_records, error_key="errors_full")

    print(make_table(vis_results, METRICS,
                     "Bbox errors (visible surface — same pixels as GT sensor depth)"))
    print(make_table(full_results, METRICS,
                     "Bbox errors (full object mask — model sees more than sensor depth)"))

    # Per-estimator average across datasets
    print("\n### Average across datasets (visible surface)\n")
    avg = dataset_avg(vis_results, METRICS)
    header = ["Estimator"] + METRICS + ["Datasets"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for est, r in sorted(avg.items()):
        vals = [f"{r.get(m, float('nan'))*100:.1f}%" for m in METRICS]
        print("| " + " | ".join([est] + vals + [str(r.get("n_datasets", "?"))]) + " |")


if __name__ == "__main__":
    main()
