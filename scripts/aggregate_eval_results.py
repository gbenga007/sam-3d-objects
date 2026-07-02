"""Aggregate JSONL eval results into a summary table.

Usage:
    python scripts/aggregate_eval_results.py \\
        artifacts/eval_depth/HAMMER_sam3d_v3_best.jsonl \\
        artifacts/eval_depth/HAMMER_moge_baseline.jsonl \\
        --output artifacts/eval_depth/summary_table.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


METRIC_MODES = [
    "depth_metric",
    "depth_scale_invariant",
    "depth_affine_invariant",
    "disparity_affine_invariant",
    "points_metric",
    "points_scale_invariant",
    "points_affine_invariant",
    "local_points",
]


def load_records(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(records: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns: {mode: {metric: mean_value}}
    Skips records with errors or NaN values.
    """
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_ok = n_err = 0
    for rec in records:
        if "error" in rec:
            n_err += 1
            continue
        n_ok += 1
        metrics = rec.get("metrics", {})
        for mode, vals in metrics.items():
            for metric, v in vals.items():
                if isinstance(v, float) and not np.isnan(v) and not np.isinf(v):
                    accum[mode][metric].append(v)

    result = {}
    for mode, subdict in accum.items():
        result[mode] = {k: float(np.mean(v)) for k, v in subdict.items() if v}

    print(f"  {n_ok} ok, {n_err} errors", file=sys.stderr)
    return result


def format_table(
    baseline_results: Dict[str, Dict[str, Dict[str, float]]],
    modes: Optional[List[str]] = None,
    metrics: List[str] = ("rel", "delta1"),
) -> str:
    if modes is None:
        all_modes: set = set()
        for res in baseline_results.values():
            all_modes.update(res.keys())
        modes = [m for m in METRIC_MODES if m in all_modes]

    col_names = [f"{m.replace('depth_','d_').replace('points_','p_').replace('_invariant','_inv').replace('_metric','_met').replace('local_','loc_')}" for m in modes]

    # Header
    lines = []
    header = "| Baseline |"
    for cn in col_names:
        for met in metrics:
            header += f" {cn}/{met} |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(modes) * len(metrics)))

    for baseline, res in sorted(baseline_results.items()):
        row = f"| {baseline} |"
        for mode in modes:
            for met in metrics:
                v = res.get(mode, {}).get(met, float("nan"))
                if np.isnan(v):
                    row += " — |"
                else:
                    row += f" {v:.4f} |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL result files")
    parser.add_argument("--dataset", default=None, help="Filter to a specific dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown file")
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Metric modes to show (default: all found)")
    args = parser.parse_args()

    baseline_results = {}
    for path in args.inputs:
        records = load_records(path)
        if args.dataset:
            records = [r for r in records if r.get("dataset") == args.dataset]

        # Infer baseline name from filename
        stem = path.stem  # e.g. "HAMMER_sam3d_v3_best"
        parts = stem.split("_")
        # Try to find the baseline name by looking for "sam3d", "moge", etc.
        baseline = stem.replace("all_datasets_", "").replace("HAMMER_", "").replace("iBims-1_", "").replace("DIODE_", "")

        print(f"[{baseline}]", file=sys.stderr)
        agg = aggregate(records)
        baseline_results[baseline] = agg

    # Print per-mode summaries
    for baseline, res in sorted(baseline_results.items()):
        print(f"\n=== {baseline} ===")
        for mode, vals in sorted(res.items()):
            line = f"  {mode}: " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(vals.items()))
            print(line)

    # Markdown table
    table = format_table(baseline_results, modes=args.modes)
    print("\n" + table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")
        print(f"\n[wrote {args.output}]")


if __name__ == "__main__":
    main()
