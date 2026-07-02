#!/usr/bin/env python
"""
GO/NO-GO Experiment A (bias curve): does the SFT'd model CORRECT MoGe-2's systematic
size-dependent scale bias, or does it just inherit it?

Contribution-2 claim ([[metric-whd-sft-direction]]): a monocular metric pointmap (MoGe-2)
over-measures small/close objects (Objectron ratio ~1.76) — a SIZE-DEPENDENT bias the RAW
anchor cannot fix. If the SFT'd model FLATTENS that bias (and beats a globally-CALIBRATED raw
anchor), the model learned something a constant can't = a genuine model contribution. If
SFT ~= calibrated-stock, there is NO model contribution and it IS just a depth swap.

Two model conditions, SAME precomputed MoGe-2 pointmaps, SAME heldout instances, SAME seed:
    --model stock                              (pretrained ss_generator, no SFT)
    --model sft --metric-checkpoint <ckpt>     (layout_sft_v1_best.pt: SFT layout transformer)
Only the trained layout weights differ ==> isolates what the SFT learned.

    python scripts/bias_curve_experiment.py --model stock
    python scripts/bias_curve_experiment.py --model sft \
        --metric-checkpoint artifacts/metric_scale/checkpoints/layout_sft_v1_best.pt
    python scripts/bias_curve_experiment.py --analyze   # after both runs

Outputs artifacts/metric_scale/metrics/bias_curve_<model>.jsonl  (+ analysis to stdout).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/conda/envs/sam3d"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import sam3d_objects  # noqa: F401  (registers hydra targets)
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset
from sam3d_objects.training.finetune_metric_scale import (
    Moge2PointmapStore, make_train_eval_subsets,
)

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
RGB_ROOTS = {"nocs_real275": f"{OMNI}/real_test", "objectron": OMNI, "arkitscenes": OMNI}
# Match train_layout_sft_v1.sh EXACTLY so the heldout split is identical.
HELDOUT_PER_SOURCE = {"nocs_real275": 64, "objectron": 200, "arkitscenes": 200}
MOGE2_DIR = "artifacts/metric_scale/moge2_pointmaps"


def load_pipeline(config_path, metric_checkpoint=None):
    config = OmegaConf.load(config_path)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    # stock => None; sft => load the SFT ss_backbone (inference-load applies it, strict=False).
    config.metric_scale_checkpoint_path = metric_checkpoint
    config.workspace_dir = os.path.dirname(config_path)
    return instantiate(config)


def build_heldout():
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=list(RGB_ROOTS), rgb_roots=RGB_ROOTS,
        split="train", max_records_per_source=3334, skip_missing_rgb=True,
    )
    _, eval_subset = make_train_eval_subsets(
        ds, train_samples=None, heldout_samples=0, seed=0, shuffle_split=False,
        split_group="record", heldout_per_source=HELDOUT_PER_SOURCE,
    )
    return ds, eval_subset


def run_condition(args):
    out_path = Path(args.output or
                    f"artifacts/metric_scale/metrics/bias_curve_{args.model}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "sft" and not args.metric_checkpoint:
        sys.exit("--model sft requires --metric-checkpoint")

    ds, eval_subset = build_heldout()
    store = Moge2PointmapStore(MOGE2_DIR)
    ckpt = None
    if args.model == "sft":
        # load_metric_scale_checkpoint joins relative paths onto workspace_dir
        # (checkpoints/hf) — absolutize so the repo-relative path resolves.
        ckpt = os.path.abspath(args.metric_checkpoint)
    pipeline = load_pipeline(args.config, ckpt)
    readout = args.readout or ("head" if args.model == "sft" else "pose")
    print(f"readout={readout}")

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(l)["uid"] for l in f if l.strip()}
        print(f"resuming: {len(done)} done")
    fout = open(out_path, "a")
    rels, per_source, skipped = [], defaultdict(list), 0
    n_total = len(eval_subset)
    for n in range(n_total):
        item = eval_subset[n]
        uid = item["uid"]
        if uid in done:
            continue
        try:
            rgba = item["image"]
            gt_dims = np.asarray(item["metric_dims"], dtype=np.float64)
            pm = store.lookup(item["image_name"])  # precomputed MoGe-2 [H,W,3] PyTorch3D
            if readout == "head":
                # Trained-system readout: MetricScaleHead dims through the (fixed)
                # stock pipeline, at the 4/1-step flow schedule the head was
                # trained with (the paper's inference protocol for the metric readout).
                out = pipeline.run(rgba, None, args.inference_seed, stage1_only=False,
                                   with_mesh_postprocess=False, with_texture_baking=False,
                                   with_layout_postprocess=False, use_vertex_color=True,
                                   stage1_inference_steps=4, stage2_inference_steps=1,
                                   pointmap=pm, decode_formats=["mesh"])
                md = out.get("metric_dimensions")
                if md is None:
                    skipped += 1
                    continue
                pred_dims = md[0].detach().cpu().float().numpy().reshape(-1)
            else:
                out = pipeline.run(rgba, None, args.inference_seed,
                                   stage1_only=True, pointmap=pm)
                scale = out["scale"][0].detach().cpu().float().numpy().reshape(-1)
                if scale.size == 1:
                    scale = np.repeat(scale, 3)
                voxel = out["voxel"]
                voxel = voxel.detach().cpu().numpy() if torch.is_tensor(voxel) else np.asarray(voxel)
                if voxel.size == 0:
                    skipped += 1
                    continue
                ext = (voxel.max(axis=0) - voxel.min(axis=0)) + 1.0 / 64.0
                pred_dims = ext * scale[:3]
            gt_iso, pred_iso = float(np.max(gt_dims)), float(np.max(pred_dims))
            rel = abs(pred_iso - gt_iso) / max(gt_iso, 1e-4)
            rec = dict(model=args.model, uid=uid, source=item["source"],
                       category=item["category"], gt_dims=gt_dims.tolist(), gt_iso=gt_iso,
                       pred_dims=pred_dims.tolist(), pred_iso=pred_iso, rel_iso=rel)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            rels.append(rel); per_source[item["source"]].append(rel)
            if (n + 1) % 20 == 0 or n < 5:
                print(f"[{n+1}/{n_total}] {item['source']:<13} gt={gt_iso:.3f} "
                      f"pred={pred_iso:.3f} rel={rel:.3f}", flush=True)
        except Exception as e:
            skipped += 1
            print(f"  [skip] {uid}: {type(e).__name__}: {e}", flush=True)
    fout.close()
    print(f"\n=== model={args.model} n={len(rels)} skipped={skipped} ===")
    if rels:
        print(f"overall iso-MAPE median={np.median(rels)*100:.1f}% mean={np.mean(rels)*100:.1f}%")
        for s in sorted(per_source):
            a = np.array(per_source[s])
            print(f"  {s:<13} n={len(a):>3} median={np.median(a)*100:.1f}% mean={np.mean(a)*100:.1f}%")


def _load(model):
    p = Path(f"artifacts/metric_scale/metrics/bias_curve_{model}.jsonl")
    if not p.exists():
        sys.exit(f"missing {p} — run --model {model} first")
    return {json.loads(l)["uid"]: json.loads(l) for l in open(p) if l.strip()}


def _slope(logsize, logratio):
    """OLS slope of log(pred/gt) on log(gt_iso): the SIZE-DEPENDENCE of the bias (0 = flat)."""
    A = np.vstack([np.ones_like(logsize), logsize]).T
    b = np.linalg.lstsq(A, logratio, rcond=None)[0]
    return float(b[1]), float(b[0])


def _mape(logratio):
    return float(np.mean(np.abs(np.exp(logratio) - 1.0)))


def analyze():
    stock, sft = _load("stock"), _load("sft")
    uids = sorted(set(stock) & set(sft))
    print(f"joined instances: {len(uids)} (stock {len(stock)}, sft {len(sft)})")
    gt = np.array([stock[u]["gt_iso"] for u in uids])
    lg = np.log(gt)
    r_stock = np.log(np.array([stock[u]["pred_iso"] for u in uids]) / gt)
    r_sft = np.log(np.array([sft[u]["pred_iso"] for u in uids]) / gt)
    src = np.array([stock[u]["source"] for u in uids])

    print("\n=== SIZE-DEPENDENT BIAS (slope of log(pred/gt) vs log(gt_iso); 0 = flat) ===")
    for name, r in [("stock(raw anchor)", r_stock), ("sft", r_sft)]:
        sl, ic = _slope(lg, r)
        print(f"  {name:<18} slope={sl:+.3f}  intercept={ic:+.3f}  raw-MAPE={_mape(r)*100:.1f}%")

    # Global calibration: remove the single best constant (median log-ratio) from each.
    sc = r_stock - np.median(r_stock)
    fc = r_sft - np.median(r_sft)
    print("\n=== AFTER GLOBAL CALIBRATION (one constant) ===")
    print(f"  stock-calibrated  slope={_slope(lg, sc)[0]:+.3f}  MAPE={_mape(sc)*100:.1f}%")
    print(f"  sft               slope={_slope(lg, r_sft)[0]:+.3f}  MAPE={_mape(r_sft)*100:.1f}%")
    print(f"  sft-calibrated    slope={_slope(lg, fc)[0]:+.3f}  MAPE={_mape(fc)*100:.1f}%")

    print("\n=== BIAS CURVE (median pred/gt ratio per GT-size quartile) ===")
    qs = np.quantile(gt, [0, .25, .5, .75, 1.0])
    print(f"  {'size bucket (m)':<22}{'n':>4}{'stock':>9}{'sft':>9}")
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (gt >= lo) & (gt <= hi if i == 3 else gt < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.3f},{hi:.3f}]{'':<6}{int(m.sum()):>4}"
              f"{np.exp(np.median(r_stock[m])):>9.2f}{np.exp(np.median(r_sft[m])):>9.2f}")

    print("\n=== PER SOURCE (median pred/gt ratio) ===")
    for s in sorted(set(src)):
        m = src == s
        print(f"  {s:<13} n={int(m.sum()):>3} stock={np.exp(np.median(r_stock[m])):.2f} "
              f"sft={np.exp(np.median(r_sft[m])):.2f}")

    stock_slope = abs(_slope(lg, r_stock)[0])
    sft_slope = abs(_slope(lg, r_sft)[0])
    print("\n=== GO/NO-GO VERDICT ===")
    flattens = sft_slope < 0.6 * stock_slope
    beats_cal = _mape(r_sft) < _mape(sc)
    print(f"  flattens size-bias (|sft slope| < 0.6*|stock|): {flattens} "
          f"({sft_slope:.3f} vs {stock_slope:.3f})")
    print(f"  beats globally-calibrated stock (MAPE):         {beats_cal} "
          f"({_mape(r_sft)*100:.1f}% vs {_mape(sc)*100:.1f}%)")
    print("  >>> " + ("MODEL CONTRIBUTION CONFIRMED (the SFT learns a size-dependent "
                      "correction a constant cannot)." if (flattens and beats_cal) else
                      "INCONCLUSIVE / NO model contribution — SFT ~= calibrated depth swap. "
                      "Reconsider the framing."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stock", "sft"])
    ap.add_argument("--metric-checkpoint", default=None)
    ap.add_argument("--config", default=str(REPO / "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--inference-seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--readout", choices=["pose", "head"], default=None,
                    help="size readout; default = head for --model sft, pose for stock")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.model:
        run_condition(args)
    else:
        ap.error("pass --model {stock,sft} or --analyze")


if __name__ == "__main__":
    main()
