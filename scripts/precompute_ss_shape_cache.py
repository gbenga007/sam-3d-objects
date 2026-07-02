#!/usr/bin/env python
"""Precompute CLEAN 25-step SS shape latents (fixed seed) per object for native-FM v3.

WHY: native_fm_v2 uses a live 4-step shape -> coarse + (without seed) high-variance canon. The
fixed seed makes canon deterministic (train==eval), but 4-step canon is still a coarse blob and
doesn't match real 25-step inference. Caching the 25-step shape gives a CLEAN, deployment-matched,
deterministic canon -> accurate PROPORTIONS + exact cancellation + the scale modality only has to
predict iso. Shape transformer is FROZEN in our SFT, so the stock pipeline's shape is the right one.

Per object (uid), seeded with shape_sample_seed(image_name) — IDENTICAL to the train/eval seed — run
sample_sparse_structure(25 steps) and store {shape, scale, translation, 6drotation_normalized,
translation_scale, canon_ext, downsample_factor}. Conditioned on the MoGe-2 pointmap (the train depth
source). Resumable (skips existing). Output: <out>/<sanitized uid>.pt + manifest.json (uid -> file).

    python scripts/precompute_ss_shape_cache.py --steps 25
"""
import argparse, json, os, sys, time
os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/opt/conda/envs/sam3d"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sam3d_objects.training.finetune_metric_scale import (
    load_pipeline, Moge2PointmapStore, _voxel_canonical_extents, shape_sample_seed, SS_FM_MODALITIES,
)
from sam3d_objects.data.dataset.metric.omninocs import OmniNOCSObjectDataset

OMNI = "/mnt/source/datasets_sam3d/OmniNOCS"
RGB = {"nocs_real275": f"{OMNI}/real_test", "objectron": OMNI, "arkitscenes": OMNI}


def sanitize(uid: str) -> str:
    return str(uid).replace("/", "__").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--output-dir", default="artifacts/metric_scale/ss_shape_cache_25")
    ap.add_argument("--moge2-dir", default="artifacts/metric_scale/moge2_pointmaps")
    ap.add_argument("--max-records-per-source", type=int, default=3334)
    ap.add_argument("--config", default=str(REPO / "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = OmniNOCSObjectDataset(
        omninocs_root=OMNI, sources=list(RGB), rgb_roots=RGB, split="train",
        max_records_per_source=args.max_records_per_source, skip_missing_rgb=True,
    )
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"dataset: {len(ds)} records; processing {n}", flush=True)
    store = Moge2PointmapStore(args.moge2_dir)
    pipeline = load_pipeline(args.config, "cuda")   # stock (frozen shape)

    manifest_path = out / "manifest.json"
    manifest = {"steps": args.steps, "frames": {}}
    if manifest_path.exists():
        manifest = json.load(open(manifest_path))
    t0 = time.time(); done = 0; skipped = 0
    for i in range(n):
        try:
            item = ds[i]
            uid = item["uid"]
            fp = out / f"{sanitize(uid)}.pt"
            if fp.exists():
                manifest["frames"][uid] = fp.name
                continue
            try:
                pm = store.lookup(item["image_name"])
            except KeyError:
                skipped += 1
                continue
            seed = shape_sample_seed(item["image_name"])
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with pipeline.device, torch.no_grad():
                pmd = pipeline.compute_pointmap(item["image"], pointmap=pm)
                ssd = pipeline.preprocess_image(item["image"], pipeline.ss_preprocessor,
                                                pointmap=pmd["pointmap"])
                ss_ret = pipeline.sample_sparse_structure(
                    ssd, inference_steps=args.steps, use_distillation=False, with_grad=False)
            rec = {k: ss_ret[k].detach().half().cpu() for k in SS_FM_MODALITIES}
            rec["canon_ext"] = _voxel_canonical_extents(ss_ret).float().cpu()
            rec["downsample_factor"] = float(ss_ret.get("downsample_factor", 1.0))
            rec["source"] = item["source"]
            tmp = fp.with_suffix(".pt.tmp")
            torch.save(rec, tmp); os.replace(tmp, fp)
            manifest["frames"][uid] = fp.name
            done += 1
            if done % 50 == 0:
                json.dump(manifest, open(manifest_path, "w"))
                rate = (time.time() - t0) / max(done, 1)
                print(f"[{i+1}/{n}] cached={done} skipped={skipped} "
                      f"{rate:.1f}s/it eta={rate*(n-i)/3600:.1f}h", flush=True)
        except Exception as e:
            skipped += 1
            print(f"[skip] idx={i}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
    json.dump(manifest, open(manifest_path, "w"))
    print(f"DONE cached={done} skipped={skipped} total_frames={len(manifest['frames'])}", flush=True)


if __name__ == "__main__":
    main()
