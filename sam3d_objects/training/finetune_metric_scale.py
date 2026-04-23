# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fine-tune the metric-scale heads on OmniNOCS NOCS-Real275.

This is intentionally NOCS-first and conservative: the SAM 3D backbone stays
frozen, SS/SLAT latents are generated under no-grad, and only the lightweight
MetricScaleHead + MetricScaleDecoder receive gradients. The script is designed
to overfit a tiny sample set before scaling to larger data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sam3d_objects.data.dataset.metric import OmniNOCSObjectDataset, OmniNOCSReal275Dataset
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import MetricScaleHead


class DeviceOnlyPipeline:
    def __init__(self, device: str):
        self.device = torch.device(device)


def collate_instances(batch: list[dict]) -> dict:
    return {
        "images": [item["image"] for item in batch],
        "metric_dims": torch.tensor(
            [item["metric_dims"] for item in batch], dtype=torch.float32
        ),
        "categories": [item["category"] for item in batch],
        "uids": [item["uid"] for item in batch],
        "mask_pixels": torch.tensor(
            [item["mask_pixels"] for item in batch], dtype=torch.long
        ),
    }


def freeze_pipeline(pipeline) -> None:
    for model in pipeline.models.values():
        if model is None:
            continue
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)


def load_pipeline(config_path: str, device: str, compile_model: bool = False):
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "Hydra/OmegaConf are required to instantiate the SAM 3D pipeline. "
            "Install the repo requirements in the active environment, including "
            "hydra-core==1.3.2."
        ) from exc

    config = OmegaConf.load(config_path)
    config.workspace_dir = os.path.dirname(config_path)
    config.compile_model = compile_model
    config.device = device
    pipeline = instantiate(config)
    freeze_pipeline(pipeline)
    return pipeline


def build_dataset(args):
    if args.dataset == "omninocs-mixed":
        rgb_roots = {
            "nocs_real275": args.rgb_root,
            "objectron": args.objectron_rgb_root,
            "arkitscenes": args.arkitscenes_rgb_root,
            "hypersim": args.hypersim_rgb_root,
        }
        dataset = OmniNOCSObjectDataset(
            omninocs_root=args.omninocs_root,
            sources=args.omninocs_sources,
            rgb_roots=rgb_roots,
            split=args.split,
            categories=args.categories,
            min_mask_pixels=args.min_mask_pixels,
            max_records=args.max_records,
            max_records_per_source=args.max_records_per_source,
            skip_missing_rgb=args.skip_missing_rgb,
        )
        print(
            "Loaded OmniNOCS mixed dataset "
            f"sources={args.omninocs_sources} records={len(dataset)} "
            f"skipped_missing_rgb={dataset.skipped_missing_rgb} "
            f"skipped_missing_mask={dataset.skipped_missing_mask}"
        )
        return dataset

    return OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        categories=args.categories,
        min_mask_pixels=args.min_mask_pixels,
        max_records=args.max_records,
    )


def predict_log_dims(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    image,
    stage1_steps: int | None,
    stage2_steps: int | None,
) -> torch.Tensor:
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(image)
        ss_input_dict = pipeline.preprocess_image(
            image, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(image, pipeline.slat_preprocessor)

        with torch.no_grad():
            ss_return_dict = pipeline.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_steps,
                use_distillation=False,
            )
            slat = pipeline.sample_slat(
                slat_input_dict,
                ss_return_dict["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
            )

        scale_token = scale_head(
            ss_return_dict["shape"].detach(),
            ss_input_dict.get("pointmap_scale"),
            ss_input_dict.get("pointmap_shift"),
        )
        return scale_decoder(
            slat.feats.detach(),
            scale_token,
            slat.coords[:, 0],
        )


def encode_metric_scale_features(
    pipeline,
    item: dict,
    metric_dims: torch.Tensor,
    stage1_steps: int | None,
    stage2_steps: int | None,
) -> dict:
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(item["image"])
        ss_input_dict = pipeline.preprocess_image(
            item["image"], pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input_dict = pipeline.preprocess_image(item["image"], pipeline.slat_preprocessor)

        with torch.no_grad():
            ss_return_dict = pipeline.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_steps,
                use_distillation=False,
            )
            slat = pipeline.sample_slat(
                slat_input_dict,
                ss_return_dict["coords"],
                inference_steps=stage2_steps,
                use_distillation=False,
            )

    pointmap_scale = ss_input_dict.get("pointmap_scale")
    pointmap_shift = ss_input_dict.get("pointmap_shift")
    return {
        "shape": ss_return_dict["shape"].detach().cpu(),
        "pointmap_scale": None if pointmap_scale is None else pointmap_scale.detach().cpu(),
        "pointmap_shift": None if pointmap_shift is None else pointmap_shift.detach().cpu(),
        "slat_feats": slat.feats.detach().cpu(),
        "batch_indices": slat.coords[:, 0].detach().cpu(),
        "metric_dims": metric_dims.detach().cpu(),
        "uid": item.get("uid"),
        "category": item.get("category"),
        "source": item.get("source"),
        "image_name": item.get("image_name"),
    }


def predict_cached_log_dims(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    features: dict,
) -> torch.Tensor:
    shape = features["shape"].to(pipeline.device)
    pointmap_scale = features["pointmap_scale"]
    pointmap_shift = features["pointmap_shift"]
    if pointmap_scale is not None:
        pointmap_scale = pointmap_scale.to(pipeline.device)
    if pointmap_shift is not None:
        pointmap_shift = pointmap_shift.to(pipeline.device)

    scale_token = scale_head(shape, pointmap_scale, pointmap_shift)
    return scale_decoder(
        features["slat_feats"].to(pipeline.device),
        scale_token,
        features["batch_indices"].to(pipeline.device),
    )


def evaluate_cached_features(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    feature_cache: list[dict],
) -> dict:
    losses = []
    rel_errors = []
    abs_errors_cm = []
    category_errors = defaultdict(list)
    category_errors_cm = defaultdict(list)
    source_errors = defaultdict(list)
    source_errors_cm = defaultdict(list)
    scale_head.eval()
    scale_decoder.eval()
    with torch.no_grad():
        for features in feature_cache:
            log_pred = predict_cached_log_dims(
                pipeline, scale_head, scale_decoder, features
            )[0]
            target = features["metric_dims"].to(pipeline.device)
            log_target = torch.log(target.clamp(min=1e-6))
            pred = torch.exp(log_pred)
            rel = (pred - target).abs() / target.clamp(min=1e-6)
            abs_cm = (pred - target).abs() * 100.0
            losses.append(F.smooth_l1_loss(log_pred, log_target).detach().cpu())
            rel_cpu = rel.detach().cpu()
            abs_cm_cpu = abs_cm.detach().cpu()
            rel_errors.append(rel_cpu)
            abs_errors_cm.append(abs_cm_cpu)
            category_errors[features.get("category", "unknown")].append(rel_cpu.mean())
            category_errors_cm[features.get("category", "unknown")].append(abs_cm_cpu.mean())
            source_errors[features.get("source", "unknown")].append(rel_cpu.mean())
            source_errors_cm[features.get("source", "unknown")].append(abs_cm_cpu.mean())
    scale_head.train()
    scale_decoder.train()
    rel_tensor = torch.stack(rel_errors)
    abs_cm_tensor = torch.stack(abs_errors_cm)
    return {
        "loss": float(torch.stack(losses).mean()),
        "mean_abs_pct": float(rel_tensor.mean()) * 100,
        "median_abs_pct": float(rel_tensor.median()) * 100,
        "axis_mean_abs_pct": [float(v) * 100 for v in rel_tensor.mean(dim=0)],
        "mean_abs_cm": float(abs_cm_tensor.mean()),
        "median_abs_cm": float(abs_cm_tensor.median()),
        "axis_mean_abs_cm": [float(v) for v in abs_cm_tensor.mean(dim=0)],
        "category_mean_abs_pct": {
            category: float(torch.stack(values).mean()) * 100
            for category, values in sorted(category_errors.items())
        },
        "category_mean_abs_cm": {
            category: float(torch.stack(values).mean())
            for category, values in sorted(category_errors_cm.items())
        },
        "source_mean_abs_pct": {
            source: float(torch.stack(values).mean()) * 100
            for source, values in sorted(source_errors.items())
        },
        "source_mean_abs_cm": {
            source: float(torch.stack(values).mean())
            for source, values in sorted(source_errors_cm.items())
        },
    }


def format_metrics(prefix: str, epoch: int | None, metrics: dict) -> str:
    epoch_part = "" if epoch is None else f" epoch={epoch}"
    axis_pct = ",".join(f"{value:.2f}" for value in metrics["axis_mean_abs_pct"])
    axis_cm = ",".join(f"{value:.2f}" for value in metrics["axis_mean_abs_cm"])
    return (
        f"{prefix}{epoch_part} "
        f"loss={metrics['loss']:.6f} "
        f"mean_abs_pct={metrics['mean_abs_pct']:.2f} "
        f"median_abs_pct={metrics['median_abs_pct']:.2f} "
        f"axis_mean_abs_pct=[{axis_pct}] "
        f"mean_abs_cm={metrics['mean_abs_cm']:.2f} "
        f"median_abs_cm={metrics['median_abs_cm']:.2f} "
        f"axis_mean_abs_cm=[{axis_cm}]"
    )


def print_category_metrics(prefix: str, metrics: dict) -> None:
    category_metrics = metrics.get("category_mean_abs_pct", {})
    if not category_metrics:
        return
    rendered = ", ".join(
        f"{category}={value:.2f}%" for category, value in category_metrics.items()
    )
    print(f"{prefix}_by_category {rendered}")
    category_metrics_cm = metrics.get("category_mean_abs_cm", {})
    if category_metrics_cm:
        rendered_cm = ", ".join(
            f"{category}={value:.2f}cm" for category, value in category_metrics_cm.items()
        )
        print(f"{prefix}_by_category_cm {rendered_cm}")
    source_metrics = metrics.get("source_mean_abs_pct", {})
    if source_metrics:
        rendered_source = ", ".join(
            f"{source}={value:.2f}%" for source, value in source_metrics.items()
        )
        print(f"{prefix}_by_source {rendered_source}")
    source_metrics_cm = metrics.get("source_mean_abs_cm", {})
    if source_metrics_cm:
        rendered_source_cm = ", ".join(
            f"{source}={value:.2f}cm" for source, value in source_metrics_cm.items()
        )
        print(f"{prefix}_by_source_cm {rendered_source_cm}")


def write_metrics(path: str | None, split: str, epoch: int | None, metrics: dict) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "epoch": epoch, **metrics}
    with open(output_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def file_metadata(path: str | None, hash_file: bool = False) -> dict | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    metadata = {
        "path": str(file_path),
        "exists": True,
        "size_bytes": file_path.stat().st_size,
    }
    if hash_file:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["sha256"] = digest.hexdigest()
    return metadata


def category_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        counts[features.get("category", "unknown")] += 1
    return dict(sorted(counts.items()))


def scene_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        image_name = features.get("image_name")
        if image_name is None:
            counts["unknown"] += 1
        else:
            counts[Path(image_name).parts[-2]] += 1
    return dict(sorted(counts.items()))


def source_counts(feature_cache: list[dict]) -> dict:
    counts = defaultdict(int)
    for features in feature_cache:
        counts[features.get("source", "unknown")] += 1
    return dict(sorted(counts.items()))


def feature_schema(feature_cache: list[dict]) -> dict:
    if not feature_cache:
        return {}
    sample = feature_cache[0]
    schema = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            schema[key] = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            schema[key] = {"type": type(value).__name__}
    return schema


def build_manifest(
    args,
    train_cache: list[dict],
    heldout_cache: list[dict],
    checkpoint_path: str | None = None,
    best_checkpoint_path: str | None = None,
    best_metrics: dict | None = None,
    cache_path: str | None = None,
) -> dict:
    return {
        "manifest_version": 1,
        "script": "sam3d_objects/training/finetune_metric_scale.py",
        "args": vars(args),
        "dataset": {
            "dataset": args.dataset,
            "omninocs_root": args.omninocs_root,
            "omninocs_sources": args.omninocs_sources,
            "annotations_root": args.annotations_root,
            "rgb_root": args.rgb_root,
            "objectron_rgb_root": args.objectron_rgb_root,
            "arkitscenes_rgb_root": args.arkitscenes_rgb_root,
            "hypersim_rgb_root": args.hypersim_rgb_root,
            "skip_missing_rgb": args.skip_missing_rgb,
            "split": args.split,
            "categories": args.categories,
            "max_records": args.max_records,
            "max_records_per_source": args.max_records_per_source,
            "min_mask_pixels": args.min_mask_pixels,
        },
        "split": {
            "split_group": args.split_group,
            "seed": args.seed,
            "shuffle_split": args.shuffle_split,
            "train_count": len(train_cache),
            "heldout_count": len(heldout_cache),
            "train_categories": category_counts(train_cache),
            "heldout_categories": category_counts(heldout_cache),
            "train_sources": source_counts(train_cache),
            "heldout_sources": source_counts(heldout_cache),
            "train_scenes": scene_counts(train_cache),
            "heldout_scenes": scene_counts(heldout_cache),
        },
        "feature_schema": {
            "train": feature_schema(train_cache),
            "heldout": feature_schema(heldout_cache),
        },
        "artifacts": {
            "feature_cache": file_metadata(cache_path, hash_file=False),
            "checkpoint": file_metadata(checkpoint_path, hash_file=True),
            "best_checkpoint": file_metadata(best_checkpoint_path, hash_file=True),
            "metrics_output": file_metadata(args.metrics_output, hash_file=True),
        },
        "best_metrics": best_metrics,
    }


def write_manifest(
    path: str | None,
    args,
    train_cache: list[dict],
    heldout_cache: list[dict],
    checkpoint_path: str | None = None,
    best_checkpoint_path: str | None = None,
    best_metrics: dict | None = None,
    cache_path: str | None = None,
) -> None:
    if path is None:
        return
    manifest = build_manifest(
        args,
        train_cache,
        heldout_cache,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_metrics=best_metrics,
        cache_path=cache_path,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote reproducibility manifest to {output_path}")


def validate_feature_cache_manifest(
    manifest_path: str | None,
    train_cache: list[dict],
    heldout_cache: list[dict],
) -> None:
    if manifest_path is None:
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    split = manifest.get("split", {})
    expected_train = split.get("train_count")
    expected_heldout = split.get("heldout_count")
    if expected_train is not None and expected_train != len(train_cache):
        raise RuntimeError(
            f"Manifest train_count={expected_train} but loaded cache has {len(train_cache)}."
        )
    if expected_heldout is not None and expected_heldout != len(heldout_cache):
        raise RuntimeError(
            f"Manifest heldout_count={expected_heldout} but loaded cache has {len(heldout_cache)}."
        )
    expected_schema = manifest.get("feature_schema", {}).get("train", {})
    actual_schema = feature_schema(train_cache)
    for key, expected in expected_schema.items():
        actual = actual_schema.get(key)
        if expected.get("type") == "tensor" and actual != expected:
            raise RuntimeError(
                f"Manifest schema mismatch for {key}: expected {expected}, got {actual}."
            )
    print(f"Validated feature cache against manifest {manifest_path}")


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    flattened = {}
    axis_names = ("width", "height", "depth")
    for key, value in metrics.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{metric_key}/{nested_key}"] = nested_value
        elif isinstance(value, list):
            names = axis_names if len(value) == len(axis_names) else range(len(value))
            for name, item in zip(names, value):
                flattened[f"{metric_key}/{name}"] = item
        else:
            flattened[metric_key] = value
    return flattened


def init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging was requested with --wandb, but wandb is not "
            "installed in the active environment. Install it or rerun without --wandb."
        ) from exc

    tags = args.wandb_tags.split(",") if args.wandb_tags else None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags,
        mode=args.wandb_mode,
        config=vars(args),
    )


def wandb_log(wandb_run, payload: dict, step: int | None = None) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


def save_metric_checkpoint(
    path: str,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    args,
    epoch: int | None = None,
    metrics: dict | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metric_scale_head": scale_head.state_dict(),
            "metric_scale_decoder": scale_decoder.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        },
        output_path,
    )


def category_mean_baseline(train_cache: list[dict], eval_cache: list[dict]) -> dict | None:
    category_dims = defaultdict(list)
    for features in train_cache:
        category_dims[features.get("category", "unknown")].append(features["metric_dims"])
    if not category_dims or not eval_cache:
        return None

    category_means = {
        category: torch.stack(values).mean(dim=0)
        for category, values in category_dims.items()
    }
    global_mean = torch.stack(
        [features["metric_dims"] for features in train_cache]
    ).mean(dim=0)

    rel_errors = []
    abs_errors_cm = []
    category_errors = defaultdict(list)
    category_errors_cm = defaultdict(list)
    source_errors = defaultdict(list)
    source_errors_cm = defaultdict(list)
    for features in eval_cache:
        category = features.get("category", "unknown")
        source = features.get("source", "unknown")
        pred = category_means.get(category, global_mean)
        target = features["metric_dims"]
        rel = (pred - target).abs() / target.clamp(min=1e-6)
        abs_cm = (pred - target).abs() * 100.0
        rel_errors.append(rel)
        abs_errors_cm.append(abs_cm)
        category_errors[category].append(rel.mean())
        category_errors_cm[category].append(abs_cm.mean())
        source_errors[source].append(rel.mean())
        source_errors_cm[source].append(abs_cm.mean())

    rel_tensor = torch.stack(rel_errors)
    abs_cm_tensor = torch.stack(abs_errors_cm)
    return {
        "loss": float("nan"),
        "mean_abs_pct": float(rel_tensor.mean()) * 100,
        "median_abs_pct": float(rel_tensor.median()) * 100,
        "axis_mean_abs_pct": [float(v) * 100 for v in rel_tensor.mean(dim=0)],
        "mean_abs_cm": float(abs_cm_tensor.mean()),
        "median_abs_cm": float(abs_cm_tensor.median()),
        "axis_mean_abs_cm": [float(v) for v in abs_cm_tensor.mean(dim=0)],
        "category_mean_abs_pct": {
            category: float(torch.stack(values).mean()) * 100
            for category, values in sorted(category_errors.items())
        },
        "category_mean_abs_cm": {
            category: float(torch.stack(values).mean())
            for category, values in sorted(category_errors_cm.items())
        },
        "source_mean_abs_pct": {
            source: float(torch.stack(values).mean()) * 100
            for source, values in sorted(source_errors.items())
        },
        "source_mean_abs_cm": {
            source: float(torch.stack(values).mean())
            for source, values in sorted(source_errors_cm.items())
        },
    }


def split_group_key(record: dict, split_group: str) -> str:
    if split_group == "record":
        return record["uid"]
    image_name = record["image_name"]
    if split_group == "image":
        return image_name
    if split_group == "scene":
        return Path(image_name).parts[-2]
    raise ValueError(f"Unsupported split group: {split_group}")


def make_train_eval_subsets(
    dataset,
    train_samples: int | None,
    heldout_samples: int,
    seed: int,
    shuffle_split: bool,
    split_group: str,
) -> tuple[Subset, Subset | None]:
    dataset_len = len(dataset)
    if split_group == "record":
        indices = list(range(dataset_len))
        if shuffle_split:
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(dataset_len, generator=generator).tolist()
        train_count = dataset_len if train_samples is None or train_samples <= 0 else train_samples
        train_count = min(train_count, dataset_len)
        eval_count = max(heldout_samples, 0)
        eval_start = train_count
        eval_stop = min(eval_start + eval_count, dataset_len)
        train_indices = indices[:train_count]
        eval_indices = indices[eval_start:eval_stop]
    else:
        grouped_indices = defaultdict(list)
        for idx, record in enumerate(dataset.records):
            grouped_indices[split_group_key(record, split_group)].append(idx)
        groups = sorted(grouped_indices)
        generator = torch.Generator().manual_seed(seed)
        if shuffle_split:
            permutation = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[idx] for idx in permutation]
        train_target = dataset_len if train_samples is None or train_samples <= 0 else train_samples
        heldout_target = max(heldout_samples, 0)
        train_indices = []
        eval_indices = []
        for group in groups:
            group_indices = grouped_indices[group]
            if len(train_indices) < train_target:
                train_indices.extend(group_indices)
            elif len(eval_indices) < heldout_target:
                eval_indices.extend(group_indices)
            if len(train_indices) >= train_target and len(eval_indices) >= heldout_target:
                break

    train_subset = Subset(dataset, train_indices)
    eval_subset = None
    if heldout_samples:
        eval_subset = Subset(dataset, eval_indices)
        if len(eval_subset) < heldout_samples:
            print(
                f"Requested {heldout_samples} held-out samples but only "
                f"{len(eval_subset)} were available after the train split."
            )
    print(
        f"Split group={split_group} train={len(train_subset)} "
        f"heldout={0 if eval_subset is None else len(eval_subset)}"
    )
    return train_subset, eval_subset


def cache_metric_scale_features(
    pipeline,
    dataset,
    min_mask_pixels: int,
    stage1_steps: int | None,
    stage2_steps: int | None,
    desc: str,
) -> list[dict]:
    feature_cache = []
    for item in tqdm(dataset, desc=desc):
        if int(item["mask_pixels"]) < min_mask_pixels:
            continue
        feature_cache.append(
            encode_metric_scale_features(
                pipeline,
                item,
                torch.as_tensor(item["metric_dims"], dtype=torch.float32),
                stage1_steps,
                stage2_steps,
            )
        )
    return feature_cache


def save_feature_cache(path: str, train_cache: list[dict], heldout_cache: list[dict], args) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train": train_cache,
            "heldout": heldout_cache,
            "args": vars(args),
        },
        cache_path,
    )
    print(
        f"Saved feature cache to {cache_path} "
        f"(train={len(train_cache)}, heldout={len(heldout_cache)})"
    )
    write_manifest(
        args.manifest_output,
        args,
        train_cache,
        heldout_cache,
        cache_path=str(cache_path),
    )


def load_feature_cache(path: str) -> tuple[list[dict], list[dict], dict]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    train_cache = cache.get("train", [])
    heldout_cache = cache.get("heldout", [])
    if not train_cache:
        raise RuntimeError(f"No train features found in cache: {path}")
    print(
        f"Loaded feature cache from {path} "
        f"(train={len(train_cache)}, heldout={len(heldout_cache)})"
    )
    return train_cache, heldout_cache, cache.get("args", {})


def load_metric_checkpoint(
    path: str,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scale_head.load_state_dict(checkpoint["metric_scale_head"])
    scale_decoder.load_state_dict(checkpoint["metric_scale_decoder"])
    print(f"Loaded metric scale checkpoint from {path}")
    return checkpoint.get("args", {})


def evaluate_and_record(
    pipeline,
    scale_head: MetricScaleHead,
    scale_decoder: MetricScaleDecoder,
    feature_cache: list[dict],
    heldout_cache: list[dict],
    metrics_output: str | None,
    wandb_run=None,
    epoch: int | None = None,
) -> dict:
    train_metrics = evaluate_cached_features(
        pipeline, scale_head, scale_decoder, feature_cache
    )
    print(format_metrics("train_eval", epoch, train_metrics))
    print_category_metrics("train_eval", train_metrics)
    write_metrics(metrics_output, "train", epoch, train_metrics)
    wandb_payload = flatten_metrics("train_eval", train_metrics)
    if heldout_cache:
        heldout_metrics = evaluate_cached_features(
            pipeline, scale_head, scale_decoder, heldout_cache
        )
        print(format_metrics("heldout_eval", epoch, heldout_metrics))
        print_category_metrics("heldout_eval", heldout_metrics)
        write_metrics(metrics_output, "heldout", epoch, heldout_metrics)
        wandb_payload.update(flatten_metrics("heldout_eval", heldout_metrics))
    else:
        heldout_metrics = None
    if epoch is not None:
        wandb_payload["epoch"] = epoch
    wandb_log(wandb_run, wandb_payload, step=epoch)
    return {"train": train_metrics, "heldout": heldout_metrics}


def default_best_output_path(output: str) -> str:
    output_path = Path(output)
    return str(output_path.with_name(f"{output_path.stem}_best{output_path.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument(
        "--dataset",
        choices=["omninocs-nocs-real275", "omninocs-mixed"],
        default="omninocs-nocs-real275",
        help="Dataset loader to use for metric-scale training.",
    )
    parser.add_argument("--omninocs-root", default="/mnt/dest/OmniNOCS")
    parser.add_argument(
        "--omninocs-sources",
        nargs="+",
        default=["nocs_real275"],
        choices=["nocs_real275", "objectron", "arkitscenes", "hypersim"],
        help="Sources to include with --dataset omninocs-mixed.",
    )
    parser.add_argument(
        "--annotations-root",
        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275",
    )
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--objectron-rgb-root", default=None)
    parser.add_argument("--arkitscenes-rgb-root", default=None)
    parser.add_argument("--hypersim-rgb-root", default=None)
    parser.add_argument(
        "--skip-missing-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For mixed OmniNOCS, skip records whose source RGB frame is unavailable.",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional per-source cap for --dataset omninocs-mixed.",
    )
    parser.add_argument("--overfit-samples", type=int, default=10)
    parser.add_argument(
        "--heldout-samples",
        type=int,
        default=0,
        help="With --cache-latents, reserve this many examples after the train split for eval.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        help="Use a deterministic random train/held-out split instead of record order.",
    )
    parser.add_argument(
        "--split-group",
        choices=["record", "image", "scene"],
        default="record",
        help="Keep grouped records together when forming train/held-out splits.",
    )
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--cache-latents",
        action="store_true",
        help="Precompute frozen SAM3D features once, then overfit only the metric heads.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="When caching latents, print same-set eval metrics every N epochs.",
    )
    parser.add_argument("--save-feature-cache", default=None)
    parser.add_argument("--load-feature-cache", default=None)
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="Load a saved metric-head checkpoint before training or eval-only metrics.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate a loaded checkpoint on cached train/held-out features, then exit.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSONL path for baseline/train/held-out eval metrics.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Optional JSON path recording dataset, split, feature schema, and artifacts.",
    )
    parser.add_argument(
        "--validate-manifest",
        default=None,
        help="Validate a loaded feature cache against a previous manifest JSON.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build and save feature cache, then exit before metric-head training.",
    )
    parser.add_argument("--output", default="checkpoints/metric_scale_overfit.pt")
    parser.add_argument(
        "--best-output",
        default=None,
        help=(
            "Optional path for the best checkpoint by held-out mean_abs_pct. "
            "Defaults to <output stem>_best<suffix> when held-out eval is available."
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training, evaluation, baseline, and checkpoint metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="sam3d-metric-scale")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode override.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    wandb_run = init_wandb(args)

    if args.load_feature_cache:
        feature_cache, heldout_cache, cache_args = load_feature_cache(args.load_feature_cache)
        validate_feature_cache_manifest(
            args.validate_manifest,
            feature_cache,
            heldout_cache,
        )
        pipeline = DeviceOnlyPipeline(args.device)
        train_dataset = None
        loader = None
    else:
        dataset = build_dataset(args)
        train_dataset, eval_dataset = make_train_eval_subsets(
            dataset,
            args.overfit_samples,
            args.heldout_samples,
            args.seed,
            args.shuffle_split,
            args.split_group,
        )

        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_instances,
        )

        pipeline = load_pipeline(args.config, args.device, args.compile_model)
        feature_cache = []
        heldout_cache = []

        if args.cache_latents:
            feature_cache = cache_metric_scale_features(
                pipeline,
                train_dataset,
                args.min_mask_pixels,
                args.stage1_steps,
                args.stage2_steps,
                "caching train features",
            )
            if not feature_cache:
                raise RuntimeError("No metric-scale training examples remained after filtering.")
            if eval_dataset is not None:
                heldout_cache = cache_metric_scale_features(
                    pipeline,
                    eval_dataset,
                    args.min_mask_pixels,
                    args.stage1_steps,
                    args.stage2_steps,
                    "caching held-out features",
                )
                if not heldout_cache:
                    print("Held-out split had no usable examples after mask filtering.")
            if args.save_feature_cache:
                save_feature_cache(
                    args.save_feature_cache, feature_cache, heldout_cache, args
                )
            if args.cache_only:
                return

    scale_head = MetricScaleHead().to(pipeline.device).train()
    scale_decoder = MetricScaleDecoder().to(pipeline.device).train()

    if args.load_checkpoint:
        load_metric_checkpoint(args.load_checkpoint, scale_head, scale_decoder)

    if args.eval_only:
        if not args.load_feature_cache:
            raise RuntimeError("--eval-only requires --load-feature-cache.")
        if not args.load_checkpoint:
            raise RuntimeError("--eval-only requires --load-checkpoint.")
        baseline_metrics = category_mean_baseline(feature_cache, heldout_cache)
        if baseline_metrics is not None:
            print(format_metrics("category_baseline_heldout", None, baseline_metrics))
            print_category_metrics("category_baseline_heldout", baseline_metrics)
            write_metrics(
                args.metrics_output,
                "category_baseline_heldout",
                None,
                baseline_metrics,
            )
            wandb_log(
                wandb_run,
                flatten_metrics("category_baseline_heldout", baseline_metrics),
            )
        eval_metrics = evaluate_and_record(
            pipeline,
            scale_head,
            scale_decoder,
            feature_cache,
            heldout_cache,
            args.metrics_output,
            wandb_run,
            None,
        )
        write_manifest(
            args.manifest_output,
            args,
            feature_cache,
            heldout_cache,
            checkpoint_path=args.load_checkpoint,
            cache_path=args.load_feature_cache,
            best_metrics=eval_metrics.get("heldout"),
        )
        if wandb_run is not None:
            wandb_run.finish()
        return

    optimizer = torch.optim.AdamW(
        list(scale_head.parameters()) + list(scale_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.cache_latents or args.load_feature_cache:
        baseline_metrics = category_mean_baseline(feature_cache, heldout_cache)
        if baseline_metrics is not None:
            print(format_metrics("category_baseline_heldout", None, baseline_metrics))
            print_category_metrics("category_baseline_heldout", baseline_metrics)
            write_metrics(
                args.metrics_output,
                "category_baseline_heldout",
                None,
                baseline_metrics,
            )
            wandb_log(
                wandb_run,
                flatten_metrics("category_baseline_heldout", baseline_metrics),
            )
        best_heldout_mean_abs_pct = float("inf")
        best_output = args.best_output or default_best_output_path(args.output)
        best_metrics = None
        for epoch in range(args.epochs):
            running_loss = 0.0
            running_count = 0
            order = torch.randperm(len(feature_cache)).tolist()
            progress = tqdm(
                range(0, len(order), args.batch_size), desc=f"epoch {epoch + 1}/{args.epochs}"
            )
            for offset in progress:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for idx in order[offset : offset + args.batch_size]:
                    features = feature_cache[idx]
                    log_pred = predict_cached_log_dims(
                        pipeline, scale_head, scale_decoder, features
                    )
                    log_target = torch.log(
                        features["metric_dims"].to(pipeline.device).clamp(min=1e-6)
                    )
                    losses.append(F.smooth_l1_loss(log_pred[0], log_target))

                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()

                batch_count = len(losses)
                running_loss += float(loss.detach().cpu()) * batch_count
                running_count += batch_count
                progress.set_postfix(loss=running_loss / max(running_count, 1))

            epoch_loss = running_loss / max(running_count, 1)
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "train_epoch/loss": epoch_loss,
                    "train_epoch/examples": running_count,
                    "train_epoch/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

            if args.eval_every and (epoch + 1) % args.eval_every == 0:
                eval_metrics = evaluate_and_record(
                    pipeline,
                    scale_head,
                    scale_decoder,
                    feature_cache,
                    heldout_cache,
                    args.metrics_output,
                    wandb_run,
                    epoch + 1,
                )
                heldout_metrics = eval_metrics.get("heldout")
                if heldout_metrics is not None:
                    heldout_mean_abs_pct = heldout_metrics["mean_abs_pct"]
                    if heldout_mean_abs_pct < best_heldout_mean_abs_pct:
                        best_heldout_mean_abs_pct = heldout_mean_abs_pct
                        best_metrics = heldout_metrics
                        save_metric_checkpoint(
                            best_output,
                            scale_head,
                            scale_decoder,
                            args,
                            epoch + 1,
                            heldout_metrics,
                        )
                        print(
                            f"Saved best metric scale checkpoint to {best_output} "
                            f"(epoch={epoch + 1}, heldout_mean_abs_pct={heldout_mean_abs_pct:.2f})"
                        )
                        wandb_log(
                            wandb_run,
                            {
                                "best/epoch": epoch + 1,
                                "best/heldout_mean_abs_pct": heldout_mean_abs_pct,
                                "best/checkpoint_path": best_output,
                            },
                            step=epoch + 1,
                        )
    else:
        for epoch in range(args.epochs):
            running_loss = 0.0
            running_count = 0
            progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for image, metric_dims, mask_pixels in zip(
                    batch["images"], batch["metric_dims"], batch["mask_pixels"]
                ):
                    if int(mask_pixels) < args.min_mask_pixels:
                        continue
                    log_pred = predict_log_dims(
                        pipeline,
                        scale_head,
                        scale_decoder,
                        image,
                        args.stage1_steps,
                        args.stage2_steps,
                    )
                    log_target = torch.log(metric_dims.to(pipeline.device).clamp(min=1e-6))
                    losses.append(F.smooth_l1_loss(log_pred[0], log_target))

                if not losses:
                    continue

                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()

                batch_count = len(losses)
                running_loss += float(loss.detach().cpu()) * batch_count
                running_count += batch_count
                progress.set_postfix(loss=running_loss / max(running_count, 1))

            epoch_loss = running_loss / max(running_count, 1)
            wandb_log(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "train_epoch/loss": epoch_loss,
                    "train_epoch/examples": running_count,
                    "train_epoch/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch + 1,
            )

    output_path = Path(args.output)
    save_metric_checkpoint(str(output_path), scale_head, scale_decoder, args)
    print(f"Saved metric scale checkpoint to {output_path}")
    write_manifest(
        args.manifest_output,
        args,
        feature_cache,
        heldout_cache,
        checkpoint_path=str(output_path),
        best_checkpoint_path=(
            best_output
            if (args.cache_latents or args.load_feature_cache) and best_metrics is not None
            else None
        ),
        best_metrics=best_metrics if args.cache_latents or args.load_feature_cache else None,
        cache_path=args.load_feature_cache or args.save_feature_cache,
    )
    wandb_log(wandb_run, {"final/checkpoint_path": str(output_path)})
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
