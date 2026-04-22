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
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sam3d_objects.data.dataset.metric import OmniNOCSReal275Dataset
from sam3d_objects.model.backbone.metric_scale_decoder import MetricScaleDecoder
from sam3d_objects.model.backbone.scale_head import MetricScaleHead


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument(
        "--annotations-root",
        default="/mnt/dest/OmniNOCS/omninocs_release_nocs_real275",
    )
    parser.add_argument("--rgb-root", default="/mnt/dest/OmniNOCS/real_test")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--overfit-samples", type=int, default=10)
    parser.add_argument("--min-mask-pixels", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-steps", type=int, default=None)
    parser.add_argument("--stage2-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output", default="checkpoints/metric_scale_overfit.pt")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    dataset = OmniNOCSReal275Dataset(
        annotations_root=args.annotations_root,
        rgb_root=args.rgb_root,
        split=args.split,
        categories=args.categories,
        min_mask_pixels=args.min_mask_pixels,
        max_records=args.max_records,
    )
    if args.overfit_samples and args.overfit_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.overfit_samples)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_instances,
    )

    pipeline = load_pipeline(args.config, args.device, args.compile_model)
    scale_head = MetricScaleHead().to(pipeline.device).train()
    scale_decoder = MetricScaleDecoder().to(pipeline.device).train()

    optimizer = torch.optim.AdamW(
        list(scale_head.parameters()) + list(scale_decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metric_scale_head": scale_head.state_dict(),
            "metric_scale_decoder": scale_decoder.state_dict(),
            "args": vars(args),
        },
        output_path,
    )
    print(f"Saved metric scale checkpoint to {output_path}")


if __name__ == "__main__":
    main()
