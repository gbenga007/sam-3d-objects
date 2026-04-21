# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Unified MetricScaleDataset combining Objectron and NOCS REAL275.

Used during fine-tuning to supervise MetricScaleHead and MetricScaleDecoder.
Each item provides a pipeline-ready RGBA image and metric ground-truth dimensions.

The dataset does NOT run MoGe or the SS/SLAT generators — those run at
fine-tuning time inside the training loop to produce the latents needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .nocs import NOCSDataset
from .objectron import ObjectronDataset


class MetricScaleDataset(Dataset):
    """
    Combines Objectron and NOCS REAL275 into a single dataset.

    Args:
        objectron_root:  path to preprocessed Objectron directory, or None to skip
        nocs_root:       path to preprocessed NOCS REAL275 directory, or None to skip
        categories:      optional list of category names to filter; None = all
        split:           "train" | "val" — uses a deterministic 90/10 per-category split
        seed:            RNG seed for reproducible splits

    Item keys:
        image         np.ndarray  [H, W, 4] uint8 — RGBA (alpha = object mask)
        metric_dims   torch.Tensor [3] float32    — [width, height, depth] in metres
        log_metric_dims torch.Tensor [3] float32  — log of the above (for loss computation)
        category      str
        uid           str
        source        str  — "objectron" | "nocs"
    """

    def __init__(
        self,
        objectron_root: str | None = None,
        nocs_root: str | None = None,
        categories: list[str] | None = None,
        split: str = "train",
        seed: int = 42,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        sub_datasets: list[Dataset] = []
        if objectron_root is not None:
            sub_datasets.append(ObjectronDataset(objectron_root, categories))
        if nocs_root is not None:
            sub_datasets.append(NOCSDataset(nocs_root, categories))

        if not sub_datasets:
            raise ValueError("Provide at least one of objectron_root or nocs_root.")

        combined = ConcatDataset(sub_datasets)

        # Deterministic 90/10 per-category split
        self.records = self._split(combined, split, seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        image = item["image"]           # [H, W, 4] uint8
        metric_dims = torch.tensor(item["metric_dims"], dtype=torch.float32)
        return {
            "image": image,
            "metric_dims": metric_dims,
            "log_metric_dims": torch.log(metric_dims.clamp(min=1e-6)),
            "pointmap_scale": torch.tensor(item["pointmap_scale"], dtype=torch.float32),
            "pointmap_shift": torch.tensor(item["pointmap_shift"], dtype=torch.float32),
            "category": item["category"],
            "uid": item["uid"],
            "source": item["source"],
        }

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split(dataset: ConcatDataset, split: str, seed: int) -> list:
        """Return a list of items after applying a deterministic per-category split."""
        rng = np.random.default_rng(seed)
        all_items: list[dict] = [dataset[i] for i in range(len(dataset))]

        # Group indices by (source, category)
        groups: dict[tuple, list[int]] = {}
        for i, item in enumerate(all_items):
            key = (item["source"], item["category"])
            groups.setdefault(key, []).append(i)

        selected: list[int] = []
        for indices in groups.values():
            indices = list(rng.permutation(indices))
            n_val = max(1, int(len(indices) * 0.1))
            if split == "val":
                selected.extend(indices[:n_val])
            else:
                selected.extend(indices[n_val:])

        return [all_items[i] for i in selected]


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def metric_scale_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that keeps images as a list (variable resolution)
    and stacks tensors normally.
    """
    return {
        "images": [item["image"] for item in batch],
        "metric_dims": torch.stack([item["metric_dims"] for item in batch]),
        "log_metric_dims": torch.stack([item["log_metric_dims"] for item in batch]),
        "pointmap_scale": torch.stack([item["pointmap_scale"] for item in batch]),
        "pointmap_shift": torch.stack([item["pointmap_shift"] for item in batch]),
        "categories": [item["category"] for item in batch],
        "uids": [item["uid"] for item in batch],
        "sources": [item["source"] for item in batch],
    }
