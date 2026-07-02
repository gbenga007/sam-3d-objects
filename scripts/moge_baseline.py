"""MoGe-v1 baseline adapter for eval_metric_depth.py.

MoGe (Ruicheng/moge-vitl) is an affine-invariant monocular point-map
estimator. It predicts shape correctly but has an unknown global scale+shift.
We expose it as both:
  - ``points_affine_invariant``  (primary, as designed by the authors)
  - ``points_scale_invariant``   (fallback)

compute_metrics then optimally aligns each prediction to GT before scoring,
giving us the MoGe-baseline "ceiling" for depth quality.

Usage via eval_metric_depth.py:
    python scripts/eval_metric_depth.py \\
        --baseline moge \\
        --datasets HAMMER iBims-1 DIODE \\
        --output results/moge_baseline.jsonl
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np

from moge.test.baseline import MGEBaselineInterface
from moge.model.v1 import MoGeModel


class MoGeBaseline(MGEBaselineInterface):
    """Thin wrapper around MoGe-v1 that satisfies the eval harness contract."""

    def __init__(self, device: str = "cuda", pretrained: str = "Ruicheng/moge-vitl"):
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(pretrained).to(self.device).eval()

    @staticmethod
    def load(device: str = "cuda", pretrained: str = "Ruicheng/moge-vitl"):
        return MoGeBaseline(device=device, pretrained=pretrained)

    def infer(
        self,
        image: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Standard MGEBaselineInterface inference.

        Args:
            image: [3, H, W] float32 in [0, 1].
            intrinsics: ignored (MoGe predicts its own intrinsics).

        Returns dict with:
          - ``points_affine_invariant``: [H, W, 3] pointmap (affine-invariant)
          - ``depth_affine_invariant``:  [H, W]    depth   (affine-invariant)
          - ``intrinsics``:              [3, 3]    normalized predicted intrinsics
          - ``mask``:                    [H, W]    MoGe validity mask
        """
        # MoGe expects [B, 3, H, W] or [3, H, W], in [0, 1] float.
        img = image.to(self.device)
        if img.ndim == 3:
            img = img.unsqueeze(0)   # [1, 3, H, W]

        out = self.model.infer(img, apply_mask=True, force_projection=True)

        points = out["points"]       # [1, H, W, 3] or [H, W, 3]
        depth  = out["depth"]        # [1, H, W]  or [H, W]
        intrins = out["intrinsics"]  # [1, 3, 3]  or [3, 3]
        mask   = out["mask"]         # bool [1, H, W] or [H, W]

        # Squeeze batch dim if present.
        if points.ndim == 4:
            points  = points[0]
            depth   = depth[0]
            mask    = mask[0]
        if intrins.ndim == 3:
            intrins = intrins[0]

        return {
            "points_affine_invariant": points,   # [H, W, 3]
            "depth_affine_invariant":  depth,    # [H, W]
            "intrinsics":              intrins,  # [3, 3]
            "mask":                    mask,     # [H, W] bool
        }
