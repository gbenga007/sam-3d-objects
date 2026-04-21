# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn


class MetricScaleDecoder(nn.Module):
    """
    Decodes physical object dimensions [width, height, depth] in meters from
    the SLAT refined latent combined with the scale conditioning token.

    Placed in the SLAT stage because it has access to both the refined 3D
    geometry (SLAT latent) and the metric scale context (scale token from SS).
    Trained with ground truth metric dimensions; all other pipeline components
    are frozen during fine-tuning.

    Outputs log-space predictions for numerical stability — exponentiate to get meters.
    """

    def __init__(
        self,
        slat_feat_dim: int = 8,
        scale_token_dim: int = 768,
        scale_proj_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.scale_proj = nn.Linear(scale_token_dim, scale_proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slat_feat_dim + scale_proj_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),  # log([width, height, depth])
        )

    def forward(
        self,
        slat_feats: torch.Tensor,
        scale_token: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            slat_feats:    [num_voxels, 8]   — SparseTensor .feats after SLAT decoding
            scale_token:   [batch, 1, 768]   — from ScaleTokenProjector
            batch_indices: [num_voxels]      — per-voxel batch index (SparseTensor coords[:, 0])

        Returns:
            log_dims: [batch, 3] — log([width, height, depth] in meters)
        """
        batch_size = scale_token.shape[0]
        device = slat_feats.device
        dtype = slat_feats.dtype

        # Mean-pool SLAT features per batch element via scatter
        pooled = torch.zeros(batch_size, slat_feats.shape[1], device=device, dtype=dtype)
        counts = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        idx = batch_indices.long().view(-1, 1).expand(-1, slat_feats.shape[1])
        pooled.scatter_add_(0, idx, slat_feats)
        counts.scatter_add_(0, batch_indices.long().view(-1, 1), torch.ones_like(counts[:1].expand(batch_indices.shape[0], 1)))
        pooled = pooled / counts.clamp(min=1)  # [batch, 8]

        scale_ctx = self.scale_proj(scale_token.squeeze(1).to(dtype))  # [batch, scale_proj_dim]
        x = torch.cat([pooled, scale_ctx], dim=1)
        return self.mlp(x)  # [batch, 3]  — log([w, h, d])

    def predict_metric_dimensions(
        self,
        slat_feats: torch.Tensor,
        scale_token: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns metric dimensions in meters (positive values)."""
        return torch.exp(self.forward(slat_feats, scale_token, batch_indices))
