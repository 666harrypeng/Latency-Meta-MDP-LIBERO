"""Lightweight diagnostic probe over frozen DINO patch histories."""

from __future__ import annotations

import math

import torch
from torch import nn

from latency_meta_mdp.legacy.vision_probe_data import PROBE_TARGET_DIM, ROBOT_PROPRIO_DIM


class TemporalVisionStateProbe(nn.Module):
    """Learned spatial pooling followed by a small temporal recurrent model."""

    def __init__(
        self,
        *,
        history_sample_count: int,
        patch_projection_dim: int,
        temporal_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.history_sample_count = history_sample_count
        self.patch_projection_dim = patch_projection_dim
        self.patch_projection = nn.Linear(384, patch_projection_dim)
        self.patch_norm = nn.LayerNorm(patch_projection_dim)
        self.spatial_position = nn.Parameter(torch.empty(196, patch_projection_dim))
        self.camera_embedding = nn.Parameter(torch.empty(2, patch_projection_dim))
        self.pool_query = nn.Parameter(torch.empty(2, patch_projection_dim))
        self.temporal = nn.GRU(
            input_size=2 * patch_projection_dim + ROBOT_PROPRIO_DIM,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.GELU(),
            nn.Linear(temporal_hidden_dim, PROBE_TARGET_DIM),
        )
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, vision_history: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
        if vision_history.ndim != 5 or tuple(vision_history.shape[1:]) != (
            self.history_sample_count,
            2,
            196,
            384,
        ):
            raise ValueError("probe vision tensor has invalid shape")
        if tuple(proprio_history.shape) != (
            vision_history.shape[0],
            self.history_sample_count,
            ROBOT_PROPRIO_DIM,
        ):
            raise ValueError("probe proprioception tensor has invalid shape")
        projected = self.patch_projection(vision_history.float())
        projected = projected + self.spatial_position[None, None, None, :, :]
        projected = projected + self.camera_embedding[None, None, :, None, :]
        projected = self.patch_norm(projected)
        scores = torch.einsum("bkcpe,ce->bkcp", projected, self.pool_query)
        scores = scores / math.sqrt(self.patch_projection_dim)
        weights = torch.softmax(scores, dim=3)
        pooled = torch.einsum("bkcp,bkcpe->bkce", weights, projected)
        fused = torch.cat(
            (
                pooled.flatten(start_dim=2),
                proprio_history.float(),
            ),
            dim=-1,
        )
        temporal, _ = self.temporal(fused)
        return self.head(temporal[:, -1])
