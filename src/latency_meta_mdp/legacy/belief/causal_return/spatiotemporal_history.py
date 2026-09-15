"""Patch-preserving motion-aware encoding of causal K6 histories."""

from __future__ import annotations

import torch
from torch import nn

from latency_meta_mdp.legacy.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryConfig,
    MotionAwareHistoryEstimate,
)


def _feedforward(*, dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * dim, dim),
        nn.Dropout(dropout),
    )


class FactorizedSpatiotemporalBlock(nn.Module):
    """Causal temporal attention followed by full-grid spatial attention."""

    def __init__(
        self,
        *,
        history_dim: int,
        attention_head_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (
            history_dim <= 0
            or attention_head_count <= 0
            or history_dim % attention_head_count
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("factorized spatiotemporal dimensions are invalid")
        self.history_dim = history_dim
        self.temporal_norm = nn.LayerNorm(history_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=history_dim,
            num_heads=attention_head_count,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_ffn_norm = nn.LayerNorm(history_dim)
        self.temporal_ffn = _feedforward(dim=history_dim, dropout=dropout)
        self.spatial_norm = nn.LayerNorm(history_dim)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=history_dim,
            num_heads=attention_head_count,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_ffn_norm = nn.LayerNorm(history_dim)
        self.spatial_ffn = _feedforward(dim=history_dim, dropout=dropout)

    def forward(
        self,
        visual_history: torch.Tensor,
        *,
        history_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual_history.ndim != 5:
            raise ValueError("factorized visual history must have five dimensions")
        batch, tick_count, camera_count, patch_count, dim = visual_history.shape
        if (
            (tick_count, camera_count, patch_count, dim) != (6, 2, 196, self.history_dim)
            or tuple(history_valid_mask.shape) != (batch, tick_count)
            or history_valid_mask.dtype is not torch.bool
        ):
            raise ValueError("factorized visual history has invalid shapes or mask")
        if not torch.all(torch.isfinite(visual_history)):
            raise ValueError("factorized visual history must be finite")
        valid = history_valid_mask.to(device=visual_history.device)
        temporal = visual_history.permute(0, 2, 3, 1, 4).reshape(
            batch * camera_count * patch_count,
            tick_count,
            dim,
        )
        temporal_normalized = self.temporal_norm(temporal)
        causal_mask = torch.triu(
            torch.ones(
                tick_count,
                tick_count,
                dtype=torch.bool,
                device=visual_history.device,
            ),
            diagonal=1,
        )
        temporal_padding = ~valid[:, None, None, :].expand(
            batch,
            camera_count,
            patch_count,
            tick_count,
        ).reshape(batch * camera_count * patch_count, tick_count)
        temporal_update, _ = self.temporal_attention(
            temporal_normalized,
            temporal_normalized,
            temporal_normalized,
            attn_mask=causal_mask,
            key_padding_mask=temporal_padding,
            need_weights=False,
        )
        temporal = temporal + temporal_update
        temporal = temporal + self.temporal_ffn(self.temporal_ffn_norm(temporal))
        spatial = temporal.reshape(
            batch,
            camera_count,
            patch_count,
            tick_count,
            dim,
        ).permute(0, 3, 1, 2, 4)
        spatial = spatial.reshape(
            batch * tick_count * camera_count,
            patch_count,
            dim,
        )
        spatial_normalized = self.spatial_norm(spatial)
        spatial_update, _ = self.spatial_attention(
            spatial_normalized,
            spatial_normalized,
            spatial_normalized,
            need_weights=False,
        )
        spatial = spatial + spatial_update
        spatial = spatial + self.spatial_ffn(self.spatial_ffn_norm(spatial))
        return spatial.reshape(
            batch,
            tick_count,
            camera_count,
            patch_count,
            dim,
        )


class MotionAwareHistoryEncoder(nn.Module):
    """Encode full-grid visual motion and robot history into two typed tokens."""

    def __init__(self, config: MotionAwareHistoryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.history_dim
        self.patch_projection = nn.Linear(config.vision_feature_dim, dim)
        self.spatial_position = nn.Parameter(torch.empty(config.patch_token_count, dim))
        self.camera_embedding = nn.Parameter(torch.empty(config.camera_count, dim))
        self.history_embedding = nn.Parameter(torch.empty(config.history_sample_count, dim))
        self.patch_norm = nn.LayerNorm(dim)
        self.spatiotemporal_blocks = nn.ModuleList(
            FactorizedSpatiotemporalBlock(
                history_dim=dim,
                attention_head_count=config.attention_head_count,
                dropout=config.dropout,
            )
            for _ in range(config.spatiotemporal_block_count)
        )
        self.robot_projection = nn.Sequential(
            nn.Linear(config.robot_state_dim, dim),
            nn.GELU(),
        )
        self.robot_gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=config.robot_gru_layer_count,
            dropout=config.dropout,
            batch_first=True,
        )
        self.pose_query = nn.Parameter(torch.empty(1, dim))
        self.motion_query = nn.Parameter(torch.empty(1, dim))
        self.query_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.attention_head_count,
            dropout=config.dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(dim)
        self.query_ffn_norm = nn.LayerNorm(dim)
        self.query_ffn = _feedforward(dim=dim, dropout=config.dropout)
        self.position_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, config.object_position_dim),
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, config.object_velocity_dim),
        )
        for parameter in (
            self.spatial_position,
            self.camera_embedding,
            self.history_embedding,
            self.pose_query,
            self.motion_query,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _validate_inputs(
        self,
        *,
        vision_history: torch.Tensor,
        robot_history_normalized: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> int:
        batch = vision_history.shape[0]
        if tuple(vision_history.shape) != (
            batch,
            self.config.history_sample_count,
            self.config.camera_count,
            self.config.patch_token_count,
            self.config.vision_feature_dim,
        ):
            raise ValueError("motion-aware vision history has an invalid shape")
        if tuple(robot_history_normalized.shape) != (
            batch,
            self.config.history_sample_count,
            self.config.robot_state_dim,
        ):
            raise ValueError("motion-aware normalized robot history has an invalid shape")
        if (
            tuple(history_valid_mask.shape) != (batch, self.config.history_sample_count)
            or history_valid_mask.dtype is not torch.bool
            or not torch.all(history_valid_mask)
        ):
            raise ValueError("motion-aware history valid mask must contain six real samples")
        if not torch.all(torch.isfinite(vision_history)) or not torch.all(
            torch.isfinite(robot_history_normalized)
        ):
            raise ValueError("motion-aware history inputs must be finite")
        return batch

    def forward(
        self,
        *,
        vision_history: torch.Tensor,
        robot_history_normalized: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> MotionAwareHistoryEstimate:
        batch = self._validate_inputs(
            vision_history=vision_history,
            robot_history_normalized=robot_history_normalized,
            history_valid_mask=history_valid_mask,
        )
        visual = self.patch_projection(vision_history.float())
        visual = visual + self.spatial_position[None, None, None]
        visual = visual + self.camera_embedding[None, None, :, None]
        visual = visual + self.history_embedding[None, :, None, None]
        visual = self.patch_norm(visual)
        for block in self.spatiotemporal_blocks:
            visual = block(visual, history_valid_mask=history_valid_mask)
        robot = self.robot_projection(robot_history_normalized.float())
        robot = robot + self.history_embedding[None]
        robot, _ = self.robot_gru(robot)
        memory = torch.cat((visual.flatten(start_dim=1, end_dim=3), robot), dim=1)
        queries = torch.cat((self.pose_query, self.motion_query), dim=0)
        queries = queries[None].expand(batch, -1, -1)
        attended, _ = self.query_attention(
            queries,
            memory,
            memory,
            need_weights=False,
        )
        context = self.query_norm(queries + attended)
        context = context + self.query_ffn(self.query_ffn_norm(context))
        return MotionAwareHistoryEstimate(
            history_context_tokens=context,
            object_position_normalized=self.position_head(context[:, 0]),
            object_velocity_normalized=self.velocity_head(context[:, 1]),
        )
