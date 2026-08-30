"""Temporally ordered information-state estimation from deployment history."""

from __future__ import annotations

import math

import torch
from torch import nn

from latency_meta_mdp.belief.causal_return.contracts import (
    CurrentStateEstimate,
    InformationStateConfig,
)


class InformationStateEstimator(nn.Module):
    """Estimate current robot/object state without future-control information."""

    def __init__(self, config: InformationStateConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.patch_projection = nn.Linear(config.vision_feature_dim, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.spatial_position = nn.Parameter(torch.empty(config.patch_token_count, dim))
        self.camera_embedding = nn.Parameter(torch.empty(config.camera_count, dim))
        self.history_position = nn.Parameter(torch.empty(config.history_sample_count, dim))
        self.spatial_pool_queries = nn.Parameter(
            torch.empty(
                config.camera_count,
                config.spatial_pool_query_count,
                dim,
            )
        )
        self.robot_projection = nn.Sequential(
            nn.Linear(config.robot_state_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.type_embedding = nn.Parameter(torch.empty(2, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.attention_head_count,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_fusion = nn.TransformerEncoder(
            layer,
            num_layers=config.temporal_fusion_layer_count,
            enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.state_queries = nn.Parameter(torch.empty(config.state_token_count, dim))
        self.state_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.attention_head_count,
            dropout=config.dropout,
            batch_first=True,
        )
        self.state_norm = nn.LayerNorm(dim)
        self.state_feedforward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)
        self.object_mean_head = nn.Linear(dim, config.object_state_dim)
        self.object_log_scale_head = nn.Linear(dim, config.object_state_dim)
        for parameter in (
            self.spatial_position,
            self.camera_embedding,
            self.history_position,
            self.spatial_pool_queries,
            self.type_embedding,
            self.state_queries,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _validate_inputs(
        self,
        *,
        vision_history: torch.Tensor,
        robot_history: torch.Tensor,
        history_time_ms: torch.Tensor,
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
            raise ValueError("information-state vision history has invalid shape")
        if tuple(robot_history.shape) != (
            batch,
            self.config.history_sample_count,
            self.config.robot_state_dim,
        ):
            raise ValueError("information-state robot history has invalid shape")
        if tuple(history_time_ms.shape) != (batch, self.config.history_sample_count):
            raise ValueError("information-state timestamps have invalid shape")
        expected = torch.arange(
            -(self.config.history_sample_count - 1) * self.config.formal_tick_ms,
            self.config.formal_tick_ms,
            self.config.formal_tick_ms,
            device=history_time_ms.device,
            dtype=history_time_ms.dtype,
        )
        if not torch.equal(history_time_ms, expected[None].expand(batch, -1)):
            raise ValueError("information-state timestamps are invalid")
        if (
            tuple(history_valid_mask.shape) != (batch, self.config.history_sample_count)
            or history_valid_mask.dtype is not torch.bool
            or not torch.all(history_valid_mask)
        ):
            raise ValueError("information-state validity mask is invalid")
        for name, value in (
            ("vision", vision_history),
            ("robot", robot_history),
        ):
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"information-state {name} input must be finite")
        return batch

    def forward(
        self,
        vision_history: torch.Tensor,
        robot_history: torch.Tensor,
        history_time_ms: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> CurrentStateEstimate:
        batch = self._validate_inputs(
            vision_history=vision_history,
            robot_history=robot_history,
            history_time_ms=history_time_ms,
            history_valid_mask=history_valid_mask,
        )
        dim = self.config.model_dim
        visual = self.patch_projection(vision_history.float())
        visual = visual + self.spatial_position[None, None, None]
        visual = visual + self.camera_embedding[None, None, :, None]
        visual = visual + self.history_position[None, :, None, None]
        continuous_time = self.time_projection(
            (history_time_ms.float() / 100.0)[..., None]
        )
        visual = visual + continuous_time[:, :, None, None]
        visual = self.patch_norm(visual)
        scores = torch.einsum(
            "bkcpe,cqe->bkcqp",
            visual,
            self.spatial_pool_queries,
        ) / math.sqrt(dim)
        weights = torch.softmax(scores, dim=-1)
        visual_tokens = torch.einsum("bkcqp,bkcpe->bkcqe", weights, visual)
        visual_tokens = visual_tokens.flatten(start_dim=2, end_dim=3)
        visual_tokens = visual_tokens + self.type_embedding[0]

        robot_tokens = self.robot_projection(robot_history.float())
        robot_tokens = robot_tokens + self.history_position[None] + continuous_time
        robot_tokens = robot_tokens + self.type_embedding[1]
        tokens = torch.cat((visual_tokens, robot_tokens[:, :, None]), dim=2)
        tokens_per_tick = tokens.shape[2]
        tokens = tokens.flatten(start_dim=1, end_dim=2)
        token_valid = history_valid_mask[:, :, None].expand(-1, -1, tokens_per_tick)
        token_valid = token_valid.reshape(batch, -1)
        token_time = torch.arange(
            self.config.history_sample_count,
            device=tokens.device,
        ).repeat_interleave(tokens_per_tick)
        temporal_mask = token_time[None, :] > token_time[:, None]
        fused = self.temporal_fusion(
            tokens,
            mask=temporal_mask,
            src_key_padding_mask=~token_valid,
        )
        fused = self.fusion_norm(fused)
        queries = self.state_queries[None].expand(batch, -1, -1)
        attended, _ = self.state_attention(
            queries,
            fused,
            fused,
            key_padding_mask=~token_valid,
            need_weights=False,
        )
        state_tokens = self.state_norm(queries + attended)
        state_tokens = self.output_norm(
            state_tokens + self.state_feedforward(state_tokens)
        )
        pooled = state_tokens.mean(dim=1)
        object_mean = self.object_mean_head(pooled)
        object_log_scale = torch.clamp(
            self.object_log_scale_head(pooled),
            min=self.config.log_scale_min,
            max=self.config.log_scale_max,
        )
        return CurrentStateEstimate(
            robot_state=robot_history[:, -1],
            object_state_mean=object_mean,
            object_state_log_scale=object_log_scale,
            state_tokens=state_tokens,
        )
