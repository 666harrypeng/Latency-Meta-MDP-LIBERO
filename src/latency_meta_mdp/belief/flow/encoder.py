"""Flow-specific causal Belief Encoder trained jointly with the vector field."""

from __future__ import annotations

import math

import torch
from torch import nn

from latency_meta_mdp.belief.flow.config import FlowBeliefConfig


class FlowBeliefEncoder(nn.Module):
    """Encode launch-time information without accepting a delay query."""

    def __init__(self, config: FlowBeliefConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.model_dim
        self.patch_projection = nn.Linear(384, dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.spatial_position = nn.Parameter(torch.empty(196, dim))
        self.camera_embedding = nn.Parameter(torch.empty(2, dim))
        self.time_embedding = nn.Parameter(torch.empty(config.history_sample_count, dim))
        self.patch_pool_queries = nn.Parameter(
            torch.empty(2, config.patch_pool_query_count, dim)
        )
        self.proprio_projection = nn.Sequential(
            nn.Linear(16, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(7, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_position = nn.Parameter(torch.empty(25, dim))
        self.latency_projection = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.type_embedding = nn.Parameter(torch.empty(4, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.transformer_head_count,
            dim_feedforward=4 * dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=config.fusion_layer_count,
            enable_nested_tensor=False,
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.belief_queries = nn.Parameter(
            torch.empty(config.belief_token_count, dim)
        )
        self.belief_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.transformer_head_count,
            dropout=config.dropout,
            batch_first=True,
        )
        self.belief_norm = nn.LayerNorm(dim)
        self.belief_feedforward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * dim, dim),
        )
        self.output_norm = nn.LayerNorm(dim)
        for parameter in (
            self.spatial_position,
            self.camera_embedding,
            self.time_embedding,
            self.patch_pool_queries,
            self.action_position,
            self.type_embedding,
            self.belief_queries,
        ):
            nn.init.normal_(parameter, std=0.02)

    def forward(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        remaining_actions: torch.Tensor,
        latency_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        batch = vision_history.shape[0]
        expected_vision = (
            batch,
            self.config.history_sample_count,
            2,
            196,
            384,
        )
        if tuple(vision_history.shape) != expected_vision:
            raise ValueError("Flow Belief vision history has invalid shape")
        if tuple(proprio_history.shape) != (
            batch,
            self.config.history_sample_count,
            16,
        ):
            raise ValueError("Flow Belief proprioception history has invalid shape")
        if tuple(remaining_actions.shape) != (batch, 25, 7):
            raise ValueError("Flow Belief remaining action buffer has invalid shape")
        if tuple(latency_probabilities.shape) != (batch, 20):
            raise ValueError("Flow Belief latency law has invalid shape")
        if not torch.allclose(
            latency_probabilities.sum(dim=-1),
            torch.ones(batch, device=latency_probabilities.device),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Flow Belief latency probabilities must sum to one")

        vision = self.patch_projection(vision_history.float())
        vision = vision + self.spatial_position[None, None, None, :, :]
        vision = vision + self.camera_embedding[None, None, :, None, :]
        vision = vision + self.time_embedding[None, :, None, None, :]
        vision = self.patch_norm(vision)
        scores = torch.einsum(
            "bkcpe,cqe->bkcqp",
            vision,
            self.patch_pool_queries,
        ) / math.sqrt(self.config.model_dim)
        weights = torch.softmax(scores, dim=-1)
        vision_tokens = torch.einsum("bkcqp,bkcpe->bkcqe", weights, vision)
        vision_tokens = vision_tokens.flatten(start_dim=1, end_dim=3)
        vision_tokens = vision_tokens + self.type_embedding[0]

        proprio_tokens = self.proprio_projection(proprio_history.float())
        proprio_tokens = proprio_tokens + self.time_embedding[None]
        proprio_tokens = proprio_tokens + self.type_embedding[1]

        action_tokens = self.action_projection(remaining_actions.float())
        action_tokens = action_tokens + self.action_position[None]
        action_tokens = action_tokens + self.type_embedding[2]

        delay_coordinate = torch.linspace(
            1.0 / 20.0,
            1.0,
            20,
            device=latency_probabilities.device,
            dtype=latency_probabilities.dtype,
        ).expand(batch, -1)
        latency_tokens = self.latency_projection(
            torch.stack((latency_probabilities, delay_coordinate), dim=-1).float()
        )
        latency_tokens = latency_tokens + self.type_embedding[3]

        fused = self.fusion(
            torch.cat(
                (vision_tokens, proprio_tokens, action_tokens, latency_tokens),
                dim=1,
            )
        )
        fused = self.fusion_norm(fused)
        queries = self.belief_queries[None].expand(batch, -1, -1)
        attended, _ = self.belief_attention(queries, fused, fused, need_weights=False)
        belief = self.belief_norm(queries + attended)
        return self.output_norm(belief + self.belief_feedforward(belief))
