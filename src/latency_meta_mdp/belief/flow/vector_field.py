"""Joint conditional vector field for normalized 22D future states."""

from __future__ import annotations

import math

import torch
from torch import nn

from latency_meta_mdp.belief.flow.config import FlowBeliefConfig


def _fourier_features(value: torch.Tensor, frequency_count: int) -> torch.Tensor:
    frequencies = 2.0 ** torch.arange(
        frequency_count,
        device=value.device,
        dtype=torch.float32,
    )
    phase = 2.0 * math.pi * value.float()[..., None] * frequencies
    return torch.cat(
        (value.float()[..., None], torch.sin(phase), torch.cos(phase)),
        dim=-1,
    )


class ResidualFiLMBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(condition_dim, 2 * hidden_dim)
        self.update = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(condition).chunk(2, dim=-1)
        modulated = self.norm(hidden) * (1.0 + torch.tanh(scale)) + shift
        return hidden + self.update(modulated)


class ConditionalStateVectorField(nn.Module):
    """Predict one joint velocity over all normalized state dimensions."""

    def __init__(self, config: FlowBeliefConfig) -> None:
        super().__init__()
        self.config = config
        condition_dim = config.model_dim
        hidden_dim = config.flow_hidden_dim
        time_feature_dim = 1 + 2 * config.flow_time_fourier_frequency_count
        delay_feature_dim = 1 + 2 * config.delay_fourier_frequency_count
        self.state_embedding = nn.Sequential(
            nn.Linear(22, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_condition = nn.Linear(22, condition_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(time_feature_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.delay_embedding = nn.Sequential(
            nn.Linear(delay_feature_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.belief_attention = nn.MultiheadAttention(
            embed_dim=condition_dim,
            num_heads=config.transformer_head_count,
            dropout=config.dropout,
            batch_first=True,
        )
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.blocks = nn.ModuleList(
            ResidualFiLMBlock(
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                dropout=config.dropout,
            )
            for _ in range(config.flow_residual_block_count)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 22),
        )

    def forward(
        self,
        *,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        belief_tokens: torch.Tensor,
        delay_ticks: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_state.ndim != 3 or noisy_state.shape[-1] != 22:
            raise ValueError("Flow noisy state must have shape [B, R, 22]")
        batch, query_count, _ = noisy_state.shape
        if tuple(flow_time.shape) != (batch, query_count):
            raise ValueError("Flow time must match state batch and query axes")
        if torch.any(flow_time < 0.0) or torch.any(flow_time > 1.0):
            raise ValueError("Flow time must lie in [0, 1]")
        if tuple(delay_ticks.shape) != (batch, query_count):
            raise ValueError("Flow delay ticks must match state batch and query axes")
        if torch.any(delay_ticks < 1) or torch.any(delay_ticks > 20):
            raise ValueError("Flow delay ticks must lie in [1, 20]")
        if tuple(belief_tokens.shape) != (
            batch,
            self.config.belief_token_count,
            self.config.model_dim,
        ):
            raise ValueError("Flow belief tokens have invalid shape")
        query = self.state_condition(noisy_state.float())
        query = query + self.time_embedding(
            _fourier_features(flow_time, self.config.flow_time_fourier_frequency_count)
        )
        query = query + self.delay_embedding(
            _fourier_features(
                delay_ticks.float() / 20.0,
                self.config.delay_fourier_frequency_count,
            )
        )
        attended, _ = self.belief_attention(
            query,
            belief_tokens,
            belief_tokens,
            need_weights=False,
        )
        condition = self.condition_norm(query + attended)
        hidden = self.state_embedding(noisy_state.float())
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)
