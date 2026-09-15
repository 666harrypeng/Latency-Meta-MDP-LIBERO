"""Joint Flow-specific Belief Encoder and conditional vector field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from latency_meta_mdp.legacy.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.legacy.belief.flow.encoder import FlowBeliefEncoder
from latency_meta_mdp.legacy.belief.flow.vector_field import ConditionalStateVectorField


@dataclass(frozen=True)
class FlowMatchingBatch:
    noise: torch.Tensor
    flow_time: torch.Tensor
    noisy_state: torch.Tensor
    target_velocity: torch.Tensor


def build_flow_matching_batch(
    *,
    target_state: torch.Tensor,
    noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    if target_state.ndim != 3 or target_state.shape[-1] != 22:
        raise ValueError("target state must have shape [B, R, 22]")
    if noise is None:
        noise = torch.randn(
            target_state.shape,
            device=target_state.device,
            dtype=target_state.dtype,
            generator=generator,
        )
    if noise.shape != target_state.shape:
        raise ValueError("Flow noise and target state shapes must agree")
    if flow_time is None:
        flow_time = torch.rand(
            target_state.shape[:2],
            device=target_state.device,
            dtype=target_state.dtype,
            generator=generator,
        )
    if flow_time.shape != target_state.shape[:2]:
        raise ValueError("Flow time must match target batch and query axes")
    if torch.any(flow_time < 0.0) or torch.any(flow_time > 1.0):
        raise ValueError("Flow time must lie in [0, 1]")
    interpolation = flow_time[..., None]
    return FlowMatchingBatch(
        noise=noise,
        flow_time=flow_time,
        noisy_state=(1.0 - interpolation) * noise + interpolation * target_state,
        target_velocity=target_state - noise,
    )


class FlowBeliefModel(nn.Module):
    def __init__(self, config: FlowBeliefConfig) -> None:
        super().__init__()
        self.encoder = FlowBeliefEncoder(config)
        self.vector_field = ConditionalStateVectorField(config)

    def forward(
        self,
        *,
        vision_history: torch.Tensor,
        proprio_history: torch.Tensor,
        remaining_actions: torch.Tensor,
        latency_probabilities: torch.Tensor,
        noisy_state: torch.Tensor,
        flow_time: torch.Tensor,
        delay_ticks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        belief = self.encoder(
            vision_history=vision_history,
            proprio_history=proprio_history,
            remaining_actions=remaining_actions,
            latency_probabilities=latency_probabilities,
        )
        velocity = self.vector_field(
            noisy_state=noisy_state,
            flow_time=flow_time,
            belief_tokens=belief,
            delay_ticks=delay_ticks,
        )
        return velocity, belief


def conditional_flow_matching_loss(
    *,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError("predicted and target Flow velocities must have identical shapes")
    return torch.mean(torch.square(predicted_velocity - target_velocity))
