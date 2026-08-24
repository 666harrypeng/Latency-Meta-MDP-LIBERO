"""Deterministic fixed-step ODE solvers for Flow Belief sampling."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def integrate_flow_ode(
    *,
    initial_state: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    solver: str,
    step_count: int,
) -> torch.Tensor:
    if initial_state.ndim < 2 or initial_state.shape[-1] != 22:
        raise ValueError("initial Flow state must end with a 22D state axis")
    if solver not in {"euler", "heun"}:
        raise ValueError("Flow solver must be euler or heun")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
        raise ValueError("Flow solver step_count must be a positive integer")
    state = initial_state
    dt = 1.0 / step_count
    time_shape = state.shape[:-1]
    for step in range(step_count):
        time = torch.full(
            time_shape,
            step * dt,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = velocity_fn(state, time)
        if velocity.shape != state.shape:
            raise ValueError("Flow velocity function returned an invalid shape")
        if solver == "euler":
            state = state + dt * velocity
            continue
        predicted = state + dt * velocity
        next_time = torch.full(
            time_shape,
            (step + 1) * dt,
            device=state.device,
            dtype=state.dtype,
        )
        next_velocity = velocity_fn(predicted, next_time)
        if next_velocity.shape != state.shape:
            raise ValueError("Flow velocity function returned an invalid Heun shape")
        state = state + 0.5 * dt * (velocity + next_velocity)
    return state


def sample_flow_belief(
    *,
    vector_field: nn.Module,
    belief_tokens: torch.Tensor,
    delay_ticks: torch.Tensor,
    noise: torch.Tensor,
    solver: str,
    step_count: int,
) -> torch.Tensor:
    if noise.ndim != 4 or noise.shape[-1] != 22:
        raise ValueError("Flow sampling noise must have shape [B, R, S, 22]")
    batch, query_count, sample_count, _ = noise.shape
    if tuple(delay_ticks.shape) != (batch, query_count):
        raise ValueError("Flow sampling delay ticks do not match noise queries")
    if belief_tokens.ndim != 3 or belief_tokens.shape[0] != batch:
        raise ValueError("Flow sampling belief tokens do not match noise batch")
    flat_state = noise.reshape(batch, query_count * sample_count, 22)
    flat_delay = (
        delay_ticks[:, :, None]
        .expand(batch, query_count, sample_count)
        .reshape(batch, query_count * sample_count)
    )

    def velocity_fn(state: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        return vector_field(
            noisy_state=state,
            flow_time=flow_time,
            belief_tokens=belief_tokens,
            delay_ticks=flat_delay,
        )

    result = integrate_flow_ode(
        initial_state=flat_state,
        velocity_fn=velocity_fn,
        solver=solver,
        step_count=step_count,
    )
    return result.reshape(batch, query_count, sample_count, 22)
