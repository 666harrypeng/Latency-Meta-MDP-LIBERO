"""Inference-time RTC guidance in OpenPI's t=1 noise, t=0 data convention.

The action buffer is transformed by the policy's own input normalization before
entering this module. No action labels, realized future delay or RGB fusion live here.
"""

from __future__ import annotations

import numpy as np


def build_rtc_weights(
    *,
    valid_mask,
    estimated_delay_ticks: int,
    active_action_dim: int,
    model_action_dim: int,
):
    """RTC Eq.5: hard-weighted prefix, exponential overlap taper, no padded guidance."""
    mask = np.asarray(valid_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("RTC buffer mask must be a boolean vector")
    overlap = int(mask.sum())
    if not np.array_equal(mask, np.arange(len(mask)) < overlap):
        raise ValueError("RTC buffer must have a contiguous valid prefix")
    if (
        type(estimated_delay_ticks) is not int
        or not 0 <= estimated_delay_ticks <= overlap
        or type(active_action_dim) is not int
        or type(model_action_dim) is not int
        or not 0 < active_action_dim <= model_action_dim
    ):
        raise ValueError("RTC estimate exceeds available controls or action dimensions are invalid")
    index = np.arange(len(mask))
    c = np.clip((overlap - index) / (overlap - estimated_delay_ticks + 1), 0, 1)
    temporal = c * np.expm1(c) / np.expm1(1.0)
    temporal[index < estimated_delay_ticks] = 1.0
    temporal[index >= overlap] = 0.0
    weights = np.zeros((len(mask), model_action_dim), dtype=np.float32)
    weights[:, :active_action_dim] = temporal[:, None]
    return weights


def estimate_clean_actions(x_t, time, velocity):
    return x_t - time * velocity


def rtc_guided_velocity(
    velocity_fn,
    x_t,
    time,
    previous_actions,
    weights,
    *,
    max_guidance_weight,
):
    """Use one denoiser VJP with the correct sign for integration toward decreasing t.

    With paper time u=1-t, the correction coefficient becomes
    ((1-t)^2+t^2)/(t*(1-t)). Clipping handles both endpoints. Zero guidance takes
    the ordinary velocity branch, including when the weight is a traced scalar.
    """
    import jax
    import jax.numpy as jnp

    if previous_actions.shape != x_t.shape or weights.shape != x_t.shape:
        raise ValueError("normalized RTC actions and weights must match the noisy action tensor")

    def native(_):
        return velocity_fn(x_t).astype(x_t.dtype)

    def guided(_):
        def clean_and_velocity(value):
            velocity = velocity_fn(value).astype(value.dtype)
            return estimate_clean_actions(value, time, velocity), velocity

        clean, pullback, velocity = jax.vjp(clean_and_velocity, x_t, has_aux=True)
        residual = (previous_actions - clean) * weights
        correction = pullback(residual)[0]
        coefficient = jnp.minimum(
            max_guidance_weight,
            (time**2 + (1 - time) ** 2) / jnp.maximum(time * (1 - time), 1e-8),
        )
        return velocity - coefficient * correction

    return jax.lax.cond(max_guidance_weight > 0, guided, native, operand=None)
