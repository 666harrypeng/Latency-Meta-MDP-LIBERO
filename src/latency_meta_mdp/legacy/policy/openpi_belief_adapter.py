"""Policy-owned, probability-aware return memory for the pinned JAX pi0.5 model.

The interface preserves all five anchors. It is a consumer of frozen JEPA outputs,
not another Belief model. Memory is encoded once and reused by every flow step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

_WIDTH = 384
_HEADS = 8
_QUERIES = 8


def _heads(x):
    return x.reshape(*x.shape[:-1], _HEADS, _WIDTH // _HEADS).swapaxes(-3, -2)


def _attend(query, key, value, log_mass=None):
    logits = jnp.einsum("...hqf,...hkf->...hqk", query, key) / (_WIDTH // _HEADS) ** 0.5
    if log_mass is not None:
        logits = logits + log_mass[:, None, None, :]
    result = jnp.einsum("...hqk,...hkf->...hqf", jax.nn.softmax(logits, axis=-1), value)
    return result.swapaxes(-3, -2).reshape(*result.shape[:-3], result.shape[-2], _WIDTH)


class ReturnBeliefAdapter(nnx.Module):
    def __init__(self, *, action_width: int, rngs: nnx.Rngs):
        self.visual_norm = nnx.LayerNorm(_WIDTH, dtype=jnp.float32, rngs=rngs)
        self.memory_norm = nnx.LayerNorm(_WIDTH, dtype=jnp.float32, rngs=rngs)
        self.hidden_norm = nnx.LayerNorm(action_width, dtype=jnp.float32, rngs=rngs)
        self.positions = nnx.Param(jax.random.normal(rngs.params(), (2, 196, _WIDTH)) * 0.02)
        self.queries = nnx.Param(jax.random.normal(rngs.params(), (_QUERIES, _WIDTH)) * 0.02)
        self.visual_key = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.visual_value = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.resample_out = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.proprio = nnx.Linear(16, _WIDTH, rngs=rngs)
        self.delay = nnx.Linear(1, _WIDTH, rngs=rngs)
        self.memory_key = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.memory_value = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.action_query = nnx.Linear(action_width, _WIDTH, rngs=rngs)
        self.action_out = nnx.Linear(_WIDTH, action_width, rngs=rngs)
        self.gate = nnx.Param(jnp.zeros((), jnp.float32))

    def encode(self, visual, proprio, delays, probabilities):
        """Encode normalized future proprio/latents; return cached K/V and log mass."""
        batch = visual.shape[0]
        if (
            visual.shape != (batch, 5, 2, 196, 384)
            or proprio.shape != (batch, 5, 16)
            or delays.shape != (batch, 5)
            or probabilities.shape != (batch, 5)
        ):
            raise ValueError(
                "return memory requires five dual-camera stride4 anchors and current16"
            )
        visual, proprio, delays, probabilities = jax.tree.map(
            jax.lax.stop_gradient, (visual, proprio, delays, probabilities)
        )
        spatial = self.visual_norm(visual) + self.positions.value
        spatial = spatial.reshape(batch, 5, 392, _WIDTH)
        queries = jnp.broadcast_to(self.queries.value, (batch, 5, _QUERIES, _WIDTH))
        resampled = queries + self.resample_out(
            _attend(
                _heads(queries),
                _heads(self.visual_key(spatial)),
                _heads(self.visual_value(spatial)),
            )
        )
        resampled = resampled + self.proprio(proprio)[:, :, None, :]
        resampled = resampled + self.delay(delays[..., None] / 20.0)[:, :, None, :]
        memory = self.memory_norm(resampled).reshape(batch, 5 * _QUERIES, _WIDTH)
        # The resampler is independent of PMF. Its only consumer weighting is
        # exact anchor mass shared across the eight separately retained slots.
        log_mass = jnp.where(
            probabilities > 0,
            jnp.log(jnp.where(probabilities > 0, probabilities, 1.0)) - jnp.log(float(_QUERIES)),
            -jnp.inf,
        )
        log_mass = jnp.repeat(log_mass, _QUERIES, axis=-1)
        return _heads(self.memory_key(memory)), _heads(self.memory_value(memory)), log_mass

    def __call__(self, hidden, memory):
        keys, values, log_mass = memory
        attended = _attend(
            _heads(self.action_query(self.hidden_norm(hidden))), keys, values, log_mass
        )
        residual = (jnp.tanh(self.gate.value) * self.action_out(attended)).astype(hidden.dtype)
        return hidden + residual


@dataclasses.dataclass(frozen=True)
class NativePolicyWithReturnBeliefLoader:
    """Keep adapter initialization while requiring the complete native parameter tree."""

    native_loader: Any
    parameter_key: str = "return_belief_adapter"

    def load(self, params):
        from openpi.shared import array_typing

        if self.parameter_key not in {"return_belief_adapter", "return_belief_prefix"}:
            raise ValueError("unknown policy-owned Belief parameter tree")
        if self.parameter_key not in params:
            raise ValueError("conditioned weight loading requires an adapter parameter tree")
        native = {key: value for key, value in params.items() if key != self.parameter_key}
        loaded = self.native_loader.load(native)
        array_typing.check_pytree_equality(
            expected=native,
            got=loaded,
            check_shapes=True,
            check_dtypes=True,
        )
        return {**loaded, self.parameter_key: params[self.parameter_key]}
