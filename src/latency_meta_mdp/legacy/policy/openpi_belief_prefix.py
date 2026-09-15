"""Typed forecast tokens inserted before the native pi0.5 VLM transformer."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from latency_meta_mdp.legacy.policy.openpi_belief_adapter import _attend, _heads

_WIDTH = 384
_ANCHORS = 5
_VIEWS = 2
_PATCHES = 196


class ReturnBeliefPrefixEncoder(nnx.Module):
    """Resample each forecast view independently; native VLM performs their fusion."""

    def __init__(self, *, vlm_width: int, queries_per_view: int = 4, rngs: nnx.Rngs):
        if type(queries_per_view) is not int or not 1 <= queries_per_view <= _PATCHES:
            raise ValueError("queries_per_view must be an integer from1 to196")
        if type(vlm_width) is not int or vlm_width <= 0:
            raise ValueError("vlm_width must be positive")
        self.vlm_width = vlm_width
        self.queries_per_view = queries_per_view
        self.visual_norm = nnx.LayerNorm(_WIDTH, dtype=jnp.float32, rngs=rngs)
        self.resample_norm = nnx.LayerNorm(_WIDTH, dtype=jnp.float32, rngs=rngs)
        self.spatial = nnx.Param(jax.random.normal(rngs.params(), (14, 14, _WIDTH)) * 0.02)
        self.queries = nnx.Param(
            jax.random.normal(rngs.params(), (queries_per_view, _WIDTH)) * 0.02
        )
        self.query = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.key = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.value = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.resample_out = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs)
        self.visual_up = nnx.Linear(_WIDTH, vlm_width, rngs=rngs)
        self.state = nnx.Linear(16, vlm_width, rngs=rngs)
        self.time = nnx.Linear(1, vlm_width, rngs=rngs)
        self.mass = nnx.Linear(1, vlm_width, rngs=rngs)
        self.law = nnx.Linear(22, vlm_width, rngs=rngs)
        self.types = nnx.Param(jax.random.normal(rngs.params(), (3, vlm_width)) * 0.02)
        self.cameras = nnx.Param(jax.random.normal(rngs.params(), (2, vlm_width)) * 0.02)
        self.slots = nnx.Param(
            jax.random.normal(rngs.params(), (queries_per_view, vlm_width)) * 0.02
        )

    @property
    def token_count(self):
        return 1 + _ANCHORS * (1 + _VIEWS * self.queries_per_view)

    def __call__(
        self, visual, proprio, delays, macro_probabilities, latency_probabilities, known_delay
    ):
        batch = visual.shape[0]
        expected = (
            (batch, _ANCHORS, _VIEWS, _PATCHES, _WIDTH),
            (batch, _ANCHORS, 16),
            (batch, _ANCHORS),
            (batch, _ANCHORS),
            (batch, 20),
            (batch, 2),
        )
        fields = (visual, proprio, delays, macro_probabilities, latency_probabilities, known_delay)
        if any(value.shape != shape for value, shape in zip(fields, expected, strict=True)):
            raise ValueError("forecast prefix fields do not match the five-anchor/D20 contract")
        visual, proprio, delays, macro_probabilities, latency_probabilities, known_delay = (
            jax.tree.map(jax.lax.stop_gradient, fields)
        )
        spatial = self.visual_norm(visual) + self.spatial.value.reshape(_PATCHES, _WIDTH)
        queries = jnp.broadcast_to(
            self.queries.value,
            (batch, _ANCHORS, _VIEWS, self.queries_per_view, _WIDTH),
        )
        resampled = queries + self.resample_out(
            _attend(
                _heads(self.query(queries)), _heads(self.key(spatial)), _heads(self.value(spatial))
            )
        )
        time = self.time(delays[..., None] / 20.0)
        visual_tokens = (
            self.visual_up(self.resample_norm(resampled))
            + self.types.value[0]
            + self.cameras.value[None, None, :, None, :]
            + self.slots.value[None, None, None, :, :]
            + time[:, :, None, None, :]
        )
        state_tokens = (
            self.state(proprio)
            + time
            + self.mass(macro_probabilities[..., None])
            + self.types.value[1]
        )
        anchor_tokens = jnp.concatenate(
            [
                state_tokens[:, :, None, :],
                visual_tokens.reshape(batch, _ANCHORS, -1, self.vlm_width),
            ],
            axis=2,
        ).reshape(batch, -1, self.vlm_width)
        law_token = self.law(jnp.concatenate([latency_probabilities, known_delay], axis=-1))
        law_token = law_token + self.types.value[2]
        return jnp.concatenate([law_token[:, None, :], anchor_tokens], axis=1)
