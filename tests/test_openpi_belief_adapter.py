"""Contract tests for the policy-owned return memory; run in the OpenPI environment."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture
def patched_openpi():
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    patches = tuple(
        Path("patches/openpi") / name
        for name in (
            "0001-filter-incomplete-action-chunks.patch",
            "0003-mask-action-tails.patch",
            "0004-return-belief-side-memory.patch",
        )
        if (Path("patches/openpi") / name).exists()
    )
    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=patches,
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root
        sys.path.remove(str(root / "src"))


def _inputs():
    import jax
    import jax.numpy as jnp

    visual = jax.random.normal(jax.random.key(1), (1, 5, 2, 196, 384)) * 0.05
    proprio = jax.random.normal(jax.random.key(2), (1, 5, 16))
    delays = jnp.array([[4, 8, 12, 16, 20]], jnp.float32)
    probabilities = jnp.array([[0.5, 0.3, 0.1, 0.1, 0]], jnp.float32)
    return visual, proprio, delays, probabilities


def _adapter():
    import flax.nnx as nnx

    from latency_meta_mdp.legacy.policy.openpi_belief_adapter import ReturnBeliefAdapter

    return ReturnBeliefAdapter(action_width=64, rngs=nnx.Rngs(7))


def test_zero_gate_is_exact_native_and_memory_retains_five_anchors():
    import jax.numpy as jnp

    adapter = _adapter()
    memory = adapter.encode(*_inputs())
    hidden = jnp.arange(50 * 64, dtype=jnp.float32).reshape(1, 50, 64)
    np.testing.assert_array_equal(adapter(hidden, memory), hidden)
    assert memory[0].shape == memory[1].shape == (1, 8, 40, 48)
    mass = jnp.exp(memory[2]).reshape(1, 5, 8).sum(-1)
    np.testing.assert_allclose(mass, _inputs()[-1], atol=1e-7)


def test_memory_does_not_inflate_small_positive_probability_mass():
    import jax.numpy as jnp

    visual, proprio, delays, pmf = _inputs()
    memory = _adapter().encode(visual, proprio, delays, pmf.at[:, 4].set(1e-35))
    np.testing.assert_allclose(
        jnp.exp(memory[2]).reshape(1, 5, 8).sum(-1)[0, 4], 1e-35, rtol=1e-5, atol=0
    )


def test_conditioned_config_builds_real_and_abstract_observations(patched_openpi):
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, active_action_dim=7, use_return_belief=True)
    abstract, _ = config.inputs_spec(batch_size=2)
    assert abstract.return_belief_visual.shape == (2, 5, 2, 196, 384)
    concrete = config.fake_obs(1)
    assert concrete.return_belief_proprio.shape == (1, 5, 16)
    with pytest.raises(ValueError, match="state-aware"):
        Pi0Config(
            pi05=True, active_action_dim=7, discrete_state_input=False, use_return_belief=True
        )


def test_nonzero_gate_responds_to_pmf_but_ignores_zero_mass_anchor():
    import jax.numpy as jnp

    adapter = _adapter()
    adapter.gate.value = jnp.array(0.5)
    visual, proprio, delays, pmf = _inputs()
    hidden = jnp.ones((1, 50, 64))
    original = adapter(hidden, adapter.encode(visual, proprio, delays, pmf))
    changed_pmf = pmf.at[:, 0].set(0.2).at[:, 1].set(0.6)
    changed = adapter(hidden, adapter.encode(visual, proprio, delays, changed_pmf))
    assert np.max(np.abs(original - changed)) > 1e-5
    zero_changed = adapter(
        hidden, adapter.encode(visual.at[:, 4].set(100), proprio.at[:, 4].set(100), delays, pmf)
    )
    np.testing.assert_allclose(zero_changed, original, atol=1e-6)


def test_paired_anchor_permutation_does_not_change_the_conditioned_result():
    import jax.numpy as jnp

    adapter = _adapter()
    adapter.gate.value = jnp.array(0.5)
    fields = _inputs()
    order = jnp.array([2, 0, 4, 1, 3])
    hidden = jnp.ones((1, 50, 64))
    original = adapter(hidden, adapter.encode(*fields))
    permuted = adapter(hidden, adapter.encode(*(x[:, order] for x in fields)))
    np.testing.assert_allclose(permuted, original, atol=1e-6)


def test_belief_input_gradients_are_blocked_and_adapter_parameters_can_learn():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    adapter = _adapter()
    adapter.gate.value = jnp.array(0.5)
    fields = _inputs()
    hidden = jnp.ones((1, 50, 64))

    def objective(*inputs):
        return jnp.square(adapter(hidden, adapter.encode(*inputs))).mean()

    grads = jax.grad(objective, argnums=(0, 1, 2, 3))(*fields)
    for grad in grads:
        np.testing.assert_array_equal(grad, 0)

    def parameter_loss(model):
        return jnp.square(model(hidden, model.encode(*fields))).mean()

    _, params_grad = nnx.value_and_grad(parameter_loss)(adapter)
    assert sum(float(jnp.sum(jnp.square(x))) for x in jax.tree.leaves(params_grad)) > 0


def test_sampling_encodes_memory_once_and_keeps_native_prefix_protocol(patched_openpi):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    from openpi.models.pi0 import Pi0

    calls = {"encode": 0, "prefix": 0}

    class Memory:
        def encode(self, *fields):
            calls["encode"] += 1
            return jnp.array(0.0)

        def __call__(self, hidden, memory):
            return hidden + memory

    class Policy:
        action_horizon = 3
        active_action_dim = 2
        return_belief_adapter = Memory()
        action_out_proj = staticmethod(lambda x: x)

        def embed_prefix(self, observation):
            calls["prefix"] += 1
            return jnp.ones((1, 2, 4)), jnp.ones((1, 2), bool), jnp.array([False, False])

        def embed_suffix(self, obs, actions, time):
            return actions, jnp.ones((1, 3), bool), jnp.array([True, False, False]), None

    def llm(tokens, *, mask, positions, **kwargs):
        if tokens[0] is not None:
            np.testing.assert_array_equal(positions, [[0, 1]])
            assert mask.shape == (1, 2, 2)
        else:
            assert mask.shape == (1, 3, 5)
        return (tokens[0], tokens[1]), None

    policy = Policy()
    policy.PaliGemma = SimpleNamespace(llm=llm)
    visual, proprio, delays, probabilities = _inputs()
    obs = Observation(
        images={
            key: jnp.zeros((1, 224, 224, 3))
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        image_masks={},
        state=jnp.zeros((1, 4)),
        return_belief_visual=visual,
        return_belief_proprio=proprio,
        return_belief_delay_ticks=delays,
        return_belief_probabilities=probabilities,
    )
    noise = jnp.ones((1, 3, 4))
    conditioned = Pi0.sample_actions(policy, jax.random.key(4), obs, noise=noise, num_steps=5)
    assert calls == {"encode": 1, "prefix": 1}
    policy.return_belief_adapter = None
    native = Pi0.sample_actions(policy, jax.random.key(4), obs, noise=noise, num_steps=5)
    np.testing.assert_array_equal(native, conditioned)


def test_weight_loader_allows_only_new_adapter_weights(patched_openpi):
    from latency_meta_mdp.legacy.policy.openpi_belief_adapter import (
        NativePolicyWithReturnBeliefLoader,
    )

    native = {"action_out_proj": {"kernel": np.ones((4, 7), np.float32)}}
    adapter = {"gate": np.zeros((), np.float32)}
    reference = {**native, "return_belief_adapter": adapter}
    loader = NativePolicyWithReturnBeliefLoader(SimpleNamespace(load=lambda params: native))
    loaded = loader.load(reference)
    np.testing.assert_array_equal(loaded["action_out_proj"]["kernel"], 1)
    assert loaded["return_belief_adapter"] is adapter
    with pytest.raises((ValueError, AssertionError)):
        NativePolicyWithReturnBeliefLoader(SimpleNamespace(load=lambda params: {})).load(reference)


def test_belief_inputs_use_clean_state_statistics_and_reject_hidden_delay(patched_openpi):
    import jax
    from openpi.models.model import ModelType, Observation, preprocess_observation
    from openpi.shared.normalize import NormStats

    from latency_meta_mdp.legacy.policy.openpi_belief_data import ReturnBeliefInputs

    stats = NormStats(mean=np.zeros(16), std=np.ones(16), q01=np.full(16, -2), q99=np.full(16, 2))
    transform = ReturnBeliefInputs(model_type=ModelType.PI05, state_norm_stats=stats)
    visual, proprio, delays, pmf = map(lambda x: np.asarray(x)[0], _inputs())
    belief = {
        "visual": visual.astype(np.float16),
        "proprio": proprio,
        "delay_ticks": delays,
        "probabilities": pmf,
    }
    raw = {
        "observation/state": np.zeros(16, np.float32),
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "prompt": "Grasp the moving ball and lift it.",
        "return_belief": belief,
    }
    out = transform(raw)
    np.testing.assert_allclose(
        out["return_belief_proprio"], (proprio + 2) / (4 + 1e-6) * 2 - 1, atol=1e-6
    )
    out.pop("prompt")
    obs = Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None], out))
    processed = preprocess_observation(None, obs, train=False)
    np.testing.assert_array_equal(processed.return_belief_visual, obs.return_belief_visual)
    with pytest.raises(ValueError, match="fields"):
        transform({**raw, "return_belief": {**belief, "realized_delay": 4}})
    with pytest.raises(ValueError, match="probabilities"):
        transform({**raw, "return_belief": {**belief, "probabilities": np.zeros(5)}})


def test_conditioned_data_transforms_preserve_native_tokens_and_action_mask(
    patched_openpi, tmp_path
):
    import dataclasses

    from openpi import transforms
    from openpi.shared import normalize

    from latency_meta_mdp.legacy.policy.openpi_belief_data import ReturnBeliefDataConfig
    from latency_meta_mdp.policy.openpi.training import _build_config
    from latency_meta_mdp.policy.profile import load_sft_profile

    profile = load_sft_profile(Path("configs/contracts/policy/pi05_state16_h50.yaml"))
    clean = _build_config(profile, 3)

    def stats(n):
        return normalize.NormStats(
            mean=np.zeros(n), std=np.ones(n), q01=np.full(n, -2), q99=np.full(n, 2)
        )

    assets = tmp_path / "assets"
    normalize.save(assets / clean.data.repo_id, {"state": stats(16), "actions": stats(7)})
    conditioned = ReturnBeliefDataConfig(
        repo_id=clean.data.repo_id, base_config=clean.data.base_config
    )
    data = conditioned.create(assets, dataclasses.replace(clean.model, use_return_belief=True))
    native_data = clean.data.create(assets, clean.model)

    def pipeline(config):
        return transforms.compose(
            [
                *config.repack_transforms.inputs,
                *config.data_transforms.inputs,
                transforms.Normalize(config.norm_stats, use_quantiles=config.use_quantile_norm),
                *config.model_transforms.inputs,
            ]
        )

    fields = dict(
        zip(
            ("visual", "proprio", "delay_ticks", "probabilities"),
            (np.asarray(x)[0] for x in _inputs()),
            strict=True,
        )
    )
    sample = {
        "image": np.zeros((224, 224, 3), np.uint8),
        "wrist_image": np.zeros((224, 224, 3), np.uint8),
        "state": np.zeros(16, np.float32),
        "actions": np.ones((50, 7), np.float32),
        "actions_is_pad": np.arange(50) >= 3,
        "prompt": "Grasp the moving ball and lift it.",
        "return_belief": fields,
    }
    output = pipeline(data)(sample)
    baseline = pipeline(native_data)(sample)
    for key in (
        "state",
        "actions",
        "action_loss_mask",
        "tokenized_prompt",
        "tokenized_prompt_mask",
    ):
        np.testing.assert_array_equal(output[key], baseline[key])
    assert output["return_belief_visual"].shape == (5, 2, 196, 384)
    assert output["return_belief_proprio"].dtype == np.float32
