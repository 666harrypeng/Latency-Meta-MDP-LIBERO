from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module", autouse=True)
def patched_openpi():
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    names = (
        "0001-filter-incomplete-action-chunks.patch",
        "0003-mask-action-tails.patch",
        "0004-return-belief-side-memory.patch",
        "0005-return-policy-data-view.patch",
    )
    patches = tuple(
        Path("patches/openpi") / name for name in names if (Path("patches/openpi") / name).exists()
    )
    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=patches,
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root
        sys.path.remove(str(root / "src"))


def test_importance_weight_scales_the_real_loss_and_is_not_a_sampling_input():
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0
    from test_openpi_masked_actions import _loss_fixture

    model, obs = _loss_fixture(jnp.ones((2, 3, 4), bool))
    actions = jnp.ones((2, 3, 4))
    weighted = dataclasses.replace(obs, action_loss_weight=jnp.array([0.25, 2.0]))
    plain = Pi0.compute_loss(model, jax.random.key(1), obs, actions)
    actual = Pi0.compute_loss(model, jax.random.key(1), weighted, actions)
    np.testing.assert_allclose(actual, plain * np.array([[0.25], [2.0]]))
    model.PaliGemma = SimpleNamespace(llm=lambda tokens, **kw: ((tokens[0], tokens[1]), None))
    model.active_action_dim = 2
    a = Pi0.sample_actions(model, jax.random.key(1), obs, noise=actions, num_steps=2)
    b = Pi0.sample_actions(model, jax.random.key(1), weighted, noise=actions, num_steps=2)
    np.testing.assert_array_equal(a, b)


def _raw():
    return {
        "observation/state": np.zeros(16, np.float32),
        "observation/image": np.zeros((8, 8, 3), np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
        "actions": np.zeros((50, 7), np.float32),
        "actions_is_pad": np.ones(50, bool),
        "action_loss_weight": np.float32(1.5),
        "prompt": "Grasp the moving ball and lift it.",
    }


def _stats():
    from openpi.shared.normalize import NormStats

    return NormStats(mean=np.zeros(16), std=np.ones(16), q01=np.full(16, -1), q99=np.ones(16))


def test_post_terminal_return_target_is_zero_loss_but_native_source_still_rejects_it():
    from openpi.models.model import ModelType

    from latency_meta_mdp.legacy.policy.openpi_belief_data import ReturnBeliefInputs
    from latency_meta_mdp.policy.openpi.data import StructuredPolicyInputs

    raw = {
        **_raw(),
        "return_belief": {
            "visual": np.zeros((5, 2, 196, 384), np.float16),
            "proprio": np.zeros((5, 16), np.float32),
            "delay_ticks": np.array([4, 8, 12, 16, 20]),
            "probabilities": np.full(5, 0.2, np.float32),
        },
    }
    transformed = ReturnBeliefInputs(model_type=ModelType.PI05, state_norm_stats=_stats())(raw)
    assert not transformed["action_loss_mask"].any()
    assert transformed["action_loss_weight"] == 1.5
    with pytest.raises(ValueError, match="real action"):
        StructuredPolicyInputs(model_type=ModelType.PI05)(raw)


def test_exact_delay_oracle_has_an_explicit_separate_input_route():
    from openpi.models.model import ModelType

    from latency_meta_mdp.legacy.policy.openpi_belief_data import (
        KnownDelayOracleInputs,
        ReturnBeliefInputs,
    )

    raw = {
        **_raw(),
        "known_delay_oracle": {
            "visual": np.ones((2, 196, 384), np.float16),
            "proprio": np.zeros(16, np.float32),
            "known_delay_ticks": 1,
        },
    }
    out = KnownDelayOracleInputs(model_type=ModelType.PI05, state_norm_stats=_stats())(raw)
    np.testing.assert_array_equal(out["return_belief_delay_ticks"], [1] * 5)
    np.testing.assert_array_equal(out["return_belief_probabilities"], [1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="fields"):
        ReturnBeliefInputs(model_type=ModelType.PI05, state_norm_stats=_stats())(raw)


def test_in_process_policy_bridge_packs_current16_and_uses_explicit_noise_stream():
    from openpi.policies.policy import Policy

    from latency_meta_mdp.runtime.policy_execution import InProcessOpenpiPolicy, PolicyObservation

    class LocalPolicy(Policy):
        def __init__(self):
            self.received = None

        def infer(self, obs, *, noise=None):
            self.received = obs
            return {"actions": noise[:, :7]}

    observation = PolicyObservation(
        formal_tick=4,
        image=np.zeros((8, 8, 3), np.uint8),
        wrist_image=np.zeros((8, 8, 3), np.uint8),
        state=np.zeros(16, np.float32),
    )
    first = LocalPolicy()
    second = LocalPolicy()
    a = InProcessOpenpiPolicy(first, noise_rng=np.random.default_rng(7))
    b = InProcessOpenpiPolicy(second, noise_rng=np.random.default_rng(7))
    belief = {"fixture": 1}
    np.testing.assert_array_equal(
        a(observation, belief)["actions"], b(observation, None)["actions"]
    )
    assert first.received["return_belief"] is belief
    assert "return_belief" not in second.received
    with pytest.raises(TypeError, match="in-process"):
        InProcessOpenpiPolicy(object(), noise_rng=np.random.default_rng(7))


def test_return_control_configs_freeze_native_weights_and_keep_clean_assets():
    import flax.nnx as nnx

    from latency_meta_mdp.legacy.policy.configs import build_return_policy_train_config
    from latency_meta_mdp.legacy.policy.openpi_belief_data import (
        KnownDelayOracleDataConfig,
        NoFutureControlDataConfig,
    )
    from latency_meta_mdp.policy.openpi.training import _build_config
    from latency_meta_mdp.policy.profile import load_sft_profile

    clean = _build_config(
        load_sft_profile(Path("configs/contracts/policy/pi05_state16_h50.yaml")), 3
    )
    config = build_return_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/fixture/clean/params"),
        view_spec={"mode": "predicted_mixture"},
        experiment_name="adapter-control",
    )
    assert config.model.use_return_belief and config.model.discrete_state_input
    assert Path(config.data.assets.assets_dir) == clean.assets_dirs
    assert config.policy_metadata["adapter_only"] is True
    assert nnx.filterlib.to_predicate(config.trainable_filter)(
        ("return_belief_adapter", "gate"), nnx.Param(0.0)
    )
    assert not nnx.filterlib.to_predicate(config.trainable_filter)(
        ("action_out_proj", "kernel"), nnx.Param(0.0)
    )
    oracle = build_return_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/fixture/clean/params"),
        view_spec={"mode": "known_delay_oracle"},
        experiment_name="oracle-control",
    )
    assert isinstance(oracle.data, KnownDelayOracleDataConfig)
    assert oracle.policy_metadata["privileged_oracle"] is True
    control = build_return_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/fixture/clean/params"),
        view_spec={"mode": "no_future_control"},
        experiment_name="no-future-control",
    )
    assert isinstance(control.data, NoFutureControlDataConfig)
    assert control.policy_metadata["privileged_oracle"] is False
    assert control.trainable_filter == config.trainable_filter


def test_no_future_control_erases_future_content_but_keeps_time_and_mass():
    from openpi.models.model import ModelType

    from latency_meta_mdp.legacy.policy.openpi_belief_data import NoFutureControlInputs

    raw = {
        **_raw(),
        "return_belief": {
            "visual": np.ones((5, 2, 196, 384), np.float16),
            "proprio": np.ones((5, 16), np.float32),
            "delay_ticks": np.array([4, 8, 12, 16, 20]),
            "probabilities": np.array([0.5, 0.2, 0.1, 0.1, 0.1], np.float32),
        },
    }
    out = NoFutureControlInputs(model_type=ModelType.PI05, state_norm_stats=_stats())(raw)
    np.testing.assert_array_equal(out["return_belief_visual"], 0)
    np.testing.assert_array_equal(out["return_belief_proprio"], 0)
    np.testing.assert_array_equal(
        out["return_belief_delay_ticks"], raw["return_belief"]["delay_ticks"]
    )
    np.testing.assert_array_equal(
        out["return_belief_probabilities"], raw["return_belief"]["probabilities"]
    )
    assert out["action_loss_weight"] == 1.5
