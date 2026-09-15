"""Native four-image forecast conditioning through the real Pi0 loss/attention path."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest_plugins = ("test_forecast_policy_cache",)

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module")
def forecast_openpi():
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=tuple(sorted(Path("patches/openpi").glob("000[1-9]-*.patch"))),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root


def test_forecast_config_has_four_explicit_views_and_rejects_old_conditioning(forecast_openpi):
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, active_action_dim=7, use_rtc_forecast=True)
    obs, actions = config.inputs_spec(batch_size=2)
    assert tuple(obs.images) == config.image_keys
    assert len(obs.images) == 4
    assert actions.shape == (2, 50, 32)
    with pytest.raises(ValueError, match="exclusive"):
        dataclasses.replace(config, use_return_belief_prefix=True)
    with pytest.raises(ValueError, match="state-aware"):
        dataclasses.replace(config, discrete_state_input=False)


def _inputs():
    return {
        "observation/image": np.full((224, 224, 3), 20, np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 40, np.uint8),
        "observation/state": np.arange(16, dtype=np.float32) / 32,
        "prompt": "Grasp the moving ball and lift it.",
        "forecast": {
            "rgb": np.stack([np.full((224, 224, 3), v, np.uint8) for v in (60, 80)]),
            "proprio": np.arange(16, dtype=np.float32) / 16,
            "query_ticks": 20,
        },
        "actions": np.full((50, 7), np.nan, np.float32),
        "actions_is_pad": np.arange(50) >= 2,
    }


def test_forecast_transform_preserves_native_inputs_and_masks_tail(forecast_openpi):
    from openpi.models.model import ModelType
    from openpi.shared.normalize import NormStats

    from latency_meta_mdp.policy.openpi.forecast import ForecastPolicyInputs

    raw = _inputs()
    raw["actions"][:2] = 0.2
    stats = NormStats(mean=np.ones(16), std=np.full(16, 2.0))
    transform = ForecastPolicyInputs(
        model_type=ModelType.PI05, state_norm_stats=stats, use_quantiles=False
    )
    data = transform(raw)
    assert [int(im[0, 0, 0]) for im in data["image"].values()] == [20, 40, 60, 80]
    np.testing.assert_array_equal(data["state"], raw["observation/state"])
    np.testing.assert_allclose(
        data["forecast_proprio"], (raw["forecast"]["proprio"] - 1) / 2, atol=1e-6
    )
    assert data["action_loss_mask"].sum() == 14
    assert np.isfinite(data["actions"]).all()
    assert "actions_is_pad" not in data
    assert "forecast_proprio" not in raw


def test_forecast_tokenizer_encodes_both_states_query_and_rejects_truncation(forecast_openpi):
    from openpi.models.tokenizer import PaligemmaTokenizer

    class CaptureSentencePiece:
        def encode(self, text, add_bos):
            self.text = text
            return [1, *text.encode()]

    tokenizer = object.__new__(PaligemmaTokenizer)
    tokenizer._tokenizer = CaptureSentencePiece()
    tokenizer._max_len = 1024
    current, future = np.zeros(16), np.full(16, 0.25)
    tokens, mask = tokenizer.tokenize_forecast("pick", current, future_state=future, query_ticks=7)
    assert "140 ms" in tokenizer._tokenizer.text
    assert "Forecast state:" in tokenizer._tokenizer.text
    assert "current main, current wrist, forecast main, forecast wrist" in tokenizer._tokenizer.text
    assert tokens.shape == mask.shape == (1024,)
    for index in range(16):
        changed = future.copy()
        changed[index] += 0.25
        other, _ = tokenizer.tokenize_forecast("pick", current, future_state=changed, query_ticks=7)
        assert not np.array_equal(tokens, other)
    tokenizer._max_len = 5
    with pytest.raises(ValueError, match="truncat"):
        tokenizer.tokenize_forecast("pick", current, future_state=future, query_ticks=7)


def test_real_attention_uses_all_views_and_is_independent_of_dict_order(
    monkeypatch, forecast_openpi
):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict
    from test_openpi_belief_prefix import _tiny_pi05

    from latency_meta_mdp.policy.openpi.training import (
        _build_config,
        build_forecast_policy_train_config,
    )
    from latency_meta_mdp.policy.profile import load_sft_profile

    cfg, _, _ = _tiny_pi05(monkeypatch, prefix=False)
    cfg = dataclasses.replace(cfg, use_rtc_forecast=True)
    model = cfg.create(jax.random.key(1))
    obs = cfg.fake_obs(batch_size=1)
    obs = dataclasses.replace(
        obs,
        images={k: jnp.full((1, 224, 224, 3), i / 10) for i, k in enumerate(cfg.image_keys)},
        image_masks={k: jnp.ones((1,), bool) for k in cfg.image_keys},
        tokenized_prompt_mask=jnp.ones((1, cfg.max_token_len), bool),
    )
    original = model.embed_prefix(obs)
    reordered = dataclasses.replace(obs, images=dict(reversed(list(obs.images.items()))))
    for a, b in zip(original, model.embed_prefix(reordered), strict=True):
        np.testing.assert_array_equal(a, b)
    assert original[0].shape[1] == 4 * 4 + cfg.max_token_len
    noise = jax.random.normal(jax.random.key(2), (1, 50, 32))
    mask = jnp.zeros_like(noise, dtype=bool).at[:, :3, :7].set(True)
    obs = dataclasses.replace(obs, action_loss_mask=mask)
    actions = jnp.where(mask, 0.1, jnp.nan)
    clean = _build_config(
        load_sft_profile(Path("configs/contracts/policy/pi05_state16_h50.yaml")), 3
    )
    train = build_forecast_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/unused/6000/params"),
        forecast_identity={
            "predictor_architecture": "jepa_direct_q20_history_stride4_w3_v1",
            "predictor_sha256": "a" * 64,
            "decoder_sha256": "b" * 64,
            "jepa_normalization_sha256": "c" * 64,
        },
        experiment_name="test",
    )
    # A fresh upstream action expert has zero AdaRMS gates. One genuine loss
    # update opens those gates; production initializes from trained clean6000.
    gradient = nnx.value_and_grad(
        lambda m: m.compute_loss(jax.random.key(3), obs, actions, train=False).mean(),
        argnums=nnx.DiffState(0, train.trainable_filter),
    )
    _, grads = gradient(model)
    nnx.update(
        model,
        optax.apply_updates(
            nnx.state(model, train.trainable_filter), jax.tree.map(lambda g: -1e-2 * g, grads)
        ),
    )
    loss = model.compute_loss(jax.random.key(3), obs, actions, train=False)
    assert np.isfinite(loss).all()
    for key in cfg.image_keys:
        changed = dataclasses.replace(obs, images={**obs.images, key: obs.images[key] + 0.5})
        other = model.compute_loss(jax.random.key(3), changed, actions, train=False)
        assert float(jnp.max(jnp.abs(other - loss))) > 1e-6, key
    original_actions = model.sample_actions(jax.random.key(4), obs, noise=noise, num_steps=2)
    changed_actions = model.sample_actions(jax.random.key(4), changed, noise=noise, num_steps=2)
    assert float(jnp.max(jnp.abs(original_actions[..., :7] - changed_actions[..., :7]))) > 1e-7
    hidden = dataclasses.replace(
        obs, image_masks={**obs.image_masks, cfg.image_keys[-1]: jnp.zeros((1,), bool)}
    )
    hidden_changed = dataclasses.replace(hidden, images=changed.images)
    np.testing.assert_array_equal(
        model.compute_loss(jax.random.key(3), hidden, actions, train=False),
        model.compute_loss(jax.random.key(3), hidden_changed, actions, train=False),
    )
    before = {
        k: np.array(v, copy=True)
        for k, v in flatten_dict(nnx.state(model).to_pure_dict(), sep="/").items()
    }
    _, grads = gradient(model)
    flat = flatten_dict(grads.to_pure_dict(), sep="/")
    assert all(np.isfinite(v).all() for v in flat.values())
    for expert in (False, True):
        assert (
            sum(
                float(np.square(v).sum())
                for k, v in flat.items()
                if k.startswith("PaliGemma/llm/layers/") and ("_1/" in k) == expert
            )
            > 0
        )
    nnx.update(
        model,
        optax.apply_updates(
            nnx.state(model, train.trainable_filter), jax.tree.map(lambda g: -1e-3 * g, grads)
        ),
    )
    after = flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    for key in before:
        if key.startswith(("PaliGemma/img/", "PaliGemma/llm/embedder/")):
            np.testing.assert_array_equal(before[key], after[key])


def test_production_loader_uses_forecast_sampler_and_native_loss_masks(
    monkeypatch, tmp_path, built_cache, forecast_openpi
):
    from openpi.shared import normalize
    from openpi.training import data_loader

    from latency_meta_mdp.data.forecast.dataset import ForecastPolicyDataset
    from latency_meta_mdp.policy.openpi.training import (
        _build_config,
        build_forecast_policy_train_config,
    )
    from latency_meta_mdp.policy.profile import load_sft_profile

    cache, record, bindings = built_cache

    class Native:
        def __len__(self):
            return record.terminal_tick

        def __getitem__(self, i):
            return {
                "image": np.zeros((224, 224, 3), np.uint8),
                "wrist_image": np.zeros((224, 224, 3), np.uint8),
                "state": record.proprio_physical[i],
                "prompt": "pick",
                "frame_index": i,
                "episode_index": 0,
            }

    dataset = ForecastPolicyDataset(
        Native(), episode_rows=list(cache.episodes.values()), cache=cache
    )
    clean = _build_config(
        load_sft_profile(Path("configs/contracts/policy/pi05_state16_h50.yaml")), 3
    )
    clean = dataclasses.replace(
        clean, assets_base_dir=str(tmp_path / "assets"), batch_size=20, num_workers=0
    )
    stats = {
        k: normalize.NormStats(
            mean=np.zeros(d), std=np.ones(d), q01=np.full(d, -2), q99=np.full(d, 2)
        )
        for k, d in (("state", 16), ("actions", 7))
    }
    normalize.save(clean.assets_dirs / clean.data.repo_id, stats)
    config = build_forecast_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/unused/params"),
        forecast_identity={
            k: bindings[k]
            for k in (
                "predictor_sha256",
                "decoder_sha256",
                "jepa_normalization_sha256",
                "predictor_architecture",
            )
        },
        forecast_view={
            "cache_root": str(cache.root),
            "policy_export_manifest": "unused-by-fixture",
            "bindings": bindings,
        },
        experiment_name="loader-test",
    )
    # Replace only the external LeRobot store construction; wrappers, transforms,
    # sampler, batching and Observation conversion are the production code.
    monkeypatch.setattr(data_loader, "create_torch_dataset", lambda *args: dataset)
    assert config.num_train_steps == 2 * len(dataset.training_sampler(batch_size=20)) // 20
    assert config.policy_metadata["forecast_training_budget_resolved"] is True
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    observation, actions = next(iter(loader))
    assert set(observation.images) == set(config.model.image_keys)
    assert actions.shape == (20, 50, 32)
    mask = np.asarray(observation.action_loss_mask)
    assert mask[:, :, :7].any(axis=(1, 2)).all()
    assert not mask[:, :, 7:].any()
    assert np.isfinite(actions).all()
    assert not hasattr(observation, "realized_delay_ticks")


def test_rtc_bridge_transports_forecast_and_normalizes_buffer_once(forecast_openpi):
    import flax.nnx as nnx
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.policy import Policy
    from openpi.shared.normalize import NormStats

    from latency_meta_mdp.data.forecast.samples import DecodedForecast
    from latency_meta_mdp.policy.openpi.forecast import ForecastPolicyInputs, ForecastTokenizePrompt
    from latency_meta_mdp.runtime.policy_execution import (
        InProcessRtcOpenpiPolicy,
        PolicyObservation,
    )
    from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext

    class Tokenizer:
        def tokenize_forecast(self, prompt, state, *, future_state, query_ticks):
            np.testing.assert_allclose(state, 0.5, atol=1e-6)
            np.testing.assert_allclose(future_state, 1.0, atol=1e-6)
            assert query_ticks == 7
            return np.ones(4, np.int32), np.ones(4, bool)

    class Model(nnx.Module):
        pi05 = True
        active_action_dim = 7
        action_horizon = 50
        image_keys = (
            "base_0_rgb",
            "left_wrist_0_rgb",
            "forecast_base_0_rgb",
            "forecast_left_wrist_0_rgb",
        )

        def sample_actions(self, rng, obs, *, rtc_previous_actions, rtc_weights, noise):
            import jax.numpy as jnp

            assert len(obs.images) == 4
            mask_penalty = jnp.stack(tuple(obs.image_masks.values())).sum() - 4
            return 2 * rtc_previous_actions + mask_penalty

    stats = {
        "state": NormStats(mean=np.zeros(16), std=np.full(16, 2)),
        "actions": NormStats(mean=np.zeros(7), std=np.full(7, 2)),
    }
    policy = Policy(
        Model(),
        transforms=[
            ForecastPolicyInputs(
                model_type=ModelType.PI05, state_norm_stats=stats["state"], use_quantiles=False
            ),
            transforms.Normalize(stats),
            ForecastTokenizePrompt(Tokenizer(), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[transforms.Unnormalize(stats)],
    )
    obs = PolicyObservation(
        10,
        np.zeros((224, 224, 3), np.uint8),
        np.zeros((224, 224, 3), np.uint8),
        np.ones(16, np.float32),
    )
    packet = DecodedForecast(
        10, 17, 7, 0, np.zeros((2, 224, 224, 3), np.uint8), np.full(16, 2, np.float32)
    )
    ctx = RtcInferenceContext(0, 10, obs, 0, np.ones((50, 7)), np.ones(50, bool), 7, packet)
    bridge = InProcessRtcOpenpiPolicy(policy, noise_rng=np.random.default_rng(1))
    np.testing.assert_allclose(bridge(obs, ctx)["actions"][:, :7], 2, atol=1e-6)
    with pytest.raises(ValueError, match="mode disagree"):
        bridge(obs, dataclasses.replace(ctx, forecast=None))
