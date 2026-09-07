"""Real JAX/Flax prefix contract, including cached native attention and trainability."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module")
def patched_openpi():
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy

    patches = tuple(sorted(Path("patches/openpi").glob("000[1-6]-*.patch")))
    assert any(p.name.startswith("0006-") for p in patches)
    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=patches,
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root
        sys.path.remove(str(root / "src"))


def _fields():
    import jax
    import jax.numpy as jnp

    pmf = jnp.full((1, 20), 0.05)
    macro = jnp.array([[0.25, 0.2, 0.2, 0.2, 0.15]])
    return (
        jax.random.normal(jax.random.key(2), (1, 5, 2, 196, 384)) * 0.05,
        jax.random.normal(jax.random.key(3), (1, 5, 16)),
        jnp.array([[4, 8, 12, 16, 20]], dtype=jnp.float32),
        macro,
        pmf,
        jnp.zeros((1, 2)),
    )


def _encoder():
    import flax.nnx as nnx

    from latency_meta_mdp.openpi_belief_prefix import ReturnBeliefPrefixEncoder

    return ReturnBeliefPrefixEncoder(vlm_width=64, queries_per_view=4, rngs=nnx.Rngs(7))


def test_prefix_preserves_anchor_view_slots_and_full_law_information():
    encoder = _encoder()
    fields = _fields()
    original = np.asarray(encoder(*fields))
    assert original.shape == (1, 46, 64)
    visual, *other = fields
    changed = np.asarray(encoder(visual.at[:, 2, 1, 3, :20].add(0.25), *other))
    # law0; each nine-token anchor = state, main4, wrist4.
    slots = slice(1 + 2 * 9 + 5, 1 + 3 * 9)
    assert np.max(np.abs(original[:, slots] - changed[:, slots])) > 1e-6
    untouched = np.ones(46, bool)
    untouched[slots] = False
    np.testing.assert_array_equal(original[:, untouched], changed[:, untouched])
    changed_fields = list(fields)
    # Move probability within the first macro anchor: macro masses stay equal.
    changed_fields[4] = fields[4].at[:, 0].add(0.02).at[:, 1].add(-0.02)
    law_changed = np.asarray(encoder(*changed_fields))
    assert np.max(np.abs(original[:, 0] - law_changed[:, 0])) > 1e-6
    np.testing.assert_array_equal(original[:, 1:], law_changed[:, 1:])


def test_prefix_input_gradients_stop_but_encoder_parameters_receive_gradients():
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    encoder = _encoder()
    fields = _fields()
    gradients = jax.grad(lambda *xs: jnp.square(encoder(*xs)).mean(), argnums=tuple(range(6)))(
        *fields
    )
    for grad in gradients:
        np.testing.assert_array_equal(grad, 0)
    _, grads = nnx.value_and_grad(lambda m: jnp.square(m(*fields)).mean())(encoder)
    assert sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(grads)) > 0


def test_prefix_and_legacy_late_modes_cannot_be_combined(patched_openpi):
    from openpi.models.pi0_config import Pi0Config

    cfg = Pi0Config(pi05=True, active_action_dim=7, use_return_belief_prefix=True)
    observation, _ = cfg.inputs_spec(batch_size=2)
    assert observation.return_belief_latency_probabilities.shape == (2, 20)
    assert observation.return_belief_known_delay.shape == (2, 2)
    with pytest.raises(ValueError, match="exclusive"):
        dataclasses.replace(cfg, use_return_belief=True)


def _tiny_pi05(monkeypatch, *, prefix):
    import flax.linen as linen
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0, pi0_config

    class FrozenVisionFixture(linen.Module):
        num_classes: int
        variant: str
        pool_type: str
        scan: bool
        dtype_mm: str

        @linen.compact
        def __call__(self, images, train=False):
            pixels = images[:, ::112, ::112].reshape(images.shape[0], 4, 3)
            return linen.Dense(self.num_classes, dtype=jnp.float32)(pixels), None

    # Only the frozen image encoder is reduced. Gemma attention/blocks and Pi0 flow are real.
    monkeypatch.setattr(pi0._siglip, "Module", FrozenVisionFixture)
    cfg = pi0_config.Pi0Config(
        pi05=True,
        active_action_dim=7,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        max_token_len=7,
        use_return_belief_prefix=prefix,
    )
    model = cfg.create(jax.random.key(12))
    observation = cfg.fake_obs(batch_size=1)
    if prefix:
        observation = dataclasses.replace(
            observation,
            **dict(
                zip(
                    (
                        "return_belief_visual",
                        "return_belief_proprio",
                        "return_belief_delay_ticks",
                        "return_belief_probabilities",
                        "return_belief_latency_probabilities",
                        "return_belief_known_delay",
                    ),
                    _fields(),
                    strict=True,
                )
            ),
        )
    observation = dataclasses.replace(
        observation,
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "left_wrist_0_rgb": jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([False]),
        },
        tokenized_prompt_mask=jnp.array([[True, True, True, False, False, False, False]]),
    )
    return cfg, model, observation


def test_native_prefix_and_padding_positions_are_preserved(monkeypatch, patched_openpi):
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    _, native, plain_obs = _tiny_pi05(monkeypatch, prefix=False)
    _, model, obs = _tiny_pi05(monkeypatch, prefix=True)
    c, cm, ca = native.embed_prefix(plain_obs)
    tokens, mask, ar = model.embed_prefix(obs)
    np.testing.assert_array_equal(tokens[:, : c.shape[1]], c)
    np.testing.assert_array_equal(mask[:, : c.shape[1]], cm)
    np.testing.assert_array_equal(ar[: c.shape[1]], ca)
    assert tokens.shape[1] == c.shape[1] + 46
    assert int(mask.sum()) == int(cm.sum()) + 46 == 57
    positions = jnp.cumsum(mask, axis=1) - 1
    np.testing.assert_array_equal(positions[:, c.shape[1] :], [np.arange(11, 57)])
    attn = make_attn_mask(mask, ar)
    valid = np.flatnonzero(np.asarray(mask[0]))
    assert np.asarray(attn[0])[np.ix_(valid, valid)].all()
    assert not np.asarray(attn[0])[:, np.flatnonzero(~np.asarray(mask[0]))].any()


def test_joint_and_cached_real_gemma_agree_with_forecast_prefix(monkeypatch, patched_openpi):
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    _, model, obs = _tiny_pi05(monkeypatch, prefix=True)
    c, cm, ca = model.embed_prefix(obs)
    noisy = jax.random.normal(jax.random.key(9), (1, 50, 32))
    a, am, aa, cond = model.embed_suffix(obs, noisy, jnp.array([0.4]))
    mask = jnp.concatenate([cm, am], axis=1)
    ar = jnp.concatenate([ca, aa])
    visibility = make_attn_mask(mask, ar)
    assert not np.asarray(visibility[0, : c.shape[1], c.shape[1] :]).any()
    joint, _ = model.PaliGemma.llm(
        [c, a],
        positions=jnp.cumsum(mask, axis=1) - 1,
        mask=visibility,
        adarms_cond=[None, cond],
    )
    _, cache = model.PaliGemma.llm(
        [c, None],
        positions=jnp.cumsum(cm, axis=1) - 1,
        mask=make_attn_mask(cm, ca),
    )
    cached, _ = model.PaliGemma.llm(
        [None, a],
        positions=cm.sum(axis=1)[:, None] + jnp.cumsum(am, axis=1) - 1,
        mask=visibility[:, c.shape[1] :],
        kv_cache=cache,
        adarms_cond=[None, cond],
    )
    np.testing.assert_allclose(joint[1], cached[1], rtol=2e-5, atol=2e-5)


def _raw_prefix():
    fields = _fields()
    return {
        "observation/state": np.zeros(16, np.float32),
        "observation/image": np.zeros((8, 8, 3), np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
        "prompt": "Grasp the moving ball and lift it.",
        "actions": np.ones((50, 7), np.float32),
        "actions_is_pad": np.arange(50) >= 7,
        "action_loss_weight": np.float32(1),
        "return_belief": dict(
            zip(
                ("visual", "proprio", "delay_ticks", "probabilities", "latency_probabilities"),
                [np.asarray(x)[0] for x in fields[:5]],
                strict=True,
            )
        ),
    }


def _prefix_transform(**kwargs):
    from openpi.models.model import ModelType
    from openpi.shared.normalize import NormStats

    from latency_meta_mdp.openpi_belief_data import PrefixReturnBeliefInputs

    stats = NormStats(mean=np.zeros(16), std=np.ones(16), q01=np.full(16, -2), q99=np.full(16, 2))
    return PrefixReturnBeliefInputs(model_type=ModelType.PI05, state_norm_stats=stats, **kwargs)


def test_prefix_transform_keeps_native_inputs_and_rejects_delay_and_pmf_weights(patched_openpi):
    raw = _raw_prefix()
    transform = _prefix_transform()
    result = transform(raw)
    np.testing.assert_array_equal(result["state"], raw["observation/state"])
    assert result["action_loss_mask"].sum() == 7 * 7
    np.testing.assert_array_equal(result["return_belief_known_delay"], 0)
    np.testing.assert_array_equal(
        result["return_belief_latency_probabilities"], raw["return_belief"]["latency_probabilities"]
    )
    expected = (raw["return_belief"]["proprio"] + 2) / (4 + 1e-6) * 2 - 1
    np.testing.assert_allclose(result["return_belief_proprio"], expected, atol=1e-6)
    with pytest.raises(ValueError, match="privileged"):
        transform({**raw, "known_delay_oracle": {"known_delay_ticks": 5}})
    with pytest.raises(ValueError, match="unit"):
        transform({**raw, "action_loss_weight": 0.01})
    with pytest.raises(ValueError, match="macro"):
        transform(
            {**raw, "return_belief": {**raw["return_belief"], "probabilities": np.full(5, 0.2)}}
        )


def test_prefix_oracle_and_no_future_transforms_preserve_full_trajectory_and_law(patched_openpi):
    raw = _raw_prefix()
    base = _prefix_transform()(raw)
    oracle = _prefix_transform(privileged=True)(
        {**raw, "known_delay_oracle": {"known_delay_ticks": 5}}
    )
    for key in (
        "return_belief_visual",
        "return_belief_proprio",
        "return_belief_latency_probabilities",
    ):
        np.testing.assert_array_equal(oracle[key], base[key])
    np.testing.assert_array_equal(oracle["return_belief_known_delay"], [1, 0.25])
    no_future = _prefix_transform(erase_future=True)(raw)
    np.testing.assert_array_equal(no_future["return_belief_visual"], 0)
    np.testing.assert_array_equal(no_future["return_belief_proprio"], 0)
    for key in ("state", "return_belief_latency_probabilities", "return_belief_probabilities"):
        np.testing.assert_array_equal(no_future[key], base[key])


def test_prefix_config_trains_both_native_transformers_and_freezes_only_input_encoders(
    patched_openpi,
):
    import flax.nnx as nnx

    from latency_meta_mdp.openpi_sft import _build_config, build_prefix_return_policy_train_config
    from latency_meta_mdp.sft_profile import load_sft_profile

    clean = _build_config(
        load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml")), 3
    )
    config = build_prefix_return_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/unused/6000/params"),
        view_spec={"mode": "predicted_mixture"},
        experiment_name="prefix-test",
    )
    assert config.model.use_return_belief_prefix and not config.model.use_return_belief
    assert config.data.return_policy_view["conditioning"] == "prefix"
    for path in [
        ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
        ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
        ("PaliGemma", "llm", "final_norm", "scale"),
        ("action_out_proj", "kernel"),
        ("time_mlp_in", "kernel"),
        ("return_belief_prefix", "law", "kernel"),
    ]:
        assert config.trainable_filter(path, nnx.Param(0.0)), path
    for path in [
        ("PaliGemma", "img", "head", "kernel"),
        ("PaliGemma", "llm", "embedder", "input_embedding"),
    ]:
        assert not config.trainable_filter(path, nnx.Param(0.0)), path


def test_real_loss_updates_prefix_vlm_and_action_expert_but_not_frozen_inputs(
    monkeypatch, patched_openpi
):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from flax.traverse_util import flatten_dict

    from latency_meta_mdp.openpi_sft import _build_config, build_prefix_return_policy_train_config
    from latency_meta_mdp.sft_profile import load_sft_profile

    tiny, model, observation = _tiny_pi05(monkeypatch, prefix=True)
    clean = _build_config(
        load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml")), 3
    )
    config = build_prefix_return_policy_train_config(
        clean_config=dataclasses.replace(
            clean, model=dataclasses.replace(tiny, use_return_belief_prefix=False)
        ),
        clean_checkpoint=Path("/unused/params"),
        view_spec={"mode": "predicted_mixture"},
        experiment_name="gradient-test",
    )
    mask = jnp.zeros((1, 50, 32), bool).at[:, :7, :7].set(True)
    observation = dataclasses.replace(
        observation, action_loss_mask=mask, action_loss_weight=jnp.ones(1)
    )
    actions = jnp.full((1, 50, 32), 0.2)
    # The real post-training starts from clean weights. A fresh upstream action
    # expert has zero AdaRMS gates, so first make one native IL update rather
    # than artificially opening those gates or expecting prefix gradients at init.
    _, native, native_observation = _tiny_pi05(monkeypatch, prefix=False)
    native_observation = dataclasses.replace(
        native_observation, action_loss_mask=mask, action_loss_weight=jnp.ones(1)
    )
    _, native_grad = nnx.value_and_grad(
        lambda m: m.compute_loss(
            jax.random.key(17), native_observation, actions, train=False
        ).mean(),
        argnums=nnx.DiffState(0, config.trainable_filter),
    )(native)
    native_selected = nnx.state(native, config.trainable_filter)
    nnx.update(
        native, optax.apply_updates(native_selected, jax.tree.map(lambda g: -1e-2 * g, native_grad))
    )
    nnx.update(model, nnx.state(native))
    before = flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    before = {k: np.array(v, copy=True) for k, v in before.items()}

    def loss(m):
        return m.compute_loss(jax.random.key(77), observation, actions, train=False).mean()

    value, gradients = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, config.trainable_filter))(
        model
    )
    assert np.isfinite(value)
    flat_grad = flatten_dict(gradients.to_pure_dict(), sep="/")
    assert all(np.isfinite(v).all() for v in flat_grad.values())
    groups = {
        "prefix": lambda p: p.startswith("return_belief_prefix/"),
        "vlm": lambda p: p.startswith("PaliGemma/llm/layers/") and "_1/" not in p,
        "action_expert": lambda p: p.startswith("PaliGemma/llm/layers/") and "_1/" in p,
    }
    for name, belongs in groups.items():
        assert sum(float(np.square(v).sum()) for p, v in flat_grad.items() if belongs(p)) > 0, name
    selected = nnx.state(model, config.trainable_filter)
    nnx.update(model, optax.apply_updates(selected, jax.tree.map(lambda g: -1e-4 * g, gradients)))
    after = flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    for path in before:
        if path.startswith(("PaliGemma/img/", "PaliGemma/llm/embedder/")):
            np.testing.assert_array_equal(before[path], after[path])
    for name, belongs in groups.items():
        assert any(not np.array_equal(before[p], after[p]) for p in before if belongs(p)), name


def test_real_data_loader_uses_uniform_valid_pair_sampler(monkeypatch, tmp_path, patched_openpi):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import data_loader
    from openpi.training.config import DataConfig
    from test_policy_return_prefix_data import _prefix_dataset

    dataset, _, _ = _prefix_dataset(tmp_path)
    monkeypatch.setattr(data_loader, "create_torch_dataset", lambda *args: dataset)
    structure = {
        "observation/state": "state",
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "actions": "actions",
        "actions_is_pad": "actions_is_pad",
        "action_loss_weight": "action_loss_weight",
        "prompt": "prompt",
        "return_belief": {
            k: f"return_belief/{k}"
            for k in ("visual", "proprio", "delay_ticks", "probabilities", "latency_probabilities")
        },
    }
    config = DataConfig(
        repo_id="fixture",
        norm_stats={},
        return_policy_view={"conditioning": "prefix"},
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform(structure)]),
        data_transforms=transforms.Group(inputs=[_prefix_transform()]),
        model_transforms=transforms.Group(
            inputs=[
                # Tokenization is tested separately; avoid a network tokenizer dependency here.
                lambda d: {k: v for k, v in d.items() if k != "prompt"},
                transforms.PadStatesAndActions(32),
            ]
        ),
    )
    loader = data_loader.create_torch_data_loader(
        config,
        Pi0Config(pi05=True, active_action_dim=7, use_return_belief_prefix=True),
        50,
        128,
        shuffle=True,
        num_batches=5,
        num_workers=0,
        seed=8,
    )
    delays = []
    batches = []
    for observation, actions in loader:
        batches.append((np.asarray(observation.state), np.asarray(actions)))
        source = np.rint(np.asarray(observation.state[:, 1]) - 1).astype(int)
        target = np.rint(np.asarray(actions[:, 0, 0]) * 60).astype(int)
        delays.extend((target - source).tolist())
        assert np.asarray(observation.action_loss_mask[:, 0, :7]).all()
        np.testing.assert_array_equal(observation.action_loss_weight, 1)
        np.testing.assert_array_equal(observation.return_belief_known_delay, 0)
    np.testing.assert_array_equal(np.bincount(delays, minlength=21)[1:], 32)
    resumed = data_loader.create_torch_data_loader(
        config,
        Pi0Config(pi05=True, active_action_dim=7, use_return_belief_prefix=True),
        50,
        128,
        shuffle=True,
        num_batches=2,
        num_workers=0,
        seed=8,
        start_batch=3,
    )
    for (observation, actions), expected in zip(resumed, batches[3:], strict=True):
        np.testing.assert_array_equal(observation.state, expected[0])
        np.testing.assert_array_equal(actions, expected[1])


def test_prefix_checkpoint_restores_optimizer_ema_and_identical_next_update(
    monkeypatch, tmp_path, patched_openpi
):
    import functools

    import jax
    import jax.numpy as jnp
    from openpi.training import checkpoints, sharding
    from openpi.training.config import DataConfig
    from openpi.training.weight_loaders import NoOpWeightLoader

    from latency_meta_mdp.openpi_sft import (
        _build_config,
        _load_train_script,
        build_prefix_return_policy_train_config,
    )
    from latency_meta_mdp.sft_profile import load_sft_profile

    tiny, _, observation = _tiny_pi05(monkeypatch, prefix=True)
    clean = _build_config(
        load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml")), 3
    )
    config = build_prefix_return_policy_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/unused/params"),
        view_spec={"mode": "predicted_mixture"},
        experiment_name="checkpoint-test",
    )
    config = dataclasses.replace(config, model=tiny, weight_loader=NoOpWeightLoader(), batch_size=1)
    trainer = _load_train_script(patched_openpi)
    mesh = sharding.make_mesh(1)
    state, _ = trainer.init_train_state(config, jax.random.key(10), mesh, resume=False)
    observation = dataclasses.replace(
        observation,
        action_loss_mask=jnp.zeros((1, 50, 32), bool).at[:, :9, :7].set(True),
        action_loss_weight=jnp.ones(1),
    )
    batch = (observation, jnp.full((1, 50, 32), 0.2))
    update = jax.jit(functools.partial(trainer.train_step, config))
    with sharding.set_mesh(mesh):
        state, _ = update(jax.random.key(11), state, batch)
        state, _ = update(jax.random.key(11), state, batch)
    assert int(state.step) == 2
    loader = type("Loader", (), {"data_config": lambda self: DataConfig(repo_id="fixture")})()
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=2, overwrite=False, resume=False
    )
    assert not resuming
    try:
        checkpoints.save_state(manager, state, loader, 2)
        manager.wait_until_finished()
        shape, _ = trainer.init_train_state(config, jax.random.key(10), mesh, resume=True)
        restored = checkpoints.restore_state(manager, shape, loader, step=2)
        for wanted, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            assert wanted.dtype == actual.dtype
            np.testing.assert_array_equal(actual, wanted)
        assert "return_belief_prefix" in restored.ema_params
        with sharding.set_mesh(mesh):
            expected, expected_info = update(jax.random.key(11), state, batch)
            actual, actual_info = update(jax.random.key(11), restored, batch)
        for wanted, got in zip(
            jax.tree.leaves((expected, expected_info)),
            jax.tree.leaves((actual, actual_info)),
            strict=True,
        ):
            np.testing.assert_array_equal(got, wanted)
        # Restore preserves the selected trainable set, including the new subtree.
        assert "return_belief_prefix" in restored.params.filter(config.trainable_filter)
    finally:
        manager.close()


def test_clean_loader_allows_only_new_prefix_initialization(patched_openpi):
    import jax.numpy as jnp

    from latency_meta_mdp.openpi_belief_adapter import NativePolicyWithReturnBeliefLoader

    native = {"native": jnp.ones((2, 3), jnp.bfloat16)}
    prefix = {"law": jnp.ones((22, 64))}

    class Loader:
        def load(self, expected):
            assert set(expected) == {"native"}
            return native

    loader = NativePolicyWithReturnBeliefLoader(Loader(), parameter_key="return_belief_prefix")
    got = loader.load({**native, "return_belief_prefix": prefix})
    assert got["return_belief_prefix"] is prefix
    with pytest.raises(ValueError):
        loader.load({"native": jnp.ones((3, 3)), "return_belief_prefix": prefix})
