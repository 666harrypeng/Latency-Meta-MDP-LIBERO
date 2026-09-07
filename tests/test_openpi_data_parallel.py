"""Use JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4."""

from __future__ import annotations

import dataclasses
import functools
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module", autouse=True)
def patched_openpi():
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy

    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=tuple(sorted(Path("patches/openpi").glob("000[1-5]-*.patch"))),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root
        sys.path.remove(str(root / "src"))


def _config(tmp_path, devices=4, batch=128):
    from latency_meta_mdp.openpi_sft import build_level_train_config
    from latency_meta_mdp.sft_launch import SFTLaunchRequest
    from latency_meta_mdp.sft_profile import load_sft_profile

    profile = load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml"))
    return build_level_train_config(
        profile=profile,
        request=SFTLaunchRequest(3, "ddp-check", "formal", False, devices, batch),
        assets_root=tmp_path / "assets",
        checkpoint_root=tmp_path / "checkpoints",
        wandb_enabled=False,
    )


def test_data_parallel_config_scales_lr_clock_and_keeps_native_contract(tmp_path):
    config = _config(tmp_path)
    assert config.fsdp_devices == 1
    assert config.batch_size == 128
    assert config.num_train_steps == config.lr_schedule.decay_steps == 6000
    assert config.lr_schedule.warmup_steps == 300
    assert config.lr_schedule.peak_lr == 5e-5
    assert config.keep_period == 2000
    assert config.model.discrete_state_input
    assert config.model.active_action_dim == 7
    assert config.policy_metadata["per_device_batch_size"] == 32
    assert config.policy_metadata["training_device_count"] == 4
    assert config.policy_metadata["training_examples"] == 768000


def test_training_rejects_unexpected_devices_and_model_sharding(
    tmp_path, patched_openpi, monkeypatch
):
    import jax

    from latency_meta_mdp.openpi_sft import run_openpi_training

    config = _config(tmp_path)
    with pytest.raises(ValueError, match="replicated"):
        run_openpi_training(
            config=dataclasses.replace(config, fsdp_devices=2), openpi_root=patched_openpi
        )
    monkeypatch.setattr(jax, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="visible devices"):
        run_openpi_training(config=config, openpi_root=patched_openpi)


def test_pinned_training_step_matches_single_device_and_replicates_optimizer(
    tmp_path, patched_openpi
):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models.model import BaseModel, Observation
    from openpi.training import sharding
    from openpi.training.weight_loaders import NoOpWeightLoader

    from latency_meta_mdp.openpi_sft import _load_train_script

    if jax.default_backend() != "cpu" or jax.device_count() != 4:
        pytest.skip("requires four logical CPU devices; this is not a GPU throughput benchmark")

    class RegressionPolicy(BaseModel):
        """Unequal sample weights/mask lengths expose incorrect replica averaging."""

        def __init__(self, rng):
            self.linear = nnx.Linear(2, 2, rngs=nnx.Rngs(rng))

        def compute_loss(self, rng, observation, actions, *, train=False):
            del rng, train
            error = (self.linear(observation.state)[:, None, :] - actions) ** 2
            mask = observation.action_loss_mask
            loss = jnp.where(mask, error, 0).sum((1, 2)) / jnp.maximum(mask.sum((1, 2)), 1)
            return loss[:, None] * observation.action_loss_weight[:, None]

        def sample_actions(self, rng, observation, **kwargs):
            raise NotImplementedError

    @dataclasses.dataclass(frozen=True)
    class RegressionConfig:
        def create(self, rng):
            return RegressionPolicy(rng)

    trainer = _load_train_script(patched_openpi)
    config = dataclasses.replace(
        _config(tmp_path, batch=8), model=RegressionConfig(), weight_loader=NoOpWeightLoader()
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    assert mesh.shape == {"batch": 4, "fsdp": 1}
    state, layout = trainer.init_train_state(config, jax.random.key(7), mesh, resume=False)
    for leaf in jax.tree.leaves(state):
        assert leaf.sharding.is_fully_replicated
    mask = np.ones((8, 3, 2), bool)
    mask[0] = False
    mask[1:4, 1:] = False
    obs = Observation(
        images={},
        image_masks={},
        state=jnp.arange(16, dtype=jnp.float32).reshape(8, 2) / 16,
        action_loss_mask=jnp.asarray(mask),
        action_loss_weight=jnp.array([1, 0.2, 1.5, 2, 0.5, 1, 1.2, 0.8]),
    )
    batch = (obs, jnp.arange(48, dtype=jnp.float32).reshape(8, 3, 2) / 48)
    data_layout = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed_step = jax.jit(
        functools.partial(trainer.train_step, config),
        in_shardings=(replicated, layout, data_layout),
        out_shardings=(layout, replicated),
    )
    single = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    reference_step = jax.jit(functools.partial(trainer.train_step, config))
    reference = jax.device_put(state, single)
    reference_batch = jax.device_put(batch, single)
    distributed_batch = jax.device_put(batch, data_layout)
    assert distributed_batch[0].state.addressable_shards[0].data.shape == (2, 2)
    # Include Adam moments and EMA across multiple updates, not just forward loss.
    for seed in (11, 12, 13):
        rng = jax.random.key(seed)
        reference, expected = reference_step(
            jax.device_put(rng, single), reference, reference_batch
        )
        state, actual = distributed_step(rng, state, distributed_batch)
        for got, wanted in zip(
            jax.tree.leaves((state, actual)), jax.tree.leaves((reference, expected)), strict=True
        ):
            assert got.sharding.is_fully_replicated
            np.testing.assert_allclose(got, wanted, rtol=2e-5, atol=2e-6)
    assert int(state.step) == 3
