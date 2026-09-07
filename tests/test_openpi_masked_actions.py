"""Run with the pinned OpenPI environment; exercise its patched real loss path."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.fixture(scope="module", autouse=True)
def patched_openpi():
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy

    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=tuple(
            path
            for name in (
                "0001-filter-incomplete-action-chunks.patch",
                "0003-mask-action-tails.patch",
            )
            if (path := Path("patches/openpi") / name).exists()
        ),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root
        sys.path.remove(str(root / "src"))


def _loss_fixture(mask):
    import jax.numpy as jnp
    from openpi.models.model import Observation

    # A cheap coupled token mixer, so padded targets can affect valid outputs if
    # the real Pi0 flow-input construction fails to remove them.
    class CoupledPolicy:
        action_horizon = 3
        PaliGemma = SimpleNamespace(
            llm=lambda tokens, **kw: (
                (tokens[0], tokens[1] + tokens[1].mean(axis=1, keepdims=True)),
                None,
            )
        )
        action_out_proj = staticmethod(lambda x: x)

        def embed_prefix(self, obs):
            return jnp.zeros((2, 1, 4)), jnp.ones((2, 1), bool), jnp.array([False])

        def embed_suffix(self, obs, actions, time):
            return actions, jnp.ones((2, 3), bool), jnp.array([True, False, False]), None

    obs = Observation(
        images={
            k: jnp.zeros((2, 224, 224, 3))
            for k in (
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
            )
        },
        image_masks={},
        state=jnp.zeros((2, 4)),
        action_loss_mask=mask,
    )
    return CoupledPolicy(), obs


def test_flow_loss_and_gradients_ignore_padding_even_with_token_coupling():
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0

    mask = jnp.zeros((2, 3, 4), bool).at[:, 0, :2].set(True)
    model, obs = _loss_fixture(mask)
    actions = jnp.ones((2, 3, 4)) * 0.25

    def loss(a):
        return Pi0.compute_loss(model, jax.random.key(3), obs, a).mean()

    poisoned = jnp.where(mask, actions, jnp.nan)
    np.testing.assert_allclose(loss(actions), loss(poisoned), rtol=1e-6)
    grad = jax.grad(loss)(actions)
    np.testing.assert_array_equal(np.asarray(grad)[~np.asarray(mask)], 0)
    assert np.isfinite(grad).all()
    assert np.any(np.asarray(grad)[np.asarray(mask)] != 0)


def test_flow_loss_all_masked_is_finite_zero_and_full_mask_matches_legacy():
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0

    model, obs = _loss_fixture(jnp.ones((2, 3, 4), bool))
    actions = jnp.ones((2, 3, 4))

    def loss(o):
        return Pi0.compute_loss(model, jax.random.key(3), o, actions)

    # CUDA can reassociate these equivalent float32 reductions by one ULP.
    np.testing.assert_array_max_ulp(
        np.asarray(loss(obs)),
        np.asarray(loss(dataclasses.replace(obs, action_loss_mask=None))),
        maxulp=2,
    )
    empty = dataclasses.replace(obs, action_loss_mask=jnp.zeros((2, 3, 4), bool))
    np.testing.assert_array_equal(loss(empty), 0)


def test_sampler_keeps_unsupervised_dimensions_at_their_training_noise():
    import jax
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0

    model, obs = _loss_fixture(None)
    model.active_action_dim = 2
    model.PaliGemma = SimpleNamespace(llm=lambda tokens, **kw: ((tokens[0], tokens[1]), None))
    noise = jnp.ones((2, 3, 4))
    result = Pi0.sample_actions(model, jax.random.key(4), obs, noise=noise, num_steps=2)
    np.testing.assert_array_equal(result[..., 2:], noise[..., 2:])
    assert not np.array_equal(result[..., :2], noise[..., :2])


def test_structured_transforms_keep_mask_and_encode_each_current_state_dimension(tmp_path):
    import jax
    from openpi import transforms
    from openpi.models.model import Observation, preprocess_observation

    from latency_meta_mdp.openpi_sft import _build_config
    from latency_meta_mdp.sft_profile import load_sft_profile

    config = _build_config(
        load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml")), 3
    )
    assert config.model.discrete_state_input is True
    assert config.model.active_action_dim == 7
    data_config = config.data.create(tmp_path, config.model)
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            *data_config.model_transforms.inputs,
        ]
    )
    raw = dict(
        image=np.zeros((224, 224, 3), np.uint8),
        wrist_image=np.full((224, 224, 3), 255, np.uint8),
        state=np.zeros(16, np.float32),
        actions=np.ones((50, 7), np.float32),
        actions_is_pad=np.arange(50) >= 2,
        prompt="Grasp the moving ball and lift it.",
    )
    batch = transform(dict(raw))
    assert batch["action_loss_mask"].shape == (50, 32)
    assert batch["action_loss_mask"].sum() == 14
    np.testing.assert_array_equal(batch["actions"][2:], 0)
    baseline = batch["tokenized_prompt"]
    for dim in range(16):
        state = raw["state"].copy()
        state[dim] = 0.8
        changed = transform({**raw, "state": state})
        assert not np.array_equal(changed["tokenized_prompt"], baseline), dim
    assert batch["tokenized_prompt_mask"].sum() < config.model.max_token_len
    assert batch["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    }
    obs = Observation.from_dict(jax.tree.map(lambda a: np.asarray(a)[None], batch))
    preprocessed = preprocess_observation(None, obs, train=False)
    np.testing.assert_array_equal(preprocessed.action_loss_mask, obs.action_loss_mask)
    # Terminal rows must never become an all-masked training source.
    with pytest.raises(ValueError, match="real action"):
        transform({**raw, "actions_is_pad": np.ones(50, bool)})


def test_lerobot_roundtrip_mask_matches_source_and_norm_stats_count_real_frames(
    tmp_path,
    monkeypatch,
    patched_openpi,
):
    from lerobot.common.datasets import lerobot_dataset
    from openpi.shared import normalize
    from openpi.training import data_loader
    from test_structured_policy_data import _source

    from latency_meta_mdp.lerobot_conversion import write_lerobot_policy_dataset
    from latency_meta_mdp.openpi_sft import _build_config, compute_openpi_norm_stats
    from latency_meta_mdp.policy_data import (
        load_structured_policy_episode,
        materialize_policy_action_target,
    )
    from latency_meta_mdp.sft_profile import load_sft_profile

    profile = load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml"))
    repo_id = profile.levels[1].repo_id
    episode = load_structured_policy_episode(_source(tmp_path), episode_id="test-L1")
    other = dataclasses.replace(
        episode, episode_id="other", actions=np.full((3, 7), -0.8, np.float32)
    )
    dataset_home = tmp_path / "datasets"
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", dataset_home)
    nested_path = write_lerobot_policy_dataset(
        episodes=[episode, other], output_dir=dataset_home / repo_id, repo_id=repo_id
    )
    config = _build_config(profile, 1)
    data = config.data.create(tmp_path / "assets", config.model)
    raw = data_loader.create_torch_dataset(data, 50, config.model)
    transformed = data_loader.transform_dataset(raw, data, skip_norm_stats=True)
    assert len(raw) == 6
    for index in range(6):
        source = episode if index < 3 else other
        target, valid = materialize_policy_action_target(source, target_start_tick=index % 3)
        sample = transformed[index]
        np.testing.assert_array_equal(
            sample["action_loss_mask"][:, :7], np.broadcast_to(valid[:, None], (50, 7))
        )
        np.testing.assert_array_equal(sample["action_loss_mask"][:, 7:], False)
        np.testing.assert_allclose(sample["actions"][:, :7], target)
    output = tmp_path / "stats/norm_stats.json"
    compute_openpi_norm_stats(
        level=1,
        repo_id=repo_id,
        dataset_root=dataset_home,
        expected_source_count=6,
        output_path=output,
        profile=profile,
        openpi_root=patched_openpi,
    )
    stats = normalize.load(output.parent)
    np.testing.assert_allclose(
        stats["actions"].mean,
        np.concatenate([episode.actions, other.actions]).mean(axis=0),
        atol=1e-7,
    )
    assert stats["state"].mean.shape == (16,)
    # The actual batch interface consumed by OpenPI training retains the mask.
    import jax

    loader = data_loader.create_torch_data_loader(
        data,
        config.model,
        50,
        2,
        skip_norm_stats=True,
        num_batches=1,
        num_workers=0,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    observation, actions = next(iter(loader))
    assert observation.action_loss_mask.shape == actions.shape == (2, 50, 32)
    from latency_meta_mdp.artifacts import sha256_file

    manifest = {
        "format_id": "metamdp_lerobot_structured_train_v1",
        "split": "train",
        "sft_profile_sha256": sha256_file(
            Path("configs/policy/pi05_structured_state16_h50_v1.yaml")
        ),
        "source_manifest_sha256": "0" * 64,
        "split_manifest_sha256": "1" * 64,
        "train_master_task_indices": [42],
        "datasets": [
            {
                "level": 1,
                "repo_id": repo_id,
                "frame_count": 6,
                "dataset_manifest": nested_path.relative_to(dataset_home).as_posix(),
                "dataset_manifest_sha256": sha256_file(nested_path),
            }
        ],
    }
    manifest_path = dataset_home / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    preparation = tmp_path / "preparation"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "latency_meta_mdp.cli.prepare_structured_pi05_sft",
            "--dataset-manifest",
            str(manifest_path),
            "--level",
            "1",
            "--output-dir",
            str(preparation),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )
    report = json.loads((preparation / "preparation.json").read_text())
    assert report["source_count"] == 6
    assert report["episode_boundary_masks_checked"] == 2
    assert report["state_stats_dim"] == 16
    assert report["action_stats_dim"] == 7
    assert report["state_tokens_enabled"] is True
    assert report["training_started"] is False
