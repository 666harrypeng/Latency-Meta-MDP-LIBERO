import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest_plugins = ("test_forecast_policy_cache",)


@pytest.fixture(scope="module")
def native_forecast_openpi():
    from latency_meta_mdp.policy.conditioning import conditioning_patches
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=conditioning_patches(Path.cwd(), "forecast_only"),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        sys.path.insert(0, str(root / "src"))
        yield root


def test_native_structure_and_future_target_schedule(native_forecast_openpi, tmp_path):
    from latency_meta_mdp.policy.config import load_training_job, resolve_policy_profile
    from latency_meta_mdp.policy.openpi.forecast_only import build_forecast_only_train_config
    from latency_meta_mdp.policy.openpi.training import build_level_train_config
    from latency_meta_mdp.policy.schedule import SFTLaunchRequest

    job = load_training_job(Path("configs/experiments/moving_ball/l2/clean.yaml"))
    clean = build_level_train_config(
        profile=resolve_policy_profile(job),
        request=SFTLaunchRequest(2, "clean", "formal", False, 8, 256),
        assets_root=tmp_path,
        checkpoint_root=tmp_path,
        wandb_enabled=False,
    )
    identity = {
        k: "a" * 64 for k in ("predictor_sha256", "decoder_sha256", "jepa_normalization_sha256")
    }
    identity["predictor_architecture"] = "jepa_direct_q20_history_stride4_w3_v1"
    (tmp_path / "manifest.json").write_text(
        json.dumps({"complete": True, "bindings": identity, "episodes": [{"frame_count": 100}]})
    )
    view = {
        "cache_root": str(tmp_path),
        "bindings": identity,
        "policy_export_manifest": "unused",
        "input_mode": "forecast_only",
    }
    config = build_forecast_only_train_config(
        clean_config=clean,
        clean_checkpoint=Path("/unused/params"),
        forecast_identity=identity,
        experiment_name="only",
        forecast_view=view,
    )
    assert dataclasses.asdict(config.model) == dataclasses.asdict(clean.model)
    assert config.model.image_keys == clean.model.image_keys
    assert config.model.max_token_len == clean.model.max_token_len
    assert config.policy_metadata["target_alignment"] == "forecast_time"
    assert config.policy_metadata["conditioning"] == "native_rtc_forecast_only_rgb_v1"
    assert config.keep_period == 10 and config.num_train_steps == 20
    native = clean.data.create(clean.assets_dirs, clean.model)
    future = config.data.create(config.assets_dirs, config.model)
    assert [type(t) for t in native.model_transforms.inputs] == [
        type(t) for t in future.model_transforms.inputs
    ]
    assert type(native.data_transforms.inputs[0]) is type(future.data_transforms.inputs[0])
    raw = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.arange(16, dtype=np.float32),
        "prompt": "pick",
    }
    a = native.data_transforms.inputs[0](raw)
    b = future.data_transforms.inputs[0](raw)
    assert set(a) == set(b)
    for key in a["image"]:
        np.testing.assert_array_equal(a["image"][key], b["image"][key])
    for transform in native.model_transforms.inputs:
        a = transform(a)
    for transform in future.model_transforms.inputs:
        b = transform(b)
    np.testing.assert_array_equal(a["tokenized_prompt"], b["tokenized_prompt"])
    np.testing.assert_array_equal(a["tokenized_prompt_mask"], b["tokenized_prompt_mask"])


def test_actual_loader_dispatches_native_forecast_view(
    native_forecast_openpi, built_cache, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from openpi.shared import normalize
    from openpi.training import data_loader

    from latency_meta_mdp.data.forecast.only_dataset import ForecastOnlyPolicyDataset
    from latency_meta_mdp.io.artifacts import sha256_file
    from latency_meta_mdp.policy.openpi.forecast_only import build_forecast_only_train_config
    from latency_meta_mdp.policy.openpi.training import _build_config
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

    monkeypatch.setattr(
        data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a: SimpleNamespace(fps=50)
    )
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", lambda *a, **kw: Native())
    nested = tmp_path / "episodes.json"
    nested.write_text(json.dumps({"episodes": list(cache.episodes.values())}))
    export = tmp_path / "manifest.json"
    export.write_text(
        json.dumps(
            {
                "format_id": "metamdp_lerobot_structured_train_v1",
                "split": "train",
                **{k: bindings[k] for k in ("source_manifest_sha256", "split_manifest_sha256")},
                "datasets": [
                    {
                        "level": record.level,
                        "dataset_manifest": nested.name,
                        "dataset_manifest_sha256": sha256_file(nested),
                    }
                ],
            }
        )
    )
    clean = _build_config(
        load_sft_profile(Path("configs/contracts/policy/pi05_state16_h50.yaml")), record.level
    )
    clean = dataclasses.replace(
        clean,
        assets_base_dir=str(tmp_path / "assets"),
        batch_size=20,
        num_workers=0,
        data=dataclasses.replace(
            clean.data,
            base_config=dataclasses.replace(clean.data.base_config, prompt_from_task=False),
        ),
    )
    normalize.save(
        clean.assets_dirs / clean.data.repo_id,
        {
            k: normalize.NormStats(
                mean=np.zeros(d), std=np.ones(d), q01=np.full(d, -2), q99=np.full(d, 2)
            )
            for k, d in (("state", 16), ("actions", 7))
        },
    )
    config = build_forecast_only_train_config(
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
        experiment_name="only-loader",
        forecast_view={
            "cache_root": str(cache.root),
            "bindings": bindings,
            "input_mode": "forecast_only",
            "policy_export_manifest": str(export),
        },
    )
    data = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data, 50, config.model)
    assert isinstance(dataset, ForecastOnlyPolicyDataset)
    assert config.keep_period == len(dataset.training_sampler(batch_size=20)) // 20
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=1, start_batch=1)
    obs, actions = next(iter(loader))
    assert actions.shape == (20, 50, 32) and obs.tokenized_prompt.shape == (20, 200)
    assert not np.asarray(obs.image_masks["right_wrist_0_rgb"]).any()
    assert np.asarray(obs.action_loss_mask)[:, 0, :7].all()
    assert not np.asarray(obs.action_loss_mask)[:, :, 7:].any()
