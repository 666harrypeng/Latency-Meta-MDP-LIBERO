import sys

import pytest

pytest.importorskip("jax")

from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
from latency_meta_mdp.policy.profile import load_sft_profile


def test_task_config_and_legacy_level_config_share_model_without_identity_aliasing(tmp_path):
    root = repository_root()
    profile = load_sft_profile(root / "configs/contracts/policy/pi05_state16_h50.yaml")
    patches = tuple(
        root / "patches/openpi" / p
        for p in (
            "0001-filter-incomplete-action-chunks.patch",
            "0002-save-completed-step-checkpoints.patch",
            "0003-mask-action-tails.patch",
        )
    )
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        from latency_meta_mdp.policy.config import resolve_policy_profile
        from latency_meta_mdp.policy.openpi.training import (
            _build_config,
            build_launch_train_config,
            build_task_train_config,
        )
        from latency_meta_mdp.policy.schedule import SFTLaunchRequest

        old = _build_config(profile, 2)
        new = build_task_train_config(
            profile,
            config_name="conveyor_test",
            repo_id="local/conveyor",
            task_id="conveyor_sort",
            action_contract_id="panda_osc_pose_delta_conveyor_v2",
            extra_metadata={"variant": "surface"},
        )
        assert old.name == profile.levels[2].config_name
        assert old.data.repo_id == profile.levels[2].repo_id
        assert old.policy_metadata["level"] == 2
        assert old.policy_metadata["action_contract_id"] == "panda_osc_pose_delta_v1"
        assert new.model == old.model
        assert new.model.discrete_state_input and new.model.active_action_dim == 7
        assert "level" not in new.policy_metadata
        assert new.policy_metadata["task_id"] == "conveyor_sort"
        assert new.policy_metadata["action_contract_id"] == "panda_osc_pose_delta_conveyor_v2"
        for level in (1, 2, 3):
            legacy = _build_config(profile, level)
            assert legacy.name == profile.levels[level].config_name
            assert legacy.data.repo_id == profile.levels[level].repo_id
            assert legacy.policy_metadata["level"] == level
        historical = load_sft_profile(
            root / "configs/legacy/policy/pi05_panda_ball_full_sft_h50_v2.yaml"
        )
        historical_config = _build_config(historical, 2)
        assert not historical_config.model.discrete_state_input
        assert historical_config.data.base_config.drop_n_last_frames == 49

        task_profile = resolve_policy_profile(
            {
                "schema_version": 2,
                "profile": root / "configs/contracts/policy/pi05_state16_h50.yaml",
                "training": root / "configs/training/policy/conveyor_clean.yaml",
                "batch_size": 256,
            },
            source_count=15539,
        )
        launch = build_launch_train_config(
            profile=task_profile,
            request=SFTLaunchRequest(
                None, "conveyor-check", "formal", False, 8, 256, task_id="conveyor_sort"
            ),
            assets_root=tmp_path / "assets",
            checkpoint_root=tmp_path / "checkpoints",
            wandb_enabled=False,
            task_parameters={
                "config_name": "conveyor_test",
                "repo_id": "local/conveyor",
                "task_id": "conveyor_sort",
                "action_contract_id": "panda_osc_pose_delta_conveyor_v2",
                "extra_metadata": {"variant": "surface"},
            },
        )
        assert launch.model == old.model
        assert launch.batch_size == 256 and launch.num_train_steps == 183
        assert launch.lr_schedule.warmup_steps == 10
        assert launch.keep_period == 61 and launch.save_interval == 15
        assert launch.data.assets.asset_id == "local/conveyor"
        assert launch.data.assets.assets_dir == str(tmp_path / "assets")
        assert launch.checkpoint_dir == tmp_path / "checkpoints/conveyor_test/conveyor-check"
        assert launch.policy_metadata["per_device_batch_size"] == 32
        assert "level" not in launch.policy_metadata
        assert not launch.wandb_enabled
