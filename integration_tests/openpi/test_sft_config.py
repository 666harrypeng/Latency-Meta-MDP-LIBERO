from __future__ import annotations

import sys
from pathlib import Path

from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree
from latency_meta_mdp.openpi_sft import register_sft_configs
from latency_meta_mdp.sft_profile import load_sft_profile

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_register_sft_configs_builds_three_full_parameter_h50_profiles(
    tmp_path: Path,
) -> None:
    profile = load_sft_profile(
        _PROJECT_ROOT / "configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"
    )
    with temporary_patched_openpi_worktree(
        openpi_root=_PROJECT_ROOT / "third_party/openpi",
        patch_path=_PROJECT_ROOT / "patches/openpi/0001-filter-incomplete-action-chunks.patch",
        expected_revision=profile.openpi_revision,
        additional_patch_paths=(
            _PROJECT_ROOT / "patches/openpi/0002-save-completed-step-checkpoints.patch",
        ),
    ) as worktree:
        sys.path.insert(0, str(worktree / "src"))
        try:
            from openpi.training.config import _CONFIGS_DICT

            names = register_sft_configs(profile)

            assert names == tuple(profile.levels[level].config_name for level in (1, 2, 3))
            configs = tuple(_CONFIGS_DICT[name] for name in names)
        finally:
            sys.path.remove(str(worktree / "src"))

    for level, config in zip((1, 2, 3), configs, strict=True):
        assert config.model.pi05 is True
        assert config.model.action_horizon == 50
        assert config.model.discrete_state_input is False
        assert config.model.paligemma_variant == "gemma_2b"
        assert config.model.action_expert_variant == "gemma_300m"
        assert config.freeze_filter.__class__.__name__ == "Nothing"
        assert config.batch_size == 192
        assert config.num_train_steps == 3_999
        assert config.save_interval == 100
        assert config.keep_period == 1_333
        assert config.policy_metadata["level"] == level
        assert config.policy_metadata["launch_trigger_horizon"] == 25

        data = config.data.create(tmp_path, config.model)
        assert data.repo_id == profile.levels[level].repo_id
        assert data.drop_n_last_frames == 49
        assert data.action_sequence_keys == ("actions",)
