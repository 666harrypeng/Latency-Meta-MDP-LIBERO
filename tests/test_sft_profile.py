from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.sft_profile import SFTProfile, load_sft_profile


def test_panda_ball_sft_profile_locks_three_level_specific_full_sft_configs() -> None:
    profile = load_sft_profile(
        Path("configs/policy/pi05_panda_ball_full_sft_v1.yaml")
    )

    assert profile.profile_id == "pi05_panda_ball_full_sft_v1"
    assert profile.full_parameter is True
    assert profile.action_horizon == 16
    assert profile.execution_horizon == 8
    assert profile.drop_n_last_frames == 15
    assert profile.fps == 50
    assert profile.state_dim == 8
    assert profile.source_action_dim == 7
    assert profile.extra_delta_transform is False
    assert profile.batch_size == 64
    assert profile.num_train_steps == 12_000
    assert profile.save_interval == 4_000
    assert profile.openpi_patch_sha256 == sha256_file(
        Path("patches/openpi/0001-filter-incomplete-action-chunks.patch")
    )
    assert set(profile.levels) == {1, 2, 3}
    assert len({level.config_name for level in profile.levels.values()}) == 3
    assert len({level.repo_id for level in profile.levels.values()}) == 3
    for level, level_profile in profile.levels.items():
        assert level_profile.config_name == f"pi05_panda_ball_l{level}_full"
        assert f"-l{level}-" in level_profile.repo_id


def test_sft_profile_rejects_a_tail_filter_that_does_not_match_horizon() -> None:
    profile = load_sft_profile(
        Path("configs/policy/pi05_panda_ball_full_sft_v1.yaml")
    )

    with pytest.raises(ValueError, match="action_horizon - 1"):
        replace(profile, drop_n_last_frames=14)


def test_sft_profile_rejects_cross_level_dataset_reuse() -> None:
    profile = load_sft_profile(
        Path("configs/policy/pi05_panda_ball_full_sft_v1.yaml")
    )
    shared = profile.levels[1]

    with pytest.raises(ValueError, match="repo ids must be unique"):
        SFTProfile(
            **{
                **profile.__dict__,
                "levels": {
                    1: shared,
                    2: replace(profile.levels[2], repo_id=shared.repo_id),
                    3: profile.levels[3],
                },
            }
        )
