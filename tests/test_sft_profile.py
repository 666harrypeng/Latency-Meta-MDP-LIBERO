from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.sft_profile import SFTProfile, load_sft_profile


def test_panda_ball_sft_profile_locks_three_level_specific_full_sft_configs() -> None:
    profile = load_sft_profile(Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"))

    assert profile.profile_id == "pi05_panda_ball_full_sft_h50_v2"
    assert profile.full_parameter is True
    assert profile.temporal_contract.contract_id == "h50_e25_d20_k6_v1"
    assert profile.action_horizon == 50
    assert profile.launch_trigger_horizon == 25
    assert profile.drop_n_last_frames == 49
    assert profile.fps == 50
    assert profile.state_dim == 8
    assert profile.source_action_dim == 7
    assert profile.extra_delta_transform is False
    assert profile.batch_size == 192
    assert profile.num_train_steps == 3_999
    assert profile.warmup_steps == 200
    assert profile.save_interval == 100
    assert profile.keep_period == 1_333
    assert profile.log_interval == 33
    assert profile.peak_learning_rate == 5e-5
    assert profile.decay_learning_rate == 5e-6
    assert profile.openpi_patch_sha256 == sha256_file(
        Path("patches/openpi/0001-filter-incomplete-action-chunks.patch")
    )
    assert set(profile.levels) == {1, 2, 3}
    assert len({level.config_name for level in profile.levels.values()}) == 3
    assert len({level.repo_id for level in profile.levels.values()}) == 3
    for level, level_profile in profile.levels.items():
        assert level_profile.config_name == f"pi05_panda_ball_l{level}_full_h50"
        assert level_profile.repo_id == (
            f"yypeng666/metamdp-robosuite-franka-moving_ball-l{level}-clean-50hz-h50-v2"
        )
        assert f"-l{level}-" in level_profile.repo_id
        assert level_profile.repo_id.endswith("-50hz-h50-v2")


def test_sft_profile_rejects_a_tail_filter_that_does_not_match_horizon() -> None:
    profile = load_sft_profile(Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"))

    with pytest.raises(ValueError, match="action_horizon - 1"):
        replace(profile, drop_n_last_frames=48)


def test_sft_profile_rejects_cross_level_dataset_reuse() -> None:
    profile = load_sft_profile(Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"))
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


def test_sft_profile_requires_milestones_to_partition_the_formal_run() -> None:
    profile = load_sft_profile(Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"))

    with pytest.raises(ValueError, match="milestone"):
        replace(profile, keep_period=1_332)

    with pytest.raises(ValueError, match="rolling"):
        replace(profile, save_interval=1_333)
