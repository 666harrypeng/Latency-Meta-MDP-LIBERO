from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.recording import RecordProfile


def _belief_episode_dir(tmp_path: Path) -> Path:
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l2-seed-000010-belief-view",
            level=2,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.BELIEF,
            camera_width=8,
            camera_height=8,
        ),
    )
    output = tmp_path / "episode"
    write_synchronized_episode_artifact(episode=episode, output_dir=output)
    return output


def test_belief_episode_view_separates_deployment_and_privileged_streams(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_data import load_belief_episode

    view = load_belief_episode(_belief_episode_dir(tmp_path))

    assert view.level == 2
    assert view.record_profile == "belief"
    assert view.boundary_count == view.transition_count + 1
    assert view.deployment.agentview_rgb.shape == (view.boundary_count, 8, 8, 3)
    assert view.deployment.wrist_rgb.shape == (view.boundary_count, 8, 8, 3)
    assert view.deployment.robot_qpos.shape == (view.boundary_count, 7)
    assert view.deployment.eef_orientation_matrix_world.shape == (
        view.boundary_count,
        3,
        3,
    )
    assert not hasattr(view.deployment, "object_pose")
    assert not hasattr(view.deployment, "commanded_motion_position")
    assert view.supervision.object_pose.shape == (view.boundary_count, 7)
    assert view.supervision.object_velocity.shape == (view.boundary_count, 6)
    assert view.supervision.commanded_motion_position.shape == (view.boundary_count, 3)
    assert view.supervision.left_pad_contact.dtype == np.bool_
    assert view.expert_actions.shape == (view.transition_count, 7)
    assert np.all(view.action_mask)


def test_belief_indices_use_unpadded_history_and_valid_future_boundaries(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_data import build_belief_sample_indices, load_belief_episode

    view = load_belief_episode(_belief_episode_dir(tmp_path))
    indices = build_belief_sample_indices(
        episode=view,
        history_ticks=6,
        delay_ticks=tuple(range(1, 21)),
    )

    assert indices
    assert all(index.history_start_tick == index.source_tick - 5 for index in indices)
    assert all(
        index.target_tick == index.source_tick + index.branch_delay_tick
        for index in indices
    )
    assert all(index.target_tick < view.boundary_count for index in indices)
    assert all(
        index.source_tick + index.branch_delay_tick <= view.transition_count
        for index in indices
    )
    expected_count = sum(
        max(0, view.transition_count - delay_tick - 6 + 2)
        for delay_tick in range(1, 21)
    )
    assert len(indices) == expected_count
    assert {index.branch_delay_tick for index in indices} == set(range(1, 21))


def test_launch_history_and_return_target_do_not_cross_privilege_boundary(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_data import (
        build_belief_sample_indices,
        build_launch_history,
        build_return_target,
        load_belief_episode,
        teacher_forced_pre_return_actions,
    )

    view = load_belief_episode(_belief_episode_dir(tmp_path))
    index = build_belief_sample_indices(
        episode=view,
        history_ticks=6,
        delay_ticks=(4,),
    )[0]
    history = build_launch_history(episode=view, index=index)
    target = build_return_target(episode=view, index=index)
    actions = teacher_forced_pre_return_actions(episode=view, index=index)

    assert history.boundary_tick.tolist() == list(range(6))
    assert history.agentview_rgb.shape == (6, 8, 8, 3)
    assert not hasattr(history, "object_pose")
    assert not hasattr(history, "branch_delay_tick")
    assert target.branch_delay_tick == 4
    assert target.target_tick == 9
    np.testing.assert_array_equal(
        target.object_pose,
        view.supervision.object_pose[target.target_tick],
    )
    np.testing.assert_array_equal(
        target.agentview_rgb,
        view.deployment.agentview_rgb[target.target_tick],
    )
    np.testing.assert_array_equal(
        actions,
        view.expert_actions[index.source_tick : index.target_tick],
    )


def test_belief_loader_rejects_nonbelief_episode(tmp_path: Path) -> None:
    from latency_meta_mdp.belief_data import load_belief_episode

    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l1-seed-000010-sft-only",
            level=1,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.SFT,
            camera_width=8,
            camera_height=8,
        ),
    )
    output = tmp_path / "sft_episode"
    write_synchronized_episode_artifact(episode=episode, output_dir=output)

    with pytest.raises(ValueError, match="BELIEF record profile"):
        load_belief_episode(output)
