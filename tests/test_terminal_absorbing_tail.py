from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.data.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.data.recording import RecordProfile
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.io.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.legacy.belief_data import load_belief_episode
from latency_meta_mdp.runtime.temporal_contract import load_temporal_contract


def _episode(tmp_path: Path):
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l2-seed-000010-absorbing-tail",
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
    return load_belief_episode(output)


def test_absorbing_tail_is_target_only_aligned_and_non_mutating(tmp_path: Path) -> None:
    from latency_meta_mdp.legacy.terminal_absorbing_tail import (
        build_terminal_absorbing_tail,
    )

    episode = _episode(tmp_path)
    contract = load_temporal_contract(Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"))
    action_contract = load_action_contract(
        Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
    )
    original_actions = episode.expert_actions.copy()
    original_images = episode.deployment.agentview_rgb.copy()

    view = build_terminal_absorbing_tail(
        episode=episode,
        temporal_contract=contract,
        action_contract=action_contract,
    )

    assert view.protocol_id == "terminal_absorbing_tail_v1"
    assert view.real_transition_count == episode.transition_count
    assert view.real_boundary_count == episode.boundary_count
    assert view.tail_tick_count == 70
    assert view.extended_transition_count == episode.transition_count + 70
    assert view.extended_boundary_count == episode.boundary_count + 70
    assert view.boundary_tick.tolist() == list(range(view.extended_boundary_count))
    assert view.boundary_time_us.tolist() == [
        tick * 20_000 for tick in range(view.extended_boundary_count)
    ]

    terminal_tick = episode.transition_count
    assert view.boundary_source_index[: episode.boundary_count].tolist() == list(
        range(episode.boundary_count)
    )
    assert np.all(view.boundary_source_index[episode.boundary_count :] == terminal_tick)
    assert np.all(view.transition_source_valid[:terminal_tick])
    assert not np.any(view.transition_source_valid[terminal_tick:])
    assert np.all(view.boundary_target_valid)
    assert not np.any(view.boundary_is_absorbing[:terminal_tick])
    assert np.all(view.boundary_is_absorbing[terminal_tick:])
    assert not np.any(view.transition_is_absorbing[:terminal_tick])
    assert np.all(view.transition_is_absorbing[terminal_tick:])

    expected_hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(view.hold_action, expected_hold)
    np.testing.assert_array_equal(view.expert_actions[:terminal_tick], episode.expert_actions)
    np.testing.assert_array_equal(
        view.expert_actions[terminal_tick:],
        np.repeat(expected_hold[None, :], 70, axis=0),
    )
    assert np.all(view.expert_phase[terminal_tick:] == "terminal_absorbing")

    extended_images = view.extend_boundary_array(episode.deployment.agentview_rgb)
    np.testing.assert_array_equal(
        extended_images[: episode.boundary_count], episode.deployment.agentview_rgb
    )
    np.testing.assert_array_equal(extended_images[-1], episode.deployment.agentview_rgb[-1])
    extended_velocity = view.extend_boundary_array(
        episode.deployment.robot_qvel,
        absorbing_value=np.zeros(7),
    )
    np.testing.assert_array_equal(
        extended_velocity[:terminal_tick], episode.deployment.robot_qvel[:terminal_tick]
    )
    np.testing.assert_array_equal(extended_velocity[terminal_tick:], 0.0)
    np.testing.assert_array_equal(episode.expert_actions, original_actions)
    np.testing.assert_array_equal(episode.deployment.agentview_rgb, original_images)


def test_absorbing_tail_rejects_failure_episode(tmp_path: Path) -> None:
    from latency_meta_mdp.legacy.terminal_absorbing_tail import (
        build_terminal_absorbing_tail,
    )

    episode = _episode(tmp_path)
    status = episode.supervision.boundary_outcome_status.copy()
    status[-1] = "failure"
    failed = replace(
        episode,
        supervision=replace(
            episode.supervision,
            boundary_outcome_status=status,
        ),
    )

    with pytest.raises(ValueError, match="successful episode"):
        build_terminal_absorbing_tail(
            episode=failed,
            temporal_contract=load_temporal_contract(
                Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml")
            ),
            action_contract=load_action_contract(
                Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
            ),
        )
