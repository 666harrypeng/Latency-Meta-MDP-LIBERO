from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.policy_data import (
    ACTION_DIM,
    ACTION_HORIZON,
    POLICY_STATE_DIM,
    load_policy_episode,
    rotation_matrix_to_rotvec,
)
from latency_meta_mdp.recording import RecordProfile


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_rotation_matrix_to_rotvec_uses_canonical_shortest_axis_angle() -> None:
    np.testing.assert_allclose(rotation_matrix_to_rotvec(np.eye(3)), np.zeros(3))
    np.testing.assert_allclose(
        rotation_matrix_to_rotvec(_rotation_z(np.pi / 2)),
        np.asarray([0.0, 0.0, np.pi / 2]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        rotation_matrix_to_rotvec(_rotation_z(np.pi)),
        np.asarray([0.0, 0.0, np.pi]),
        atol=1e-7,
    )


def test_load_policy_episode_builds_same_tick_8d_state_and_complete_h16_sources(
    tmp_path: Path,
) -> None:
    episode = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l1-seed-000010-attempt-002",
            level=1,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.SFT,
            camera_width=8,
            camera_height=8,
        ),
    )
    episode_dir = tmp_path / episode.metadata.episode_id
    write_synchronized_episode_artifact(episode=episode, output_dir=episode_dir)

    policy_episode = load_policy_episode(episode_dir)

    transition_count = len(episode.transitions)
    assert policy_episode.state.shape == (transition_count, POLICY_STATE_DIM)
    assert policy_episode.actions.shape == (transition_count, ACTION_DIM)
    assert policy_episode.action_horizon == ACTION_HORIZON
    assert policy_episode.agentview_rgb.shape == (transition_count, 8, 8, 3)
    assert policy_episode.wrist_rgb.shape == (transition_count, 8, 8, 3)
    assert policy_episode.valid_action_chunk_sources.tolist() == list(
        range(transition_count - ACTION_HORIZON + 1)
    )
    assert policy_episode.source_formal_tick.tolist() == list(range(transition_count))
    assert policy_episode.source_time_us.tolist() == [
        tick * 20_000 for tick in range(transition_count)
    ]

    initial = episode.boundaries[0].deployment
    np.testing.assert_allclose(policy_episode.state[0, :3], initial.eef_position_world)
    np.testing.assert_allclose(
        policy_episode.state[0, 3:6],
        rotation_matrix_to_rotvec(initial.eef_orientation_matrix_world),
    )
    np.testing.assert_allclose(policy_episode.state[0, 6:], initial.gripper_qpos)
    np.testing.assert_allclose(
        policy_episode.actions,
        np.stack([transition.expert_action for transition in episode.transitions]),
    )
    assert not hasattr(policy_episode, "object_pose")
    assert not hasattr(policy_episode, "privileged")


def test_load_policy_episode_rejects_legacy_artifact_schema(tmp_path: Path) -> None:
    episode_dir = tmp_path / "legacy"
    episode_dir.mkdir()
    (episode_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "format_id": "synchronized_episode_npz_v2"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw-v3"):
        load_policy_episode(episode_dir)
