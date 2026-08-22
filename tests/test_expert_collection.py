from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.outcomes import OutcomeStatus
from latency_meta_mdp.recording import RecordProfile


def test_collect_expert_episode_builds_one_complete_synchronized_pilot() -> None:
    collection = importlib.import_module("latency_meta_mdp.expert_collection")
    spec = collection.ExpertEpisodeSpec(
        episode_id="l1-seed-000010-attempt-000",
        level=1,
        scene_seed=10,
        motion_seed=10,
        expert_seed=10,
        record_profile=RecordProfile.PILOT_DEBUG,
        camera_width=32,
        camera_height=32,
    )

    episode = collection.collect_expert_episode(
        project_root=Path.cwd(),
        spec=spec,
    )

    episode.validate_complete()
    assert episode.terminal_status is OutcomeStatus.SUCCESS
    assert episode.boundaries[-1].time_us < 4_000_000
    assert len(episode.transitions) == len(episode.boundaries) - 1
    assert episode.metadata.episode_id == spec.episode_id
    assert episode.metadata.level == 1
    assert episode.metadata.scene_seed == 10
    assert episode.metadata.motion_seed == 10
    assert episode.metadata.expert_seed == 10
    assert set(episode.metadata.config_sha256) == {
        "runtime",
        "task",
        "motion",
        "control",
        "expert",
    }
    assert all(len(value) == 64 for value in episode.metadata.config_sha256.values())

    initial = episode.boundaries[0]
    assert initial.time_us == 0
    assert initial.control_debug is not None
    assert initial.control_debug.applied_reference is None
    assert initial.control_debug.applied_reference_source_formal_tick is None
    assert initial.control_debug.nullspace_joint_position_error is None
    assert initial.control_debug.eef_position_error is None
    assert initial.control_debug.eef_orientation_error_rotvec is None
    assert set(initial.deployment.images) == {"agentview", "robot0_eye_in_hand"}
    assert all(
        camera.rgb.shape == (32, 32, 3)
        and camera.source_time_us == boundary.time_us
        for boundary in episode.boundaries
        for camera in boundary.deployment.images.values()
    )

    for target_tick, boundary in enumerate(episode.boundaries[1:], start=1):
        debug = boundary.control_debug
        assert debug is not None
        assert debug.applied_reference_source_formal_tick == target_tick - 1
        assert debug.nullspace_joint_position_error is not None
        assert debug.eef_position_error is not None
        assert debug.eef_orientation_error_rotvec is not None
        np.testing.assert_array_equal(
            debug.applied_reference,
            episode.transitions[target_tick - 1].expert_action,
        )

    assert all(boundary.privileged is not None for boundary in episode.boundaries)
    assert all(
        boundary.privileged.object_velocity.shape == (6,)
        and boundary.deployment.gripper_qpos.shape == (2,)
        and boundary.deployment.gripper_qvel.shape == (2,)
        and boundary.deployment.eef_position_world.shape == (3,)
        and boundary.deployment.eef_orientation_matrix_world.shape == (3, 3)
        for boundary in episode.boundaries
        if boundary.privileged is not None
    )
    assert [event.kind.value for event in episode.physical_events] == [
        "first_contact",
        "stable_grasp",
        "handoff",
        "lift_threshold",
        "success",
    ]
    assert all(
        event.time_us == event.physics_step_index * episode.metadata.physics_dt_us
        for event in episode.physical_events
    )

    deployment = episode.deployment_view()
    assert not hasattr(deployment.boundaries[0], "privileged")
    assert not hasattr(deployment.transitions[0], "expert_audit")


def test_write_synchronized_episode_artifact_is_lossless_and_no_overwrite(
    tmp_path: Path,
) -> None:
    collection = importlib.import_module("latency_meta_mdp.expert_collection")
    artifact_io = importlib.import_module("latency_meta_mdp.episode_artifacts")
    episode = collection.collect_expert_episode(
        project_root=Path.cwd(),
        spec=collection.ExpertEpisodeSpec(
            episode_id="l1-seed-000010-attempt-001",
            level=1,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.PILOT_DEBUG,
            camera_width=8,
            camera_height=8,
        ),
    )
    output_dir = tmp_path / episode.metadata.episode_id

    manifest_path = artifact_io.write_synchronized_episode_artifact(
        episode=episode,
        output_dir=output_dir,
    )

    assert manifest_path == output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["format_id"] == "synchronized_episode_npz_v3"
    assert manifest["episode_id"] == episode.metadata.episode_id
    assert manifest["boundary_count"] == len(episode.boundaries)
    assert manifest["transition_count"] == len(episode.transitions)
    assert set(manifest["artifacts"]) == {"arrays.npz", "events.json", "metadata.json"}
    assert all(len(value) == 64 for value in manifest["artifacts"].values())

    with np.load(output_dir / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["boundary_time_us"].tolist() == [
            boundary.time_us for boundary in episode.boundaries
        ]
        assert arrays["agentview_rgb"].shape == (len(episode.boundaries), 8, 8, 3)
        assert arrays["robot0_eye_in_hand_rgb"].shape == (
            len(episode.boundaries),
            8,
            8,
            3,
        )
        assert arrays["gripper_qpos"].shape == (len(episode.boundaries), 2)
        assert arrays["gripper_qvel"].shape == (len(episode.boundaries), 2)
        assert arrays["eef_position_world"].shape == (len(episode.boundaries), 3)
        assert arrays["eef_orientation_matrix_world"].shape == (
            len(episode.boundaries),
            3,
            3,
        )
        np.testing.assert_array_equal(
            arrays["gripper_qpos"],
            np.stack(
                [boundary.deployment.gripper_qpos for boundary in episode.boundaries]
            ),
        )
        np.testing.assert_array_equal(
            arrays["gripper_qvel"],
            np.stack(
                [boundary.deployment.gripper_qvel for boundary in episode.boundaries]
            ),
        )
        np.testing.assert_array_equal(
            arrays["eef_position_world"],
            np.stack(
                [boundary.deployment.eef_position_world for boundary in episode.boundaries]
            ),
        )
        np.testing.assert_array_equal(
            arrays["eef_orientation_matrix_world"],
            np.stack(
                [
                    boundary.deployment.eef_orientation_matrix_world
                    for boundary in episode.boundaries
                ]
            ),
        )
        assert arrays["control_reference_valid"].tolist() == [False] + [True] * len(
            episode.transitions
        )
        assert np.all(np.isnan(arrays["applied_reference"][0]))
        np.testing.assert_array_equal(
            arrays["applied_reference"][1],
            episode.transitions[0].expert_action,
        )

    events = json.loads((output_dir / "events.json").read_text(encoding="utf-8"))
    assert [event["kind"] for event in events["events"]] == [
        event.kind.value for event in episode.physical_events
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        artifact_io.write_synchronized_episode_artifact(
            episode=episode,
            output_dir=output_dir,
        )
