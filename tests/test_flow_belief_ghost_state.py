from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.belief.flow.ghost_state import (
    reconstruct_return_state,
    select_sample_medoid,
    valid_sample_mask,
)

_CONFIG = Path("configs/analysis/flow_belief_agentview_ghost_v1.yaml")


def test_ghost_config_locks_agentview_render_contract() -> None:
    config = load_flow_belief_ghost_config(_CONFIG)

    assert config.renderer_id == "flow_belief_agentview_ghost_v1"
    assert config.camera_name == "agentview"
    assert (config.width, config.height) == (256, 256)
    assert config.display_delay_ticks == (1, 5, 10, 15, 20)
    assert config.overlay_alpha == 0.55
    assert config.ground_truth_rgb == (0, 114, 178)
    assert config.prediction_rgb == (213, 94, 0)
    assert config.medoid_metric == "normalized_l2"

    with pytest.raises(ValueError, match="agentview"):
        replace(config, camera_name="robot0_eye_in_hand")


def test_return_state_reconstruction_preserves_nuisance_coordinates() -> None:
    predicted = np.arange(22, dtype=np.float64) / 100.0
    predicted[14] = 0.06
    target_gripper_qpos = np.asarray([0.041, -0.039])
    target_gripper_qvel = np.asarray([0.02, -0.01])
    target_object_pose = np.asarray([0.1, -0.2, 0.85, 0.5, 0.5, 0.5, 0.5])
    target_object_velocity = np.asarray([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])

    state = reconstruct_return_state(
        predicted_state=predicted,
        target_gripper_qpos=target_gripper_qpos,
        target_gripper_qvel=target_gripper_qvel,
        target_object_pose=target_object_pose,
        target_object_velocity=target_object_velocity,
    )

    np.testing.assert_allclose(state.robot_qpos, predicted[:7])
    np.testing.assert_allclose(state.robot_qvel, predicted[7:14])
    np.testing.assert_allclose(state.gripper_qpos, [0.031, -0.029])
    np.testing.assert_allclose(state.gripper_qvel, [0.08, -0.07])
    np.testing.assert_allclose(state.object_qpos[:3], predicted[16:19])
    np.testing.assert_allclose(state.object_qpos[3:], target_object_pose[3:])
    np.testing.assert_allclose(state.object_qvel[:3], predicted[19:22])
    np.testing.assert_allclose(state.object_qvel[3:], target_object_velocity[3:])
    assert state.robot_qpos.flags.writeable is False


def test_valid_sample_mask_rejects_joint_gripper_and_workspace_violations() -> None:
    samples = np.zeros((5, 22), dtype=np.float64)
    samples[:, 14] = 0.04
    samples[:, 16:19] = [0.0, 0.0, 0.9]
    samples[1, 0] = 2.0
    samples[2, 14] = -0.01
    samples[3, 16] = 1.5
    samples[4, 21] = np.nan
    joint_ranges = np.tile(np.asarray([-1.0, 1.0]), (7, 1))
    object_bounds = np.asarray([[-0.5, 0.5], [-0.5, 0.5], [0.8, 1.2]])

    valid = valid_sample_mask(
        physical_samples=samples,
        joint_ranges=joint_ranges,
        gripper_width_range=(0.0, 0.08),
        object_position_bounds=object_bounds,
    )

    np.testing.assert_array_equal(valid, [True, False, False, False, False])


def test_medoid_uses_only_valid_samples_and_breaks_ties_by_index() -> None:
    samples = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )

    assert select_sample_medoid(samples, np.asarray([True, True, True, False])) == 0
    with pytest.raises(ValueError, match="valid sample"):
        select_sample_medoid(samples, np.zeros(4, dtype=np.bool_))
