from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief_data import (
    BeliefDeploymentStream,
    BeliefEpisodeView,
    BeliefSupervisionStream,
)
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.temporal_contract import load_temporal_contract


def _episode(*, transition_count: int = 100) -> BeliefEpisodeView:
    boundary_count = transition_count + 1
    tick = np.arange(boundary_count, dtype=np.int64)
    qpos = tick[:, None] + np.arange(7)[None, :] / 10
    qvel = 2 * qpos
    gripper_qpos = np.stack((0.01 * tick + 0.04, -0.005 * tick - 0.04), axis=1)
    gripper_qvel = np.stack((0.02 * tick, -0.01 * tick), axis=1)
    object_pose = np.zeros((boundary_count, 7), dtype=float)
    object_pose[:, :3] = np.stack((tick, 2 * tick, 3 * tick), axis=1)
    object_pose[:, 3] = 1.0
    object_velocity = np.zeros((boundary_count, 6), dtype=float)
    object_velocity[:, :3] = np.stack((4 * tick, 5 * tick, 6 * tick), axis=1)
    phases = np.asarray(
        ["pregrasp"] * 30
        + ["approach"] * 25
        + ["close"] * 15
        + ["lift"] * (transition_count - 70),
    )
    actions = np.arange(transition_count * 7, dtype=float).reshape(transition_count, 7)
    return BeliefEpisodeView(
        episode_id="synthetic-l2",
        task_id="dynamic_grasp_lift",
        instruction="Grasp the moving ball and lift it.",
        level=2,
        record_profile="belief",
        deployment=BeliefDeploymentStream(
            boundary_tick=tick,
            boundary_time_us=tick * 20_000,
            agentview_rgb=np.zeros((boundary_count, 2, 2, 3), dtype=np.uint8),
            wrist_rgb=np.zeros((boundary_count, 2, 2, 3), dtype=np.uint8),
            robot_qpos=qpos,
            robot_qvel=qvel,
            gripper_qpos=gripper_qpos,
            gripper_qvel=gripper_qvel,
            eef_position_world=np.zeros((boundary_count, 3)),
            eef_orientation_matrix_world=np.repeat(
                np.eye(3)[None, :, :], boundary_count, axis=0
            ),
        ),
        supervision=BeliefSupervisionStream(
            object_pose=object_pose,
            object_velocity=object_velocity,
            commanded_motion_position=np.zeros((boundary_count, 3)),
            commanded_motion_velocity=np.zeros((boundary_count, 3)),
            commanded_motion_acceleration=np.zeros((boundary_count, 3)),
            commanded_motion_segment_index=np.zeros(boundary_count, dtype=np.int64),
            left_pad_contact=tick >= 55,
            right_pad_contact=tick >= 56,
            handoff_state=np.where(tick >= 70, "physical", "driven"),
            relative_geometry=np.stack((7 * tick, 8 * tick, 9 * tick), axis=1),
            boundary_outcome_status=np.asarray(
                ["running"] * transition_count + ["success"]
            ),
        ),
        transition_source_tick=np.arange(transition_count, dtype=np.int64),
        transition_target_tick=np.arange(1, transition_count + 1, dtype=np.int64),
        expert_actions=actions,
        action_mask=np.ones((transition_count, 7), dtype=np.bool_),
        expert_phase=phases,
    )


def test_checked_in_return_belief_audit_config_is_valid() -> None:
    from latency_meta_mdp.return_belief_geometry import load_return_belief_audit_config

    config = load_return_belief_audit_config(
        Path("configs/analysis/return_belief_geometry_v1.yaml")
    )

    assert config.analysis_id == "return_belief_geometry_v1"
    assert config.central_probability_mass == 0.9
    assert config.action_prefix_ticks == 5
    assert config.summary_quantiles == (0.0, 0.1, 0.5, 0.9, 1.0)


def test_return_state_stream_uses_joint_gripper_width_and_ball_state() -> None:
    from latency_meta_mdp.return_belief_geometry import build_return_state_stream

    episode = _episode()
    states = build_return_state_stream(episode)

    assert states.shape == (101, 22)
    np.testing.assert_array_equal(states[:, :7], episode.deployment.robot_qpos)
    np.testing.assert_array_equal(states[:, 7:14], episode.deployment.robot_qvel)
    np.testing.assert_allclose(
        states[:, 14],
        episode.deployment.gripper_qpos[:, 0]
        - episode.deployment.gripper_qpos[:, 1],
    )
    np.testing.assert_allclose(
        states[:, 15],
        episode.deployment.gripper_qvel[:, 0]
        - episode.deployment.gripper_qvel[:, 1],
    )
    np.testing.assert_array_equal(
        states[:, 16:19], episode.supervision.object_pose[:, :3]
    )
    np.testing.assert_array_equal(
        states[:, 19:22], episode.supervision.object_velocity[:, :3]
    )


def test_return_contexts_preserve_delay_cloud_and_only_complete_action_bundles() -> None:
    from latency_meta_mdp.return_belief_geometry import build_return_contexts

    episode = _episode()
    contract = load_temporal_contract(
        Path("configs/temporal/h50_e25_d20_k6_v1.yaml")
    )
    law = load_latency_law(
        Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml")
    )

    contexts = build_return_contexts(
        episode=episode,
        temporal_contract=contract,
        latency_law=law,
    )

    assert len(contexts) == 51
    first = contexts[0]
    assert first.source_tick == 25
    assert first.source_phase == "pregrasp"
    assert first.delay_ticks.tolist() == list(range(1, 21))
    assert first.target_ticks.tolist() == list(range(26, 46))
    np.testing.assert_allclose(first.probabilities, law.probabilities)
    assert first.future_states.shape == (20, 22)
    assert first.relative_positions.shape == (20, 3)
    assert first.action_targets is not None
    assert first.action_targets.shape == (20, 50, 7)
    np.testing.assert_array_equal(
        first.action_targets[0], episode.expert_actions[26:76]
    )
    np.testing.assert_array_equal(
        first.action_targets[-1], episode.expert_actions[45:95]
    )

    assert contexts[5].source_tick == 30
    assert contexts[5].action_targets is not None
    assert contexts[6].source_tick == 31
    assert contexts[6].action_targets is None
    assert contexts[-1].source_tick == 75
    assert contexts[-1].target_ticks[-1] == 95


def _context(
    states: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    phases: np.ndarray | None = None,
    contacts: np.ndarray | None = None,
    handoffs: np.ndarray | None = None,
    actions: np.ndarray | None = None,
):
    from latency_meta_mdp.return_belief_geometry import ReturnContext

    branch_count = len(states)
    weights = (
        np.full(branch_count, 1.0 / branch_count)
        if probabilities is None
        else probabilities
    )
    return ReturnContext(
        episode_id="geometry",
        level=2,
        source_tick=25,
        source_phase="pregrasp",
        delay_ticks=np.arange(1, branch_count + 1),
        probabilities=weights,
        target_ticks=25 + np.arange(1, branch_count + 1),
        future_states=states,
        relative_positions=states[:, 16:19],
        return_phases=(
            np.asarray(["pregrasp"] * branch_count) if phases is None else phases
        ),
        return_contact=(
            np.zeros(branch_count, dtype=np.bool_) if contacts is None else contacts
        ),
        return_physical_handoff=(
            np.zeros(branch_count, dtype=np.bool_) if handoffs is None else handoffs
        ),
        action_targets=actions,
    )


def test_state_geometry_distinguishes_zero_straight_and_curved_clouds() -> None:
    from latency_meta_mdp.return_belief_geometry import (
        StateNormalization,
        measure_state_geometry,
    )

    normalization = StateNormalization(mean=np.zeros(22), scale=np.ones(22))
    zero = measure_state_geometry(
        _context(np.zeros((5, 22))),
        normalization=normalization,
        central_probability_mass=0.9,
    )
    assert zero.combined_rms_spread == 0.0
    assert zero.pc1_explained_ratio == 0.0
    assert zero.effective_rank == 0.0
    assert zero.affine_residual_fraction == 0.0
    assert zero.pc1_gaussian_quantile_rmse == 0.0
    assert zero.central_tortuosity == 1.0

    straight_states = np.zeros((5, 22))
    straight_states[:, 0] = np.arange(5)
    straight = measure_state_geometry(
        _context(straight_states),
        normalization=normalization,
        central_probability_mass=0.9,
    )
    assert straight.pc1_explained_ratio == 1.0
    assert straight.pc12_explained_ratio == 1.0
    assert straight.effective_rank == 1.0
    assert straight.affine_residual_fraction == 0.0
    assert straight.pc1_gaussian_quantile_rmse > 0.0
    assert straight.central_tortuosity == 1.0

    angles = np.linspace(0.0, np.pi / 2.0, 9)
    curved_states = np.zeros((9, 22))
    curved_states[:, 0] = np.cos(angles)
    curved_states[:, 1] = np.sin(angles)
    curved = measure_state_geometry(
        _context(curved_states),
        normalization=normalization,
        central_probability_mass=0.9,
    )
    assert 0.5 < curved.pc1_explained_ratio < 1.0
    assert curved.pc12_explained_ratio == 1.0
    assert curved.affine_residual_fraction > 0.0
    assert curved.central_tortuosity > 1.0


def test_state_geometry_reports_physical_spread_and_transition_probabilities() -> None:
    from latency_meta_mdp.return_belief_geometry import (
        StateNormalization,
        measure_state_geometry,
    )

    states = np.zeros((2, 22))
    states[1, :7] = 2.0
    states[1, 16:19] = 0.2
    context = _context(
        states,
        probabilities=np.asarray([0.25, 0.75]),
        phases=np.asarray(["pregrasp", "approach"]),
        contacts=np.asarray([False, True]),
        handoffs=np.asarray([False, True]),
    )
    metrics = measure_state_geometry(
        context,
        normalization=StateNormalization(mean=np.zeros(22), scale=np.ones(22)),
        central_probability_mass=0.9,
    )

    assert metrics.robot_qpos_rms_spread == np.sqrt(0.75)
    assert metrics.object_position_rms_spread == np.sqrt(0.0075)
    assert metrics.phase_crossing_probability == 0.75
    assert metrics.contact_probability == 0.75
    assert metrics.physical_handoff_probability == 0.75


def test_action_compatibility_distinguishes_identical_and_opposite_strategies() -> None:
    from latency_meta_mdp.return_belief_geometry import measure_action_compatibility

    states = np.zeros((2, 22))
    identical_actions = np.zeros((2, 50, 7))
    identical_actions[:, :, 0] = 0.5
    identical_actions[:, :, -1] = -1.0
    identical = measure_action_compatibility(
        _context(states, actions=identical_actions),
        prefix_ticks=5,
    )
    assert identical is not None
    assert identical.translation_rms_deviation == 0.0
    assert identical.rotation_rms_deviation == 0.0
    assert identical.gripper_disagreement_probability == 0.0
    assert identical.prefix_opposite_direction_rate == 0.0
    assert identical.effective_rank == 0.0

    opposite_actions = identical_actions.copy()
    opposite_actions[0, :, 0] = 1.0
    opposite_actions[1, :, 0] = -1.0
    opposite_actions[1, :, -1] = 1.0
    opposite = measure_action_compatibility(
        _context(states, actions=opposite_actions),
        prefix_ticks=5,
    )
    assert opposite is not None
    assert opposite.translation_rms_deviation == 1.0
    assert opposite.rotation_rms_deviation == 0.0
    assert opposite.gripper_disagreement_probability == 0.5
    assert opposite.prefix_opposite_direction_rate == 1.0
    assert opposite.effective_rank == pytest.approx(1.0)

    assert measure_action_compatibility(
        _context(states, actions=None),
        prefix_ticks=5,
    ) is None
