from __future__ import annotations

import numpy as np
import pytest

from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.legacy.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.legacy.belief.flow.ghost_environment import GhostEnvironmentAdapter
from latency_meta_mdp.legacy.belief.flow.ghost_state import reconstruct_return_state
from latency_meta_mdp.legacy.belief_data import load_belief_episode
from latency_meta_mdp.legacy.return_belief_geometry import build_absorbing_return_state_stream
from latency_meta_mdp.legacy.terminal_absorbing_tail import build_terminal_absorbing_tail
from latency_meta_mdp.runtime.temporal_contract import load_temporal_contract

_PROJECT_ROOT = repository_root()
_EPISODE = _PROJECT_ROOT / (
    "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/episodes/L1/seed_001180"
)


def test_ground_truth_state_reconstruction_matches_recorded_agentview() -> None:
    if not _EPISODE.is_dir():
        pytest.skip("ghost environment integration requires formal L1 seed 1180")
    config = load_flow_belief_ghost_config(
        _PROJECT_ROOT / "configs/legacy/analysis/flow_belief_agentview_ghost_v1.yaml"
    )
    episode = load_belief_episode(_EPISODE)
    temporal = load_temporal_contract(
        _PROJECT_ROOT / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"
    )
    action_contract = load_action_contract(
        _PROJECT_ROOT / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
    )
    tail = build_terminal_absorbing_tail(
        episode=episode,
        temporal_contract=temporal,
        action_contract=action_contract,
    )
    source_tick = 25
    target_tick = source_tick + 1
    return_states = build_absorbing_return_state_stream(tail)
    gripper_qpos = tail.extend_boundary_array(episode.deployment.gripper_qpos)
    gripper_qvel = tail.extend_boundary_array(episode.deployment.gripper_qvel)
    object_pose = tail.extend_boundary_array(episode.supervision.object_pose)
    object_velocity = tail.extend_boundary_array(episode.supervision.object_velocity)
    agentview = tail.extend_boundary_array(episode.deployment.agentview_rgb)
    state = reconstruct_return_state(
        predicted_state=return_states[target_tick],
        target_gripper_qpos=gripper_qpos[target_tick],
        target_gripper_qvel=gripper_qvel[target_tick],
        target_object_pose=object_pose[target_tick],
        target_object_velocity=object_velocity[target_tick],
    )
    env = make_dynamic_grasp_lift_environment(
        spec=load_task_spec(
            _PROJECT_ROOT / "configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml"
        ),
        seed=episode.scene_seed,
        offscreen=True,
        controller_config=action_contract.to_robosuite_config(),
    )
    try:
        adapter = GhostEnvironmentAdapter(
            env=env,
            camera_name=config.camera_name,
            width=config.width,
            height=config.height,
        )
        rendered = adapter.render_state(
            state=state,
            sim_time_seconds=target_tick * temporal.formal_tick_us / 1_000_000,
        )
        repeated = adapter.render_state(
            state=state,
            sim_time_seconds=target_tick * temporal.formal_tick_us / 1_000_000,
        )
        eef_position = adapter.forward_state(
            state=state,
            sim_time_seconds=target_tick * temporal.formal_tick_us / 1_000_000,
        )
    finally:
        env.close()

    rgb_mae = float(
        np.mean(np.abs(rendered.rgb.astype(np.float32) - agentview[target_tick].astype(np.float32)))
    )
    assert rgb_mae <= config.ground_truth_rgb_mae_max
    assert np.count_nonzero(rendered.robot_mask) > 1_000
    assert np.count_nonzero(rendered.ball_mask) > 10
    np.testing.assert_array_equal(rendered.rgb, repeated.rgb)
    np.testing.assert_array_equal(rendered.robot_mask, repeated.robot_mask)
    np.testing.assert_array_equal(rendered.ball_mask, repeated.ball_mask)
    np.testing.assert_allclose(rendered.robot_qpos, state.robot_qpos)
    np.testing.assert_allclose(rendered.object_position, state.object_qpos[:3])
    np.testing.assert_allclose(eef_position, rendered.eef_position)
    assert adapter.joint_ranges.shape == (7, 2)
    assert adapter.gripper_width_range[0] >= 0.0
    assert adapter.gripper_width_range[1] > adapter.gripper_width_range[0]
