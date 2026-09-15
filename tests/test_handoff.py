from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.data.recording import HandoffState as RecordedHandoffState
from latency_meta_mdp.envs.backend import FormalStepExecutor, PreparedPhysicsPoint, RoboSuitePlant
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.motion import (
    ConstantVelocityProfile,
    DrivenBallWorld,
    build_motion_profile,
    load_motion_config,
)
from latency_meta_mdp.envs.outcomes import (
    EpisodeOutcomeTracker,
    OutcomeCriteria,
    TerminalReason,
)
from latency_meta_mdp.envs.snapshots import BoundarySnapshotter
from latency_meta_mdp.envs.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.runtime.handoff import (
    HandoffAwareBallWorld,
    HandoffState,
    OneWayHandoff,
    PandaBallContactDetector,
)
from latency_meta_mdp.runtime.timing import ClockLedger

_CONTROL_CONFIG = Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
_TASK_CONFIG = Path("configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml")


def _environment():
    contract = load_action_contract(_CONTROL_CONFIG)
    env = make_dynamic_grasp_lift_environment(
        spec=load_task_spec(_TASK_CONFIG),
        seed=7,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    return contract, env


def _criteria() -> OutcomeCriteria:
    return OutcomeCriteria(
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        stable_grasp_dwell_us=40_000,
        lift_height_m=0.10,
        lift_dwell_us=100_000,
        grasp_deadline_us=3_000_000,
        lift_timeout_us=10_000_000,
    )


def test_contact_detector_reads_left_and_right_pads_from_current_mujoco_contacts() -> None:
    assert RecordedHandoffState is HandoffState
    _contract, env = _environment()
    try:
        detector = PandaBallContactDetector(env)
        empty = detector.sample(time_us=0, physics_step_index=0, formal_tick_index=0)
        assert empty.left_pad_contact is False
        assert empty.right_pad_contact is False
        assert empty.bilateral_contact is False

        robot = env.robots[0]
        grip_site_id = robot.eef_site_id[robot.arms[0]]
        qpos = np.array(env.sim.data.get_joint_qpos(env.ball.joints[0]), copy=True)
        qpos[:3] = env.sim.data.site_xpos[grip_site_id]
        env.sim.data.set_joint_qpos(env.ball.joints[0], qpos)
        env.sim.data.set_joint_qvel(env.ball.joints[0], np.zeros(6))
        env.sim.forward()
        env.sim.step1()

        contact = detector.sample(time_us=0, physics_step_index=0, formal_tick_index=0)
        assert contact.left_pad_contact is True
        assert contact.right_pad_contact is True
        assert contact.bilateral_contact is True
        assert contact.left_contact_count >= 1
        assert contact.right_contact_count >= 1
    finally:
        env.close()


def test_stable_contact_commits_one_boundary_handoff_and_preserves_twist() -> None:
    contract, env = _environment()
    try:
        robot = env.robots[0]
        grip_site_id = robot.eef_site_id[robot.arms[0]]
        grip_position = np.array(env.sim.data.site_xpos[grip_site_id], copy=True)
        tracker = EpisodeOutcomeTracker(_criteria())
        coordinator = OneWayHandoff(
            env=env,
            action_contract=contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=tracker,
        )
        profile = ConstantVelocityProfile(
            start_xy=grip_position[:2],
            end_xy=grip_position[:2] + np.array([0.003, 0.0]),
            velocity_xy=np.array([0.001, 0.0]),
            anchor_time_us=3_000_000,
            workspace_z=float(grip_position[2]),
        )
        world = HandoffAwareBallWorld(
            driver=DrivenBallWorld(profile=profile, motion_level=1),
            handoff=coordinator,
        )
        executor = FormalStepExecutor(
            plant=RoboSuitePlant(
                env=env,
                snapshotter=BoundarySnapshotter(camera_names=(), width=16, height=16),
                world_writer=world,
                physics_point_observer=coordinator.on_physics_point,
                control_observer=coordinator.on_control_applied,
            ),
            ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
        )
        executor.initialize()
        close_action = contract.compose_action(
            arm_reference=np.zeros(6),
            gripper_command=contract.gripper_close_command,
        )

        first = executor.step_formal(close_action)
        second = executor.step_formal(close_action)
        released = executor.step_formal(close_action)

        assert first.time_us == 20_000
        assert second.time_us == 40_000
        assert released.time_us == 60_000
        assert coordinator.state is HandoffState.PHYSICAL
        assert coordinator.release_time_us == 60_000
        assert world.apply_count == 31
        np.testing.assert_allclose(
            coordinator.release_qvel,
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            atol=1e-15,
            rtol=0,
        )
        np.testing.assert_allclose(
            released.object_qpos,
            coordinator.release_qpos,
            atol=1e-15,
            rtol=0,
        )

        executor.step_formal(close_action)
        assert world.apply_count == 31
        assert coordinator.release_time_us == 60_000
        assert tracker.handoff_us == 60_000
    finally:
        env.close()


def test_coordinator_is_the_only_grasp_deadline_authority() -> None:
    contract, env = _environment()
    try:
        tracker = EpisodeOutcomeTracker(_criteria())
        coordinator = OneWayHandoff(
            env=env,
            action_contract=contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=tracker,
        )
        motion_config = load_motion_config(
            Path("configs/tasks/moving_ball/motion/dynamic_grasp_lift_l1.yaml")
        )
        profile = build_motion_profile(
            config=motion_config,
            seed=7,
            workspace_z=env.task_spec.ball_initial_position[2],
        )
        executor = FormalStepExecutor(
            plant=RoboSuitePlant(
                env=env,
                snapshotter=BoundarySnapshotter(camera_names=(), width=16, height=16),
                world_writer=HandoffAwareBallWorld(
                    driver=DrivenBallWorld(profile=profile, motion_level=1),
                    handoff=coordinator,
                ),
                physics_point_observer=coordinator.on_physics_point,
                control_observer=coordinator.on_control_applied,
            ),
            ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
        )
        executor.initialize()
        hold_open = contract.compose_action(
            arm_reference=np.zeros(6),
            gripper_command=contract.gripper_open_command,
        )
        for _ in range(150):
            executor.step_formal(hold_open)

        assert tracker.terminal_reason is TerminalReason.GRASP_DEADLINE_MISSED
        assert env.terminal_reason == TerminalReason.GRASP_DEADLINE_MISSED.value
        assert coordinator.state is HandoffState.FAILURE
        assert env.task_failure is True
        assert env.done is True
    finally:
        env.close()


def test_soft_reset_invalidates_old_episode_until_explicit_rebind() -> None:
    contract, env = _environment()
    try:
        coordinator = OneWayHandoff(
            env=env,
            action_contract=contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=EpisodeOutcomeTracker(_criteria()),
        )
        assert env.external_outcome_authority is coordinator

        env.reset()
        assert env.external_outcome_authority is None
        point = PreparedPhysicsPoint(
            physics_step_index=0,
            formal_tick_index=0,
            time_us=0,
            at_formal_boundary=True,
        )
        with pytest.raises(RuntimeError, match="reset or rebound"):
            coordinator.on_physics_point(point)

        fresh_tracker = EpisodeOutcomeTracker(_criteria())
        coordinator.reset_episode(fresh_tracker)
        assert env.external_outcome_authority is coordinator
        assert coordinator.outcome_tracker is fresh_tracker
        assert coordinator.state is HandoffState.DRIVEN
        assert coordinator.release_time_us is None
        assert coordinator.release_qpos is None
        assert coordinator.release_qvel is None
        coordinator.on_physics_point(point)

        replacement = OneWayHandoff(
            env=env,
            action_contract=contract,
            contact_detector=PandaBallContactDetector(env),
            outcome_tracker=EpisodeOutcomeTracker(_criteria()),
        )
        assert env.external_outcome_authority is replacement
        with pytest.raises(RuntimeError, match="reset or rebound"):
            coordinator.on_physics_point(
                PreparedPhysicsPoint(
                    physics_step_index=1,
                    formal_tick_index=0,
                    time_us=2_000,
                    at_formal_boundary=False,
                )
            )
    finally:
        env.close()
