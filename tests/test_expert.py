from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from latency_meta_mdp.backend import FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.expert import (
    ExpertObservation,
    ExpertPhase,
    ScriptedBallExpert,
    load_expert_config,
)
from latency_meta_mdp.handoff import (
    HandoffAwareBallWorld,
    HandoffState,
    OneWayHandoff,
    PandaBallContactDetector,
)
from latency_meta_mdp.motion import (
    DrivenBallWorld,
    StationaryProfile,
    build_motion_profile,
    load_motion_config,
)
from latency_meta_mdp.outcomes import EpisodeOutcomeTracker, OutcomeCriteria, OutcomeStatus
from latency_meta_mdp.snapshots import BoundarySnapshotter
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger

_CONTROL_CONFIG = Path("configs/control/panda_osc_pose_delta_v1.yaml")
_EXPERT_CONFIG = Path("configs/expert/panda_ball_feedback_v1.yaml")
_MOTION_ROOT = Path("configs/motion")
_TASK_CONFIG = Path("configs/task/dynamic_grasp_lift_l0.yaml")


def _run_episode(*, level: int, seed: int):
    contract = load_action_contract(_CONTROL_CONFIG)
    task_spec = load_task_spec(_TASK_CONFIG)
    env = make_dynamic_grasp_lift_environment(
        spec=task_spec,
        seed=seed,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    criteria = OutcomeCriteria(
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        stable_grasp_dwell_us=40_000,
        lift_height_m=0.10,
        lift_dwell_us=100_000,
        grasp_deadline_us=None if level == 0 else 3_000_000,
        lift_timeout_us=10_000_000,
    )
    tracker = EpisodeOutcomeTracker(criteria)
    handoff = OneWayHandoff(
        env=env,
        action_contract=contract,
        contact_detector=PandaBallContactDetector(env),
        outcome_tracker=tracker,
    )
    if level == 0:
        profile = StationaryProfile(
            position_xy=np.array(task_spec.ball_initial_position[:2]),
            workspace_z=task_spec.ball_initial_position[2],
        )
    else:
        profile = build_motion_profile(
            config=load_motion_config(
                _MOTION_ROOT / f"dynamic_grasp_lift_l{level}.yaml"
            ),
            seed=seed,
            workspace_z=task_spec.ball_initial_position[2],
        )
    executor = FormalStepExecutor(
        plant=RoboSuitePlant(
            env=env,
            snapshotter=BoundarySnapshotter(camera_names=(), width=16, height=16),
            world_writer=HandoffAwareBallWorld(
                driver=DrivenBallWorld(profile=profile, motion_level=level),
                handoff=handoff,
            ),
            physics_point_observer=handoff.on_physics_point,
            control_observer=handoff.on_control_applied,
        ),
        ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
    )
    expert = ScriptedBallExpert(
        action_contract=contract,
        config=load_expert_config(_EXPERT_CONFIG),
    )
    snapshot = executor.initialize()
    initial_snapshot = snapshot
    decisions = []
    try:
        for _ in range(700):
            arm = env.robots[0].part_controllers["right"]
            observation = ExpertObservation.from_snapshot(
                snapshot,
                world_to_base_rotation=arm.origin_ori.T,
            )
            decision = expert.next_action(
                observation=observation,
                handoff_state=handoff.state,
            )
            decisions.append(decision)
            snapshot = executor.step_formal(decision.action)
            if env.done:
                break
        return env, tracker, handoff, expert, decisions, snapshot, initial_snapshot
    except BaseException:
        env.close()
        raise


def test_expert_does_not_read_privileged_world_velocity() -> None:
    env, _tracker, _handoff, _expert, _decisions, _snapshot, initial = _run_episode(
        level=0,
        seed=7,
    )
    try:
        contract = load_action_contract(_CONTROL_CONFIG)
        config = load_expert_config(_EXPERT_CONFIG)
        baseline = ScriptedBallExpert(action_contract=contract, config=config)
        corrupted = ScriptedBallExpert(action_contract=contract, config=config)
        hidden_a = replace(
            initial,
            commanded_world=MappingProxyType({"target_velocity": np.zeros(3)}),
        )
        hidden_b = replace(
            initial,
            commanded_world=MappingProxyType({"target_velocity": np.full(3, 999.0)}),
        )
        observation_a = ExpertObservation.from_snapshot(
            hidden_a,
            world_to_base_rotation=np.eye(3),
        )
        observation_b = ExpertObservation.from_snapshot(
            hidden_b,
            world_to_base_rotation=np.eye(3),
        )

        action_a = baseline.next_action(
            observation=observation_a,
            handoff_state=HandoffState.DRIVEN,
        ).action
        action_b = corrupted.next_action(
            observation=observation_b,
            handoff_state=HandoffState.DRIVEN,
        ).action

        np.testing.assert_allclose(action_a, action_b, atol=0, rtol=0)
        assert {field.name for field in fields(observation_a)} == {
            "physics_step_index",
            "formal_tick_index",
            "time_us",
            "ball_position_world",
            "eef_position_world",
            "world_to_base_rotation",
        }
    finally:
        env.close()


def test_expert_completes_static_physical_grasp_and_lift() -> None:
    env, tracker, handoff, expert, decisions, _snapshot, _initial = _run_episode(
        level=0,
        seed=7,
    )
    try:
        assert tracker.status is OutcomeStatus.SUCCESS
        assert handoff.state is HandoffState.PHYSICAL
        assert tracker.handoff_us is not None
        assert tracker.terminal_time_us is not None
        assert expert.phase is ExpertPhase.LIFT
        phases = [decision.phase for decision in decisions]
        assert phases[0] is ExpertPhase.PREGRASP
        assert ExpertPhase.APPROACH in phases
        assert ExpertPhase.CLOSE in phases
        assert ExpertPhase.LIFT in phases
        assert all(decision.action.shape == (7,) for decision in decisions)
        assert decisions[0].source_time_us == 0
        assert decisions[0].source_formal_tick == 0
        assert decisions[0].history_sample_count == 1
        contract = load_action_contract(_CONTROL_CONFIG)
        for decision in decisions:
            _arm_action, gripper = contract.split_action(decision.action)
            expected = (
                contract.gripper_close_command
                if decision.phase in {ExpertPhase.CLOSE, ExpertPhase.LIFT}
                else contract.gripper_open_command
            )
            assert gripper == expected
    finally:
        env.close()


def test_expert_rotates_world_position_error_into_the_base_frame() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)
    expert = ScriptedBallExpert(
        action_contract=contract,
        config=load_expert_config(_EXPERT_CONFIG),
    )
    observation = ExpertObservation(
        physics_step_index=0,
        formal_tick_index=0,
        time_us=0,
        ball_position_world=np.array([1.0, 0.0, 0.0]),
        eef_position_world=np.zeros(3),
        world_to_base_rotation=np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    decision = expert.next_action(
        observation=observation,
        handoff_state=HandoffState.DRIVEN,
    )

    np.testing.assert_allclose(decision.action[:3], [0.0, -1.0, 1.0], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("level", "seed"),
    [(level, seed) for level in (1, 2, 3) for seed in range(10, 20)],
)
def test_expert_intercepts_dynamic_seeds_before_deadline(level: int, seed: int) -> None:
    env, tracker, handoff, expert, _decisions, _snapshot, _initial = _run_episode(
        level=level,
        seed=seed,
    )
    try:
        assert tracker.status is OutcomeStatus.SUCCESS
        assert tracker.handoff_us is not None
        assert tracker.handoff_us < 3_000_000
        assert handoff.state is HandoffState.PHYSICAL
        assert expert.phase is ExpertPhase.LIFT
    finally:
        env.close()
