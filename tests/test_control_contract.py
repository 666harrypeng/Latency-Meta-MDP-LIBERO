from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.backend import AppliedControlSample, FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.control import ActionRepresentation, load_action_contract
from latency_meta_mdp.snapshots import BoundarySnapshotter
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger

_TASK_CONFIG = Path("configs/task/dynamic_grasp_lift_l0.yaml")
_CONTROL_CONFIG = Path("configs/control/panda_osc_pose_delta_v1.yaml")


def test_delta_eef_contract_builds_the_seven_dimensional_panda_controller() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)
    spec = load_task_spec(_TASK_CONFIG)
    env = make_dynamic_grasp_lift_environment(
        spec=spec,
        seed=7,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    try:
        arm = env.robots[0].part_controllers["right"]
        contract.verify_runtime(env)
        assert contract.contract_id == "panda_osc_pose_delta_v1"
        assert contract.representation is ActionRepresentation.DELTA_EEF_POSE
        assert contract.action_dim == 7
        assert contract.actuator_dim == 9
        assert env.sim.data.ctrl.shape == (contract.actuator_dim,)
        assert env.action_dim == contract.action_dim
        np.testing.assert_allclose(env.action_spec[0], -np.ones(7), atol=0, rtol=0)
        np.testing.assert_allclose(env.action_spec[1], np.ones(7), atol=0, rtol=0)
        np.testing.assert_allclose(
            arm.output_min,
            [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            arm.output_max,
            [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            atol=0,
            rtol=0,
        )
        assert arm.input_type == "delta"
        assert arm.input_ref_frame == "base"
        assert arm.interpolator_pos is None
        assert arm.interpolator_ori is None
        assert contract.physics_steps_per_action == 10
    finally:
        env.close()


def test_action_contract_composes_splits_and_scales_normalized_delta_actions() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)
    normalized_delta = np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.0])

    action = contract.compose_action(
        arm_reference=normalized_delta,
        gripper_command=-1.0,
    )
    recovered_arm, recovered_gripper = contract.split_action(action)
    scaled = contract.scale_arm_action(recovered_arm)

    np.testing.assert_allclose(recovered_arm, normalized_delta, atol=0, rtol=0)
    np.testing.assert_allclose(
        scaled,
        [0.05, -0.05, 0.0, 0.25, -0.25, 0.0],
        atol=1e-15,
        rtol=0,
    )
    assert recovered_gripper == -1.0
    assert action.flags.writeable is False


def test_action_contract_rejects_wrong_dimensions_and_non_normalized_commands() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)

    with pytest.raises(ValueError, match="arm reference dimension"):
        contract.compose_action(arm_reference=np.zeros(7), gripper_command=-1.0)
    with pytest.raises(ValueError, match="arm reference bounds"):
        contract.compose_action(arm_reference=np.full(6, 1.01), gripper_command=-1.0)
    with pytest.raises(ValueError, match="gripper command"):
        contract.compose_action(arm_reference=np.zeros(6), gripper_command=2.0)


def test_zero_delta_holds_pose_for_exactly_ten_physics_steps() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)
    env = make_dynamic_grasp_lift_environment(
        spec=load_task_spec(_TASK_CONFIG),
        seed=7,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    try:
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        control_samples: list[AppliedControlSample] = []
        executor = FormalStepExecutor(
            plant=RoboSuitePlant(
                env=env,
                snapshotter=BoundarySnapshotter(camera_names=(), width=16, height=16),
                control_observer=control_samples.append,
            ),
            ledger=ledger,
        )
        initial = executor.initialize()
        final = executor.step_formal(
            contract.compose_action(
                arm_reference=np.zeros(6),
                gripper_command=contract.gripper_open_command,
            )
        )

        assert ledger.physics_step_index == 10
        assert ledger.formal_tick_index == 1
        assert ledger.time_us == 20_000
        assert np.linalg.norm(final.eef_pos - initial.eef_pos) < 1e-8
        assert np.max(np.abs(final.robot_qpos - initial.robot_qpos)) < 1e-8
        assert [sample.time_us for sample in control_samples] == list(range(0, 20_000, 2_000))
        assert [sample.physics_step_index for sample in control_samples] == list(range(10))
        assert [sample.policy_step for sample in control_samples] == [True] + [False] * 9
        assert all(sample.action.flags.writeable is False for sample in control_samples)
        assert all(sample.actuator_ctrl.flags.writeable is False for sample in control_samples)
        assert all(sample.robot_qpos.flags.writeable is False for sample in control_samples)
    finally:
        env.close()


def test_runtime_verification_rejects_controller_parameter_drift() -> None:
    contract = load_action_contract(_CONTROL_CONFIG)
    env = make_dynamic_grasp_lift_environment(
        spec=load_task_spec(_TASK_CONFIG),
        seed=7,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    try:
        arm = env.robots[0].part_controllers["right"]
        arm.output_max[0] = 0.04
        with pytest.raises(ValueError, match="output scale"):
            contract.verify_runtime(env)
        arm.output_max[0] = 0.05

        arm.kp[0] = 149.0
        with pytest.raises(ValueError, match="controller gains"):
            contract.verify_runtime(env)
        arm.kp[0] = 150.0

        arm.uncoupling = False
        with pytest.raises(ValueError, match="uncoupling"):
            contract.verify_runtime(env)
    finally:
        env.close()
