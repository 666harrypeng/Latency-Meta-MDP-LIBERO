"""Deterministic physical calibration for the selected Panda action contract."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.envs.backend import AppliedControlSample, FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.envs.control import ActionContract, load_action_contract
from latency_meta_mdp.envs.snapshots import BoundarySnapshotter
from latency_meta_mdp.envs.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.runtime.timing import ClockLedger

_PROJECT_ROOT = repository_root()


def _implementation_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _implementation_source_sha256() -> str:
    paths = (
        _PROJECT_ROOT / "pyproject.toml",
        _PROJECT_ROOT / "uv.lock",
        _PROJECT_ROOT / "src/latency_meta_mdp/envs/backend.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/envs/control.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/envs/control_calibration.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/envs/snapshots.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/envs/task.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/runtime/timing.py",
        _PROJECT_ROOT / "src/latency_meta_mdp/runtime/diagnostics/calibrate_control.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(_PROJECT_ROOT).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlCalibrationSpec:
    schema_version: int
    calibration_id: str
    seed: int
    hold_ticks: int
    translation_ticks: int
    settle_ticks: int
    rotation_ticks: int
    gripper_ticks: int
    translation_command: float
    rotation_command: float
    max_hold_eef_drift_m: float
    max_hold_joint_drift_rad: float
    min_positive_x_velocity_mps: float
    max_positive_x_velocity_mps: float
    min_negative_x_velocity_mps: float
    max_negative_x_velocity_mps: float
    max_translation_goal_error_m: float
    min_rotation_response_rad: float
    max_rotation_goal_error_rad: float
    max_arm_saturated_physics_step_fraction: float
    min_joint_limit_margin_rad: float
    max_return_position_error_m: float
    max_closed_gripper_width_m: float
    min_reopened_gripper_width_m: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("control calibration schema_version must be 1")
        if self.calibration_id != "panda_osc_pose_delta_calibration_v1":
            raise ValueError("unsupported control calibration_id")
        for name in (
            "seed",
            "hold_ticks",
            "translation_ticks",
            "settle_ticks",
            "rotation_ticks",
            "gripper_ticks",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is invalid")
        for name in (
            "translation_command",
            "rotation_command",
            "max_hold_eef_drift_m",
            "max_hold_joint_drift_rad",
            "min_positive_x_velocity_mps",
            "max_positive_x_velocity_mps",
            "max_translation_goal_error_m",
            "min_rotation_response_rad",
            "max_rotation_goal_error_rad",
            "max_arm_saturated_physics_step_fraction",
            "min_joint_limit_margin_rad",
            "max_return_position_error_m",
            "max_closed_gripper_width_m",
            "min_reopened_gripper_width_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.min_positive_x_velocity_mps < self.max_positive_x_velocity_mps:
            raise ValueError("positive velocity bounds are invalid")
        if not self.min_negative_x_velocity_mps < self.max_negative_x_velocity_mps < 0:
            raise ValueError("negative velocity bounds are invalid")
        if not np.isfinite(self.min_negative_x_velocity_mps) or not np.isfinite(
            self.max_negative_x_velocity_mps
        ):
            raise ValueError("negative velocity bounds must be finite")
        if not 0 < self.translation_command <= 1 or not 0 < self.rotation_command <= 1:
            raise ValueError("calibration commands must be normalized")

    @property
    def total_ticks(self) -> int:
        return (
            self.hold_ticks
            + 2 * self.translation_ticks
            + 3 * self.settle_ticks
            + self.rotation_ticks
            + 2 * self.gripper_ticks
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ControlCalibrationSpec:
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("control calibration config fields are invalid")
        return cls(**raw)


def load_control_calibration_spec(path: Path) -> ControlCalibrationSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("control calibration config must be a YAML mapping")
    return ControlCalibrationSpec.from_mapping(raw)


@dataclass(frozen=True)
class _Phase:
    name: str
    ticks: int
    arm_action: tuple[float, float, float, float, float, float]
    gripper_action: float


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = np.clip((np.trace(left @ right.T) - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cosine))


def _hash_trace(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _phase_plan(
    contract: ActionContract,
    spec: ControlCalibrationSpec,
) -> tuple[_Phase, ...]:
    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    positive_x = (spec.translation_command, 0.0, 0.0, 0.0, 0.0, 0.0)
    negative_x = (-spec.translation_command, 0.0, 0.0, 0.0, 0.0, 0.0)
    positive_z_rotation = (0.0, 0.0, 0.0, 0.0, 0.0, spec.rotation_command)
    return (
        _Phase("hold_open", spec.hold_ticks, zero, contract.gripper_open_command),
        _Phase("translate_positive_x", spec.translation_ticks, positive_x, -1.0),
        _Phase("settle_after_positive_x", spec.settle_ticks, zero, -1.0),
        _Phase("translate_negative_x", spec.translation_ticks, negative_x, -1.0),
        _Phase("settle_after_negative_x", spec.settle_ticks, zero, -1.0),
        _Phase("rotate_positive_z", spec.rotation_ticks, positive_z_rotation, -1.0),
        _Phase("settle_after_rotation", spec.settle_ticks, zero, -1.0),
        _Phase("close_gripper", spec.gripper_ticks, zero, contract.gripper_close_command),
        _Phase("open_gripper", spec.gripper_ticks, zero, contract.gripper_open_command),
    )


def run_control_calibration(
    *,
    control_config_path: Path,
    calibration_config_path: Path,
    task_config_path: Path,
) -> dict[str, Any]:
    import mujoco
    import robosuite

    contract = load_action_contract(control_config_path)
    calibration = load_control_calibration_spec(calibration_config_path)
    task_spec = load_task_spec(task_config_path)
    env = make_dynamic_grasp_lift_environment(
        spec=task_spec,
        seed=calibration.seed,
        offscreen=False,
        controller_config=contract.to_robosuite_config(),
    )
    try:
        contract.verify_runtime(env)
        robot = env.robots[0]
        arm = robot.part_controllers["right"]
        arm_actuator_indexes = np.asarray(robot._ref_arm_joint_actuator_indexes, dtype=int)
        arm_ctrl_low = env.sim.model.actuator_ctrlrange[arm_actuator_indexes, 0]
        arm_ctrl_high = env.sim.model.actuator_ctrlrange[arm_actuator_indexes, 1]
        joint_indexes = np.asarray(robot._ref_joint_indexes, dtype=int)
        joint_ranges = env.sim.model.jnt_range[joint_indexes]
        physics_ctrl: list[np.ndarray] = []
        physics_joint_margin: list[float] = []
        boundary_joint_margin: list[float] = []
        physics_saturation: list[np.ndarray] = []
        physics_trace: list[dict[str, Any]] = []
        boundary_trace: list[dict[str, Any]] = []
        trace_values: list[np.ndarray] = []

        def observe_control(sample: AppliedControlSample) -> None:
            ctrl = np.array(sample.actuator_ctrl[arm_actuator_indexes], copy=True)
            joint_qpos = np.array(sample.robot_qpos, copy=True)
            saturated = np.isclose(ctrl, arm_ctrl_low, atol=1e-9) | np.isclose(
                ctrl, arm_ctrl_high, atol=1e-9
            )
            joint_margin = float(
                np.min(
                    np.minimum(
                        joint_qpos - joint_ranges[:, 0],
                        joint_ranges[:, 1] - joint_qpos,
                    )
                )
            )
            physics_ctrl.append(ctrl)
            physics_joint_margin.append(joint_margin)
            physics_saturation.append(saturated)
            physics_trace.append(
                {
                    "action": sample.action.tolist(),
                    "actuator_ctrl": sample.actuator_ctrl.tolist(),
                    "formal_tick_index": sample.formal_tick_index,
                    "joint_limit_margin_rad": joint_margin,
                    "physics_step_index": sample.physics_step_index,
                    "policy_step": sample.policy_step,
                    "robot_qpos": sample.robot_qpos.tolist(),
                    "saturated_arm_joints": saturated.tolist(),
                    "time_us": sample.time_us,
                }
            )
            trace_values.extend([sample.action, sample.actuator_ctrl, joint_qpos])

        plant = RoboSuitePlant(
            env=env,
            snapshotter=BoundarySnapshotter(camera_names=(), width=16, height=16),
            control_observer=observe_control,
        )
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        executor = FormalStepExecutor(plant=plant, ledger=ledger)
        initial = executor.initialize()
        initial_gripper_qpos = np.array(
            env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
        )
        boundary_trace.append(
            {
                "desired_eef_orientation_matrix": initial.eef_xmat.tolist(),
                "desired_eef_position_m": initial.eef_pos.tolist(),
                "eef_orientation_matrix": initial.eef_xmat.tolist(),
                "eef_orientation_error_rad": 0.0,
                "eef_position_error_m": 0.0,
                "eef_position_m": initial.eef_pos.tolist(),
                "formal_tick_index": initial.formal_tick_index,
                "gripper_width_m": float(abs(initial_gripper_qpos[0] - initial_gripper_qpos[1])),
                "phase": "initial",
                "physics_step_index": initial.physics_step_index,
                "robot_qpos": initial.robot_qpos.tolist(),
                "robot_qvel": initial.robot_qvel.tolist(),
                "time_us": initial.time_us,
            }
        )
        boundary_joint_margin.append(
            float(
                np.min(
                    np.minimum(
                        initial.robot_qpos - joint_ranges[:, 0],
                        joint_ranges[:, 1] - initial.robot_qpos,
                    )
                )
            )
        )
        current = initial
        phase_reports: dict[str, dict[str, Any]] = {}

        for phase in _phase_plan(contract, calibration):
            start = current
            position_errors: list[float] = []
            orientation_errors: list[float] = []
            eef_displacements: list[float] = []
            joint_displacements: list[float] = []
            action = contract.compose_action(
                arm_reference=np.asarray(phase.arm_action),
                gripper_command=phase.gripper_action,
            )
            for _ in range(phase.ticks):
                current = executor.step_formal(action)
                boundary_joint_margin.append(
                    float(
                        np.min(
                            np.minimum(
                                current.robot_qpos - joint_ranges[:, 0],
                                joint_ranges[:, 1] - current.robot_qpos,
                            )
                        )
                    )
                )
                desired_position = arm.origin_pos + arm.origin_ori @ arm.goal_pos
                desired_orientation = arm.origin_ori @ arm.goal_ori
                position_errors.append(float(np.linalg.norm(current.eef_pos - desired_position)))
                orientation_errors.append(_rotation_distance(desired_orientation, current.eef_xmat))
                current_gripper_qpos = np.array(
                    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
                )
                boundary_trace.append(
                    {
                        "desired_eef_orientation_matrix": desired_orientation.tolist(),
                        "desired_eef_position_m": desired_position.tolist(),
                        "eef_orientation_matrix": current.eef_xmat.tolist(),
                        "eef_orientation_error_rad": orientation_errors[-1],
                        "eef_position_error_m": position_errors[-1],
                        "eef_position_m": current.eef_pos.tolist(),
                        "formal_tick_index": current.formal_tick_index,
                        "gripper_width_m": float(
                            abs(current_gripper_qpos[0] - current_gripper_qpos[1])
                        ),
                        "phase": phase.name,
                        "physics_step_index": current.physics_step_index,
                        "robot_qpos": current.robot_qpos.tolist(),
                        "robot_qvel": current.robot_qvel.tolist(),
                        "time_us": current.time_us,
                    }
                )
                eef_displacements.append(float(np.linalg.norm(current.eef_pos - start.eef_pos)))
                joint_displacements.append(
                    float(np.max(np.abs(current.robot_qpos - start.robot_qpos)))
                )
                trace_values.extend(
                    [current.robot_qpos, current.robot_qvel, current.eef_pos, current.eef_xmat]
                )
            duration_seconds = phase.ticks * contract.formal_tick_us / 1_000_000
            displacement = current.eef_pos - start.eef_pos
            gripper_qpos = np.array(
                env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
            )
            phase_reports[phase.name] = {
                "average_velocity_xyz_mps": (displacement / duration_seconds).tolist(),
                "displacement_xyz_m": displacement.tolist(),
                "end_eef_position_m": current.eef_pos.tolist(),
                "end_gripper_width_m": float(abs(gripper_qpos[0] - gripper_qpos[1])),
                "max_eef_displacement_m": max(eef_displacements),
                "max_joint_displacement_rad": max(joint_displacements),
                "max_orientation_goal_error_rad": max(orientation_errors),
                "max_position_goal_error_m": max(position_errors),
                "rotation_response_rad": _rotation_distance(current.eef_xmat, start.eef_xmat),
                "ticks": phase.ticks,
            }

        hold = phase_reports["hold_open"]
        positive = phase_reports["translate_positive_x"]
        negative = phase_reports["translate_negative_x"]
        rotation = phase_reports["rotate_positive_z"]
        closed = phase_reports["close_gripper"]
        reopened = phase_reports["open_gripper"]
        return_reference = phase_reports["settle_after_negative_x"]
        saturation_matrix = np.stack(physics_saturation)
        arm_saturated_physics_step_fraction = float(np.mean(np.any(saturation_matrix, axis=1)))
        arm_saturated_actuator_step_fraction = float(np.mean(saturation_matrix))
        minimum_joint_margin = min((*physics_joint_margin, *boundary_joint_margin))
        blockers: list[str] = []

        def block(condition: bool, name: str) -> None:
            if condition:
                blockers.append(name)

        positive_velocity = float(positive["average_velocity_xyz_mps"][0])
        negative_velocity = float(negative["average_velocity_xyz_mps"][0])
        translation_goal_error = max(
            float(positive["max_position_goal_error_m"]),
            float(negative["max_position_goal_error_m"]),
        )
        return_error = float(
            np.linalg.norm(np.asarray(return_reference["end_eef_position_m"]) - initial.eef_pos)
        )
        hold_eef_drift = float(hold["max_eef_displacement_m"])
        hold_joint_drift = float(hold["max_joint_displacement_rad"])
        block(hold_eef_drift > calibration.max_hold_eef_drift_m, "hold_eef_drift")
        block(
            hold_joint_drift > calibration.max_hold_joint_drift_rad,
            "hold_joint_drift",
        )
        block(
            not calibration.min_positive_x_velocity_mps
            <= positive_velocity
            <= calibration.max_positive_x_velocity_mps,
            "positive_x_velocity",
        )
        block(
            not calibration.min_negative_x_velocity_mps
            <= negative_velocity
            <= calibration.max_negative_x_velocity_mps,
            "negative_x_velocity",
        )
        block(
            translation_goal_error > calibration.max_translation_goal_error_m,
            "translation_goal_error",
        )
        block(
            float(rotation["rotation_response_rad"]) < calibration.min_rotation_response_rad,
            "rotation_response",
        )
        block(
            float(rotation["max_orientation_goal_error_rad"])
            > calibration.max_rotation_goal_error_rad,
            "rotation_goal_error",
        )
        block(
            arm_saturated_physics_step_fraction
            > calibration.max_arm_saturated_physics_step_fraction,
            "arm_torque_saturation",
        )
        block(
            minimum_joint_margin < calibration.min_joint_limit_margin_rad,
            "joint_limit_margin",
        )
        block(return_error > calibration.max_return_position_error_m, "return_position_error")
        block(
            float(closed["end_gripper_width_m"]) > calibration.max_closed_gripper_width_m,
            "gripper_close",
        )
        block(
            float(reopened["end_gripper_width_m"]) < calibration.min_reopened_gripper_width_m,
            "gripper_reopen",
        )
        block(ledger.formal_tick_index != calibration.total_ticks, "formal_tick_count")
        block(
            ledger.physics_step_index
            != calibration.total_ticks * contract.physics_steps_per_action,
            "physics_step_count",
        )

        return {
            "actuator_dim": contract.actuator_dim,
            "action_dim": contract.action_dim,
            "arm_saturated_actuator_step_fraction": arm_saturated_actuator_step_fraction,
            "arm_saturated_physics_step_fraction": arm_saturated_physics_step_fraction,
            "boundary_trace": boundary_trace,
            "blockers": blockers,
            "calibration_config_sha256": sha256_file(calibration_config_path),
            "calibration_id": calibration.calibration_id,
            "closed_gripper_width_m": float(closed["end_gripper_width_m"]),
            "contract_id": contract.contract_id,
            "control_config_sha256": sha256_file(control_config_path),
            "eligible": not blockers,
            "final_time_us": ledger.time_us,
            "formal_tick_count": ledger.formal_tick_index,
            "hold_eef_drift_m": hold_eef_drift,
            "hold_joint_drift_rad": hold_joint_drift,
            "implementation_revision": _implementation_revision(),
            "implementation_source_sha256": _implementation_source_sha256(),
            "minimum_joint_limit_margin_rad": minimum_joint_margin,
            "mujoco_version": mujoco.__version__,
            "negative_x_average_velocity_mps": negative_velocity,
            "phase_reports": phase_reports,
            "physics_trace": physics_trace,
            "physics_control_sample_count": len(physics_ctrl),
            "physics_step_count": ledger.physics_step_index,
            "positive_x_average_velocity_mps": positive_velocity,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "reopened_gripper_width_m": float(reopened["end_gripper_width_m"]),
            "return_position_error_m": return_error,
            "robosuite_version": robosuite.__version__,
            "schema_version": 1,
            "seed": calibration.seed,
            "task_config_sha256": sha256_file(task_config_path),
            "trace_sha256": _hash_trace(trace_values),
        }
    finally:
        env.close()
