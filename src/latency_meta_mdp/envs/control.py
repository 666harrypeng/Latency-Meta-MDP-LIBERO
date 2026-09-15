"""Versioned policy-action contracts for the Panda control stack."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ActionRepresentation(str, Enum):
    DELTA_EEF_POSE = "delta_eef_pose"


def _float_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain numbers")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ActionContract:
    schema_version: int
    contract_id: str
    representation: ActionRepresentation
    physics_dt_us: int
    formal_tick_us: int
    actuator_dim: int
    controller_type: str
    input_type: str
    input_reference_frame: str
    rotation_representation: str
    arm_input_low: np.ndarray
    arm_input_high: np.ndarray
    arm_output_low: np.ndarray
    arm_output_high: np.ndarray
    gripper_input_low: float
    gripper_input_high: float
    gripper_open_command: float
    gripper_close_command: float
    interpolation: None
    kp: float
    damping_ratio: float
    uncouple_position_orientation: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("control schema_version must be 1")
        if self.contract_id != "panda_osc_pose_delta_v1":
            raise ValueError("unsupported action contract_id")
        if self.representation is not ActionRepresentation.DELTA_EEF_POSE:
            raise ValueError("the Panda v1 contract must use delta EEF pose actions")
        if self.physics_dt_us != 2_000 or self.formal_tick_us != 20_000:
            raise ValueError("action contract must use the certified 2 ms / 20 ms clock")
        if self.formal_tick_us % self.physics_dt_us:
            raise ValueError("formal tick must be an integer multiple of physics dt")
        if self.actuator_dim != 9:
            raise ValueError("Panda actuator_dim must be 9")
        if self.controller_type != "OSC_POSE" or self.input_type != "delta":
            raise ValueError("Panda v1 must use the delta OSC_POSE controller")
        if self.input_reference_frame != "base":
            raise ValueError("Panda v1 actions must use the robot base frame")
        if self.rotation_representation != "rotation_vector":
            raise ValueError("Panda v1 rotation actions must use rotation vectors")
        if self.interpolation is not None:
            raise ValueError("Panda v1 must use the frame-correct native OSC hold")
        if self.kp <= 0 or self.damping_ratio <= 0:
            raise ValueError("controller gains must be positive")
        if type(self.uncouple_position_orientation) is not bool:
            raise TypeError("uncouple_position_orientation must be boolean")
        for low_name, high_name in (
            ("arm_input_low", "arm_input_high"),
            ("arm_output_low", "arm_output_high"),
        ):
            low = np.array(getattr(self, low_name), dtype=float, copy=True)
            high = np.array(getattr(self, high_name), dtype=float, copy=True)
            if low.shape != (6,) or high.shape != (6,) or not np.all(low < high):
                raise ValueError(f"{low_name}/{high_name} must be ordered 6D vectors")
            if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                raise ValueError(f"{low_name}/{high_name} must be finite")
            low.setflags(write=False)
            high.setflags(write=False)
            object.__setattr__(self, low_name, low)
            object.__setattr__(self, high_name, high)
        if not self.gripper_input_low < self.gripper_input_high:
            raise ValueError("gripper input bounds must be ordered")
        if self.gripper_open_command != -1.0 or self.gripper_close_command != 1.0:
            raise ValueError("Panda gripper commands must use -1=open and +1=close")

    @property
    def arm_dim(self) -> int:
        return 6

    @property
    def gripper_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return self.arm_dim + self.gripper_dim

    @property
    def physics_steps_per_action(self) -> int:
        return self.formal_tick_us // self.physics_dt_us

    @property
    def action_low(self) -> np.ndarray:
        result = np.concatenate([self.arm_input_low, [self.gripper_input_low]])
        result.setflags(write=False)
        return result

    @property
    def action_high(self) -> np.ndarray:
        result = np.concatenate([self.arm_input_high, [self.gripper_input_high]])
        result.setflags(write=False)
        return result

    def compose_action(self, *, arm_reference: Any, gripper_command: float) -> np.ndarray:
        arm = np.asarray(arm_reference, dtype=float)
        if arm.shape != (self.arm_dim,):
            raise ValueError("arm reference dimension does not match the action contract")
        if not np.all(np.isfinite(arm)):
            raise ValueError("arm reference must be finite")
        if np.any(arm < self.arm_input_low) or np.any(arm > self.arm_input_high):
            raise ValueError("arm reference bounds are violated")
        if (
            not np.isfinite(gripper_command)
            or not self.gripper_input_low <= gripper_command <= self.gripper_input_high
        ):
            raise ValueError("gripper command is outside the action contract")
        action = np.concatenate([arm, [float(gripper_command)]])
        action.setflags(write=False)
        return action

    def split_action(self, action: Any) -> tuple[np.ndarray, float]:
        vector = np.asarray(action, dtype=float)
        if vector.shape != (self.action_dim,) or not np.all(np.isfinite(vector)):
            raise ValueError("action does not match the action contract")
        if np.any(vector < self.action_low) or np.any(vector > self.action_high):
            raise ValueError("action is outside the action contract")
        arm = np.array(vector[: self.arm_dim], copy=True)
        arm.setflags(write=False)
        return arm, float(vector[-1])

    def scale_arm_action(self, normalized_action: Any) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=float)
        if action.shape != (self.arm_dim,):
            raise ValueError("arm reference dimension does not match the action contract")
        clipped = np.clip(action, self.arm_input_low, self.arm_input_high)
        input_bias = 0.5 * (self.arm_input_high + self.arm_input_low)
        input_weight = 0.5 * (self.arm_input_high - self.arm_input_low)
        output_bias = 0.5 * (self.arm_output_high + self.arm_output_low)
        output_weight = 0.5 * (self.arm_output_high - self.arm_output_low)
        scaled = (clipped - input_bias) / input_weight * output_weight + output_bias
        scaled.setflags(write=False)
        return scaled

    def to_robosuite_config(self) -> dict[str, Any]:
        from robosuite.controllers import load_composite_controller_config
        from robosuite.controllers.parts.controller_factory import load_part_controller_config

        config = load_composite_controller_config(robot="Panda")
        arm = load_part_controller_config(default_controller=self.controller_type)
        arm.update(
            {
                "input_max": self.arm_input_high.tolist(),
                "input_min": self.arm_input_low.tolist(),
                "output_max": self.arm_output_high.tolist(),
                "output_min": self.arm_output_low.tolist(),
                "kp": self.kp,
                "damping_ratio": self.damping_ratio,
                "input_type": self.input_type,
                "input_ref_frame": self.input_reference_frame,
                "interpolation": self.interpolation,
                "uncouple_pos_ori": self.uncouple_position_orientation,
                "gripper": {"type": "GRIP"},
            }
        )
        config["body_parts"]["right"] = arm
        return deepcopy(config)

    def verify_runtime(self, env: Any) -> None:
        def require_array(actual: Any, expected: np.ndarray, message: str) -> None:
            if not np.allclose(np.asarray(actual), expected, atol=0, rtol=0):
                raise ValueError(message)

        if env.action_dim != self.action_dim:
            raise ValueError("runtime action dimension does not match the action contract")
        if env.sim.data.ctrl.shape != (self.actuator_dim,):
            raise ValueError("runtime actuator dimension does not match the action contract")
        if abs(float(env.sim.model.opt.timestep) - self.physics_dt_us / 1_000_000) > 1e-12:
            raise ValueError("runtime physics dt does not match the action contract")
        if env.control_freq != 1_000_000 / self.formal_tick_us:
            raise ValueError("runtime control frequency does not match the action contract")
        require_array(env.action_spec[0], self.action_low, "runtime action lower bounds drifted")
        require_array(env.action_spec[1], self.action_high, "runtime action upper bounds drifted")
        robot = env.robots[0]
        arm = robot.part_controllers["right"]
        if (
            arm.name != self.controller_type
            or arm.input_type != self.input_type
            or arm.input_ref_frame != self.input_reference_frame
        ):
            raise ValueError("runtime arm controller does not match the action contract")
        if arm.interpolator_pos is not None or arm.interpolator_ori is not None:
            raise ValueError("runtime must not enable the upstream OSC interpolator")
        require_array(arm.input_min, self.arm_input_low, "runtime input scale drifted")
        require_array(arm.input_max, self.arm_input_high, "runtime input scale drifted")
        require_array(arm.output_min, self.arm_output_low, "runtime output scale drifted")
        require_array(arm.output_max, self.arm_output_high, "runtime output scale drifted")
        expected_kp = np.full(6, self.kp)
        expected_kd = 2 * np.sqrt(expected_kp) * self.damping_ratio
        require_array(arm.kp, expected_kp, "runtime controller gains drifted")
        require_array(arm.kd, expected_kd, "runtime controller gains drifted")
        if arm.uncoupling is not self.uncouple_position_orientation:
            raise ValueError("runtime OSC uncoupling drifted")
        if arm._goal_update_mode != "achieved":
            raise ValueError("runtime OSC goal update mode must be achieved")
        gripper_controller = robot.part_controllers["right_gripper"]
        gripper_model = robot.gripper["right"]
        if type(gripper_controller).__name__ != "SimpleGripController":
            raise ValueError("runtime gripper controller drifted")
        if type(gripper_model).__name__ != "PandaGripper" or gripper_model.speed != 0.2:
            raise ValueError("runtime gripper model drifted")
        if robot.composite_controller._action_split_indexes["right_gripper"] != (6, 7):
            raise ValueError("runtime gripper action slice drifted")
        require_array(
            gripper_controller.input_min,
            np.full(2, self.gripper_input_low),
            "runtime gripper limits drifted",
        )
        require_array(
            gripper_controller.input_max,
            np.full(2, self.gripper_input_high),
            "runtime gripper limits drifted",
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ActionContract:
        expected = {
            "actuator_dim",
            "arm_input_high",
            "arm_input_low",
            "arm_output_high",
            "arm_output_low",
            "contract_id",
            "controller_type",
            "damping_ratio",
            "formal_tick_us",
            "gripper_close_command",
            "gripper_input_high",
            "gripper_input_low",
            "gripper_open_command",
            "input_reference_frame",
            "input_type",
            "interpolation",
            "kp",
            "physics_dt_us",
            "representation",
            "rotation_representation",
            "schema_version",
            "uncouple_position_orientation",
        }
        if set(raw) != expected:
            raise ValueError("control config fields are invalid")
        return cls(
            schema_version=raw["schema_version"],
            contract_id=raw["contract_id"],
            representation=ActionRepresentation(raw["representation"]),
            physics_dt_us=raw["physics_dt_us"],
            formal_tick_us=raw["formal_tick_us"],
            actuator_dim=raw["actuator_dim"],
            controller_type=raw["controller_type"],
            input_type=raw["input_type"],
            input_reference_frame=raw["input_reference_frame"],
            rotation_representation=raw["rotation_representation"],
            arm_input_low=_float_vector(raw["arm_input_low"], name="arm_input_low", length=6),
            arm_input_high=_float_vector(raw["arm_input_high"], name="arm_input_high", length=6),
            arm_output_low=_float_vector(raw["arm_output_low"], name="arm_output_low", length=6),
            arm_output_high=_float_vector(raw["arm_output_high"], name="arm_output_high", length=6),
            gripper_input_low=float(raw["gripper_input_low"]),
            gripper_input_high=float(raw["gripper_input_high"]),
            gripper_open_command=float(raw["gripper_open_command"]),
            gripper_close_command=float(raw["gripper_close_command"]),
            interpolation=raw["interpolation"],
            kp=float(raw["kp"]),
            damping_ratio=float(raw["damping_ratio"]),
            uncouple_position_orientation=raw["uncouple_position_orientation"],
        )


def load_action_contract(path: Path) -> ActionContract:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("control config must be a YAML mapping")
    return ActionContract.from_mapping(raw)
