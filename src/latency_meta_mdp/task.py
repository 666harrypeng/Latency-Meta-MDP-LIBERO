"""RoboSuite-native moving-ball grasp-and-lift task setup."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _float_tuple(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"{name} must contain only numbers")
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class TaskSpec:
    schema_version: int
    task_id: str
    level: int
    instruction: str
    table_full_size: tuple[float, float, float]
    table_offset: tuple[float, float, float]
    ball_radius_m: float
    ball_density_kg_m3: float
    ball_friction: tuple[float, float, float]
    ball_rgba: tuple[float, float, float, float]
    ball_initial_position: tuple[float, float, float]
    lift_success_height_m: float
    agentview_resource_id: str
    agentview_position: tuple[float, float, float]
    agentview_quaternion_wxyz: tuple[float, float, float, float]
    agentview_fovy_degrees: float
    policy_camera_names: tuple[str, str]

    @property
    def ball_diameter_m(self) -> float:
        return 2.0 * self.ball_radius_m

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TaskSpec:
        expected = {
            "agentview_fovy_degrees",
            "agentview_position",
            "agentview_quaternion_wxyz",
            "agentview_resource_id",
            "ball_density_kg_m3",
            "ball_friction",
            "ball_initial_position",
            "ball_radius_m",
            "ball_rgba",
            "instruction",
            "level",
            "lift_success_height_m",
            "policy_camera_names",
            "schema_version",
            "table_full_size",
            "table_offset",
            "task_id",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown:
            raise ValueError(f"unknown task config fields: {unknown}")
        if missing:
            raise ValueError(f"missing task config fields: {missing}")
        if raw["schema_version"] != 1:
            raise ValueError("task config schema_version must be 1")
        if raw["task_id"] != "dynamic_grasp_lift" or raw["level"] != 0:
            raise ValueError("this task config must describe dynamic_grasp_lift level 0")
        if not isinstance(raw["instruction"], str) or not raw["instruction"].strip():
            raise ValueError("instruction must be a non-empty string")
        if raw["policy_camera_names"] != ["agentview", "robot0_eye_in_hand"]:
            raise ValueError("policy cameras must be official agentview plus Panda eye-in-hand")
        if raw["agentview_resource_id"] != "libero_tabletop_agentview_v1":
            raise ValueError(
                "agentview_resource_id must select the approved LIBERO camera resource"
            )

        scalars: dict[str, float] = {}
        for name in (
            "agentview_fovy_degrees",
            "ball_density_kg_m3",
            "ball_radius_m",
            "lift_success_height_m",
        ):
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            scalars[name] = float(value)
            if scalars[name] <= 0:
                raise ValueError(f"{name} must be positive")

        spec = cls(
            schema_version=1,
            task_id="dynamic_grasp_lift",
            level=0,
            instruction=raw["instruction"],
            table_full_size=_float_tuple(raw["table_full_size"], name="table_full_size", length=3),
            table_offset=_float_tuple(raw["table_offset"], name="table_offset", length=3),
            ball_radius_m=scalars["ball_radius_m"],
            ball_density_kg_m3=scalars["ball_density_kg_m3"],
            ball_friction=_float_tuple(raw["ball_friction"], name="ball_friction", length=3),
            ball_rgba=_float_tuple(raw["ball_rgba"], name="ball_rgba", length=4),
            ball_initial_position=_float_tuple(
                raw["ball_initial_position"], name="ball_initial_position", length=3
            ),
            lift_success_height_m=scalars["lift_success_height_m"],
            agentview_resource_id=raw["agentview_resource_id"],
            agentview_position=_float_tuple(
                raw["agentview_position"], name="agentview_position", length=3
            ),
            agentview_quaternion_wxyz=_float_tuple(
                raw["agentview_quaternion_wxyz"],
                name="agentview_quaternion_wxyz",
                length=4,
            ),
            agentview_fovy_degrees=scalars["agentview_fovy_degrees"],
            policy_camera_names=(raw["policy_camera_names"][0], raw["policy_camera_names"][1]),
        )
        spec.validate_geometry()
        return spec

    def validate_geometry(self) -> None:
        if abs(float(np.linalg.norm(self.agentview_quaternion_wxyz)) - 1.0) > 1e-9:
            raise ValueError("agentview_quaternion_wxyz must be unit length")
        expected_ball_z = self.table_offset[2] + self.ball_radius_m + 0.003
        if abs(self.ball_initial_position[2] - expected_ball_z) > 1e-12:
            raise ValueError("ball center height is inconsistent with table and radius")
        half_x, half_y = self.table_full_size[0] / 2, self.table_full_size[1] / 2
        if (
            abs(self.ball_initial_position[0]) + self.ball_radius_m >= half_x
            or abs(self.ball_initial_position[1]) + self.ball_radius_m >= half_y
        ):
            raise ValueError("ball lies outside the table")
        if any(value < 0 for value in self.ball_friction):
            raise ValueError("ball friction values must be non-negative")
        if any(not 0 <= value <= 1 for value in self.ball_rgba):
            raise ValueError("ball RGBA values must lie in [0, 1]")


@dataclass(frozen=True)
class BallGoalMeasurements:
    lift_height_m: float
    linear_speed_mps: float
    angular_speed_radps: float
    current_two_pad_grasp: bool


def load_task_spec(path: Path) -> TaskSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task config must be a YAML mapping")
    return TaskSpec.from_mapping(raw)


def _make_environment_class() -> type[Any]:
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BallObject
    from robosuite.models.tasks import ManipulationTask
    from robosuite.utils.observables import Observable, sensor

    class DynamicGraspLiftEnv(ManipulationEnv):
        """Single-Panda physical grasp-and-lift task with a spherical target."""

        def __init__(self, *, spec: TaskSpec, seed: int, offscreen: bool, controller: Any) -> None:
            self.task_spec = spec
            self.table_full_size = spec.table_full_size
            self.table_friction = (1.0, 5e-3, 1e-4)
            self.table_offset = np.asarray(spec.table_offset)
            self.motion_deadline_us: int | None = None
            self.task_failure = False
            self.terminal_reason: str | None = None
            self.external_outcome_authority: Any | None = None
            super().__init__(
                robots="Panda",
                controller_configs=controller,
                initialization_noise=None,
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=offscreen,
                render_camera="agentview",
                control_freq=50,
                lite_physics=True,
                horizon=10_000,
                ignore_done=True,
                hard_reset=False,
                camera_names=list(spec.policy_camera_names),
                camera_heights=256,
                camera_widths=256,
                camera_depths=False,
                camera_segmentations=None,
                seed=seed,
            )

        def _load_model(self) -> None:
            super()._load_model()
            base_position = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(base_position)
            arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
            arena.set_origin([0, 0, 0])
            arena.set_camera(
                camera_name="agentview",
                pos=self.task_spec.agentview_position,
                quat=self.task_spec.agentview_quaternion_wxyz,
                camera_attribs={"fovy": str(self.task_spec.agentview_fovy_degrees)},
            )
            self.ball = BallObject(
                name="ball",
                size=[self.task_spec.ball_radius_m],
                density=self.task_spec.ball_density_kg_m3,
                friction=list(self.task_spec.ball_friction),
                rgba=list(self.task_spec.ball_rgba),
                joints="default",
            )
            self.model = ManipulationTask(
                mujoco_arena=arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=self.ball,
            )

        def _destroy_sim(self) -> None:
            from latency_meta_mdp.snapshots import make_offscreen_context_current

            # MuJoCo frees GL resources in the current context, which may belong to GT replay.
            make_offscreen_context_current(self)
            super()._destroy_sim()

        def _setup_references(self) -> None:
            super()._setup_references()
            self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)
            self.task_object = self.ball
            self.task_object_body_id = self.ball_body_id

        def _setup_observables(self) -> dict[str, Observable]:
            observables = super()._setup_observables()

            @sensor(modality="object")
            def ball_pos(_obs_cache: dict[str, Any]) -> np.ndarray:
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            @sensor(modality="object")
            def ball_quat(_obs_cache: dict[str, Any]) -> np.ndarray:
                return np.array(self.sim.data.body_xquat[self.ball_body_id])

            for observable_sensor in (ball_pos, ball_quat):
                observables[observable_sensor.__name__] = Observable(
                    name=observable_sensor.__name__,
                    sensor=observable_sensor,
                    sampling_rate=self.control_freq,
                )
            return observables

        def _reset_internal(self) -> None:
            super()._reset_internal()
            self.motion_deadline_us = None
            self.task_failure = False
            self.terminal_reason = None
            self.external_outcome_authority = None
            qpos = np.concatenate(
                [np.asarray(self.task_spec.ball_initial_position), np.array([1.0, 0.0, 0.0, 0.0])]
            )
            self.sim.data.set_joint_qpos(self.ball.joints[0], qpos)
            self.sim.data.set_joint_qvel(self.ball.joints[0], np.zeros(6, dtype=float))
            self.sim.forward()

        def current_grasp(self) -> bool:
            return bool(self._check_grasp(self.robots[0].gripper, self.ball))

        def goal_measurements(self) -> BallGoalMeasurements:
            center = np.array(self.sim.data.body_xpos[self.ball_body_id])
            velocity = np.array(self.sim.data.get_joint_qvel(self.ball.joints[0]))
            resting_center_z = self.task_spec.table_offset[2] + self.task_spec.ball_radius_m
            return BallGoalMeasurements(
                lift_height_m=float(center[2] - resting_center_z),
                linear_speed_mps=float(np.linalg.norm(velocity[:3])),
                angular_speed_radps=float(np.linalg.norm(velocity[3:])),
                current_two_pad_grasp=self.current_grasp(),
            )

        def backend_goal_reached(self) -> bool:
            measurements = self.goal_measurements()
            return (
                measurements.current_two_pad_grasp
                and measurements.lift_height_m + 1e-12 >= self.task_spec.lift_success_height_m
            )

        def mark_motion_deadline(self, time_us: int) -> None:
            if self.motion_deadline_us is None:
                self.motion_deadline_us = time_us

        def evaluate_boundary_terminal(self, time_us: int) -> None:
            if self.external_outcome_authority is not None:
                return
            if (
                self.motion_deadline_us is not None
                and time_us >= self.motion_deadline_us
                and not self.current_grasp()
            ):
                self.task_failure = True
                self.terminal_reason = "motion_deadline_ungrasped"
                self.done = True

        def _check_success(self) -> bool:
            return self.backend_goal_reached()

        def reward(self, action: Any = None) -> float:
            del action
            return float(self.backend_goal_reached())

    return DynamicGraspLiftEnv


def make_dynamic_grasp_lift_environment(
    *,
    spec: TaskSpec,
    seed: int,
    offscreen: bool,
    controller_config: dict[str, Any] | None = None,
) -> Any:
    import mujoco
    from robosuite.controllers import load_composite_controller_config

    environment_class = _make_environment_class()
    controller = (
        load_composite_controller_config(robot="Panda")
        if controller_config is None
        else deepcopy(controller_config)
    )
    env = environment_class(
        spec=spec,
        seed=seed,
        offscreen=offscreen,
        controller=controller,
    )
    env.sim.model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    env.reset()
    env.sim.forward()
    return env
