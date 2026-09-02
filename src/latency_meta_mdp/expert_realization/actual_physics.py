"""RoboSuite adapters for exact 2 ms structured-expert safety measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from latency_meta_mdp.backend import CompletedPhysicsStep, PreparedPhysicsPoint
from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
from latency_meta_mdp.expert_realization.safety import (
    CompiledSafetyGeometry,
    CompletedPhysicsSafetySample,
    PhysicsContactObservation,
    PhysicsSafetyAccumulator,
    PhysicsSafetySummary,
    PreparedPhysicsSafetySample,
    classify_contact_pair,
)


@dataclass(frozen=True)
class PreparedRuntimeSafetyMeasurement:
    minimum_environment_clearance_m: float
    minimum_joint_position_margin_rad: float
    maximum_joint_velocity_fraction: float
    contacts: tuple[PhysicsContactObservation, ...]

    def __post_init__(self) -> None:
        if (
            type(self.minimum_environment_clearance_m) is not float
            or not np.isfinite(self.minimum_environment_clearance_m)
            or self.minimum_environment_clearance_m < 0.0
        ):
            raise ValueError("minimum_environment_clearance_m must be non-negative finite")
        if (
            type(self.minimum_joint_position_margin_rad) is not float
            or not np.isfinite(self.minimum_joint_position_margin_rad)
        ):
            raise ValueError("minimum_joint_position_margin_rad must be finite")
        if (
            type(self.maximum_joint_velocity_fraction) is not float
            or not np.isfinite(self.maximum_joint_velocity_fraction)
            or self.maximum_joint_velocity_fraction < 0.0
        ):
            raise ValueError("maximum_joint_velocity_fraction must be non-negative finite")
        if type(self.contacts) is not tuple or any(
            not isinstance(item, PhysicsContactObservation) for item in self.contacts
        ):
            raise TypeError("contacts must be a tuple of PhysicsContactObservation")


PreparedRuntimeReader = Callable[
    [Any, PreparedPhysicsPoint, CompiledSafetyGeometry],
    PreparedRuntimeSafetyMeasurement,
]
CompletedRuntimeReader = Callable[
    [Any, CompletedPhysicsStep, tuple[int, ...]],
    tuple[tuple[int, float], ...],
]


class RuntimePhysicsSafetyMonitor:
    """Pair step1 geometry with step2 forces under the action interval's phase."""

    def __init__(
        self,
        *,
        env: Any,
        geometry: CompiledSafetyGeometry,
        prepared_reader: PreparedRuntimeReader | None = None,
        completed_reader: CompletedRuntimeReader | None = None,
    ) -> None:
        self.env = env
        self.geometry = geometry
        self._prepared_reader = prepared_reader or read_prepared_runtime_safety
        self._completed_reader = completed_reader or read_completed_contact_forces
        self._accumulator = PhysicsSafetyAccumulator(physics_dt_us=2_000)
        self._pending_point: PreparedPhysicsPoint | None = None
        self._pending_measurement: PreparedRuntimeSafetyMeasurement | Any | None = None
        self._active_phase: StructuredExpertPhase | None = None

    def set_active_interval_phase(self, phase: StructuredExpertPhase) -> None:
        if not isinstance(phase, StructuredExpertPhase):
            raise TypeError("active interval phase must be StructuredExpertPhase")
        self._active_phase = phase

    def on_prepared(self, point: PreparedPhysicsPoint) -> None:
        if not isinstance(point, PreparedPhysicsPoint):
            raise TypeError("point must be PreparedPhysicsPoint")
        if self._pending_point is not None:
            raise RuntimeError("a prepared runtime safety point is already pending")
        self._pending_point = point
        self._pending_measurement = self._prepared_reader(self.env, point, self.geometry)

    def on_completed(self, point: CompletedPhysicsStep) -> None:
        if not isinstance(point, CompletedPhysicsStep):
            raise TypeError("point must be CompletedPhysicsStep")
        if self._pending_point is None or self._pending_measurement is None:
            raise RuntimeError("completed runtime safety point has no prepared measurement")
        if self._active_phase is None:
            raise RuntimeError("completed runtime safety point has no active interval phase")
        prepared_point = self._pending_point
        measurement = self._pending_measurement
        contacts = tuple(measurement.contacts)
        self._accumulator.observe_prepared(
            PreparedPhysicsSafetySample(
                physics_step_index=prepared_point.physics_step_index,
                time_us=prepared_point.time_us,
                phase=self._active_phase,
                minimum_environment_clearance_m=float(
                    measurement.minimum_environment_clearance_m
                ),
                minimum_joint_position_margin_rad=float(
                    measurement.minimum_joint_position_margin_rad
                ),
                maximum_joint_velocity_fraction=float(
                    measurement.maximum_joint_velocity_fraction
                ),
                contacts=contacts,
            )
        )
        forces = self._completed_reader(
            self.env,
            point,
            tuple(contact.contact_index for contact in contacts),
        )
        self._accumulator.observe_completed(
            CompletedPhysicsSafetySample(
                physics_step_index=point.physics_step_index,
                time_us=point.time_us,
                normal_forces_n=forces,
            )
        )
        self._pending_point = None
        self._pending_measurement = None

    def finalize(self) -> PhysicsSafetySummary:
        # Every formal step prepares its terminal boundary for a possible next action.
        # That final step1 has no paired control/step2 and is intentionally not measured.
        if self._pending_point is not None and not self._pending_point.at_formal_boundary:
            raise RuntimeError("cannot discard a non-boundary prepared physics point")
        self._pending_point = None
        self._pending_measurement = None
        return self._accumulator.finalize()


def _geom_ids(model: Any, names: tuple[str, ...] | list[str], *, role: str) -> frozenset[int]:
    result = frozenset(int(model.geom_name2id(name)) for name in names)
    if not result:
        raise ValueError(f"{role} geometry set is empty")
    return result


def compile_runtime_safety_geometry(
    env: Any,
    *,
    joint_names: tuple[str, ...],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity_limit: np.ndarray,
) -> CompiledSafetyGeometry:
    """Compile exact geom and Panda joint IDs from one realized RoboSuite model."""
    expected_joint_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    if joint_names != expected_joint_names:
        raise ValueError("safety joint_names must use the certified Panda order")
    robot = env.robots[0]
    arm_name = robot.arms[0]
    gripper = robot.gripper[arm_name]
    model = env.sim.model
    runtime_joint_names = tuple(robot.robot_model.joints)
    expected_runtime_names = tuple(f"robot0_joint{index}" for index in range(1, 8))
    if runtime_joint_names != expected_runtime_names:
        raise ValueError("RoboSuite runtime joint order differs from the planning bridge")
    joint_ids = np.asarray(
        [model.joint_name2id(name) for name in runtime_joint_names], dtype=np.int64
    )
    qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    qvel_indices = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    if not np.array_equal(qpos_indices, np.asarray(model.jnt_qposadr[joint_ids])) or (
        not np.array_equal(qvel_indices, np.asarray(model.jnt_dofadr[joint_ids]))
    ):
        raise ValueError("RoboSuite robot state indexes differ from MuJoCo joint addresses")
    lower = np.asarray(joint_lower)
    upper = np.asarray(joint_upper)
    velocity = np.asarray(joint_velocity_limit)
    if (
        lower.dtype != np.float64
        or upper.dtype != np.float64
        or velocity.dtype != np.float64
        or lower.shape != (7,)
        or upper.shape != (7,)
        or velocity.shape != (7,)
    ):
        raise ValueError("planning-bridge joint limits must be float64[7]")
    runtime_ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    if not np.allclose(runtime_ranges[:, 0], lower, atol=1.0e-12, rtol=0.0) or (
        not np.allclose(runtime_ranges[:, 1], upper, atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("RoboSuite and planning-bridge joint position limits differ")

    left_pad = _geom_ids(
        model,
        list(gripper.important_geoms["left_fingerpad"]),
        role="left pad",
    )
    right_pad = _geom_ids(
        model,
        list(gripper.important_geoms["right_fingerpad"]),
        role="right pad",
    )
    ball = _geom_ids(model, list(env.ball.contact_geoms), role="ball")
    grasp_contact = _geom_ids(
        model,
        list(gripper.important_geoms["left_finger"])
        + list(gripper.important_geoms["right_finger"]),
        role="grasp contact",
    )
    movable = _geom_ids(
        model,
        list(robot.robot_model.contact_geoms) + list(gripper.contact_geoms),
        role="movable robot",
    )
    fixed_support = _geom_ids(
        model,
        list(robot.robot_model.base.contact_geoms),
        role="fixed robot support",
    )
    assembly = movable | fixed_support
    collision_enabled = frozenset(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0
    )
    environment = collision_enabled - assembly - ball
    return CompiledSafetyGeometry(
        left_pad_geom_ids=left_pad,
        right_pad_geom_ids=right_pad,
        grasp_contact_geom_ids=grasp_contact,
        ball_geom_ids=ball,
        movable_robot_geom_ids=movable,
        robot_assembly_geom_ids=assembly,
        environment_geom_ids=environment,
        robot_qpos_indices=qpos_indices,
        robot_qvel_indices=qvel_indices,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
    )


def _verify_runtime_clock(env: Any, *, time_us: int) -> None:
    actual = round(float(env.sim.data.time) * 1_000_000)
    if actual != time_us:
        raise RuntimeError(f"runtime safety clock mismatch: expected {time_us}, got {actual}")


def read_prepared_runtime_safety(
    env: Any,
    point: PreparedPhysicsPoint,
    geometry: CompiledSafetyGeometry,
) -> PreparedRuntimeSafetyMeasurement:
    """Read geometry, contacts, and joint state after the current ``mj_step1``."""
    import mujoco

    _verify_runtime_clock(env, time_us=point.time_us)
    data = env.sim.data
    model = env.sim.model
    raw_model = model._model
    raw_data = data._data
    contacts = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        kind = classify_contact_pair(geometry, int(contact.geom1), int(contact.geom2))
        if kind is not None:
            contacts.append(
                PhysicsContactObservation(
                    contact_index=contact_index,
                    kind=kind,
                    penetration_m=float(max(0.0, -float(contact.dist))),
                )
            )
    distances = []
    for robot_geom in sorted(geometry.movable_robot_geom_ids):
        for environment_geom in sorted(geometry.environment_geom_ids):
            distance = float(
                mujoco.mj_geomDistance(
                    raw_model,
                    raw_data,
                    robot_geom,
                    environment_geom,
                    1.0,
                    None,
                )
            )
            if np.isfinite(distance):
                distances.append(max(0.0, distance))
    if not distances:
        raise RuntimeError("MuJoCo did not provide any finite robot-environment distance")
    qpos = np.asarray(data.qpos[geometry.robot_qpos_indices], dtype=np.float64)
    qvel = np.asarray(data.qvel[geometry.robot_qvel_indices], dtype=np.float64)
    joint_margin = np.minimum(
        qpos - geometry.joint_lower,
        geometry.joint_upper - qpos,
    )
    return PreparedRuntimeSafetyMeasurement(
        minimum_environment_clearance_m=float(min(distances)),
        minimum_joint_position_margin_rad=float(np.min(joint_margin)),
        maximum_joint_velocity_fraction=float(
            np.max(np.abs(qvel) / geometry.joint_velocity_limit)
        ),
        contacts=tuple(contacts),
    )


def read_completed_contact_forces(
    env: Any,
    point: CompletedPhysicsStep,
    contact_indices: tuple[int, ...],
) -> tuple[tuple[int, float], ...]:
    """Read solved normal forces after the matching ``mj_step2``."""
    import mujoco

    _verify_runtime_clock(env, time_us=point.time_us)
    raw_model = env.sim.model._model
    raw_data = env.sim.data._data
    contact_count = int(env.sim.data.ncon)
    rows = []
    for contact_index in contact_indices:
        if not 0 <= contact_index < contact_count:
            raise RuntimeError("prepared contact index disappeared before force readout")
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(raw_model, raw_data, contact_index, wrench)
        if not np.all(np.isfinite(wrench)) or wrench[0] < -1.0e-12:
            raise RuntimeError("MuJoCo returned an invalid normal contact force")
        rows.append((contact_index, float(max(0.0, wrench[0]))))
    return tuple(rows)
