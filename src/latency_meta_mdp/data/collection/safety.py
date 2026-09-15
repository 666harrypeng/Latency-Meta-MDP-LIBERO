"""Physical-safety metrics for structured expert references and MuJoCo rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import numpy as np

from latency_meta_mdp.data.collection.executor import StructuredExpertPhase


class ContactKind(str, Enum):
    PAD_BALL = "pad_ball"
    OTHER_ROBOT_BALL = "other_robot_ball"
    ROBOT_ENVIRONMENT = "robot_environment"
    ROBOT_SELF = "robot_self"


@dataclass(frozen=True)
class CompiledSafetyGeometry:
    left_pad_geom_ids: frozenset[int]
    right_pad_geom_ids: frozenset[int]
    grasp_contact_geom_ids: frozenset[int]
    ball_geom_ids: frozenset[int]
    movable_robot_geom_ids: frozenset[int]
    robot_assembly_geom_ids: frozenset[int]
    environment_geom_ids: frozenset[int]
    robot_qpos_indices: np.ndarray
    robot_qvel_indices: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    joint_velocity_limit: np.ndarray

    def __post_init__(self) -> None:
        set_names = (
            "left_pad_geom_ids",
            "right_pad_geom_ids",
            "grasp_contact_geom_ids",
            "ball_geom_ids",
            "movable_robot_geom_ids",
            "robot_assembly_geom_ids",
            "environment_geom_ids",
        )
        for name in set_names:
            values = getattr(self, name)
            if (
                type(values) is not frozenset
                or not values
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise ValueError(f"{name} must be a non-empty frozenset of non-negative integers")
        pads = self.left_pad_geom_ids | self.right_pad_geom_ids
        if self.left_pad_geom_ids & self.right_pad_geom_ids:
            raise ValueError("left and right pad geometry must be disjoint")
        if not pads <= self.grasp_contact_geom_ids:
            raise ValueError("pad geometry must belong to the grasp-contact geometry")
        if not self.grasp_contact_geom_ids <= self.movable_robot_geom_ids:
            raise ValueError("grasp-contact geometry must belong to the movable robot")
        if not pads <= self.movable_robot_geom_ids:
            raise ValueError("pad geometry must belong to the movable robot")
        if not self.movable_robot_geom_ids <= self.robot_assembly_geom_ids:
            raise ValueError("movable robot geometry must belong to the robot assembly")
        if self.ball_geom_ids & self.robot_assembly_geom_ids:
            raise ValueError("ball and robot assembly geometry must be disjoint")
        if self.environment_geom_ids & (self.robot_assembly_geom_ids | self.ball_geom_ids):
            raise ValueError("environment geometry must be disjoint from robot and ball")
        array_contracts = {
            "robot_qpos_indices": (np.dtype(np.int64), False),
            "robot_qvel_indices": (np.dtype(np.int64), False),
            "joint_lower": (np.dtype(np.float64), True),
            "joint_upper": (np.dtype(np.float64), True),
            "joint_velocity_limit": (np.dtype(np.float64), True),
        }
        for name, (dtype, finite) in array_contracts.items():
            value = np.asarray(getattr(self, name))
            if value.dtype != dtype or value.shape != (7,):
                raise ValueError(f"{name} must have dtype {dtype} and shape (7,)")
            if finite and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if np.any(self.robot_qpos_indices < 0) or np.any(self.robot_qvel_indices < 0):
            raise ValueError("robot state indices must be non-negative")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("joint_lower must be strictly below joint_upper")
        if np.any(self.joint_velocity_limit <= 0.0):
            raise ValueError("joint_velocity_limit must be positive")


def classify_contact_pair(
    geometry: CompiledSafetyGeometry,
    geom1: int,
    geom2: int,
) -> ContactKind | None:
    if not isinstance(geometry, CompiledSafetyGeometry):
        raise TypeError("geometry must be CompiledSafetyGeometry")
    if type(geom1) is not int or type(geom2) is not int or geom1 < 0 or geom2 < 0:
        raise ValueError("contact geometry IDs must be non-negative integers")
    pair = {geom1, geom2}
    if pair & geometry.grasp_contact_geom_ids and pair & geometry.ball_geom_ids:
        return ContactKind.PAD_BALL
    if pair & geometry.robot_assembly_geom_ids and pair & geometry.ball_geom_ids:
        return ContactKind.OTHER_ROBOT_BALL
    if pair & geometry.robot_assembly_geom_ids and pair & geometry.environment_geom_ids:
        return ContactKind.ROBOT_ENVIRONMENT
    if (
        geom1 != geom2
        and geom1 in geometry.robot_assembly_geom_ids
        and (geom2 in geometry.robot_assembly_geom_ids)
    ):
        return ContactKind.ROBOT_SELF
    return None


@dataclass(frozen=True)
class PhysicsContactObservation:
    contact_index: int
    kind: ContactKind
    penetration_m: float

    def __post_init__(self) -> None:
        if type(self.contact_index) is not int or self.contact_index < 0:
            raise ValueError("contact_index must be non-negative")
        if not isinstance(self.kind, ContactKind):
            raise TypeError("kind must be ContactKind")
        if type(self.penetration_m) is not float or not np.isfinite(self.penetration_m):
            raise ValueError("penetration_m must be finite float")
        if self.penetration_m < 0.0:
            raise ValueError("penetration_m must be non-negative")


@dataclass(frozen=True)
class PreparedPhysicsSafetySample:
    physics_step_index: int
    time_us: int
    phase: StructuredExpertPhase
    minimum_environment_clearance_m: float
    minimum_joint_position_margin_rad: float
    maximum_joint_velocity_fraction: float
    contacts: tuple[PhysicsContactObservation, ...]

    def __post_init__(self) -> None:
        if (
            type(self.physics_step_index) is not int
            or self.physics_step_index < 0
            or self.time_us != self.physics_step_index * 2_000
        ):
            raise ValueError("prepared physics sample clock is invalid")
        if not isinstance(self.phase, StructuredExpertPhase):
            raise TypeError("phase must be StructuredExpertPhase")
        for name in (
            "minimum_environment_clearance_m",
            "maximum_joint_velocity_fraction",
        ):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float")
        if type(self.minimum_joint_position_margin_rad) is not float or not np.isfinite(
            self.minimum_joint_position_margin_rad
        ):
            raise ValueError("minimum_joint_position_margin_rad must be a finite float")
        if type(self.contacts) is not tuple or any(
            not isinstance(contact, PhysicsContactObservation) for contact in self.contacts
        ):
            raise TypeError("contacts must be a tuple of PhysicsContactObservation")
        indices = [contact.contact_index for contact in self.contacts]
        if len(set(indices)) != len(indices):
            raise ValueError("prepared contact indices must be unique")


@dataclass(frozen=True)
class CompletedPhysicsSafetySample:
    physics_step_index: int
    time_us: int
    normal_forces_n: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if (
            type(self.physics_step_index) is not int
            or self.physics_step_index <= 0
            or self.time_us != self.physics_step_index * 2_000
        ):
            raise ValueError("completed physics sample clock is invalid")
        if type(self.normal_forces_n) is not tuple:
            raise TypeError("normal_forces_n must be a tuple")
        indices = []
        for value in self.normal_forces_n:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError("normal force rows must be (contact_index, force_n) tuples")
            index, force = value
            if type(index) is not int or index < 0:
                raise ValueError("normal force contact index must be non-negative")
            if type(force) is not float or not np.isfinite(force) or force < 0.0:
                raise ValueError("normal force must be a non-negative finite float")
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("normal force contact indices must be unique")


@dataclass(frozen=True)
class PhysicsSafetySummary:
    minimum_non_contact_environment_clearance_m: float
    maximum_intentional_contact_penetration_m: float
    maximum_pad_ball_impulse_ns: float
    unintended_pregrasp_ball_contacts: int
    other_link_ball_contacts: int
    robot_environment_contacts: int
    robot_self_contacts: int
    minimum_joint_position_margin_rad: float
    maximum_joint_velocity_fraction: float


class PhysicsSafetyAccumulator:
    def __init__(self, *, physics_dt_us: int) -> None:
        if type(physics_dt_us) is not int or physics_dt_us != 2_000:
            raise ValueError("physics_dt_us must equal 2000")
        self.physics_dt_us = physics_dt_us
        self._pending: PreparedPhysicsSafetySample | None = None
        self._last_completed_step = 0
        self._completed_count = 0
        self._minimum_clearance = np.inf
        self._minimum_joint_margin = np.inf
        self._maximum_velocity_fraction = 0.0
        self._maximum_penetration = 0.0
        self._maximum_pad_impulse = 0.0
        self._unintended_pad_contacts = 0
        self._other_link_ball_contacts = 0
        self._robot_environment_contacts = 0
        self._robot_self_contacts = 0

    def observe_prepared(self, sample: PreparedPhysicsSafetySample) -> None:
        if not isinstance(sample, PreparedPhysicsSafetySample):
            raise TypeError("sample must be PreparedPhysicsSafetySample")
        if self._pending is not None:
            raise RuntimeError("a prepared physics point is already pending")
        if sample.physics_step_index != self._last_completed_step:
            raise ValueError("prepared physics sample is not the next split-step point")
        self._pending = sample

    def observe_completed(self, sample: CompletedPhysicsSafetySample) -> None:
        if not isinstance(sample, CompletedPhysicsSafetySample):
            raise TypeError("sample must be CompletedPhysicsSafetySample")
        prepared = self._pending
        if prepared is None:
            raise RuntimeError("completed physics step has no pending prepared point")
        if sample.physics_step_index != prepared.physics_step_index + 1 or (
            sample.time_us != prepared.time_us + self.physics_dt_us
        ):
            raise ValueError("completed physics sample does not match prepared point")
        contacts = {contact.contact_index: contact for contact in prepared.contacts}
        forces = dict(sample.normal_forces_n)
        if set(contacts) != set(forces):
            raise ValueError("completed contact-force indices do not match prepared contacts")
        self._minimum_clearance = min(
            self._minimum_clearance,
            prepared.minimum_environment_clearance_m,
        )
        self._minimum_joint_margin = min(
            self._minimum_joint_margin,
            prepared.minimum_joint_position_margin_rad,
        )
        self._maximum_velocity_fraction = max(
            self._maximum_velocity_fraction,
            prepared.maximum_joint_velocity_fraction,
        )
        for index, contact in contacts.items():
            impulse = forces[index] * self.physics_dt_us / 1_000_000
            if contact.kind is ContactKind.PAD_BALL:
                self._maximum_pad_impulse = max(self._maximum_pad_impulse, impulse)
                if contact_is_allowed(contact.kind, prepared.phase):
                    self._maximum_penetration = max(
                        self._maximum_penetration,
                        contact.penetration_m,
                    )
                else:
                    self._unintended_pad_contacts += 1
            elif contact.kind is ContactKind.OTHER_ROBOT_BALL:
                self._other_link_ball_contacts += 1
            elif contact.kind is ContactKind.ROBOT_ENVIRONMENT:
                self._robot_environment_contacts += 1
            elif contact.kind is ContactKind.ROBOT_SELF:
                self._robot_self_contacts += 1
        self._pending = None
        self._last_completed_step = sample.physics_step_index
        self._completed_count += 1

    def finalize(self) -> PhysicsSafetySummary:
        if self._pending is not None:
            raise RuntimeError("cannot finalize with a pending prepared physics point")
        if self._completed_count == 0:
            raise RuntimeError("cannot finalize an empty physics safety trace")
        return PhysicsSafetySummary(
            minimum_non_contact_environment_clearance_m=float(self._minimum_clearance),
            maximum_intentional_contact_penetration_m=float(self._maximum_penetration),
            maximum_pad_ball_impulse_ns=float(self._maximum_pad_impulse),
            unintended_pregrasp_ball_contacts=self._unintended_pad_contacts,
            other_link_ball_contacts=self._other_link_ball_contacts,
            robot_environment_contacts=self._robot_environment_contacts,
            robot_self_contacts=self._robot_self_contacts,
            minimum_joint_position_margin_rad=float(self._minimum_joint_margin),
            maximum_joint_velocity_fraction=float(self._maximum_velocity_fraction),
        )


def contact_is_allowed(kind: ContactKind, phase: StructuredExpertPhase) -> bool:
    if not isinstance(kind, ContactKind) or not isinstance(phase, StructuredExpertPhase):
        raise TypeError("contact admission requires typed kind and phase")
    return bool(
        kind is ContactKind.PAD_BALL
        and phase in {StructuredExpertPhase.CLOSE_STABILIZE, StructuredExpertPhase.LIFT}
    )


@dataclass(frozen=True)
class SweptClearanceReport:
    safe: bool
    minimum_distance_m: float
    minimum_clearance_m: float
    minimum_time_us: int
    sample_count: int

    def __post_init__(self) -> None:
        if type(self.safe) is not bool:
            raise TypeError("safe must be boolean")
        for name in ("minimum_distance_m", "minimum_clearance_m"):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite float")
        if type(self.minimum_time_us) is not int or self.minimum_time_us < 0:
            raise ValueError("minimum_time_us must be non-negative")
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("sample_count must be at least two")


def _trajectory(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float64
        or array.ndim != 2
        or array.shape[1:] != (3,)
        or len(array) < 2
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be finite float64[N,3]")
    return array


def evaluate_moving_target_swept_clearance(
    *,
    reference_positions_world: np.ndarray,
    object_positions_world: np.ndarray,
    reference_start_time_us: int,
    minimum_center_distance_m: float,
) -> SweptClearanceReport:
    """Evaluate paired 50 Hz paths on every intervening certified 2 ms physics point."""
    reference = _trajectory(reference_positions_world, name="reference_positions_world")
    objects = _trajectory(object_positions_world, name="object_positions_world")
    if reference.shape != objects.shape:
        raise ValueError("reference and object trajectories must be aligned")
    if type(reference_start_time_us) is not int or reference_start_time_us < 0:
        raise ValueError("reference_start_time_us must be non-negative")
    if reference_start_time_us % 20_000:
        raise ValueError("reference_start_time_us must lie on the formal grid")
    if (
        type(minimum_center_distance_m) is not float
        or not np.isfinite(minimum_center_distance_m)
        or minimum_center_distance_m <= 0.0
    ):
        raise ValueError("minimum_center_distance_m must be a positive finite float")

    distances = []
    times = []
    for interval in range(len(reference) - 1):
        for substep in range(10):
            alpha = substep / 10.0
            reference_point = (1.0 - alpha) * reference[interval] + alpha * reference[interval + 1]
            object_point = (1.0 - alpha) * objects[interval] + alpha * objects[interval + 1]
            distances.append(float(np.linalg.norm(reference_point - object_point)))
            times.append(reference_start_time_us + interval * 20_000 + substep * 2_000)
    distances.append(float(np.linalg.norm(reference[-1] - objects[-1])))
    times.append(reference_start_time_us + (len(reference) - 1) * 20_000)
    minimum_index = int(np.argmin(distances))
    minimum_distance = distances[minimum_index]
    return SweptClearanceReport(
        safe=minimum_distance >= minimum_center_distance_m,
        minimum_distance_m=float(minimum_distance),
        minimum_clearance_m=float(minimum_distance - minimum_center_distance_m),
        minimum_time_us=times[minimum_index],
        sample_count=len(distances),
    )


@dataclass(frozen=True)
class ActualRolloutSafetyReport:
    terminal_success: bool
    physical_handoff: bool
    phase_order_valid: bool
    minimum_non_contact_environment_clearance_m: float
    maximum_intentional_contact_penetration_m: float
    maximum_pad_ball_impulse_ns: float
    unintended_pregrasp_ball_contacts: int
    other_link_ball_contacts: int
    robot_environment_contacts: int
    robot_self_contacts: int
    minimum_joint_position_margin_rad: float
    maximum_joint_velocity_fraction: float
    maximum_eef_speed_mps: float
    maximum_eef_acceleration_mps2: float
    maximum_eef_jerk_mps3: float
    maximum_reference_tracking_error_m: float
    maximum_pregrasp_translation_error_m: float
    maximum_pregrasp_rotation_error_degrees: float
    minimum_osc_action: float
    maximum_osc_action: float
    pre_handoff_saturation_fraction: float

    def __post_init__(self) -> None:
        for name in ("terminal_success", "physical_handoff", "phase_order_valid"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name in (
            "unintended_pregrasp_ball_contacts",
            "other_link_ball_contacts",
            "robot_environment_contacts",
            "robot_self_contacts",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        excluded = {
            "terminal_success",
            "physical_handoff",
            "phase_order_valid",
            "unintended_pregrasp_ball_contacts",
            "other_link_ball_contacts",
            "robot_environment_contacts",
            "robot_self_contacts",
        }
        for item in fields(self):
            if item.name in excluded:
                continue
            value = getattr(self, item.name)
            if type(value) is not float or not np.isfinite(value):
                raise ValueError(f"{item.name} must be a finite float")
        nonnegative = {
            "minimum_non_contact_environment_clearance_m",
            "maximum_intentional_contact_penetration_m",
            "maximum_pad_ball_impulse_ns",
            "maximum_joint_velocity_fraction",
            "maximum_eef_speed_mps",
            "maximum_eef_acceleration_mps2",
            "maximum_eef_jerk_mps3",
            "maximum_reference_tracking_error_m",
            "maximum_pregrasp_translation_error_m",
            "maximum_pregrasp_rotation_error_degrees",
            "pre_handoff_saturation_fraction",
        }
        if any(getattr(self, name) < 0.0 for name in nonnegative):
            raise ValueError("rollout safety magnitudes must be non-negative")
        if self.minimum_osc_action > self.maximum_osc_action:
            raise ValueError("OSC action extrema are reversed")

    def to_mapping(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def phase_order_is_valid(phase_sequence: tuple[str, ...]) -> bool:
    if type(phase_sequence) is not tuple or any(type(value) is not str for value in phase_sequence):
        raise TypeError("phase_sequence must be a tuple of strings")
    canonical = tuple(phase.value for phase in StructuredExpertPhase)
    if not phase_sequence or set(phase_sequence) != set(canonical):
        return False
    ranks = {name: index for index, name in enumerate(canonical)}
    try:
        values = tuple(ranks[name] for name in phase_sequence)
    except KeyError:
        return False
    return (
        values[0] == 0
        and values[-1] == len(canonical) - 1
        and all(left <= right for left, right in zip(values, values[1:]))
    )


def _maximum_vector_norm(values: np.ndarray) -> float:
    return 0.0 if not len(values) else float(np.max(np.linalg.norm(values, axis=1)))


def build_actual_rollout_safety_report(
    *,
    rollout: Any,
    physics: PhysicsSafetySummary,
) -> ActualRolloutSafetyReport:
    """Combine 2 ms physics measurements with 50 Hz rollout-level metrics."""
    if not isinstance(physics, PhysicsSafetySummary):
        raise TypeError("physics must be PhysicsSafetySummary")
    positions = np.asarray(rollout.eef_positions_world)
    object_positions = np.asarray(rollout.object_positions_world)
    actions = np.asarray(rollout.actions)
    decisions = tuple(rollout.decisions)
    if (
        positions.dtype != np.float64
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or len(positions) < 2
        or object_positions.dtype != np.float64
        or object_positions.shape != positions.shape
        or actions.dtype != np.float64
        or actions.shape != (len(positions) - 1, 7)
    ):
        raise ValueError("rollout trace does not satisfy the 50 Hz metric contract")
    velocity = np.diff(positions, axis=0) / 0.02
    acceleration = np.diff(velocity, axis=0) / 0.02
    jerk = np.diff(acceleration, axis=0) / 0.02
    outside_contact = tuple(
        item
        for item in decisions
        if item.phase in {StructuredExpertPhase.SMOOTH_APPROACH, StructuredExpertPhase.GRASP_FUNNEL}
    )

    def maximum_error(rows: tuple[Any, ...], field: str) -> float:
        return max(
            (float(np.linalg.norm(np.asarray(getattr(row, field)))) for row in rows),
            default=0.0,
        )

    handoff_tick = rollout.physical_handoff_tick
    pre_handoff = tuple(
        item for item in decisions if handoff_tick is None or item.source_formal_tick < handoff_tick
    )
    first_funnel = next(
        (item for item in decisions if item.phase is StructuredExpertPhase.GRASP_FUNNEL),
        None,
    )
    if first_funnel is None:
        entry_translation_error = 0.0
        entry_rotation_error_degrees = 0.0
    else:
        entry_tick = first_funnel.source_formal_tick
        if not 0 <= entry_tick < len(positions):
            raise ValueError("funnel-entry decision lies outside the rollout trace")
        preceding = next(
            (
                item
                for item in reversed(decisions)
                if item.source_formal_tick == entry_tick - 1
                and item.phase is StructuredExpertPhase.SMOOTH_APPROACH
            ),
            None,
        )
        if preceding is None:
            raise ValueError("funnel entry lacks its preceding pregrasp decision")
        entry_translation_error = float(
            np.linalg.norm(positions[entry_tick] - preceding.target_eef_position_world)
        )
        entry_rotation_error_degrees = float(
            np.degrees(np.linalg.norm(first_funnel.orientation_error_rotvec_base))
        )
    return ActualRolloutSafetyReport(
        terminal_success=rollout.terminal_status == "success",
        physical_handoff=handoff_tick is not None,
        phase_order_valid=phase_order_is_valid(tuple(rollout.phase_sequence)),
        minimum_non_contact_environment_clearance_m=(
            physics.minimum_non_contact_environment_clearance_m
        ),
        maximum_intentional_contact_penetration_m=(
            physics.maximum_intentional_contact_penetration_m
        ),
        maximum_pad_ball_impulse_ns=physics.maximum_pad_ball_impulse_ns,
        unintended_pregrasp_ball_contacts=physics.unintended_pregrasp_ball_contacts,
        other_link_ball_contacts=physics.other_link_ball_contacts,
        robot_environment_contacts=physics.robot_environment_contacts,
        robot_self_contacts=physics.robot_self_contacts,
        minimum_joint_position_margin_rad=physics.minimum_joint_position_margin_rad,
        maximum_joint_velocity_fraction=physics.maximum_joint_velocity_fraction,
        maximum_eef_speed_mps=_maximum_vector_norm(velocity),
        maximum_eef_acceleration_mps2=_maximum_vector_norm(acceleration),
        maximum_eef_jerk_mps3=_maximum_vector_norm(jerk),
        maximum_reference_tracking_error_m=maximum_error(outside_contact, "position_error_base"),
        maximum_pregrasp_translation_error_m=entry_translation_error,
        maximum_pregrasp_rotation_error_degrees=entry_rotation_error_degrees,
        minimum_osc_action=float(np.min(actions)),
        maximum_osc_action=float(np.max(actions)),
        pre_handoff_saturation_fraction=float(
            0.0
            if not pre_handoff
            else sum(int(item.saturated) for item in pre_handoff) / len(pre_handoff)
        ),
    )
