"""Structured-reference conversion into the certified Panda OSC action contract."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.expert_realization.selector import SelectedReference
from latency_meta_mdp.expert_realization.trajectory_intent import PlannedMotionIntent
from latency_meta_mdp.handoff import HandoffState
from latency_meta_mdp.snapshots import BoundarySnapshot


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite float64[3]")
    return array


def _rotation(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.dtype != np.float64
        or matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9, rtol=0.0)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-9, rtol=0.0)
    ):
        raise ValueError(f"{name} must be a proper float64 rotation matrix")
    return matrix


@dataclass(frozen=True)
class OscActionResult:
    action: np.ndarray
    position_error_base: np.ndarray
    orientation_error_rotvec_base: np.ndarray
    saturated: bool

    def __post_init__(self) -> None:
        for name, shape in (
            ("action", (7,)),
            ("position_error_base", (3,)),
            ("orientation_error_rotvec_base", (3,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64 or value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite float64{shape}")
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        if type(self.saturated) is not bool:
            raise TypeError("saturated must be boolean")


def osc_action_from_reference(
    *,
    achieved_position_world: np.ndarray,
    achieved_orientation_world: np.ndarray,
    target_position_world: np.ndarray,
    target_orientation_world: np.ndarray,
    world_to_base_rotation: np.ndarray,
    action_contract: ActionContract,
    gripper_command: float,
    translation_clip_m: float,
) -> OscActionResult:
    """Convert one achieved-to-target world pose error into normalized base-frame OSC input."""
    if not isinstance(action_contract, ActionContract):
        raise TypeError("action_contract must be an ActionContract")
    if (
        type(translation_clip_m) is not float
        or not np.isfinite(translation_clip_m)
        or translation_clip_m <= 0.0
    ):
        raise ValueError("translation_clip_m must be a positive finite float")
    achieved_position = _vector(
        achieved_position_world, name="achieved_position_world"
    )
    target_position = _vector(target_position_world, name="target_position_world")
    achieved_orientation = _rotation(
        achieved_orientation_world, name="achieved_orientation_world"
    )
    target_orientation = _rotation(
        target_orientation_world, name="target_orientation_world"
    )
    world_to_base = _rotation(
        world_to_base_rotation, name="world_to_base_rotation"
    )

    position_error_base = world_to_base @ (target_position - achieved_position)
    commanded_position = np.array(position_error_base, copy=True)
    position_norm = float(np.linalg.norm(commanded_position))
    if position_norm > translation_clip_m:
        commanded_position *= translation_clip_m / position_norm

    achieved_orientation_base = world_to_base @ achieved_orientation
    target_orientation_base = world_to_base @ target_orientation
    orientation_delta_base = target_orientation_base @ achieved_orientation_base.T
    orientation_error = Rotation.from_matrix(orientation_delta_base).as_rotvec()
    desired_output = np.concatenate([commanded_position, orientation_error])

    input_bias = 0.5 * (action_contract.arm_input_high + action_contract.arm_input_low)
    input_weight = 0.5 * (action_contract.arm_input_high - action_contract.arm_input_low)
    output_bias = 0.5 * (action_contract.arm_output_high + action_contract.arm_output_low)
    output_weight = 0.5 * (action_contract.arm_output_high - action_contract.arm_output_low)
    normalized_unclipped = (
        (desired_output - output_bias) / output_weight * input_weight + input_bias
    )
    normalized = np.clip(
        normalized_unclipped,
        action_contract.arm_input_low,
        action_contract.arm_input_high,
    ).astype(np.float64, copy=False)
    saturated = bool(not np.array_equal(normalized, normalized_unclipped))
    action = action_contract.compose_action(
        arm_reference=normalized,
        gripper_command=gripper_command,
    ).astype(np.float64, copy=False)
    return OscActionResult(
        action=action,
        position_error_base=np.asarray(position_error_base, dtype=np.float64),
        orientation_error_rotvec_base=np.asarray(orientation_error, dtype=np.float64),
        saturated=saturated,
    )


class StructuredExpertPhase(str, Enum):
    SMOOTH_APPROACH = "smooth_approach"
    GRASP_FUNNEL = "grasp_funnel"
    CLOSE_STABILIZE = "close_stabilize"
    LIFT = "lift"


class SemanticExecutionFailure(RuntimeError):
    """A frozen realization violated its event-gated execution contract."""


@dataclass(frozen=True)
class StructuredExpertDecision:
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int
    phase: StructuredExpertPhase
    reference_kind: str
    selected_reference_index: int
    target_eef_position_world: np.ndarray
    target_eef_orientation_world: np.ndarray
    estimated_object_velocity_world: np.ndarray
    action: np.ndarray
    position_error_base: np.ndarray
    orientation_error_rotvec_base: np.ndarray
    saturated: bool

    def __post_init__(self) -> None:
        if (
            type(self.source_formal_tick) is not int
            or self.source_formal_tick < 0
            or self.source_physics_step != 10 * self.source_formal_tick
            or self.source_time_us != 20_000 * self.source_formal_tick
        ):
            raise ValueError("structured expert decision clock is invalid")
        if not isinstance(self.phase, StructuredExpertPhase):
            raise TypeError("phase must be StructuredExpertPhase")
        if self.reference_kind != "selected_reference":
            raise ValueError("post-prefix decisions must use selected_reference")
        if type(self.selected_reference_index) is not int or self.selected_reference_index < 0:
            raise ValueError("selected_reference_index must be non-negative")
        for name, shape in (
            ("target_eef_position_world", (3,)),
            ("target_eef_orientation_world", (3, 3)),
            ("estimated_object_velocity_world", (3,)),
            ("action", (7,)),
            ("position_error_base", (3,)),
            ("orientation_error_rotvec_base", (3,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64 or value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite float64{shape}")
            result = np.array(value, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        _rotation(
            self.target_eef_orientation_world,
            name="target_eef_orientation_world",
        )
        if type(self.saturated) is not bool:
            raise TypeError("saturated must be boolean")


class StructuredExpertExecutor:
    """Execute one smooth approach and one canonical event-gated grasp funnel."""

    def __init__(
        self,
        *,
        action_contract: ActionContract,
        intent: PlannedMotionIntent,
        reference: SelectedReference,
        decision_source_tick: int,
        world_to_base_rotation: np.ndarray,
        history_ticks: int = 6,
    ) -> None:
        if not isinstance(action_contract, ActionContract):
            raise TypeError("action_contract must be an ActionContract")
        if not isinstance(intent, PlannedMotionIntent):
            raise TypeError("intent must be a PlannedMotionIntent")
        if not isinstance(reference, SelectedReference):
            raise TypeError("reference must be a SelectedReference")
        if reference.expert_realization_key.task_instance_id != intent.task_instance_id:
            raise ValueError("reference and intent task identities do not match")
        if type(decision_source_tick) is not int or decision_source_tick != 5:
            raise ValueError("decision_source_tick must equal 5")
        if type(history_ticks) is not int or history_ticks < 2:
            raise ValueError("history_ticks must be an integer of at least two")
        expected_duration = (
            intent.approach.funnel_entry_target_tick - decision_source_tick
        ) * 0.02
        if (
            len(reference.timestamps_seconds) < 2
            or reference.timestamps_seconds[0] != 0.0
            or not np.allclose(
                np.diff(reference.timestamps_seconds),
                0.02,
                atol=1.0e-12,
                rtol=0.0,
            )
            or not np.isclose(
                reference.timestamps_seconds[-1], expected_duration, atol=1.0e-12, rtol=0.0
            )
        ):
            raise ValueError("selected reference does not match the frozen 50 Hz arrival schedule")
        if not np.array_equal(
            reference.fixed_orientation_world,
            intent.approach.fixed_orientation_world,
        ):
            raise ValueError("selected reference orientation does not match the intent")
        self.action_contract = action_contract
        self.intent = intent
        self.reference = reference
        self.decision_source_tick = decision_source_tick
        self.world_to_base_rotation = np.array(
            _rotation(world_to_base_rotation, name="world_to_base_rotation"),
            copy=True,
        )
        self.world_to_base_rotation.setflags(write=False)
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=history_ticks)
        self._last_tick: int | None = None
        self._previous_eef_position: np.ndarray | None = None
        self._close_started_tick: int | None = None
        self._first_unilateral_contact_tick: int | None = None
        self._grasp_offset_world: np.ndarray | None = None
        self._lift_start_position_world: np.ndarray | None = None
        self._funnel_start_tick: int | None = None
        self._effective_close_target_tick: int | None = None

    def _dynamic_entry_target(
        self,
        snapshot: BoundarySnapshot,
        estimated_object_velocity: np.ndarray,
    ) -> np.ndarray:
        return (
            np.asarray(snapshot.object_body_pos, dtype=np.float64)
            + estimated_object_velocity * self.intent.strategy.prediction_lead_seconds
            + np.array(
                [0.0, 0.0, self.intent.strategy.funnel_entry_height_m],
                dtype=np.float64,
            )
        )

    def _funnel_target(
        self,
        snapshot: BoundarySnapshot,
        estimated_object_velocity: np.ndarray,
    ) -> np.ndarray:
        if self._funnel_start_tick is None or self._effective_close_target_tick is None:
            raise RuntimeError("canonical funnel has not been initialized")
        progress = min(
            1.0,
            (snapshot.formal_tick_index + 1 - self._funnel_start_tick)
            / self.intent.strategy.funnel_descent_ticks,
        )
        centering_fraction = 0.35
        descent_progress = max(
            0.0,
            (progress - centering_fraction) / (1.0 - centering_fraction),
        )
        descent = (
            10.0 * descent_progress**3
            - 15.0 * descent_progress**4
            + 6.0 * descent_progress**5
        )
        height = self.intent.strategy.funnel_entry_height_m * (1.0 - descent)
        return (
            np.asarray(snapshot.object_body_pos, dtype=np.float64)
            + estimated_object_velocity * self.intent.strategy.prediction_lead_seconds
            + np.array([0.0, 0.0, height], dtype=np.float64)
        )

    def _entry_is_achieved(
        self,
        snapshot: BoundarySnapshot,
        estimated_object_velocity: np.ndarray,
    ) -> bool:
        current_entry = (
            np.asarray(snapshot.object_body_pos, dtype=np.float64)
            + np.array(
                [0.0, 0.0, self.intent.strategy.funnel_entry_height_m],
                dtype=np.float64,
            )
        )
        error = np.asarray(snapshot.eef_pos, dtype=np.float64) - current_entry
        return bool(np.linalg.norm(error) <= 0.015)

    def _estimated_object_velocity(self, snapshot: BoundarySnapshot) -> np.ndarray:
        tick = snapshot.formal_tick_index
        if self._last_tick is None:
            if tick != self.decision_source_tick:
                raise ValueError("structured executor must start at the decision source tick")
        elif tick != self._last_tick + 1:
            raise ValueError("structured executor requires every consecutive formal boundary")
        self._last_tick = tick
        self._history.append(
            (tick, np.array(snapshot.object_body_pos, dtype=np.float64, copy=True))
        )
        if len(self._history) < 2:
            return np.zeros(3, dtype=np.float64)
        start_tick, start_position = self._history[0]
        end_tick, end_position = self._history[-1]
        return (end_position - start_position) / ((end_tick - start_tick) * 0.02)

    def _close_geometry_is_ready(
        self,
        snapshot: BoundarySnapshot,
        estimated_object_velocity: np.ndarray,
    ) -> bool:
        eef = np.asarray(snapshot.eef_pos, dtype=np.float64)
        object_position = np.asarray(snapshot.object_body_pos, dtype=np.float64)
        relative = eef - object_position
        centered_error = abs(
            float(np.dot(relative, self.intent.grasp_funnel.centered_pad_axis_world))
        )
        if self._previous_eef_position is None:
            relative_speed = np.inf
        else:
            eef_velocity = (eef - self._previous_eef_position) / 0.02
            relative_speed = float(np.linalg.norm(eef_velocity - estimated_object_velocity))
        return bool(
            np.linalg.norm(relative) <= 0.025
            and centered_error <= 0.010
            and relative_speed <= 0.25
        )

    def _update_contact_gate(
        self,
        *,
        source_tick: int,
        left_pad_contact: bool,
        right_pad_contact: bool,
    ) -> None:
        if left_pad_contact and right_pad_contact:
            self._first_unilateral_contact_tick = None
            return
        if left_pad_contact or right_pad_contact:
            if self._first_unilateral_contact_tick is None:
                self._first_unilateral_contact_tick = source_tick
            elapsed = source_tick - self._first_unilateral_contact_tick
            if elapsed >= self.intent.grasp_funnel.bilateral_contact_acquisition_ticks:
                raise SemanticExecutionFailure(
                    "unilateral pad contact did not become bilateral inside the acquisition window"
                )

    def next_action(
        self,
        *,
        snapshot: BoundarySnapshot,
        handoff_state: HandoffState,
        left_pad_contact: bool,
        right_pad_contact: bool,
    ) -> StructuredExpertDecision:
        if not isinstance(snapshot, BoundarySnapshot):
            raise TypeError("snapshot must be a BoundarySnapshot")
        if not isinstance(handoff_state, HandoffState):
            raise TypeError("handoff_state must be a HandoffState")
        if handoff_state is HandoffState.FAILURE:
            raise RuntimeError("structured expert cannot act after handoff failure")
        if type(left_pad_contact) is not bool or type(right_pad_contact) is not bool:
            raise TypeError("pad-contact flags must be boolean")
        estimated_velocity = self._estimated_object_velocity(snapshot)
        source_tick = snapshot.formal_tick_index
        last_reference_index = len(self.reference.timestamps_seconds) - 1
        funnel = self.intent.grasp_funnel
        self._update_contact_gate(
            source_tick=source_tick,
            left_pad_contact=left_pad_contact,
            right_pad_contact=right_pad_contact,
        )

        entry_target_tick = self.intent.approach.funnel_entry_target_tick
        entry_deadline_tick = self.intent.approach.funnel_entry_deadline_tick
        if source_tick < entry_target_tick:
            phase = StructuredExpertPhase.SMOOTH_APPROACH
            reference_index = source_tick - self.decision_source_tick + 1
            if not 1 <= reference_index <= last_reference_index:
                raise RuntimeError("selected reference index is outside the frozen schedule")
            target = self.reference.eef_positions_world[reference_index]
            gripper_command = self.action_contract.gripper_open_command
        elif self._funnel_start_tick is None and not self._entry_is_achieved(
            snapshot,
            estimated_velocity,
        ):
            if source_tick > entry_deadline_tick:
                raise SemanticExecutionFailure("smooth approach missed the funnel entry deadline")
            phase = StructuredExpertPhase.SMOOTH_APPROACH
            reference_index = last_reference_index
            target = self._dynamic_entry_target(snapshot, estimated_velocity)
            gripper_command = self.action_contract.gripper_open_command
        elif self._close_started_tick is None:
            if self._funnel_start_tick is None:
                self._funnel_start_tick = source_tick
                self._effective_close_target_tick = (
                    source_tick + self.intent.strategy.funnel_descent_ticks
                )
            assert self._effective_close_target_tick is not None
            effective_close_deadline = (
                funnel.handoff_deadline_tick
                - funnel.close_dwell_ticks
                - 2
            )
            if source_tick > effective_close_deadline:
                raise SemanticExecutionFailure("canonical grasp missed the close deadline")
            reference_index = last_reference_index
            ready = self._close_geometry_is_ready(snapshot, estimated_velocity)
            if source_tick >= self._effective_close_target_tick and ready:
                self._close_started_tick = source_tick
                phase = StructuredExpertPhase.CLOSE_STABILIZE
                target = np.asarray(snapshot.object_body_pos, dtype=np.float64) + (
                    estimated_velocity * self.intent.strategy.prediction_lead_seconds
                )
                gripper_command = self.action_contract.gripper_close_command
            else:
                phase = StructuredExpertPhase.GRASP_FUNNEL
                if source_tick < self._effective_close_target_tick:
                    target = self._funnel_target(snapshot, estimated_velocity)
                else:
                    target = np.asarray(snapshot.object_body_pos, dtype=np.float64) + (
                        estimated_velocity * self.intent.strategy.prediction_lead_seconds
                    )
                gripper_command = self.action_contract.gripper_open_command
        else:
            reference_index = last_reference_index
            close_dwell_complete = (
                source_tick - self._close_started_tick >= funnel.close_dwell_ticks
            )
            if handoff_state is HandoffState.PHYSICAL and close_dwell_complete:
                phase = StructuredExpertPhase.LIFT
                if self._grasp_offset_world is None or self._lift_start_position_world is None:
                    self._grasp_offset_world = np.array(
                        np.asarray(snapshot.eef_pos, dtype=np.float64)
                        - np.asarray(snapshot.object_body_pos, dtype=np.float64),
                        dtype=np.float64,
                        copy=True,
                    )
                    self._lift_start_position_world = np.asarray(
                        snapshot.eef_pos,
                        dtype=np.float64,
                    )
                target = (
                    self._lift_start_position_world
                    + funnel.lift_relative_displacement_world
                )
            else:
                phase = StructuredExpertPhase.CLOSE_STABILIZE
                target = np.asarray(snapshot.object_body_pos, dtype=np.float64) + (
                    estimated_velocity * self.intent.strategy.prediction_lead_seconds
                )
            gripper_command = self.action_contract.gripper_close_command

        result = osc_action_from_reference(
            achieved_position_world=np.asarray(snapshot.eef_pos, dtype=np.float64),
            achieved_orientation_world=np.asarray(snapshot.eef_xmat, dtype=np.float64),
            target_position_world=np.asarray(target, dtype=np.float64),
            target_orientation_world=self.intent.approach.fixed_orientation_world,
            world_to_base_rotation=self.world_to_base_rotation,
            action_contract=self.action_contract,
            gripper_command=gripper_command,
            translation_clip_m=self.intent.tracking_error_clip_m,
        )
        self._previous_eef_position = np.asarray(
            snapshot.eef_pos,
            dtype=np.float64,
        ).copy()
        return StructuredExpertDecision(
            source_physics_step=snapshot.physics_step_index,
            source_formal_tick=source_tick,
            source_time_us=snapshot.time_us,
            phase=phase,
            reference_kind="selected_reference",
            selected_reference_index=reference_index,
            target_eef_position_world=np.asarray(target, dtype=np.float64),
            target_eef_orientation_world=self.intent.approach.fixed_orientation_world,
            estimated_object_velocity_world=np.asarray(estimated_velocity, dtype=np.float64),
            action=result.action,
            position_error_base=result.position_error_base,
            orientation_error_rotvec_base=result.orientation_error_rotvec_base,
            saturated=result.saturated,
        )
