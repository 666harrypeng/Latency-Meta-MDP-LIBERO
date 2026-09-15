"""Causal feedback expert for moving-ball interception, grasp, and lift."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.envs.control import ActionContract
from latency_meta_mdp.envs.outcomes import OutcomeStatus
from latency_meta_mdp.envs.snapshots import BoundarySnapshot
from latency_meta_mdp.runtime.handoff import HandoffState


class ExpertPhase(str, Enum):
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    CLOSE = "close"
    LIFT = "lift"


@dataclass(frozen=True)
class ExpertObservation:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    ball_position_world: np.ndarray
    eef_position_world: np.ndarray
    world_to_base_rotation: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.physics_step_index < 0
            or self.formal_tick_index < 0
            or self.time_us < 0
            or self.physics_step_index != self.formal_tick_index * 10
            or self.time_us != self.formal_tick_index * 20_000
        ):
            raise ValueError("expert observation indices do not match the formal clock")
        for name in ("ball_position_world", "eef_position_world"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3D vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        rotation = np.array(self.world_to_base_rotation, dtype=float, copy=True)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9, rtol=0)
            or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-9
        ):
            raise ValueError("world_to_base_rotation must be a proper rotation matrix")
        rotation.setflags(write=False)
        object.__setattr__(self, "world_to_base_rotation", rotation)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BoundarySnapshot,
        *,
        world_to_base_rotation: np.ndarray,
    ) -> ExpertObservation:
        return cls(
            physics_step_index=snapshot.physics_step_index,
            formal_tick_index=snapshot.formal_tick_index,
            time_us=snapshot.time_us,
            ball_position_world=snapshot.object_body_pos,
            eef_position_world=snapshot.eef_pos,
            world_to_base_rotation=world_to_base_rotation,
        )


@dataclass(frozen=True)
class ExpertConfig:
    schema_version: int
    expert_id: str
    action_contract_id: str
    history_ticks: int
    lead_time_seconds: float
    pregrasp_height_m: float
    pregrasp_tolerance_m: float
    grasp_tolerance_m: float
    lift_offset_m: float
    max_translation_goal_offset_m: float
    collection_max_duration_us: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.expert_id != "panda_ball_feedback_v1":
            raise ValueError("unsupported expert schema or identifier")
        if self.action_contract_id != "panda_osc_pose_delta_v1":
            raise ValueError("expert action contract is invalid")
        if (
            isinstance(self.history_ticks, bool)
            or not isinstance(self.history_ticks, int)
            or self.history_ticks < 2
        ):
            raise ValueError("history_ticks must be an integer of at least two")
        for name in (
            "lead_time_seconds",
            "pregrasp_height_m",
            "pregrasp_tolerance_m",
            "grasp_tolerance_m",
            "lift_offset_m",
            "max_translation_goal_offset_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.collection_max_duration_us, bool)
            or not isinstance(self.collection_max_duration_us, int)
            or self.collection_max_duration_us <= 0
            or self.collection_max_duration_us % 20_000
        ):
            raise ValueError("collection_max_duration_us must be a positive formal-grid time")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ExpertConfig:
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("expert config fields are invalid")
        return cls(**raw)


def load_expert_config(path: Path) -> ExpertConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("expert config must be a YAML mapping")
    return ExpertConfig.from_mapping(raw)


def is_qualified_expert_episode(
    *,
    status: OutcomeStatus,
    terminal_time_us: int | None,
    max_duration_us: int,
) -> bool:
    """Return whether a successful expert rollout is admissible for collection."""

    if not isinstance(status, OutcomeStatus):
        raise TypeError("status must be an OutcomeStatus")
    if (
        isinstance(max_duration_us, bool)
        or not isinstance(max_duration_us, int)
        or max_duration_us <= 0
        or max_duration_us % 20_000
    ):
        raise ValueError("max_duration_us must be a positive formal-grid time")
    if terminal_time_us is not None and (
        isinstance(terminal_time_us, bool)
        or not isinstance(terminal_time_us, int)
        or terminal_time_us < 0
        or terminal_time_us % 20_000
    ):
        raise ValueError("terminal_time_us must be a non-negative formal-grid time")
    return bool(
        status is OutcomeStatus.SUCCESS
        and terminal_time_us is not None
        and terminal_time_us < max_duration_us
    )


@dataclass(frozen=True)
class ExpertPhaseEvent:
    phase: ExpertPhase
    time_us: int


@dataclass(frozen=True)
class ExpertDecision:
    source_physics_step: int
    source_formal_tick: int
    source_time_us: int
    history_start_time_us: int
    history_sample_count: int
    action: np.ndarray
    phase: ExpertPhase
    target_eef_position: np.ndarray
    estimated_object_velocity: np.ndarray

    def __post_init__(self) -> None:
        for name in ("action", "target_eef_position", "estimated_object_velocity"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class ScriptedBallExpert:
    """Generate teacher actions from current and rolling causal object geometry."""

    def __init__(self, *, action_contract: ActionContract, config: ExpertConfig) -> None:
        if action_contract.contract_id != config.action_contract_id:
            raise ValueError("expert and action contract identifiers do not match")
        translation_axis_limit = float(
            np.min(
                np.minimum(
                    np.abs(action_contract.arm_output_low[:3]),
                    np.abs(action_contract.arm_output_high[:3]),
                )
            )
        )
        if config.max_translation_goal_offset_m > translation_axis_limit:
            raise ValueError("expert translation goal offset must fit inside every controller axis")
        self.action_contract = action_contract
        self.config = config
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=config.history_ticks)
        self.phase = ExpertPhase.PREGRASP
        self.phase_events: list[ExpertPhaseEvent] = [
            ExpertPhaseEvent(phase=ExpertPhase.PREGRASP, time_us=0)
        ]
        self._last_time_us: int | None = None
        self._lift_target: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self.phase = ExpertPhase.PREGRASP
        self.phase_events = [ExpertPhaseEvent(phase=ExpertPhase.PREGRASP, time_us=0)]
        self._last_time_us = None
        self._lift_target = None

    def _transition(self, phase: ExpertPhase, time_us: int) -> None:
        if phase is self.phase:
            return
        self.phase = phase
        self.phase_events.append(ExpertPhaseEvent(phase=phase, time_us=time_us))

    def _update_history(self, observation: ExpertObservation) -> np.ndarray:
        if self._last_time_us is None:
            if observation.time_us != 0:
                raise ValueError("expert episode must start at time zero")
        elif observation.time_us != self._last_time_us + 20_000:
            raise ValueError("expert requires every 20 ms boundary")
        self._last_time_us = observation.time_us
        self._history.append(
            (observation.time_us, np.array(observation.ball_position_world, copy=True))
        )
        if len(self._history) < 2:
            return np.zeros(3)
        start_time, start_position = self._history[0]
        end_time, end_position = self._history[-1]
        return (end_position - start_position) / ((end_time - start_time) / 1_000_000)

    def next_action(
        self,
        *,
        observation: ExpertObservation,
        handoff_state: HandoffState,
    ) -> ExpertDecision:
        if handoff_state is HandoffState.FAILURE:
            raise RuntimeError("expert cannot act after handoff failure")
        estimated_velocity = self._update_history(observation)
        ball_position = np.asarray(observation.ball_position_world)
        eef_position = np.asarray(observation.eef_position_world)
        predicted_ball = ball_position + estimated_velocity * self.config.lead_time_seconds

        if handoff_state is HandoffState.PHYSICAL and self.phase is not ExpertPhase.LIFT:
            self._lift_target = eef_position + np.array([0.0, 0.0, self.config.lift_offset_m])
            self._transition(ExpertPhase.LIFT, observation.time_us)

        if handoff_state is HandoffState.DRIVEN:
            current_pregrasp = ball_position + np.array([0.0, 0.0, self.config.pregrasp_height_m])
            if (
                self.phase is ExpertPhase.PREGRASP
                and np.linalg.norm(eef_position - current_pregrasp)
                < self.config.pregrasp_tolerance_m
            ):
                self._transition(ExpertPhase.APPROACH, observation.time_us)
            if (
                self.phase is ExpertPhase.APPROACH
                and np.linalg.norm(eef_position - ball_position) < self.config.grasp_tolerance_m
            ):
                self._transition(ExpertPhase.CLOSE, observation.time_us)

        if self.phase is ExpertPhase.PREGRASP:
            target = predicted_ball + np.array([0.0, 0.0, self.config.pregrasp_height_m])
            gripper = self.action_contract.gripper_open_command
        elif self.phase is ExpertPhase.APPROACH:
            target = predicted_ball
            gripper = self.action_contract.gripper_open_command
        elif self.phase is ExpertPhase.CLOSE:
            target = predicted_ball
            gripper = self.action_contract.gripper_close_command
        else:
            if self._lift_target is None:
                raise RuntimeError("lift phase requires a captured handoff target")
            target = self._lift_target
            gripper = self.action_contract.gripper_close_command

        position_error_base = observation.world_to_base_rotation @ (target - eef_position)
        max_translation_goal_offset_m = self.config.max_translation_goal_offset_m
        error_norm = float(np.linalg.norm(position_error_base))
        if error_norm > max_translation_goal_offset_m:
            position_error_base = position_error_base * (max_translation_goal_offset_m / error_norm)
        normalized_translation = np.clip(
            position_error_base / self.action_contract.arm_output_high[:3],
            self.action_contract.arm_input_low[:3],
            self.action_contract.arm_input_high[:3],
        )
        arm_action = np.concatenate([normalized_translation, np.zeros(3)])
        action = self.action_contract.compose_action(
            arm_reference=arm_action,
            gripper_command=gripper,
        )
        return ExpertDecision(
            source_physics_step=observation.physics_step_index,
            source_formal_tick=observation.formal_tick_index,
            source_time_us=observation.time_us,
            history_start_time_us=self._history[0][0],
            history_sample_count=len(self._history),
            action=action,
            phase=self.phase,
            target_eef_position=target,
            estimated_object_velocity=estimated_velocity,
        )
