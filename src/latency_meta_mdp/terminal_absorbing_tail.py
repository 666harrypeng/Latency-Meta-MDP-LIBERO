"""Lazy target-only absorbing continuation for successful episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from latency_meta_mdp.belief_data import BeliefEpisodeView
from latency_meta_mdp.control import ActionContract
from latency_meta_mdp.temporal_contract import TemporalContract


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TerminalAbsorbingTailView:
    protocol_id: str
    episode: BeliefEpisodeView
    tail_tick_count: int
    hold_action: np.ndarray
    boundary_tick: np.ndarray
    boundary_time_us: np.ndarray
    boundary_source_index: np.ndarray
    transition_source_valid: np.ndarray
    boundary_target_valid: np.ndarray
    boundary_is_absorbing: np.ndarray
    transition_is_absorbing: np.ndarray
    expert_actions: np.ndarray
    expert_phase: np.ndarray

    def __post_init__(self) -> None:
        if self.protocol_id != "terminal_absorbing_tail_v1":
            raise ValueError("unsupported terminal absorbing-tail protocol")
        if (
            isinstance(self.tail_tick_count, bool)
            or not isinstance(self.tail_tick_count, int)
            or self.tail_tick_count <= 0
        ):
            raise ValueError("tail_tick_count must be a positive integer")
        boundary_count = self.episode.boundary_count + self.tail_tick_count
        transition_count = self.episode.transition_count + self.tail_tick_count
        shapes = {
            "hold_action": (7,),
            "boundary_tick": (boundary_count,),
            "boundary_time_us": (boundary_count,),
            "boundary_source_index": (boundary_count,),
            "transition_source_valid": (transition_count,),
            "boundary_target_valid": (boundary_count,),
            "boundary_is_absorbing": (boundary_count,),
            "transition_is_absorbing": (transition_count,),
            "expert_actions": (transition_count, 7),
            "expert_phase": (transition_count,),
        }
        for name, expected in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} does not match the absorbing-tail contract")
            object.__setattr__(self, name, _readonly(value))
        if not np.all(np.isfinite(self.hold_action)) or not np.all(
            np.isfinite(self.expert_actions)
        ):
            raise ValueError("absorbing-tail actions must be finite")

    @property
    def real_boundary_count(self) -> int:
        return self.episode.boundary_count

    @property
    def real_transition_count(self) -> int:
        return self.episode.transition_count

    @property
    def extended_boundary_count(self) -> int:
        return len(self.boundary_tick)

    @property
    def extended_transition_count(self) -> int:
        return len(self.expert_actions)

    def extend_boundary_array(
        self,
        values: np.ndarray,
        *,
        absorbing_value: Any | None = None,
    ) -> np.ndarray:
        source = np.asarray(values)
        if source.shape[0] != self.real_boundary_count:
            raise ValueError("boundary array does not match the real episode")
        result = np.array(source[self.boundary_source_index], copy=True)
        if absorbing_value is not None:
            replacement = np.asarray(absorbing_value, dtype=source.dtype)
            if replacement.shape != source.shape[1:]:
                raise ValueError("absorbing value does not match the boundary value shape")
            result[self.real_transition_count :] = replacement
        return _readonly(result)


def build_terminal_absorbing_tail(
    *,
    episode: BeliefEpisodeView,
    temporal_contract: TemporalContract,
    action_contract: ActionContract,
) -> TerminalAbsorbingTailView:
    if str(episode.supervision.boundary_outcome_status[-1]) != "success":
        raise ValueError("absorbing tail requires a successful episode")
    if action_contract.formal_tick_us != temporal_contract.formal_tick_us:
        raise ValueError("action and temporal contracts must share the formal clock")
    tail_count = temporal_contract.target_tail_ticks
    real_transitions = episode.transition_count
    real_boundaries = episode.boundary_count
    extended_transitions = real_transitions + tail_count
    extended_boundaries = real_boundaries + tail_count
    hold_action = action_contract.compose_action(
        arm_reference=np.zeros(action_contract.arm_dim),
        gripper_command=action_contract.gripper_close_command,
    )
    boundary_source_index = np.concatenate(
        (
            np.arange(real_boundaries, dtype=np.int64),
            np.full(tail_count, real_transitions, dtype=np.int64),
        )
    )
    source_valid = np.zeros(extended_transitions, dtype=np.bool_)
    source_valid[:real_transitions] = True
    boundary_is_absorbing = np.zeros(extended_boundaries, dtype=np.bool_)
    boundary_is_absorbing[real_transitions:] = True
    transition_is_absorbing = np.zeros(extended_transitions, dtype=np.bool_)
    transition_is_absorbing[real_transitions:] = True
    return TerminalAbsorbingTailView(
        protocol_id="terminal_absorbing_tail_v1",
        episode=episode,
        tail_tick_count=tail_count,
        hold_action=hold_action,
        boundary_tick=np.arange(extended_boundaries, dtype=np.int64),
        boundary_time_us=(
            np.arange(extended_boundaries, dtype=np.int64)
            * temporal_contract.formal_tick_us
        ),
        boundary_source_index=boundary_source_index,
        transition_source_valid=source_valid,
        boundary_target_valid=np.ones(extended_boundaries, dtype=np.bool_),
        boundary_is_absorbing=boundary_is_absorbing,
        transition_is_absorbing=transition_is_absorbing,
        expert_actions=np.concatenate(
            (
                episode.expert_actions,
                np.repeat(hold_action[None, :], tail_count, axis=0),
            ),
            axis=0,
        ),
        expert_phase=np.concatenate(
            (
                episode.expert_phase,
                np.full(tail_count, "terminal_absorbing", dtype="<U18"),
            )
        ),
    )
