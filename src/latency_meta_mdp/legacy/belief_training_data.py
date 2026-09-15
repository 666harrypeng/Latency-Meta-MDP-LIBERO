"""Lazy model-ready contexts for predictive return-belief training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from latency_meta_mdp.legacy.return_belief_geometry import (
    RETURN_STATE_DIM,
    build_absorbing_return_state_stream,
)
from latency_meta_mdp.legacy.terminal_absorbing_tail import TerminalAbsorbingTailView
from latency_meta_mdp.runtime.latency_law import TruncatedBetaLatencyLaw
from latency_meta_mdp.runtime.temporal_contract import TemporalContract

ROBOT_PROPRIO_DIM = 16


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


class InteractionMode(IntEnum):
    FREE = 0
    CONTACT = 1
    GRASPED = 2
    TERMINAL = 3


@dataclass(frozen=True)
class BeliefTrainingIndex:
    episode_id: str
    level: int
    history_start_tick: int
    source_tick: int
    source_phase: str

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or self.level not in (1, 2, 3)
            or self.history_start_tick < 0
            or self.source_tick < self.history_start_tick
        ):
            raise ValueError("belief training index is invalid")


@dataclass(frozen=True)
class BeliefTrainingContext:
    index: BeliefTrainingIndex
    instruction: str
    history_ticks: np.ndarray
    agentview_history: np.ndarray
    wrist_history: np.ndarray
    robot_proprio_history: np.ndarray
    remaining_actions: np.ndarray
    latency_probabilities: np.ndarray
    target_delay_ticks: np.ndarray
    target_states: np.ndarray
    target_interaction_mode: np.ndarray
    target_absorbing: np.ndarray

    def __post_init__(self) -> None:
        history_count = len(self.history_ticks)
        delay_count = len(self.target_delay_ticks)
        if not self.instruction.strip():
            raise ValueError("belief training context requires an instruction")
        if (
            self.agentview_history.ndim != 4
            or self.wrist_history.ndim != 4
            or self.agentview_history.shape[0] != history_count
            or self.wrist_history.shape[0] != history_count
            or self.agentview_history.shape[-1] != 3
            or self.wrist_history.shape[-1] != 3
        ):
            raise ValueError("belief camera histories are invalid")
        shapes = {
            "history_ticks": (history_count,),
            "robot_proprio_history": (history_count, ROBOT_PROPRIO_DIM),
            "remaining_actions": (25, 7),
            "latency_probabilities": (delay_count,),
            "target_delay_ticks": (delay_count,),
            "target_states": (delay_count, RETURN_STATE_DIM),
            "target_interaction_mode": (delay_count,),
            "target_absorbing": (delay_count,),
        }
        for name, expected in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} does not match the belief context contract")
        if (
            not np.all(np.isfinite(self.robot_proprio_history))
            or not np.all(np.isfinite(self.remaining_actions))
            or not np.all(np.isfinite(self.latency_probabilities))
            or not np.all(np.isfinite(self.target_states))
            or not np.isclose(self.latency_probabilities.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("belief training context contains invalid numeric values")
        for name in (
            "history_ticks",
            "agentview_history",
            "wrist_history",
            "robot_proprio_history",
            "remaining_actions",
            "latency_probabilities",
            "target_delay_ticks",
            "target_states",
            "target_interaction_mode",
            "target_absorbing",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class DelayQueryBatch:
    delay_ticks: np.ndarray
    target_states: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.delay_ticks)
        shapes = {
            "delay_ticks": (count,),
            "target_states": (count, RETURN_STATE_DIM),
            "interaction_mode": (count,),
            "absorbing": (count,),
        }
        for name, expected in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} does not match the delay-query shape")
            object.__setattr__(self, name, _readonly(value))


def _robot_proprio_stream(tail_view: TerminalAbsorbingTailView) -> np.ndarray:
    deployment = tail_view.episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    values = np.concatenate(
        (
            deployment.robot_qpos,
            deployment.robot_qvel,
            width[:, None],
            width_velocity[:, None],
        ),
        axis=1,
    )
    if values.shape != (tail_view.real_boundary_count, ROBOT_PROPRIO_DIM):
        raise ValueError("episode cannot provide the 16D robot proprioception stream")
    return _readonly(values, dtype=np.float64)


def build_belief_training_indices(
    *,
    tail_view: TerminalAbsorbingTailView,
    temporal_contract: TemporalContract,
) -> tuple[BeliefTrainingIndex, ...]:
    minimum = max(
        temporal_contract.launch_trigger_horizon,
        temporal_contract.history_sample_count - 1,
    )
    maximum = tail_view.real_transition_count - 1
    if maximum < minimum:
        raise ValueError("episode has no valid belief training source")
    episode = tail_view.episode
    return tuple(
        BeliefTrainingIndex(
            episode_id=episode.episode_id,
            level=episode.level,
            history_start_tick=(source_tick - temporal_contract.history_sample_count + 1),
            source_tick=source_tick,
            source_phase=str(episode.expert_phase[source_tick]),
        )
        for source_tick in range(minimum, maximum + 1)
        if tail_view.transition_source_valid[source_tick]
    )


def _interaction_modes(
    *,
    absorbing: np.ndarray,
    physical_handoff: np.ndarray,
    contact: np.ndarray,
) -> np.ndarray:
    modes = np.full(len(absorbing), InteractionMode.FREE, dtype=np.int8)
    modes[contact] = InteractionMode.CONTACT
    modes[physical_handoff] = InteractionMode.GRASPED
    modes[absorbing] = InteractionMode.TERMINAL
    return modes


def materialize_belief_training_context(
    *,
    index: BeliefTrainingIndex,
    tail_view: TerminalAbsorbingTailView,
    temporal_contract: TemporalContract,
    latency_law: TruncatedBetaLatencyLaw,
) -> BeliefTrainingContext:
    episode = tail_view.episode
    if index.episode_id != episode.episode_id or index.level != episode.level:
        raise ValueError("belief index and episode identity disagree")
    if index.source_tick >= tail_view.real_transition_count:
        raise ValueError("belief source must be a real preterminal transition")
    if latency_law.delay_ticks != tuple(range(1, temporal_contract.maximum_delay_ticks + 1)):
        raise ValueError("latency law and temporal contract delay support disagree")
    history = slice(index.history_start_tick, index.source_tick + 1)
    history_ticks = np.arange(index.history_start_tick, index.source_tick + 1, dtype=np.int64)
    proprio = _robot_proprio_stream(tail_view)[history]
    remaining_stop = index.source_tick + temporal_contract.remaining_buffer_coverage
    remaining_actions = tail_view.expert_actions[index.source_tick : remaining_stop]
    delays = np.asarray(latency_law.delay_ticks, dtype=np.int64)
    targets = index.source_tick + delays
    state = build_absorbing_return_state_stream(tail_view)
    left_contact = tail_view.extend_boundary_array(episode.supervision.left_pad_contact)
    right_contact = tail_view.extend_boundary_array(episode.supervision.right_pad_contact)
    handoff = tail_view.extend_boundary_array(episode.supervision.handoff_state)
    absorbing = tail_view.boundary_is_absorbing[targets]
    contact = left_contact[targets] | right_contact[targets]
    return BeliefTrainingContext(
        index=index,
        instruction=episode.instruction,
        history_ticks=history_ticks,
        agentview_history=episode.deployment.agentview_rgb[history],
        wrist_history=episode.deployment.wrist_rgb[history],
        robot_proprio_history=proprio,
        remaining_actions=remaining_actions,
        latency_probabilities=latency_law.probabilities,
        target_delay_ticks=delays,
        target_states=state[targets],
        target_interaction_mode=_interaction_modes(
            absorbing=absorbing,
            physical_handoff=handoff[targets] == "physical",
            contact=contact,
        ),
        target_absorbing=absorbing,
    )


def sample_delay_queries(
    *,
    context: BeliefTrainingContext,
    rng: np.random.Generator,
    query_count: int,
) -> DelayQueryBatch:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("query_count must be a positive integer")
    rows = rng.choice(
        len(context.target_delay_ticks),
        size=query_count,
        replace=True,
        p=context.latency_probabilities,
    )
    return DelayQueryBatch(
        delay_ticks=context.target_delay_ticks[rows],
        target_states=context.target_states[rows],
        interaction_mode=context.target_interaction_mode[rows],
        absorbing=context.target_absorbing[rows],
    )
