"""Warm-start sharp-replacement action-chunk client."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import yaml

from latency_meta_mdp.envs.control import ActionContract
from latency_meta_mdp.runtime.latency_harness import Arrival, LaunchContext, LogicalLatencyHarness
from latency_meta_mdp.runtime.temporal_contract import TemporalContract, load_temporal_contract

TObservation = TypeVar("TObservation")


@dataclass(frozen=True)
class ChunkDecisionState:
    """Deployment-visible state at a no-pending decision boundary."""

    formal_tick: int
    active_cursor: int
    remaining_actions: int
    actions_consumed: int
    executable_controls: np.ndarray
    latency_probabilities: np.ndarray | None = None
    unread_action_buffer: np.ndarray | None = None
    unread_action_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        controls = np.array(self.executable_controls, copy=True)
        controls.setflags(write=False)
        object.__setattr__(self, "executable_controls", controls)
        for name in ("unread_action_buffer", "unread_action_mask"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        if self.latency_probabilities is not None:
            probability = np.array(self.latency_probabilities, dtype=np.float32, copy=True)
            if (
                probability.shape != (20,)
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or not np.isclose(probability.sum(), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError("decision-state latency law must be a normalized D20 PMF")
            probability.setflags(write=False)
            object.__setattr__(self, "latency_probabilities", probability)


@dataclass(frozen=True)
class ActionChunkClientConfig:
    schema_version: int
    protocol_id: str
    temporal_contract: TemporalContract
    bootstrap_mode: str
    handoff_strategy: str
    arrival_write_mode: str
    chunk_alignment: str
    max_pending_requests: int
    starvation_action: str

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.protocol_id != "sharp_return_time_chunk_v2":
            raise ValueError("unsupported action-chunk client schema or protocol")
        if not isinstance(self.temporal_contract, TemporalContract):
            raise TypeError("temporal_contract must be a TemporalContract")
        if self.bootstrap_mode != "warm_start":
            raise ValueError("only warm_start bootstrap is supported")
        if self.handoff_strategy != "sharp_replace":
            raise ValueError("only sharp_replace handoff is supported")
        if self.arrival_write_mode != "from_arrival":
            raise ValueError("only from_arrival chunk writes are supported")
        if self.chunk_alignment != "return_time":
            raise ValueError("only return_time chunk alignment is supported")
        if self.max_pending_requests != 1:
            raise ValueError("the client supports exactly one pending request")
        if self.starvation_action != "hold":
            raise ValueError("only explicit hold starvation is supported")

    @property
    def prediction_horizon(self) -> int:
        return self.temporal_contract.prediction_horizon

    @property
    def launch_trigger_horizon(self) -> int:
        return self.temporal_contract.launch_trigger_horizon

    @property
    def guaranteed_delay_ticks(self) -> int:
        return self.temporal_contract.remaining_buffer_coverage


def load_action_chunk_client_config(path: Path) -> ActionChunkClientConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(ActionChunkClientConfig.__dataclass_fields__):
        raise ValueError("action-chunk client config fields are invalid")
    temporal_path = raw["temporal_contract"]
    if not isinstance(temporal_path, str) or not temporal_path:
        raise ValueError("temporal_contract must be a non-empty relative path")
    resolved_temporal_path = (path.parent / temporal_path).resolve()
    raw["temporal_contract"] = load_temporal_contract(resolved_temporal_path)
    return ActionChunkClientConfig(**raw)


class ChunkClientEventKind(str, Enum):
    BOOTSTRAP_LAUNCH = "bootstrap_launch"
    BOOTSTRAP_RETURN = "bootstrap_return"
    BOOTSTRAP_INSTALL = "bootstrap_install"
    CHUNK_INSTALL = "chunk_install"
    CHUNK_ACTION_EXECUTE = "chunk_action_execute"


@dataclass(frozen=True)
class ChunkClientEvent:
    kind: ChunkClientEventKind
    formal_tick: int | None
    time_us: int | None
    chunk_id: int
    source_request_id: int | None
    old_chunk_id: int | None = None
    discarded_action_count: int | None = None
    installed_chunk_index: int | None = None
    executed_chunk_index: int | None = None
    action: np.ndarray | None = None
    wall_start_ns: int | None = None
    wall_end_ns: int | None = None
    wall_duration_ns: int | None = None
    simulation_time_before_us: int | None = None
    simulation_time_after_us: int | None = None

    def __post_init__(self) -> None:
        for name in ("chunk_id", "source_request_id", "old_chunk_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative when present")
        for name in (
            "discarded_action_count",
            "installed_chunk_index",
            "executed_chunk_index",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative when present")
        if self.formal_tick is None:
            if self.time_us is not None:
                raise ValueError("pre-episode chunk events cannot carry simulated time")
        elif (
            isinstance(self.formal_tick, bool)
            or not isinstance(self.formal_tick, int)
            or self.formal_tick < 0
            or self.time_us != self.formal_tick * 20_000
        ):
            raise ValueError("chunk event does not match the formal grid")
        if self.action is not None:
            action = np.array(self.action, dtype=float, copy=True)
            if action.ndim != 1 or not np.all(np.isfinite(action)):
                raise ValueError("chunk event action must be a finite vector")
            action.setflags(write=False)
            object.__setattr__(self, "action", action)
        wall_values = (self.wall_start_ns, self.wall_end_ns, self.wall_duration_ns)
        simulation_values = (
            self.simulation_time_before_us,
            self.simulation_time_after_us,
        )
        if self.kind is ChunkClientEventKind.BOOTSTRAP_RETURN:
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*wall_values, *simulation_values)
            ):
                raise ValueError("bootstrap return requires non-negative timing fields")
            if self.wall_end_ns - self.wall_start_ns != self.wall_duration_ns:
                raise ValueError("bootstrap wall duration does not match start/end")
        elif any(value is not None for value in (*wall_values, *simulation_values)):
            raise ValueError("bootstrap timing fields are only valid on bootstrap_return")


@dataclass(frozen=True)
class BootstrapRecord:
    protocol_id: str
    installed_chunk_id: int
    simulation_time_before_us: int
    simulation_time_after_us: int
    wall_start_ns: int
    wall_end_ns: int
    wall_duration_ns: int

    def __post_init__(self) -> None:
        if self.protocol_id not in {"sharp_return_time_chunk_v2", "rtc_observation_time_h50_v1"}:
            raise ValueError("bootstrap record protocol is invalid")
        for name in (
            "installed_chunk_id",
            "simulation_time_before_us",
            "simulation_time_after_us",
            "wall_start_ns",
            "wall_end_ns",
            "wall_duration_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.wall_end_ns - self.wall_start_ns != self.wall_duration_ns:
            raise ValueError("bootstrap record wall duration does not match start/end")


class SharpActionChunkClient(Generic[TObservation]):
    """Execute immutable chunks and replace unread actions at exact arrival boundaries."""

    def __init__(
        self,
        *,
        action_contract: ActionContract,
        config: ActionChunkClientConfig,
        harness: LogicalLatencyHarness[TObservation, np.ndarray],
        simulation_time_reader: Callable[[], int],
        monotonic_ns: Callable[[], int],
        on_chunk_install: Callable[[ChunkClientEvent], None] | None = None,
    ) -> None:
        if action_contract.formal_tick_us != harness.formal_tick_us:
            raise ValueError("chunk client and harness must share one formal clock")
        if config.temporal_contract.formal_tick_us != harness.formal_tick_us:
            raise ValueError("client config and harness must share one formal clock")
        if not callable(simulation_time_reader) or not callable(monotonic_ns):
            raise TypeError("client clocks must be callable")
        self.action_contract = action_contract
        self.config = config
        self.harness = harness
        self._simulation_time_reader = simulation_time_reader
        self._monotonic_ns = monotonic_ns
        self._on_chunk_install = on_chunk_install
        self._active_actions: np.ndarray | None = None
        self._active_chunk_id: int | None = None
        self._active_source_request_id: int | None = None
        self._active_cursor = 0
        self._actions_consumed = 0
        self._next_chunk_id = 0
        self._chunk_events: list[ChunkClientEvent] = []
        self._bootstrapped = False
        self._faulted = False
        self.last_gripper_command = action_contract.gripper_open_command

    @property
    def active_actions(self) -> np.ndarray | None:
        if self._active_actions is None:
            return None
        result = np.array(self._active_actions, copy=True)
        result.setflags(write=False)
        return result

    @property
    def active_chunk_id(self) -> int | None:
        return self._active_chunk_id

    @property
    def active_cursor(self) -> int:
        return self._active_cursor

    @property
    def actions_consumed_since_activation(self) -> int:
        return self._actions_consumed

    @property
    def chunk_events(self) -> tuple[ChunkClientEvent, ...]:
        return tuple(self._chunk_events)

    @property
    def faulted(self) -> bool:
        return self._faulted or self.harness.faulted

    def _require_healthy(self) -> None:
        if self.faulted:
            raise RuntimeError("sharp action-chunk client is faulted")

    def _validated_chunk(self, value: np.ndarray) -> np.ndarray:
        chunk = np.asarray(value, dtype=float)
        expected = (self.config.prediction_horizon, self.action_contract.action_dim)
        if chunk.shape != expected or not np.all(np.isfinite(chunk)):
            raise ValueError(f"action chunk must be a finite array with shape {expected}")
        validated = []
        for action in chunk:
            arm, gripper = self.action_contract.split_action(action)
            validated.append(
                self.action_contract.compose_action(
                    arm_reference=arm,
                    gripper_command=gripper,
                )
            )
        result = np.stack(validated)
        result.setflags(write=False)
        return result

    def _install_chunk(
        self,
        *,
        chunk: np.ndarray,
        source_request_id: int | None,
        formal_tick: int | None,
        event_kind: ChunkClientEventKind,
    ) -> None:
        validated = self._validated_chunk(chunk)
        old_chunk_id = self._active_chunk_id
        discarded = (
            0
            if self._active_actions is None
            else self.config.prediction_horizon - self._active_cursor
        )
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._active_actions = validated
        self._active_chunk_id = chunk_id
        self._active_source_request_id = source_request_id
        self._active_cursor = 0
        self._actions_consumed = 0
        self._chunk_events.append(
            ChunkClientEvent(
                kind=event_kind,
                formal_tick=formal_tick,
                time_us=None if formal_tick is None else formal_tick * 20_000,
                chunk_id=chunk_id,
                source_request_id=source_request_id,
                old_chunk_id=old_chunk_id,
                discarded_action_count=discarded,
                installed_chunk_index=0,
            )
        )
        if self._on_chunk_install is not None:
            self._on_chunk_install(self._chunk_events[-1])

    def bootstrap(
        self,
        *,
        observation: TObservation,
        infer: Callable[[TObservation], np.ndarray],
    ) -> BootstrapRecord:
        self._require_healthy()
        if self._bootstrapped:
            self._faulted = True
            raise RuntimeError("warm bootstrap may only run once")
        if not callable(infer):
            self._faulted = True
            raise TypeError("bootstrap inference callback must be callable")
        prospective_chunk_id = self._next_chunk_id
        try:
            simulation_before_us = self._simulation_time_reader()
            if (
                isinstance(simulation_before_us, bool)
                or not isinstance(simulation_before_us, int)
                or simulation_before_us != 0
            ):
                raise RuntimeError("warm bootstrap requires simulation time zero")
            self._chunk_events.append(
                ChunkClientEvent(
                    kind=ChunkClientEventKind.BOOTSTRAP_LAUNCH,
                    formal_tick=None,
                    time_us=None,
                    chunk_id=prospective_chunk_id,
                    source_request_id=None,
                )
            )
            wall_start_ns = self._monotonic_ns()
            chunk = infer(observation)
            simulation_after_us = self._simulation_time_reader()
            wall_end_ns = self._monotonic_ns()
            if (
                isinstance(wall_start_ns, bool)
                or not isinstance(wall_start_ns, int)
                or isinstance(wall_end_ns, bool)
                or not isinstance(wall_end_ns, int)
                or wall_start_ns < 0
                or wall_end_ns < wall_start_ns
            ):
                raise RuntimeError("bootstrap wall clock returned invalid timestamps")
            if (
                isinstance(simulation_after_us, bool)
                or not isinstance(simulation_after_us, int)
                or simulation_after_us < 0
            ):
                raise RuntimeError("bootstrap simulation time reader returned invalid time")
            self._chunk_events.append(
                ChunkClientEvent(
                    kind=ChunkClientEventKind.BOOTSTRAP_RETURN,
                    formal_tick=None,
                    time_us=None,
                    chunk_id=prospective_chunk_id,
                    source_request_id=None,
                    wall_start_ns=wall_start_ns,
                    wall_end_ns=wall_end_ns,
                    wall_duration_ns=wall_end_ns - wall_start_ns,
                    simulation_time_before_us=simulation_before_us,
                    simulation_time_after_us=simulation_after_us,
                )
            )
            if simulation_after_us != simulation_before_us:
                raise RuntimeError("simulation advanced during bootstrap inference")
            self._install_chunk(
                chunk=chunk,
                source_request_id=None,
                formal_tick=None,
                event_kind=ChunkClientEventKind.BOOTSTRAP_INSTALL,
            )
        except BaseException:
            self._faulted = True
            raise
        self._bootstrapped = True
        if self._active_chunk_id is None:
            self._faulted = True
            raise RuntimeError("warm bootstrap did not install an active chunk")
        return BootstrapRecord(
            protocol_id=self.config.protocol_id,
            installed_chunk_id=self._active_chunk_id,
            simulation_time_before_us=simulation_before_us,
            simulation_time_after_us=simulation_after_us,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            wall_duration_ns=wall_end_ns - wall_start_ns,
        )

    def _activate_arrival(self, arrival: Arrival[np.ndarray]) -> None:
        self.harness.mark_eligible(arrival)
        try:
            validated = self._validated_chunk(arrival.payload)
        except BaseException:
            self._faulted = True
            self.harness.mark_faulted()
            raise
        self.harness.mark_activated(arrival)
        self._install_chunk(
            chunk=validated,
            source_request_id=arrival.request_id,
            formal_tick=arrival.arrival_formal_tick,
            event_kind=ChunkClientEventKind.CHUNK_INSTALL,
        )

    def _hold_action(self) -> np.ndarray:
        return self.action_contract.compose_action(
            arm_reference=np.zeros(self.action_contract.arm_dim),
            gripper_command=self.last_gripper_command,
        )

    def unread_buffer(self) -> tuple[np.ndarray, np.ndarray]:
        """Full known buffer for Meta decisions, with explicit hold outside its valid prefix."""
        horizon = self.config.prediction_horizon
        remaining = (
            np.empty((0, self.action_contract.action_dim))
            if self._active_actions is None
            else self._active_actions[self._active_cursor : self._active_cursor + horizon]
        )
        result = np.repeat(self._hold_action()[None], horizon, axis=0)
        if len(remaining):
            result[: len(remaining)] = remaining
            result[len(remaining) :, -1] = remaining[-1, -1]
        result.setflags(write=False)
        mask = np.arange(horizon) < len(remaining)
        mask.setflags(write=False)
        return result, mask

    def executable_prefix(self) -> np.ndarray:
        """D20 prefix for JEPA; Meta decisions separately retain the complete known buffer."""
        return self.unread_buffer()[0][: self.config.temporal_contract.maximum_delay_ticks]

    def run_boundary(
        self,
        *,
        formal_tick: int,
        observation: TObservation,
        infer: Callable[[LaunchContext[TObservation]], np.ndarray],
        execute: Callable[[np.ndarray], None],
        decide_launch: Callable[[ChunkDecisionState], bool] | None = None,
    ) -> np.ndarray:
        self._require_healthy()
        if not self._bootstrapped:
            self._faulted = True
            raise RuntimeError("warm bootstrap is required before formal execution")
        try:
            arrival = self.harness.open_boundary(formal_tick)
            if arrival is not None:
                self._activate_arrival(arrival)
            launch = False
            if not self.harness.pending:
                if decide_launch is None:
                    launch = self._actions_consumed >= self.config.launch_trigger_horizon
                else:
                    full_buffer, buffer_mask = self.unread_buffer()
                    state = ChunkDecisionState(
                        formal_tick=formal_tick,
                        active_cursor=self._active_cursor,
                        remaining_actions=max(
                            0, self.config.prediction_horizon - self._active_cursor
                        ),
                        actions_consumed=self._actions_consumed,
                        executable_controls=self.executable_prefix(),
                        unread_action_buffer=full_buffer,
                        unread_action_mask=buffer_mask,
                    )
                    launch = decide_launch(state)
                    if type(launch) is not bool:
                        raise TypeError("launch decision must be boolean")
            if launch:
                immediate = self.harness.launch(observation=observation, infer=infer)
                if immediate is not None:
                    self._activate_arrival(immediate)
            if (
                self._active_actions is None
                or self._active_cursor >= self.config.prediction_horizon
            ):
                action = self._hold_action()
                self.harness.mark_starvation(action=action)
                execute(action)
                self.harness.mark_execute(request_id=None, action=action)
            else:
                action_index = self._active_cursor
                action = self._active_actions[action_index]
                execute(action)
                self.harness.mark_buffered_execute(
                    source_request_id=self._active_source_request_id,
                    action=action,
                )
                self._chunk_events.append(
                    ChunkClientEvent(
                        kind=ChunkClientEventKind.CHUNK_ACTION_EXECUTE,
                        formal_tick=formal_tick,
                        time_us=formal_tick * 20_000,
                        chunk_id=self._active_chunk_id,
                        source_request_id=self._active_source_request_id,
                        executed_chunk_index=action_index,
                        action=action,
                    )
                )
                self._active_cursor += 1
                self._actions_consumed += 1
            _arm, gripper = self.action_contract.split_action(action)
            self.last_gripper_command = gripper
            self.harness.close_boundary()
            return action
        except BaseException:
            self._faulted = True
            if not self.harness.faulted:
                self.harness.mark_faulted()
            raise
