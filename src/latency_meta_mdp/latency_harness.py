"""Causal logical-latency request lifecycle on the 20 ms formal clock."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

TObservation = TypeVar("TObservation")
TPayload = TypeVar("TPayload")


class HarnessEventKind(str, Enum):
    REFERENCE_BOUNDARY = "reference_boundary"
    LAUNCH = "launch"
    MEASURED_RETURN = "measured_return"
    ARRIVAL = "arrival"
    ELIGIBLE_ACTIVATION = "eligible_activation"
    ACTUAL_ACTIVATION = "actual_activation"
    EXECUTE = "execute"
    STARVATION = "starvation"


@dataclass(frozen=True)
class HarnessEvent:
    kind: HarnessEventKind
    formal_tick: int
    time_us: int
    request_id: int | None = None
    launch_formal_tick: int | None = None
    arrival_formal_tick: int | None = None
    realized_delay_ticks: int | None = None
    wall_start_ns: int | None = None
    wall_end_ns: int | None = None
    wall_duration_ns: int | None = None
    action: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.formal_tick, bool)
            or not isinstance(self.formal_tick, int)
            or self.formal_tick < 0
            or self.time_us != self.formal_tick * 20_000
        ):
            raise ValueError("harness event does not match the 20 ms formal grid")
        for name in (
            "request_id",
            "launch_formal_tick",
            "arrival_formal_tick",
            "realized_delay_ticks",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer when present")
        wall_values = (self.wall_start_ns, self.wall_end_ns, self.wall_duration_ns)
        if self.kind is HarnessEventKind.MEASURED_RETURN:
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in wall_values
            ):
                raise ValueError("measured return requires non-negative wall-clock fields")
            if self.wall_end_ns - self.wall_start_ns != self.wall_duration_ns:
                raise ValueError("wall duration does not match wall start/end")
        elif any(value is not None for value in wall_values):
            raise ValueError("wall-clock fields are only valid on measured_return")
        if self.action is not None:
            action = np.array(self.action, dtype=float, copy=True)
            if action.ndim != 1 or not np.all(np.isfinite(action)):
                raise ValueError("event action must be a finite vector")
            action.setflags(write=False)
            object.__setattr__(self, "action", action)


@dataclass(frozen=True)
class LaunchContext(Generic[TObservation]):
    request_id: int
    launch_formal_tick: int
    launch_time_us: int
    observation: TObservation


@dataclass(frozen=True)
class Arrival(Generic[TPayload]):
    request_id: int
    launch_formal_tick: int
    arrival_formal_tick: int
    realized_delay_ticks: int
    payload: TPayload
    _owner_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "launch_formal_tick",
            "arrival_formal_tick",
            "realized_delay_ticks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.arrival_formal_tick != (
            self.launch_formal_tick + self.realized_delay_ticks
        ):
            raise ValueError("arrival tick does not equal launch tick plus realized delay")


@dataclass(frozen=True)
class FixedDelaySampler:
    delay_ticks: int

    def __post_init__(self) -> None:
        if isinstance(self.delay_ticks, bool) or not isinstance(self.delay_ticks, int):
            raise TypeError("delay_ticks must be an integer")
        if self.delay_ticks < 0:
            raise ValueError("delay_ticks must be non-negative")

    def __call__(self) -> int:
        return self.delay_ticks


@dataclass(frozen=True)
class _PendingRequest(Generic[TPayload]):
    request_id: int
    launch_formal_tick: int
    arrival_formal_tick: int
    realized_delay_ticks: int
    payload: TPayload


class LogicalLatencyHarness(Generic[TObservation, TPayload]):
    """Enforce one causal request lifecycle without advancing simulated time."""

    def __init__(
        self,
        *,
        formal_tick_us: int,
        delay_sampler: Callable[[], int],
        simulation_time_reader: Callable[[], int],
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if formal_tick_us != 20_000:
            raise ValueError("logical latency harness requires the locked 20 ms formal tick")
        if not callable(delay_sampler):
            raise TypeError("delay_sampler must be callable")
        if not callable(simulation_time_reader):
            raise TypeError("simulation_time_reader must be callable")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        self.formal_tick_us = formal_tick_us
        self._delay_sampler = delay_sampler
        self._simulation_time_reader = simulation_time_reader
        self._monotonic_ns = monotonic_ns
        self._owner_token = object()
        self._events: list[HarnessEvent] = []
        self._pending: _PendingRequest[TPayload] | None = None
        self._next_request_id = 0
        self._last_closed_tick: int | None = None
        self._current_tick: int | None = None
        self._boundary_open = False
        self._launched_this_boundary = False
        self._faulted = False

    @property
    def events(self) -> tuple[HarnessEvent, ...]:
        return tuple(self._events)

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def pending_arrival_formal_tick(self) -> int | None:
        return None if self._pending is None else self._pending.arrival_formal_tick

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def current_formal_tick(self) -> int:
        self._require_healthy()
        if not self._boundary_open or self._current_tick is None:
            self._fail("no formal boundary is open")
        return self._current_tick

    def _require_healthy(self) -> None:
        if self._faulted:
            raise RuntimeError("logical latency harness is faulted")

    def _fail(self, message: str) -> None:
        self._faulted = True
        raise RuntimeError(message)

    def _read_simulation_time(self) -> int:
        value = self._simulation_time_reader()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self._fail("simulation time reader must return non-negative integer microseconds")
        return value

    def _emit(self, kind: HarnessEventKind, **kwargs: Any) -> None:
        if self._current_tick is None:
            self._fail("cannot emit an event without a current formal boundary")
        self._events.append(
            HarnessEvent(
                kind=kind,
                formal_tick=self._current_tick,
                time_us=self._current_tick * self.formal_tick_us,
                **kwargs,
            )
        )

    def _arrival_from_pending(self, pending: _PendingRequest[TPayload]) -> Arrival[TPayload]:
        arrival = Arrival(
            request_id=pending.request_id,
            launch_formal_tick=pending.launch_formal_tick,
            arrival_formal_tick=pending.arrival_formal_tick,
            realized_delay_ticks=pending.realized_delay_ticks,
            payload=pending.payload,
            _owner_token=self._owner_token,
        )
        self._emit(
            HarnessEventKind.ARRIVAL,
            request_id=arrival.request_id,
            launch_formal_tick=arrival.launch_formal_tick,
            arrival_formal_tick=arrival.arrival_formal_tick,
            realized_delay_ticks=arrival.realized_delay_ticks,
        )
        return arrival

    def open_boundary(self, formal_tick: int) -> Arrival[TPayload] | None:
        self._require_healthy()
        if self._boundary_open:
            self._fail("previous formal boundary is still open")
        if isinstance(formal_tick, bool) or not isinstance(formal_tick, int) or formal_tick < 0:
            self._fail("formal tick must be a non-negative integer")
        if self._last_closed_tick is None:
            if formal_tick != 0:
                self._fail("logical latency harness must start at tick zero")
        if self._pending is not None and self._pending.arrival_formal_tick < formal_tick:
            self._fail("pending request missed its exact arrival boundary")
        if self._last_closed_tick is not None and formal_tick != self._last_closed_tick + 1:
            self._fail("formal boundaries must advance consecutively")
        expected_time_us = formal_tick * self.formal_tick_us
        if self._read_simulation_time() != expected_time_us:
            self._fail("simulation time does not match the opened formal boundary")
        self._current_tick = formal_tick
        self._boundary_open = True
        self._launched_this_boundary = False
        self._emit(HarnessEventKind.REFERENCE_BOUNDARY)
        if self._pending is None or self._pending.arrival_formal_tick > formal_tick:
            return None
        pending = self._pending
        self._pending = None
        return self._arrival_from_pending(pending)

    def launch(
        self,
        *,
        observation: TObservation,
        infer: Callable[[LaunchContext[TObservation]], TPayload],
    ) -> Arrival[TPayload] | None:
        self._require_healthy()
        if not self._boundary_open or self._current_tick is None:
            self._fail("launch requires an open formal boundary")
        if self._pending is not None:
            self._fail("cannot launch while another request is pending")
        if self._launched_this_boundary:
            self._fail("at most one request may launch in one formal boundary")
        if not callable(infer):
            self._fail("inference callback must be callable")

        request_id = self._next_request_id
        self._next_request_id += 1
        launch_tick = self._current_tick
        self._launched_this_boundary = True
        self._emit(
            HarnessEventKind.LAUNCH,
            request_id=request_id,
            launch_formal_tick=launch_tick,
        )
        context = LaunchContext(
            request_id=request_id,
            launch_formal_tick=launch_tick,
            launch_time_us=launch_tick * self.formal_tick_us,
            observation=observation,
        )
        try:
            wall_start_ns = self._monotonic_ns()
            simulation_before_us = self._read_simulation_time()
            payload = infer(context)
            simulation_after_us = self._read_simulation_time()
            wall_end_ns = self._monotonic_ns()
        except BaseException:
            self._faulted = True
            raise
        if (
            isinstance(wall_start_ns, bool)
            or not isinstance(wall_start_ns, int)
            or isinstance(wall_end_ns, bool)
            or not isinstance(wall_end_ns, int)
            or wall_start_ns < 0
            or wall_end_ns < wall_start_ns
        ):
            self._fail("monotonic wall clock returned invalid timestamps")
        self._emit(
            HarnessEventKind.MEASURED_RETURN,
            request_id=request_id,
            launch_formal_tick=launch_tick,
            wall_start_ns=wall_start_ns,
            wall_end_ns=wall_end_ns,
            wall_duration_ns=wall_end_ns - wall_start_ns,
        )
        if simulation_after_us != simulation_before_us:
            self._fail("simulation advanced during inference")

        try:
            delay_ticks = self._delay_sampler()
        except BaseException:
            self._faulted = True
            raise
        if isinstance(delay_ticks, bool) or not isinstance(delay_ticks, int):
            self._fail("realized delay must be an integer")
        if delay_ticks < 0:
            self._fail("realized delay must be non-negative")
        arrival_tick = launch_tick + delay_ticks
        pending = _PendingRequest(
            request_id=request_id,
            launch_formal_tick=launch_tick,
            arrival_formal_tick=arrival_tick,
            realized_delay_ticks=delay_ticks,
            payload=payload,
        )
        if delay_ticks == 0:
            return self._arrival_from_pending(pending)
        self._pending = pending
        return None

    def close_boundary(self) -> None:
        self._require_healthy()
        if not self._boundary_open or self._current_tick is None:
            self._fail("no formal boundary is open")
        self._last_closed_tick = self._current_tick
        self._boundary_open = False
