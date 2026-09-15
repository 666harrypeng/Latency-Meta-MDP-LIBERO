"""Observation-indexed plans and causal delay history on the native control clock."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.data.forecast.samples import DecodedForecast
from latency_meta_mdp.runtime.temporal_contract import TemporalContract, load_temporal_contract

RTC_PROTOCOL_ID = "rtc_observation_time_h50_v1"


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _immutable(value, *, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimedActionPlan:
    origin_tick: int
    actions: np.ndarray
    valid_mask: np.ndarray
    request_id: int | None
    buffer_version: int
    protocol_id: str = RTC_PROTOCOL_ID

    def __post_init__(self):
        if self.protocol_id != RTC_PROTOCOL_ID:
            raise ValueError("plan requires the observation-indexed RTC protocol")
        _nonnegative_integer(self.origin_tick, "origin_tick")
        _nonnegative_integer(self.buffer_version, "buffer_version")
        if self.request_id is not None:
            _nonnegative_integer(self.request_id, "request_id")
        actions = _immutable(self.actions, dtype=float)
        mask = _immutable(self.valid_mask)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("plan requires finite H50 controller-native 7D actions")
        if mask.shape != (50,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("plan requires a nonempty boolean validity mask")
        if not np.array_equal(mask, np.arange(50) < mask.sum()):
            raise ValueError("plan validity must be a contiguous prefix")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "valid_mask", mask)

    @property
    def valid_until_tick(self):
        return self.origin_tick + int(self.valid_mask.sum())

    def remaining_from(self, *, formal_tick: int) -> np.ndarray:
        _nonnegative_integer(formal_tick, "formal_tick")
        if formal_tick < self.origin_tick:
            raise ValueError("cannot execute a plan before its observation origin")
        return self.actions[formal_tick - self.origin_tick : int(self.valid_mask.sum())]


class RollingDelayHistory:
    """Configured initial estimates followed only by completed request observations."""

    def __init__(self, *, capacity: int, initial_delays: tuple[int, ...]):
        if type(capacity) is not int or capacity <= 0 or not initial_delays:
            raise ValueError("history requires positive capacity and declared initial delays")
        if len(initial_delays) > capacity:
            raise ValueError("initial delays exceed history capacity")
        for delay in initial_delays:
            _nonnegative_integer(delay, "initial delay")
        self._delays = deque(initial_delays, maxlen=capacity)
        self._last_request_id = -1
        self._last_completion_tick = -1

    @property
    def delays(self) -> tuple[int, ...]:
        return tuple(self._delays)

    def estimate_ticks(self) -> int:
        return max(self._delays)

    def record_completed(self, *, request_id: int, origin_tick: int, completion_tick: int):
        for name, value in (
            ("request_id", request_id),
            ("origin_tick", origin_tick),
            ("completion_tick", completion_tick),
        ):
            _nonnegative_integer(value, name)
        if (
            request_id <= self._last_request_id
            or completion_tick < origin_tick
            or completion_tick < self._last_completion_tick
        ):
            raise ValueError("completed requests must have ordered identities and valid times")
        self._delays.append(completion_tick - origin_tick)
        self._last_request_id = request_id
        self._last_completion_tick = completion_tick


@dataclass(frozen=True)
class RtcActionChunkClientConfig:
    schema_version: int
    protocol_id: str
    temporal_contract: TemporalContract
    delay_history_capacity: int
    initial_delay_ticks: tuple[int, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported RTC client schema")
        if self.protocol_id != RTC_PROTOCOL_ID:
            raise ValueError("unsupported RTC protocol")
        if not isinstance(self.temporal_contract, TemporalContract):
            raise TypeError("RTC requires a typed temporal contract")
        if self.prediction_horizon != 50 or self.maximum_delay_ticks != 20:
            raise ValueError("RTC v1 requires H50 and D20")
        delays = tuple(self.initial_delay_ticks)
        RollingDelayHistory(capacity=self.delay_history_capacity, initial_delays=delays)
        if max(delays) > self.maximum_delay_ticks:
            raise ValueError("initial RTC estimate exceeds declared delay support")
        object.__setattr__(self, "initial_delay_ticks", delays)

    @property
    def prediction_horizon(self):
        return self.temporal_contract.prediction_horizon

    @property
    def launch_trigger_horizon(self):
        return self.temporal_contract.launch_trigger_horizon

    @property
    def maximum_delay_ticks(self):
        return self.temporal_contract.maximum_delay_ticks


def load_rtc_client_config(path: Path) -> RtcActionChunkClientConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != set(
        RtcActionChunkClientConfig.__dataclass_fields__
    ):
        raise ValueError("invalid RTC configuration fields")
    raw["temporal_contract"] = load_temporal_contract(
        (path.parent / raw["temporal_contract"]).resolve()
    )
    return RtcActionChunkClientConfig(**raw)


@dataclass(frozen=True)
class RtcDecisionState:
    formal_tick: int
    plan_origin_tick: int
    buffer_version: int
    active_cursor: int
    remaining_actions: int
    actions_consumed: int
    estimated_delay_ticks: int
    delay_history_ticks: tuple[int, ...]
    unread_action_buffer: np.ndarray
    unread_action_mask: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "unread_action_buffer", _immutable(self.unread_action_buffer))
        object.__setattr__(self, "unread_action_mask", _immutable(self.unread_action_mask))

    @property
    def plan_age(self):
        return self.formal_tick - self.plan_origin_tick

    @property
    def executable_controls(self):
        return self.unread_action_buffer[:20]


@dataclass(frozen=True)
class RtcForecastContext:
    """Public decision snapshot, before a request has been assigned an identity."""

    origin_tick: int
    observation: Any
    buffer_version: int
    previous_actions: np.ndarray
    previous_action_mask: np.ndarray
    estimated_delay_ticks: int

    def __post_init__(self):
        _nonnegative_integer(self.origin_tick, "origin_tick")
        _nonnegative_integer(self.buffer_version, "buffer_version")
        if (
            type(self.estimated_delay_ticks) is not int
            or not 0 <= self.estimated_delay_ticks <= 20
            or self.observation.formal_tick != self.origin_tick
        ):
            raise ValueError("forecast decision snapshot has invalid time indices")
        object.__setattr__(self, "previous_actions", _immutable(self.previous_actions))
        object.__setattr__(self, "previous_action_mask", _immutable(self.previous_action_mask))

    @classmethod
    def from_decision(cls, observation, state):
        return cls(
            state.formal_tick,
            observation,
            state.buffer_version,
            state.unread_action_buffer,
            state.unread_action_mask,
            state.estimated_delay_ticks,
        )


@dataclass(frozen=True)
class RtcInferenceContext:
    request_id: int
    origin_tick: int
    observation: Any
    buffer_version: int
    previous_actions: np.ndarray
    previous_action_mask: np.ndarray
    estimated_delay_ticks: int
    forecast: DecodedForecast | None = None

    def __post_init__(self):
        object.__setattr__(self, "previous_actions", _immutable(self.previous_actions))
        object.__setattr__(self, "previous_action_mask", _immutable(self.previous_action_mask))
        if self.forecast is not None and (
            not isinstance(self.forecast, DecodedForecast)
            or self.forecast.source_tick != self.origin_tick
            or self.forecast.requested_query_ticks != self.estimated_delay_ticks
            or self.forecast.buffer_version != self.buffer_version
        ):
            raise ValueError("forecast does not match the request source/buffer/query")
