"""Authoritative parameterized timing contract for H50/E25 deployment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONTRACT_ID = re.compile(
    r"^h(?P<h>[1-9][0-9]*)_e(?P<e>[1-9][0-9]*)_d(?P<d>[1-9][0-9]*)_k(?P<k>[1-9][0-9]*)_v1$"
)


@dataclass(frozen=True)
class TickInterval:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("tick interval must be non-negative and non-empty")

    @property
    def count(self) -> int:
        return self.maximum - self.minimum + 1

    def to_mapping(self) -> dict[str, int]:
        return {
            "count": self.count,
            "maximum": self.maximum,
            "minimum": self.minimum,
        }


@dataclass(frozen=True)
class TemporalContract:
    schema_version: int
    contract_id: str
    formal_tick_us: int
    control_frequency_hz: int
    prediction_horizon: int
    launch_trigger_horizon: int
    maximum_delay_ticks: int
    history_sample_count: int

    def __post_init__(self) -> None:
        identifier = _CONTRACT_ID.fullmatch(self.contract_id)
        if self.schema_version != 1 or identifier is None:
            raise ValueError("unsupported temporal contract schema or identifier")
        for name in (
            "formal_tick_us",
            "control_frequency_hz",
            "prediction_horizon",
            "launch_trigger_horizon",
            "maximum_delay_ticks",
            "history_sample_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        encoded = tuple(int(identifier.group(name)) for name in ("h", "e", "d", "k"))
        configured = (
            self.prediction_horizon,
            self.launch_trigger_horizon,
            self.maximum_delay_ticks,
            self.history_sample_count,
        )
        if encoded != configured:
            raise ValueError("temporal contract identifier does not match configured values")
        if self.formal_tick_us * self.control_frequency_hz != 1_000_000:
            raise ValueError("formal tick and control frequency are inconsistent")
        if (self.formal_tick_us, self.control_frequency_hz) != (20_000, 50):
            raise ValueError("temporal contract requires the certified 50 Hz formal clock")
        if not self.launch_trigger_horizon < self.prediction_horizon:
            raise ValueError("launch trigger must be shorter than prediction horizon")
        if self.maximum_delay_ticks > self.remaining_buffer_coverage:
            raise ValueError("maximum delay exceeds launch-time buffer coverage")

    @property
    def chunk_duration_us(self) -> int:
        return self.prediction_horizon * self.formal_tick_us

    @property
    def launch_trigger_duration_us(self) -> int:
        return self.launch_trigger_horizon * self.formal_tick_us

    @property
    def remaining_buffer_coverage(self) -> int:
        return self.prediction_horizon - self.launch_trigger_horizon

    @property
    def remaining_buffer_duration_us(self) -> int:
        return self.remaining_buffer_coverage * self.formal_tick_us

    @property
    def history_span_us(self) -> int:
        return (self.history_sample_count - 1) * self.formal_tick_us

    @property
    def target_tail_ticks(self) -> int:
        return self.prediction_horizon + self.maximum_delay_ticks

    @property
    def no_starvation_guaranteed(self) -> bool:
        return self.maximum_delay_ticks <= self.remaining_buffer_coverage

    @staticmethod
    def _episode_action_count(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("episode_action_count must be a positive integer")
        return value

    def belief_source_interval(self, *, episode_action_count: int) -> TickInterval:
        total = self._episode_action_count(episode_action_count)
        minimum = max(self.launch_trigger_horizon, self.history_sample_count - 1)
        maximum = total - max(
            self.remaining_buffer_coverage,
            self.maximum_delay_ticks,
        )
        if maximum < minimum:
            raise ValueError("episode has no valid belief source under temporal contract")
        return TickInterval(minimum=minimum, maximum=maximum)

    def sft_source_interval(self, *, episode_action_count: int) -> TickInterval:
        total = self._episode_action_count(episode_action_count)
        maximum = total - self.prediction_horizon
        if maximum < 0:
            raise ValueError("episode has no complete SFT prediction horizon")
        return TickInterval(minimum=0, maximum=maximum)

    def return_chunk_source_interval(
        self,
        *,
        episode_action_count: int,
        delay_tick: int,
    ) -> TickInterval:
        total = self._episode_action_count(episode_action_count)
        if (
            isinstance(delay_tick, bool)
            or not isinstance(delay_tick, int)
            or not 1 <= delay_tick <= self.maximum_delay_ticks
        ):
            raise ValueError("delay_tick is outside the temporal contract")
        belief = self.belief_source_interval(episode_action_count=total)
        maximum = min(
            belief.maximum,
            total - self.prediction_horizon - delay_tick,
        )
        if maximum < belief.minimum:
            raise ValueError("episode has no valid return-time chunk branch")
        return TickInterval(minimum=belief.minimum, maximum=maximum)

    def teacher_chunk_bounds(
        self,
        *,
        source_tick: int,
        episode_action_count: int,
    ) -> tuple[int, int]:
        interval = self.belief_source_interval(episode_action_count=episode_action_count)
        if source_tick < interval.minimum or source_tick > interval.maximum:
            raise ValueError("source_tick is outside the belief source interval")
        start = source_tick - self.launch_trigger_horizon
        return start, start + self.prediction_horizon

    def teacher_remaining_bounds(
        self,
        *,
        source_tick: int,
        episode_action_count: int,
    ) -> tuple[int, int]:
        self.teacher_chunk_bounds(
            source_tick=source_tick,
            episode_action_count=episode_action_count,
        )
        return source_tick, source_tick + self.remaining_buffer_coverage


def load_temporal_contract(path: Path) -> TemporalContract:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(TemporalContract.__dataclass_fields__):
        raise ValueError("temporal contract config fields are invalid")
    return TemporalContract(**raw)
