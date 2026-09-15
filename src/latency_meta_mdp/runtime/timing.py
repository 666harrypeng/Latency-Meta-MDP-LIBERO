"""Integer time authority for project-owned simulator stepping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TimingEvent:
    """One successfully committed clock event."""

    kind: Literal["physics_step", "formal_boundary", "compatibility_boundary"]
    physics_step_index: int
    formal_tick_index: int
    time_us: int


class ClockLedger:
    """Track simulator time using integers and validate MuJoCo's float time."""

    def __init__(
        self,
        *,
        physics_dt_us: int,
        formal_tick_us: int,
        representation_tolerance_seconds: float = 1e-12,
    ) -> None:
        if isinstance(physics_dt_us, bool) or not isinstance(physics_dt_us, int):
            raise TypeError("physics_dt_us must be an integer")
        if isinstance(formal_tick_us, bool) or not isinstance(formal_tick_us, int):
            raise TypeError("formal_tick_us must be an integer")
        if physics_dt_us <= 0 or formal_tick_us <= 0:
            raise ValueError("clock intervals must be positive")
        if formal_tick_us % physics_dt_us:
            raise ValueError("formal_tick_us must be an integer multiple of physics_dt_us")
        if (
            not math.isfinite(representation_tolerance_seconds)
            or representation_tolerance_seconds < 0
        ):
            raise ValueError("representation tolerance must be finite and non-negative")
        self.physics_dt_us = physics_dt_us
        self.formal_tick_us = formal_tick_us
        self.representation_tolerance_seconds = representation_tolerance_seconds
        self._physics_step_index = 0

    @property
    def physics_steps_per_tick(self) -> int:
        return self.formal_tick_us // self.physics_dt_us

    @property
    def physics_step_index(self) -> int:
        return self._physics_step_index

    @property
    def formal_tick_index(self) -> int:
        return self.physics_step_index // self.physics_steps_per_tick

    @property
    def compatibility_tick_index(self) -> int:
        return self.formal_tick_index // 5

    @property
    def time_us(self) -> int:
        return self.physics_step_index * self.physics_dt_us

    @property
    def at_formal_boundary(self) -> bool:
        return self.physics_step_index % self.physics_steps_per_tick == 0

    @property
    def at_compatibility_boundary(self) -> bool:
        return (
            self.formal_tick_index > 0
            and self.at_formal_boundary
            and self.formal_tick_index % 5 == 0
        )

    def validate_sim_time(self, sim_time_seconds: float, *, step_index: int | None = None) -> None:
        candidate_step = self.physics_step_index if step_index is None else step_index
        expected = candidate_step * self.physics_dt_us / 1_000_000
        # MuJoCo accumulates its float clock by repeated dt additions. Bound their
        # roundoff with gamma_n = n*u/(1-n*u), including dt/expected conversion;
        # integer steps remain authoritative and a missing physics step still fails.
        roundoff = (candidate_step + 2) * math.ulp(1.0) / 2
        tolerance = max(self.representation_tolerance_seconds, expected * roundoff / (1 - roundoff))
        if (
            not math.isfinite(sim_time_seconds)
            or abs(float(sim_time_seconds) - expected) > tolerance
        ):
            raise ValueError(
                "simulator time mismatch: "
                f"expected {expected:.12f}s at physics step {candidate_step}, "
                f"got {float(sim_time_seconds):.12f}s"
            )

    def record_successful_step(self, *, sim_time_seconds: float) -> list[TimingEvent]:
        """Commit one physics step only after its observed simulator time is valid."""
        candidate_step = self.physics_step_index + 1
        self.validate_sim_time(sim_time_seconds, step_index=candidate_step)
        self._physics_step_index = candidate_step
        events = [
            TimingEvent(
                kind="physics_step",
                physics_step_index=self.physics_step_index,
                formal_tick_index=self.formal_tick_index,
                time_us=self.time_us,
            )
        ]
        if self.at_formal_boundary:
            events.append(
                TimingEvent(
                    kind="formal_boundary",
                    physics_step_index=self.physics_step_index,
                    formal_tick_index=self.formal_tick_index,
                    time_us=self.time_us,
                )
            )
        if self.at_compatibility_boundary:
            events.append(
                TimingEvent(
                    kind="compatibility_boundary",
                    physics_step_index=self.physics_step_index,
                    formal_tick_index=self.formal_tick_index,
                    time_us=self.time_us,
                )
            )
        return events
