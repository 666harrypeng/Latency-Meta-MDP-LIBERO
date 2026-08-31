"""Timestamped grasp-and-lift episode outcome tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class OutcomeStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class TerminalReason(str, Enum):
    LIFT_SUCCEEDED = "lift_succeeded"
    GRASP_DEADLINE_MISSED = "grasp_deadline_missed"
    LIFT_DEADLINE_MISSED = "lift_deadline_missed"
    OBJECT_OUT_OF_BOUNDS = "object_out_of_bounds"
    CONTROL_FAILURE = "control_failure"


@dataclass(frozen=True)
class OutcomeCriteria:
    physics_dt_us: int
    formal_tick_us: int
    stable_grasp_dwell_us: int
    lift_height_m: float
    lift_dwell_us: int
    grasp_deadline_us: int | None
    lift_timeout_us: int

    def __post_init__(self) -> None:
        integer_fields = {
            "physics_dt_us": self.physics_dt_us,
            "formal_tick_us": self.formal_tick_us,
            "stable_grasp_dwell_us": self.stable_grasp_dwell_us,
            "lift_dwell_us": self.lift_dwell_us,
            "lift_timeout_us": self.lift_timeout_us,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.formal_tick_us % self.physics_dt_us:
            raise ValueError("formal_tick_us must be an integer multiple of physics_dt_us")
        if self.stable_grasp_dwell_us % self.physics_dt_us:
            raise ValueError("stable_grasp_dwell_us must align with the physics grid")
        if self.lift_dwell_us % self.formal_tick_us:
            raise ValueError("lift_dwell_us must align with the formal grid")
        if self.lift_timeout_us % self.formal_tick_us:
            raise ValueError("lift_timeout_us must align with the formal grid")
        if self.grasp_deadline_us is not None:
            if (
                isinstance(self.grasp_deadline_us, bool)
                or not isinstance(self.grasp_deadline_us, int)
                or self.grasp_deadline_us <= 0
                or self.grasp_deadline_us % self.formal_tick_us
            ):
                raise ValueError("grasp_deadline_us must be a positive formal-grid time")
        if not np.isfinite(self.lift_height_m) or self.lift_height_m <= 0:
            raise ValueError("lift_height_m must be finite and positive")


@dataclass(frozen=True)
class OutcomeEvent:
    kind: str
    time_us: int
    terminal_reason: TerminalReason | None = None


class EpisodeOutcomeTracker:
    """Track causal task milestones without embedding robot-control policy."""

    def __init__(self, criteria: OutcomeCriteria) -> None:
        self.criteria = criteria
        self.status = OutcomeStatus.RUNNING
        self.terminal_reason: TerminalReason | None = None
        self.terminal_time_us: int | None = None
        self.first_contact_us: int | None = None
        self.stable_grasp_us: int | None = None
        self.handoff_us: int | None = None
        self._bilateral_closing_start_us: int | None = None
        self._lift_dwell_start_us: int | None = None
        self._latest_time_us: int | None = None
        self._current_bilateral_closing = False
        self._current_bilateral_contact = False
        self._bilateral_contact_since_last_boundary = True
        self._last_contact_time_us: int | None = None
        self._last_boundary_time_us: int | None = None
        self._events: list[OutcomeEvent] = []

    @property
    def events(self) -> tuple[OutcomeEvent, ...]:
        return tuple(self._events)

    def fingerprint_payload(self) -> dict[str, object]:
        """Return a detached serialization of every causal tracker field."""

        return {
            "criteria": {
                "physics_dt_us": self.criteria.physics_dt_us,
                "formal_tick_us": self.criteria.formal_tick_us,
                "stable_grasp_dwell_us": self.criteria.stable_grasp_dwell_us,
                "lift_height_m": self.criteria.lift_height_m,
                "lift_dwell_us": self.criteria.lift_dwell_us,
                "grasp_deadline_us": self.criteria.grasp_deadline_us,
                "lift_timeout_us": self.criteria.lift_timeout_us,
            },
            "status": self.status.value,
            "terminal_reason": (
                None if self.terminal_reason is None else self.terminal_reason.value
            ),
            "terminal_time_us": self.terminal_time_us,
            "first_contact_us": self.first_contact_us,
            "stable_grasp_us": self.stable_grasp_us,
            "handoff_us": self.handoff_us,
            "bilateral_closing_start_us": self._bilateral_closing_start_us,
            "lift_dwell_start_us": self._lift_dwell_start_us,
            "latest_time_us": self._latest_time_us,
            "current_bilateral_closing": self._current_bilateral_closing,
            "current_bilateral_contact": self._current_bilateral_contact,
            "bilateral_contact_since_last_boundary": (
                self._bilateral_contact_since_last_boundary
            ),
            "last_contact_time_us": self._last_contact_time_us,
            "last_boundary_time_us": self._last_boundary_time_us,
            "events": [
                {
                    "kind": event.kind,
                    "time_us": event.time_us,
                    "terminal_reason": (
                        None if event.terminal_reason is None else event.terminal_reason.value
                    ),
                }
                for event in self._events
            ],
        }

    @property
    def handoff_eligible(self) -> bool:
        return bool(
            self.status is OutcomeStatus.RUNNING
            and self.handoff_us is None
            and self._current_bilateral_closing
            and self._bilateral_closing_start_us is not None
            and self._latest_time_us is not None
            and self._last_contact_time_us == self._latest_time_us
            and self.stable_grasp_us is not None
            and self._latest_time_us - self._bilateral_closing_start_us
            >= self.criteria.stable_grasp_dwell_us
        )

    def _validate_time(self, time_us: int, *, interval_us: int, grid_name: str) -> None:
        if isinstance(time_us, bool) or not isinstance(time_us, int) or time_us < 0:
            raise ValueError("time_us must be a non-negative integer")
        if time_us % interval_us:
            raise ValueError(f"time_us must align with the {grid_name} grid")
        if self._latest_time_us is not None and time_us < self._latest_time_us:
            raise ValueError("outcome observations must use monotonic time")
        self._latest_time_us = time_us

    def _require_running(self) -> None:
        if self.status is not OutcomeStatus.RUNNING:
            raise RuntimeError("episode outcome is already terminal")

    def _emit(
        self,
        kind: str,
        time_us: int,
        terminal_reason: TerminalReason | None = None,
    ) -> None:
        self._events.append(
            OutcomeEvent(kind=kind, time_us=time_us, terminal_reason=terminal_reason)
        )

    def observe_contact(
        self,
        *,
        time_us: int,
        any_pad_contact: bool,
        bilateral_contact: bool,
        closing_command: bool,
    ) -> None:
        self._require_running()
        if self._last_contact_time_us is None and time_us != 0:
            raise ValueError("contact cadence must start at time zero")
        if self._last_contact_time_us is not None and time_us < self._last_contact_time_us:
            raise ValueError("outcome observations must use monotonic time")
        if (
            self._last_contact_time_us is not None
            and time_us != self._last_contact_time_us + self.criteria.physics_dt_us
        ):
            raise ValueError("contact must be observed at every physics step")
        self._validate_time(
            time_us,
            interval_us=self.criteria.physics_dt_us,
            grid_name="physics",
        )
        self._last_contact_time_us = time_us
        if bilateral_contact and not any_pad_contact:
            raise ValueError("bilateral contact implies at least one pad contact")
        if any_pad_contact and self.first_contact_us is None:
            self.first_contact_us = time_us
            self._emit("first_contact", time_us)

        current = bool(bilateral_contact and closing_command)
        if current:
            if not self._current_bilateral_closing:
                self._bilateral_closing_start_us = time_us
            if (
                self.stable_grasp_us is None
                and self._bilateral_closing_start_us is not None
                and time_us - self._bilateral_closing_start_us
                >= self.criteria.stable_grasp_dwell_us
            ):
                self.stable_grasp_us = time_us
                self._emit("stable_grasp", time_us)
        else:
            self._bilateral_closing_start_us = None
        self._current_bilateral_closing = current
        self._current_bilateral_contact = bool(bilateral_contact)
        self._bilateral_contact_since_last_boundary = bool(
            self._bilateral_contact_since_last_boundary and bilateral_contact
        )

    def _commit_handoff(self, *, time_us: int) -> None:
        if (
            self.criteria.grasp_deadline_us is not None
            and time_us > self.criteria.grasp_deadline_us
        ):
            raise RuntimeError("handoff occurred after the grasp deadline")
        if self.handoff_us is not None:
            raise RuntimeError("handoff is irreversible and may only be committed once")
        if self._last_contact_time_us != time_us:
            raise RuntimeError("handoff requires contact from the same physics point")
        if not self.handoff_eligible:
            raise RuntimeError("handoff requires current stable bilateral closing contact")
        self.handoff_us = time_us
        self._emit("handoff", time_us)

    def evaluate_boundary(
        self,
        *,
        time_us: int,
        lift_height_m: float,
        request_handoff: bool,
    ) -> bool:
        self._require_running()
        if self._last_boundary_time_us is None and time_us != 0:
            raise ValueError("boundary cadence must start at time zero")
        if (
            self._last_boundary_time_us is not None
            and time_us != self._last_boundary_time_us + self.criteria.formal_tick_us
        ):
            raise ValueError("outcome must be evaluated at every formal boundary")
        if self._last_contact_time_us != time_us:
            raise RuntimeError("boundary evaluation requires contact from the same physics point")
        self._validate_time(
            time_us,
            interval_us=self.criteria.formal_tick_us,
            grid_name="formal",
        )
        self._last_boundary_time_us = time_us
        if not np.isfinite(lift_height_m):
            raise ValueError("lift_height_m must be finite")

        handoff_committed = False
        if request_handoff:
            self._commit_handoff(time_us=time_us)
            handoff_committed = True

        if self.handoff_us is None:
            if (
                self.criteria.grasp_deadline_us is not None
                and time_us >= self.criteria.grasp_deadline_us
            ):
                self._terminate(
                    status=OutcomeStatus.FAILURE,
                    reason=TerminalReason.GRASP_DEADLINE_MISSED,
                    time_us=time_us,
                )
            self._bilateral_contact_since_last_boundary = self._current_bilateral_contact
            return handoff_committed

        lift_qualifies = bool(
            self._bilateral_contact_since_last_boundary
            and self._current_bilateral_contact
            and lift_height_m + 1e-12 >= self.criteria.lift_height_m
        )
        if lift_qualifies:
            if self._lift_dwell_start_us is None:
                self._lift_dwell_start_us = time_us
                self._emit("lift_threshold", time_us)
            if time_us - self._lift_dwell_start_us >= self.criteria.lift_dwell_us:
                self._terminate(
                    status=OutcomeStatus.SUCCESS,
                    reason=TerminalReason.LIFT_SUCCEEDED,
                    time_us=time_us,
                )
                self._bilateral_contact_since_last_boundary = self._current_bilateral_contact
                return handoff_committed
        else:
            self._lift_dwell_start_us = None

        if time_us >= self.handoff_us + self.criteria.lift_timeout_us:
            self._terminate(
                status=OutcomeStatus.FAILURE,
                reason=TerminalReason.LIFT_DEADLINE_MISSED,
                time_us=time_us,
            )
        self._bilateral_contact_since_last_boundary = self._current_bilateral_contact
        return handoff_committed

    def fail_at_boundary(self, *, time_us: int, reason: TerminalReason) -> None:
        self._require_running()
        if self._last_boundary_time_us is None:
            expected_time_us = 0
        else:
            expected_time_us = self._last_boundary_time_us + self.criteria.formal_tick_us
        if time_us != expected_time_us:
            raise ValueError("failure must commit at the next formal boundary")
        if self._last_contact_time_us != time_us:
            raise RuntimeError("failure requires contact sampling at the same physics point")
        self._validate_time(
            time_us,
            interval_us=self.criteria.formal_tick_us,
            grid_name="formal",
        )
        if not isinstance(reason, TerminalReason) or reason is TerminalReason.LIFT_SUCCEEDED:
            raise ValueError("failure reason must be a non-success TerminalReason")
        self._terminate(status=OutcomeStatus.FAILURE, reason=reason, time_us=time_us)

    def _terminate(
        self,
        *,
        status: OutcomeStatus,
        reason: TerminalReason,
        time_us: int,
    ) -> None:
        self.status = status
        self.terminal_reason = reason
        self.terminal_time_us = time_us
        self._emit(
            "success" if status is OutcomeStatus.SUCCESS else "failure",
            time_us,
            terminal_reason=reason,
        )
