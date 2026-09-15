from __future__ import annotations

from collections.abc import Callable

import pytest

from latency_meta_mdp.envs.outcomes import (
    EpisodeOutcomeTracker,
    OutcomeCriteria,
    OutcomeStatus,
    TerminalReason,
)


def _criteria(*, grasp_deadline_us: int | None = 3_000_000) -> OutcomeCriteria:
    return OutcomeCriteria(
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        stable_grasp_dwell_us=40_000,
        lift_height_m=0.10,
        lift_dwell_us=100_000,
        grasp_deadline_us=grasp_deadline_us,
        lift_timeout_us=10_000_000,
    )


def _advance(
    tracker: EpisodeOutcomeTracker,
    *,
    start_us: int,
    end_us: int,
    bilateral_contact: Callable[[int], bool],
    closing_command: Callable[[int], bool] | None = None,
    lift_height_m: Callable[[int], float] | None = None,
    handoff_time_us: int | None = None,
) -> None:
    closing = closing_command or bilateral_contact
    lift = lift_height_m or (lambda _time_us: 0.0)
    for time_us in range(start_us, end_us + 1, tracker.criteria.physics_dt_us):
        contact = bilateral_contact(time_us)
        tracker.observe_contact(
            time_us=time_us,
            any_pad_contact=contact,
            bilateral_contact=contact,
            closing_command=closing(time_us),
        )
        if time_us % tracker.criteria.formal_tick_us == 0:
            tracker.evaluate_boundary(
                time_us=time_us,
                lift_height_m=lift(time_us),
                request_handoff=time_us == handoff_time_us,
            )
        if tracker.status is not OutcomeStatus.RUNNING:
            return


def test_atomic_boundary_commits_handoff_from_fresh_stable_contact() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    _advance(
        tracker,
        start_us=0,
        end_us=40_000,
        bilateral_contact=lambda _time_us: True,
        handoff_time_us=40_000,
    )

    assert tracker.handoff_us == 40_000
    assert tracker.first_contact_us == 0
    assert tracker.stable_grasp_us == 40_000
    assert [event.kind for event in tracker.events] == [
        "first_contact",
        "stable_grasp",
        "handoff",
    ]


def test_broken_contact_resets_handoff_eligibility() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    with pytest.raises(RuntimeError, match="stable bilateral closing contact"):
        _advance(
            tracker,
            start_us=0,
            end_us=60_000,
            bilateral_contact=lambda time_us: time_us != 20_000,
            handoff_time_us=60_000,
        )


def test_dynamic_episode_fails_at_grasp_deadline_without_handoff() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    _advance(
        tracker,
        start_us=0,
        end_us=3_000_000,
        bilateral_contact=lambda _time_us: False,
    )

    assert tracker.status is OutcomeStatus.FAILURE
    assert tracker.terminal_reason is TerminalReason.GRASP_DEADLINE_MISSED
    assert tracker.terminal_time_us == 3_000_000


def test_handoff_at_exact_grasp_deadline_precedes_deadline_failure() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    _advance(
        tracker,
        start_us=0,
        end_us=3_000_000,
        bilateral_contact=lambda time_us: time_us >= 2_960_000,
        handoff_time_us=3_000_000,
    )

    assert tracker.status is OutcomeStatus.RUNNING
    assert tracker.handoff_us == 3_000_000


def test_lift_timeout_is_measured_from_handoff_not_episode_start() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    def contact(time_us: int) -> bool:
        return time_us >= 2_960_000

    _advance(
        tracker,
        start_us=0,
        end_us=12_980_000,
        bilateral_contact=contact,
        handoff_time_us=3_000_000,
    )
    assert tracker.status is OutcomeStatus.RUNNING

    _advance(
        tracker,
        start_us=12_982_000,
        end_us=13_000_000,
        bilateral_contact=contact,
    )
    assert tracker.status is OutcomeStatus.FAILURE
    assert tracker.terminal_reason is TerminalReason.LIFT_DEADLINE_MISSED


def test_success_requires_fresh_continuous_contact_and_lift_dwell() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())

    _advance(
        tracker,
        start_us=0,
        end_us=1_220_000,
        bilateral_contact=lambda time_us: not 1_080_000 <= time_us < 1_100_000,
        lift_height_m=lambda time_us: 0.10 if time_us >= 1_000_000 else 0.0,
        handoff_time_us=40_000,
    )

    assert tracker.status is OutcomeStatus.SUCCESS
    assert tracker.terminal_reason is TerminalReason.LIFT_SUCCEEDED
    assert tracker.terminal_time_us == 1_220_000


def test_l0_smoke_can_disable_the_grasp_deadline() -> None:
    tracker = EpisodeOutcomeTracker(_criteria(grasp_deadline_us=None))

    _advance(
        tracker,
        start_us=0,
        end_us=3_000_000,
        bilateral_contact=lambda _time_us: False,
    )

    assert tracker.status is OutcomeStatus.RUNNING


def test_contact_cadence_starts_at_zero_and_rejects_gaps() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())
    with pytest.raises(ValueError, match="start at time zero"):
        tracker.observe_contact(
            time_us=2_000,
            any_pad_contact=False,
            bilateral_contact=False,
            closing_command=False,
        )

    tracker.observe_contact(
        time_us=0,
        any_pad_contact=False,
        bilateral_contact=False,
        closing_command=False,
    )
    with pytest.raises(ValueError, match="every physics step"):
        tracker.observe_contact(
            time_us=4_000,
            any_pad_contact=False,
            bilateral_contact=False,
            closing_command=False,
        )


def test_boundary_requires_same_time_contact_and_contiguous_cadence() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())
    tracker.observe_contact(
        time_us=0,
        any_pad_contact=False,
        bilateral_contact=False,
        closing_command=False,
    )
    tracker.evaluate_boundary(time_us=0, lift_height_m=0.0, request_handoff=False)
    for time_us in range(2_000, 40_001, 2_000):
        tracker.observe_contact(
            time_us=time_us,
            any_pad_contact=False,
            bilateral_contact=False,
            closing_command=False,
        )

    with pytest.raises(ValueError, match="every formal boundary"):
        tracker.evaluate_boundary(time_us=40_000, lift_height_m=0.0, request_handoff=False)

    fresh = EpisodeOutcomeTracker(_criteria())
    fresh.observe_contact(
        time_us=0,
        any_pad_contact=False,
        bilateral_contact=False,
        closing_command=False,
    )
    fresh.evaluate_boundary(time_us=0, lift_height_m=0.0, request_handoff=False)
    with pytest.raises(RuntimeError, match="same physics point"):
        fresh.evaluate_boundary(time_us=20_000, lift_height_m=0.0, request_handoff=False)


def test_failure_terminal_commit_requires_a_formal_boundary() -> None:
    tracker = EpisodeOutcomeTracker(_criteria())
    tracker.observe_contact(
        time_us=0,
        any_pad_contact=False,
        bilateral_contact=False,
        closing_command=False,
    )
    tracker.evaluate_boundary(time_us=0, lift_height_m=0.0, request_handoff=False)

    with pytest.raises(ValueError, match="formal boundary"):
        tracker.fail_at_boundary(
            time_us=2_000,
            reason=TerminalReason.OBJECT_OUT_OF_BOUNDS,
        )

    for time_us in range(2_000, 20_001, 2_000):
        tracker.observe_contact(
            time_us=time_us,
            any_pad_contact=False,
            bilateral_contact=False,
            closing_command=False,
        )
    tracker.fail_at_boundary(
        time_us=20_000,
        reason=TerminalReason.OBJECT_OUT_OF_BOUNDS,
    )
    assert tracker.status is OutcomeStatus.FAILURE
    assert tracker.terminal_time_us == 20_000
