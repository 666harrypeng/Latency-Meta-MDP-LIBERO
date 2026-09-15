from __future__ import annotations

import pytest

from latency_meta_mdp.runtime.timing import ClockLedger


def test_five_hundred_steps_are_one_second_and_fifty_formal_ticks() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)

    for step in range(1, 501):
        ledger.record_successful_step(sim_time_seconds=step * 0.002)

    assert ledger.physics_step_index == 500
    assert ledger.formal_tick_index == 50
    assert ledger.compatibility_tick_index == 10
    assert ledger.time_us == 1_000_000
    assert ledger.at_formal_boundary
    assert ledger.at_compatibility_boundary


def test_formal_and_compatibility_boundaries_use_integer_steps() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)

    for step in range(1, 51):
        events = ledger.record_successful_step(sim_time_seconds=step * 0.002)

    assert ledger.physics_step_index == 50
    assert ledger.formal_tick_index == 5
    assert ledger.compatibility_tick_index == 1
    assert [event.kind for event in events] == [
        "physics_step",
        "formal_boundary",
        "compatibility_boundary",
    ]


def test_sim_time_mismatch_does_not_advance_ledger() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)

    with pytest.raises(ValueError, match="simulator time mismatch"):
        ledger.record_successful_step(sim_time_seconds=0.003)

    assert ledger.physics_step_index == 0
    assert ledger.formal_tick_index == 0
    assert ledger.time_us == 0


def test_non_integral_formal_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        ClockLedger(physics_dt_us=3_000, formal_tick_us=20_000)


def test_repeated_float_timesteps_preserve_long_episode_integer_clock() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    simulator_time = 0.0
    for _ in range(10_000):
        simulator_time += 0.002
        ledger.record_successful_step(sim_time_seconds=simulator_time)
    assert ledger.time_us == 20_000_000
    assert ledger.formal_tick_index == 1000
    with pytest.raises(ValueError, match="simulator time mismatch"):
        ledger.record_successful_step(sim_time_seconds=simulator_time + 0.004)
    assert ledger.physics_step_index == 10_000


@pytest.mark.parametrize("simulator_time", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_simulator_time_cannot_commit_a_step(simulator_time) -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    with pytest.raises(ValueError, match="simulator time mismatch"):
        ledger.record_successful_step(sim_time_seconds=simulator_time)
    assert ledger.physics_step_index == 0
