from __future__ import annotations

import importlib
from dataclasses import fields

import numpy as np
import pytest


class _SimulationClock:
    def __init__(self) -> None:
        self.time_us = 0

    def read(self) -> int:
        return self.time_us


def _module():
    return importlib.import_module("latency_meta_mdp.latency_harness")


def _make_harness(*, delay_ticks: int, wall_times: tuple[int, ...] = (100, 160)):
    module = _module()
    clock = _SimulationClock()
    wall_clock = iter(wall_times)
    harness = module.LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=module.FixedDelaySampler(delay_ticks),
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    return module, harness, clock


def _advance_empty_boundary(harness, clock: _SimulationClock, tick: int) -> None:
    clock.time_us = tick * 20_000
    assert harness.open_boundary(tick) is None
    harness.close_boundary()


def test_fixed_delay_arrives_only_at_the_exact_future_boundary() -> None:
    module, harness, clock = _make_harness(delay_ticks=7)
    for tick in range(5):
        _advance_empty_boundary(harness, clock, tick)

    clock.time_us = 5 * 20_000
    assert harness.open_boundary(5) is None
    assert harness.launch(
        observation=("immutable", 5),
        infer=lambda context: (context.launch_formal_tick, "payload"),
    ) is None
    assert harness.pending is True
    harness.close_boundary()

    for tick in range(6, 12):
        _advance_empty_boundary(harness, clock, tick)

    clock.time_us = 12 * 20_000
    arrival = harness.open_boundary(12)
    assert arrival is not None
    assert arrival.request_id == 0
    assert arrival.launch_formal_tick == 5
    assert arrival.arrival_formal_tick == 12
    assert arrival.realized_delay_ticks == 7
    assert arrival.payload == (5, "payload")
    assert harness.pending is False
    assert [event.kind for event in harness.events[-2:]] == [
        module.HarnessEventKind.REFERENCE_BOUNDARY,
        module.HarnessEventKind.ARRIVAL,
    ]


def test_zero_delay_returns_an_immediate_same_boundary_arrival() -> None:
    module, harness, _clock = _make_harness(delay_ticks=0)

    assert harness.open_boundary(0) is None
    arrival = harness.launch(observation="obs", infer=lambda context: "action")

    assert arrival is not None
    assert arrival.launch_formal_tick == 0
    assert arrival.arrival_formal_tick == 0
    assert arrival.realized_delay_ticks == 0
    assert arrival.payload == "action"
    assert harness.pending is False
    assert [event.kind for event in harness.events] == [
        module.HarnessEventKind.REFERENCE_BOUNDARY,
        module.HarnessEventKind.LAUNCH,
        module.HarnessEventKind.MEASURED_RETURN,
        module.HarnessEventKind.ARRIVAL,
    ]


def test_launch_context_structurally_excludes_realized_delay() -> None:
    module, harness, _clock = _make_harness(delay_ticks=4)
    seen_fields: set[str] = set()

    def infer(context):
        nonlocal seen_fields
        seen_fields = {field.name for field in fields(context)}
        return "payload"

    harness.open_boundary(0)
    harness.launch(observation="obs", infer=infer)

    assert seen_fields == {
        "request_id",
        "launch_formal_tick",
        "launch_time_us",
        "observation",
    }
    assert "realized_delay_ticks" not in seen_fields
    assert "arrival_formal_tick" not in seen_fields
    assert module.LaunchContext.__dataclass_params__.frozen is True


def test_delay_is_sampled_only_after_inference_returns() -> None:
    module = _module()
    clock = _SimulationClock()
    order: list[str] = []

    def infer(context):
        del context
        order.append("inference")
        return "payload"

    def sample_delay() -> int:
        assert order == ["inference"]
        order.append("delay")
        return 2

    wall_clock = iter((10, 20))
    harness = module.LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=sample_delay,
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    harness.open_boundary(0)
    harness.launch(observation="obs", infer=infer)

    assert order == ["inference", "delay"]


def test_pending_request_blocks_second_launch_before_inference() -> None:
    _module_value, harness, clock = _make_harness(delay_ticks=2)
    calls = 0

    def infer(context):
        nonlocal calls
        del context
        calls += 1
        return "payload"

    harness.open_boundary(0)
    harness.launch(observation="first", infer=infer)
    harness.close_boundary()
    clock.time_us = 20_000
    harness.open_boundary(1)

    with pytest.raises(RuntimeError, match="pending"):
        harness.launch(observation="second", infer=infer)

    assert calls == 1
    assert harness.faulted is True


def test_inference_that_advances_simulation_faults_the_harness() -> None:
    _module_value, harness, clock = _make_harness(delay_ticks=0)
    harness.open_boundary(0)

    def invalid_inference(context):
        del context
        clock.time_us = 20_000
        return "payload"

    with pytest.raises(RuntimeError, match="advanced during inference"):
        harness.launch(observation="obs", infer=invalid_inference)

    assert harness.faulted is True
    with pytest.raises(RuntimeError, match="faulted"):
        harness.close_boundary()


def test_skipping_a_due_arrival_boundary_faults_instead_of_shifting_it() -> None:
    _module_value, harness, clock = _make_harness(delay_ticks=1)
    harness.open_boundary(0)
    harness.launch(observation="obs", infer=lambda context: "payload")
    harness.close_boundary()
    clock.time_us = 40_000

    with pytest.raises(RuntimeError, match="arrival boundary"):
        harness.open_boundary(2)

    assert harness.faulted is True


@pytest.mark.parametrize("delay", [-1, True, 1.5])
def test_fixed_delay_rejects_non_integer_or_negative_values(delay: object) -> None:
    module = _module()

    with pytest.raises((TypeError, ValueError), match="delay"):
        module.FixedDelaySampler(delay)


def test_wall_duration_is_diagnostic_and_does_not_change_arrival_tick() -> None:
    module, harness, _clock = _make_harness(
        delay_ticks=3,
        wall_times=(1_000, 9_001_000),
    )
    harness.open_boundary(0)
    harness.launch(observation="obs", infer=lambda context: "payload")

    measured = next(
        event
        for event in harness.events
        if event.kind is module.HarnessEventKind.MEASURED_RETURN
    )
    assert measured.wall_start_ns == 1_000
    assert measured.wall_end_ns == 9_001_000
    assert measured.wall_duration_ns == 9_000_000
    assert harness.pending_arrival_formal_tick == 3


def test_boundary_and_event_times_use_the_exact_formal_grid() -> None:
    _module_value, harness, clock = _make_harness(delay_ticks=0)
    for tick in range(3):
        _advance_empty_boundary(harness, clock, tick)

    assert [(event.formal_tick, event.time_us) for event in harness.events] == [
        (0, 0),
        (1, 20_000),
        (2, 40_000),
    ]


def test_boundary_must_start_at_zero_and_advance_consecutively() -> None:
    _module_value, harness, clock = _make_harness(delay_ticks=0)
    clock.time_us = 20_000

    with pytest.raises(RuntimeError, match="start at tick zero"):
        harness.open_boundary(1)

    assert harness.faulted is True


def test_activation_rejects_an_arrival_owned_by_another_harness() -> None:
    _module_value, first, _first_clock = _make_harness(delay_ticks=0)
    _module_value, second, _second_clock = _make_harness(delay_ticks=0)
    first.open_boundary(0)
    arrival = first.launch(observation="obs", infer=lambda context: "payload")
    assert arrival is not None
    second.open_boundary(0)

    with pytest.raises(RuntimeError, match="another harness"):
        second.mark_eligible(arrival)

    assert second.faulted is True


def test_activation_and_execution_may_only_be_recorded_once() -> None:
    _module_value, harness, _clock = _make_harness(delay_ticks=0)
    harness.open_boundary(0)
    arrival = harness.launch(observation="obs", infer=lambda context: "payload")
    assert arrival is not None
    harness.mark_eligible(arrival)
    harness.mark_activated(arrival)
    harness.mark_execute(request_id=arrival.request_id, action=np.zeros(7))

    with pytest.raises(RuntimeError, match="execute"):
        harness.mark_execute(request_id=arrival.request_id, action=np.zeros(7))

    assert harness.faulted is True


def test_invalid_event_action_faults_the_harness() -> None:
    _module_value, harness, _clock = _make_harness(delay_ticks=0)
    harness.open_boundary(0)

    with pytest.raises(ValueError, match="finite vector"):
        harness.mark_starvation(action=np.full(7, np.nan))

    assert harness.faulted is True
