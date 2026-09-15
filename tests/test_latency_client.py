from __future__ import annotations

import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.runtime.latency_harness import FixedDelaySampler, LogicalLatencyHarness

_CONTROL_CONFIG = Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")


class _SimulationClock:
    def __init__(self) -> None:
        self.time_us = 0

    def read(self) -> int:
        return self.time_us


def _client_module():
    return importlib.import_module("latency_meta_mdp.runtime.latency_client")


def _make_client(delay_ticks: int):
    module = _client_module()
    clock = _SimulationClock()
    wall_clock = itertools.count(start=100, step=10)
    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=FixedDelaySampler(delay_ticks),
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    client = module.OneStepLatencyClient(
        action_contract=load_action_contract(_CONTROL_CONFIG),
        harness=harness,
    )
    return module, client, harness, clock


def _event_kinds(harness: LogicalLatencyHarness) -> list[str]:
    return [event.kind.value for event in harness.events]


def test_zero_delay_client_activates_and_executes_new_action_same_boundary() -> None:
    _module_value, client, harness, _clock = _make_client(0)
    executed: list[np.ndarray] = []
    requested = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    actual = client.run_boundary(
        formal_tick=0,
        observation="obs",
        launch=True,
        infer=lambda context: requested,
        execute=lambda action: executed.append(action.copy()),
    )

    np.testing.assert_array_equal(actual, requested)
    assert actual.flags.writeable is False
    assert len(executed) == 1
    np.testing.assert_array_equal(executed[0], requested)
    assert _event_kinds(harness) == [
        "reference_boundary",
        "launch",
        "measured_return",
        "arrival",
        "eligible_activation",
        "actual_activation",
        "execute",
    ]


def test_starvation_holds_arm_and_preserves_last_gripper_command() -> None:
    _module_value, client, harness, clock = _make_client(2)
    executed: list[np.ndarray] = []
    close_action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    at_zero = client.run_boundary(
        formal_tick=0,
        observation="t0",
        launch=True,
        infer=lambda context: close_action,
        execute=lambda action: executed.append(action.copy()),
    )
    clock.time_us = 20_000
    at_one = client.run_boundary(
        formal_tick=1,
        observation="t1",
        launch=True,
        infer=lambda context: (_ for _ in ()).throw(
            AssertionError("pending request must block client launch")
        ),
        execute=lambda action: executed.append(action.copy()),
    )
    clock.time_us = 40_000
    at_two = client.run_boundary(
        formal_tick=2,
        observation="t2",
        launch=True,
        infer=lambda context: (_ for _ in ()).throw(
            AssertionError("arrival boundary must not launch a replacement")
        ),
        execute=lambda action: executed.append(action.copy()),
    )
    clock.time_us = 60_000
    at_three = client.run_boundary(
        formal_tick=3,
        observation="t3",
        launch=False,
        infer=lambda context: close_action,
        execute=lambda action: executed.append(action.copy()),
    )

    np.testing.assert_array_equal(at_zero, [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_array_equal(at_one, [0, 0, 0, 0, 0, 0, -1])
    np.testing.assert_array_equal(at_two, close_action)
    np.testing.assert_array_equal(at_three, [0, 0, 0, 0, 0, 0, 1])
    assert client.last_gripper_command == 1.0
    assert len(executed) == 4
    assert _event_kinds(harness).count("launch") == 1
    assert _event_kinds(harness).count("arrival") == 1
    assert _event_kinds(harness).count("starvation") == 3
    assert _event_kinds(harness).count("execute") == 4


@pytest.mark.parametrize(
    "invalid_action",
    (
        np.zeros(6),
        np.array([0.0, 0.0, 0.0, np.nan, 0.0, 0.0, -1.0]),
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
    ),
)
def test_malformed_arrival_action_faults_before_execution(
    invalid_action: np.ndarray,
) -> None:
    _module_value, client, harness, _clock = _make_client(0)
    executed: list[np.ndarray] = []

    with pytest.raises(ValueError):
        client.run_boundary(
            formal_tick=0,
            observation="obs",
            launch=True,
            infer=lambda context: invalid_action,
            execute=lambda action: executed.append(action.copy()),
        )

    assert executed == []
    assert harness.faulted is True


def test_execution_exception_faults_without_updating_gripper_state() -> None:
    _module_value, client, harness, _clock = _make_client(0)
    close_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ZeroDivisionError):
        client.run_boundary(
            formal_tick=0,
            observation="obs",
            launch=True,
            infer=lambda context: close_action,
            execute=lambda action: 1 / 0,
        )

    assert client.last_gripper_command == -1.0
    assert harness.faulted is True


def test_client_rejects_a_harness_with_a_different_formal_clock() -> None:
    module = _client_module()
    contract = load_action_contract(_CONTROL_CONFIG)

    class _WrongClockHarness:
        formal_tick_us = 10_000

    with pytest.raises(ValueError, match="formal clock"):
        module.OneStepLatencyClient(
            action_contract=contract,
            harness=_WrongClockHarness(),
        )
