from __future__ import annotations

import importlib
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.latency_harness import FixedDelaySampler, LogicalLatencyHarness

_CONTROL_CONFIG = Path("configs/control/panda_osc_pose_delta_v1.yaml")
_CLIENT_CONFIG = Path("configs/client/sharp_chunk_h16_e8_v1.yaml")


class _SimulationClock:
    def __init__(self) -> None:
        self.time_us = 0

    def read(self) -> int:
        return self.time_us


def _module():
    return importlib.import_module("latency_meta_mdp.action_chunk_client")


def _chunk(base: float, *, gripper: float = -1.0, horizon: int = 16) -> np.ndarray:
    result = np.zeros((horizon, 7), dtype=float)
    result[:, 0] = base + np.arange(horizon) / 100
    result[:, -1] = gripper
    return result


def _make_client(delay_ticks: int):
    module = _module()
    clock = _SimulationClock()
    wall_clock = itertools.count(start=100, step=10)
    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=FixedDelaySampler(delay_ticks),
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    client = module.SharpActionChunkClient(
        action_contract=load_action_contract(_CONTROL_CONFIG),
        config=module.load_action_chunk_client_config(_CLIENT_CONFIG),
        harness=harness,
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    client.bootstrap(observation="bootstrap", infer=lambda observation: _chunk(0.0))
    return module, client, harness, clock


def _run_ticks(client, clock: _SimulationClock, *, through: int, inferred_chunk: np.ndarray):
    executed: list[np.ndarray] = []
    for tick in range(through + 1):
        clock.time_us = tick * 20_000
        client.run_boundary(
            formal_tick=tick,
            observation=f"obs-{tick}",
            infer=lambda context: inferred_chunk,
            execute=lambda action: executed.append(action.copy()),
        )
    return executed


def test_default_chunk_client_config_locks_h16_e8_sharp_warm_start() -> None:
    module = _module()

    config = module.load_action_chunk_client_config(_CLIENT_CONFIG)

    assert config.prediction_horizon == 16
    assert config.execution_horizon == 8
    assert config.bootstrap_mode == "warm_start"
    assert config.handoff_strategy == "sharp_replace"
    assert config.arrival_write_mode == "from_arrival"
    assert config.max_pending_requests == 1
    assert config.starvation_action == "hold"
    assert config.guaranteed_delay_ticks == 8


@pytest.mark.parametrize(("horizon", "execution"), ((15, 8), (16, 1), (16, 16)))
def test_chunk_horizons_are_validated_hyperparameters(horizon: int, execution: int) -> None:
    module = _module()
    default = module.load_action_chunk_client_config(_CLIENT_CONFIG)

    configured = replace(
        default,
        prediction_horizon=horizon,
        execution_horizon=execution,
    )

    assert configured.guaranteed_delay_ticks == horizon - execution


@pytest.mark.parametrize(
    ("updates", "error"),
    (
        ({"prediction_horizon": 0}, ValueError),
        ({"execution_horizon": 0}, ValueError),
        ({"prediction_horizon": 8, "execution_horizon": 9}, ValueError),
        ({"prediction_horizon": True}, TypeError),
        ({"handoff_strategy": "blend"}, ValueError),
        ({"arrival_write_mode": "skip_stale"}, ValueError),
        ({"max_pending_requests": 2}, ValueError),
    ),
)
def test_chunk_config_rejects_invalid_protocol_values(
    updates: dict[str, object],
    error: type[Exception],
) -> None:
    module = _module()
    default = module.load_action_chunk_client_config(_CLIENT_CONFIG)

    with pytest.raises(error):
        replace(default, **updates)


def test_h16_e8_launches_at_tick_eight_after_indices_zero_through_seven() -> None:
    _module_value, client, harness, clock = _make_client(delay_ticks=3)
    executed = _run_ticks(client, clock, through=8, inferred_chunk=_chunk(0.5))

    np.testing.assert_allclose([action[0] for action in executed[:8]], np.arange(8) / 100)
    assert [event.formal_tick for event in harness.events if event.kind.value == "launch"] == [8]
    assert client.active_cursor == 9
    assert client.actions_consumed_since_activation == 9


def test_delay_three_sharply_installs_new_index_zero_at_tick_eleven() -> None:
    module, client, _harness, clock = _make_client(delay_ticks=3)
    executed = _run_ticks(client, clock, through=11, inferred_chunk=_chunk(0.5))

    np.testing.assert_allclose([action[0] for action in executed[:11]], np.arange(11) / 100)
    assert executed[11][0] == pytest.approx(0.5)
    installs = [
        event
        for event in client.chunk_events
        if event.kind is module.ChunkClientEventKind.CHUNK_INSTALL
    ]
    assert len(installs) == 1
    assert installs[0].formal_tick == 11
    assert installs[0].discarded_action_count == 5
    assert installs[0].installed_chunk_index == 0
    assert client.active_cursor == 1
    assert client.actions_consumed_since_activation == 1


def test_zero_delay_installs_and_executes_new_index_zero_at_tick_eight() -> None:
    module, client, _harness, clock = _make_client(delay_ticks=0)
    executed = _run_ticks(client, clock, through=8, inferred_chunk=_chunk(0.5))

    assert executed[7][0] == pytest.approx(0.07)
    assert executed[8][0] == pytest.approx(0.5)
    install = next(
        event
        for event in client.chunk_events
        if event.kind is module.ChunkClientEventKind.CHUNK_INSTALL
    )
    assert install.formal_tick == 8
    assert install.discarded_action_count == 8


@pytest.mark.parametrize(
    ("delay_ticks", "through", "expected_starvation"),
    ((8, 16, ()), (9, 17, (16,))),
)
def test_coverage_boundary_distinguishes_delay_eight_and_nine(
    delay_ticks: int,
    through: int,
    expected_starvation: tuple[int, ...],
) -> None:
    _module_value, client, harness, clock = _make_client(delay_ticks=delay_ticks)
    executed = _run_ticks(
        client,
        clock,
        through=through,
        inferred_chunk=_chunk(0.5, gripper=1.0),
    )

    starvation_ticks = tuple(
        event.formal_tick for event in harness.events if event.kind.value == "starvation"
    )
    assert starvation_ticks == expected_starvation
    if expected_starvation:
        np.testing.assert_array_equal(executed[16], [0, 0, 0, 0, 0, 0, -1])
    assert executed[through][0] == pytest.approx(0.5)


def test_hold_preserves_last_executed_gripper_command() -> None:
    _module_value, client, harness, clock = _make_client(delay_ticks=9)
    initial = _chunk(0.0, gripper=1.0)
    client = _module_value.SharpActionChunkClient(
        action_contract=load_action_contract(_CONTROL_CONFIG),
        config=_module_value.load_action_chunk_client_config(_CLIENT_CONFIG),
        harness=harness,
        simulation_time_reader=clock.read,
        monotonic_ns=itertools.count(start=1000, step=10).__next__,
    )
    client.bootstrap(observation="bootstrap", infer=lambda observation: initial)
    executed = _run_ticks(client, clock, through=16, inferred_chunk=_chunk(0.5))

    np.testing.assert_array_equal(executed[16], [0, 0, 0, 0, 0, 0, 1])
    assert client.active_cursor == 16
    assert client.actions_consumed_since_activation == 16


@pytest.mark.parametrize(
    "invalid_chunk",
    (
        np.zeros((15, 7)),
        np.full((16, 7), np.nan),
        np.full((16, 7), 2.0),
    ),
)
def test_malformed_arrival_faults_without_replacing_old_chunk(
    invalid_chunk: np.ndarray,
) -> None:
    _module_value, client, _harness, clock = _make_client(delay_ticks=0)
    for tick in range(8):
        clock.time_us = tick * 20_000
        client.run_boundary(
            formal_tick=tick,
            observation=f"obs-{tick}",
            infer=lambda context: invalid_chunk,
            execute=lambda action: None,
        )
    clock.time_us = 160_000

    with pytest.raises(ValueError):
        client.run_boundary(
            formal_tick=8,
            observation="obs-8",
            infer=lambda context: invalid_chunk,
            execute=lambda action: None,
        )

    assert client.faulted is True
    assert client.active_chunk_id == 0
    assert client.active_cursor == 8


def test_installed_chunks_are_copied_and_read_only() -> None:
    _module_value, client, _harness, _clock = _make_client(delay_ticks=0)
    installed = client.active_actions
    assert installed is not None
    assert installed.flags.writeable is False

    original = _chunk(0.8)
    module = _module()
    clock = _SimulationClock()
    wall_clock = itertools.count(start=100, step=10)
    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=FixedDelaySampler(0),
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    copied_client = module.SharpActionChunkClient(
        action_contract=load_action_contract(_CONTROL_CONFIG),
        config=module.load_action_chunk_client_config(_CLIENT_CONFIG),
        harness=harness,
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    copied_client.bootstrap(observation="bootstrap", infer=lambda observation: original)
    original[:] = 0

    assert copied_client.active_actions is not None
    assert copied_client.active_actions[0, 0] == pytest.approx(0.8)

    exposed = copied_client.active_actions
    assert exposed is not None
    exposed.setflags(write=True)
    exposed[:] = -0.5
    assert copied_client.active_actions is not None
    assert copied_client.active_actions[0, 0] == pytest.approx(0.8)


def test_chunk_event_rejects_negative_discard_or_action_index() -> None:
    module, client, _harness, clock = _make_client(delay_ticks=0)
    _run_ticks(client, clock, through=8, inferred_chunk=_chunk(0.5))
    install = next(
        event
        for event in client.chunk_events
        if event.kind is module.ChunkClientEventKind.CHUNK_INSTALL
    )
    execution = client.chunk_events[-1]

    with pytest.raises(ValueError, match="discarded_action_count"):
        replace(install, discarded_action_count=-1)
    with pytest.raises(ValueError, match="executed_chunk_index"):
        replace(execution, executed_chunk_index=-1)


def test_arrival_resets_consumption_and_prevents_same_boundary_relaunch() -> None:
    _module_value, client, harness, clock = _make_client(delay_ticks=1)
    _run_ticks(client, clock, through=9, inferred_chunk=_chunk(0.5))

    launch_ticks = [event.formal_tick for event in harness.events if event.kind.value == "launch"]
    assert launch_ticks == [8]
    assert client.active_cursor == 1
    assert client.actions_consumed_since_activation == 1
