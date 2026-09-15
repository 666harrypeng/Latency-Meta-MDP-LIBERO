from __future__ import annotations

import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.runtime.latency_harness import FixedDelaySampler, LogicalLatencyHarness

_CONTROL_CONFIG = Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
_CLIENT_CONFIG = Path("configs/runtime/client/sharp_return_time_h50_e25_v1.yaml")


class _SimulationClock:
    def __init__(self) -> None:
        self.time_us = 0

    def read(self) -> int:
        return self.time_us


def _module():
    return importlib.import_module("latency_meta_mdp.runtime.action_chunk_client")


def _chunk() -> np.ndarray:
    result = np.zeros((50, 7), dtype=float)
    result[:, 0] = np.arange(50) / 100
    result[:, -1] = -1.0
    return result


def _unbootstrapped_client():
    module = _module()
    clock = _SimulationClock()
    wall_clock = iter((100, 180, 200, 280))
    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=FixedDelaySampler(0),
        simulation_time_reader=clock.read,
        monotonic_ns=itertools.count(start=1000, step=10).__next__,
    )
    client = module.SharpActionChunkClient(
        action_contract=load_action_contract(_CONTROL_CONFIG),
        config=module.load_action_chunk_client_config(_CLIENT_CONFIG),
        harness=harness,
        simulation_time_reader=clock.read,
        monotonic_ns=wall_clock.__next__,
    )
    return module, client, harness, clock


def test_warm_bootstrap_installs_chunk_while_simulation_remains_at_zero() -> None:
    module, client, harness, clock = _unbootstrapped_client()

    record = client.bootstrap(
        observation="initial",
        infer=lambda observation: _chunk(),
    )

    assert clock.time_us == 0
    assert record.simulation_time_before_us == 0
    assert record.simulation_time_after_us == 0
    assert record.wall_start_ns == 100
    assert record.wall_end_ns == 180
    assert record.wall_duration_ns == 80
    assert record.installed_chunk_id == 0
    assert record.protocol_id == "sharp_return_time_chunk_v2"
    assert client.active_cursor == 0
    assert client.actions_consumed_since_activation == 0
    assert [event.kind for event in client.chunk_events] == [
        module.ChunkClientEventKind.BOOTSTRAP_LAUNCH,
        module.ChunkClientEventKind.BOOTSTRAP_RETURN,
        module.ChunkClientEventKind.BOOTSTRAP_INSTALL,
    ]
    assert harness.events == ()


def test_bootstrap_callback_that_advances_simulation_faults_before_installation() -> None:
    _module_value, client, harness, clock = _unbootstrapped_client()

    def invalid_inference(observation):
        del observation
        clock.time_us = 20_000
        return _chunk()

    with pytest.raises(RuntimeError, match="advanced during bootstrap"):
        client.bootstrap(observation="initial", infer=invalid_inference)

    assert client.active_actions is None
    assert client.active_chunk_id is None
    assert client.faulted is True
    assert harness.events == ()


def test_malformed_bootstrap_chunk_leaves_no_partial_active_state() -> None:
    _module_value, client, _harness, _clock = _unbootstrapped_client()

    with pytest.raises(ValueError, match="shape"):
        client.bootstrap(
            observation="initial",
            infer=lambda observation: np.zeros((49, 7)),
        )

    assert client.active_actions is None
    assert client.active_chunk_id is None
    assert client.active_cursor == 0
    assert client.actions_consumed_since_activation == 0
    assert client.faulted is True


def test_bootstrap_may_run_exactly_once() -> None:
    _module_value, client, _harness, _clock = _unbootstrapped_client()
    client.bootstrap(observation="initial", infer=lambda observation: _chunk())

    with pytest.raises(RuntimeError, match="only run once"):
        client.bootstrap(observation="again", infer=lambda observation: _chunk())

    assert client.faulted is True


def test_formal_execution_before_bootstrap_is_rejected_without_opening_boundary() -> None:
    _module_value, client, harness, _clock = _unbootstrapped_client()

    with pytest.raises(RuntimeError, match="bootstrap is required"):
        client.run_boundary(
            formal_tick=0,
            observation="obs",
            infer=lambda context: _chunk(),
            execute=lambda action: None,
        )

    assert harness.events == ()
    assert client.faulted is True


def test_first_formal_action_is_bootstrap_chunk_index_zero() -> None:
    module, client, harness, _clock = _unbootstrapped_client()
    initial = _chunk()
    client.bootstrap(observation="initial", infer=lambda observation: initial)
    executed: list[np.ndarray] = []

    action = client.run_boundary(
        formal_tick=0,
        observation="obs-0",
        infer=lambda context: (_ for _ in ()).throw(AssertionError("tick zero must not launch")),
        execute=lambda selected: executed.append(selected.copy()),
    )

    np.testing.assert_array_equal(action, initial[0])
    np.testing.assert_array_equal(executed[0], initial[0])
    assert client.chunk_events[-1].kind is module.ChunkClientEventKind.CHUNK_ACTION_EXECUTE
    assert client.chunk_events[-1].executed_chunk_index == 0
    assert all(event.kind.value not in {"launch", "arrival"} for event in harness.events)
