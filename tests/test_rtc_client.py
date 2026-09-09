import itertools
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.latency_harness import HarnessEventKind, LogicalLatencyHarness


def _actions(base=0.0):
    actions = np.zeros((50, 7))
    actions[:, 0] = base + np.arange(50) / 100
    actions[:, -1] = -1
    return actions


def _client(delay, initial=8):
    from latency_meta_mdp.rtc_client import RtcActionChunkClient
    from latency_meta_mdp.rtc_protocol import load_rtc_client_config

    clock = SimpleNamespace(tick=0)
    wall = itertools.count(0, 10).__next__
    harness = LogicalLatencyHarness(
        formal_tick_us=20_000,
        delay_sampler=lambda: delay,
        simulation_time_reader=lambda: clock.tick * 20_000,
        monotonic_ns=wall,
    )
    config = replace(
        load_rtc_client_config(Path("configs/client/rtc_observation_time_h50_v1.yaml")),
        initial_delay_ticks=(initial,),
        delay_history_capacity=2,
    )
    client = RtcActionChunkClient(
        action_contract=load_action_contract(Path("configs/control/panda_osc_pose_delta_v1.yaml")),
        config=config,
        harness=harness,
        simulation_time_reader=lambda: clock.tick * 20_000,
        monotonic_ns=wall,
    )
    record = client.bootstrap(observation=None, infer=lambda obs: _actions())
    assert record.protocol_id == config.protocol_id
    return client, clock, harness


def _infer(context):
    from latency_meta_mdp.rtc_protocol import TimedActionPlan

    return TimedActionPlan(
        origin_tick=context.origin_tick,
        request_id=context.request_id,
        buffer_version=context.buffer_version,
        actions=_actions(0.5),
        valid_mask=np.ones(50, dtype=bool),
    )


@pytest.mark.parametrize("delay", [0, 4, 8, 12, 20])
def test_arrival_uses_elapsed_index_and_pending_executes_original_buffer(delay):
    client, clock, harness = _client(delay)
    output = []
    contexts = []
    states = []

    def infer(context):
        contexts.append(context)
        return _infer(context)

    def decide(state):
        states.append(state)
        return state.formal_tick == 0

    for tick in range(delay + 2):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick,
            observation=None,
            infer=infer,
            execute=lambda action: output.append(action.copy()),
            decide_launch=decide,
        )
    if delay:
        np.testing.assert_array_equal(np.array(output[:delay]), _actions()[:delay])
    np.testing.assert_allclose(output[delay][0], 0.5 + delay / 100)
    assert len(contexts) == 1
    assert contexts[0].estimated_delay_ticks == 8
    np.testing.assert_array_equal(contexts[0].previous_actions, _actions())
    assert not contexts[0].previous_actions.flags.writeable
    assert not hasattr(contexts[0], "realized_delay_ticks")
    assert not hasattr(contexts[0], "arrival_formal_tick")
    installs = [e for e in client.events if e.kind.value == "chunk_install"]
    assert installs[0].installed_chunk_index == delay
    assert client.delay_history.delays == (8, delay)
    assert [e.formal_tick for e in harness.events if e.kind is HarnessEventKind.LAUNCH] == [0]
    assert all(s.estimated_delay_ticks == 8 for s in states if s.formal_tick < delay)


def test_e25_uses_plan_origin_not_actions_executed_after_arrival():
    client, clock, harness = _client(20)
    for tick in range(76):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick, observation=None, infer=_infer, execute=lambda a: None
        )
    assert [e.formal_tick for e in harness.events if e.kind is HarnessEventKind.LAUNCH] == [
        25,
        50,
        75,
    ]
    assert not any(e.kind is HarnessEventKind.STARVATION for e in harness.events)


def test_held_scheduler_reaches_explicit_hold_without_executing_padding():
    client, clock, harness = _client(4)
    output = []
    for tick in range(52):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick,
            observation=None,
            infer=_infer,
            decide_launch=lambda state: False,
            execute=lambda action: output.append(action.copy()),
        )
    np.testing.assert_array_equal(output[50], [0, 0, 0, 0, 0, 0, -1])
    assert sum(e.kind is HarnessEventKind.STARVATION for e in harness.events) == 2


@pytest.mark.parametrize(
    "field,value", [("origin_tick", 1), ("request_id", 99), ("buffer_version", 9)]
)
def test_response_identity_cannot_be_rebound_to_another_snapshot(field, value):
    client, _, _ = _client(0)
    with pytest.raises(ValueError, match="identity"):
        client.run_boundary(
            formal_tick=0,
            observation=None,
            infer=lambda ctx: replace(_infer(ctx), **{field: value}),
            decide_launch=lambda state: True,
            execute=lambda action: None,
        )


def test_out_of_scope_timeout_does_not_cancel_or_relaunch_pending_request():
    client, clock, harness = _client(30)
    for tick in range(21):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick,
            observation=None,
            infer=_infer,
            decide_launch=lambda state: True,
            execute=lambda action: None,
        )
    clock.tick = 21
    with pytest.raises(TimeoutError):
        client.run_boundary(
            formal_tick=21,
            observation=None,
            infer=_infer,
            decide_launch=lambda state: True,
            execute=lambda action: None,
        )
    assert harness.pending
    assert sum(e.kind is HarnessEventKind.LAUNCH for e in harness.events) == 1
    assert client.timeout_count == 1


def test_client_rejects_untyped_legacy_result_and_out_of_range_controls():
    for infer in [lambda ctx: _actions(), lambda ctx: replace(_infer(ctx), actions=_actions(2))]:
        client, _, _ = _client(0)
        with pytest.raises((TypeError, ValueError)):
            client.run_boundary(
                formal_tick=0,
                observation=None,
                infer=infer,
                decide_launch=lambda state: True,
                execute=lambda action: None,
            )


def _runtime(delay, scheduler, *, interval=1, alignment="observation_time"):
    from test_policy_execution import _observation

    from latency_meta_mdp.policy_execution import LogicalPolicyRuntime, PhysicalStepResult
    from latency_meta_mdp.rtc_protocol import load_rtc_client_config

    clock = SimpleNamespace(tick=0)
    contexts = []

    def policy(obs, context):
        contexts.append(context)
        return _actions(0.5)

    runtime = LogicalPolicyRuntime(
        action_contract=load_action_contract(Path("configs/control/panda_osc_pose_delta_v1.yaml")),
        client_config=load_rtc_client_config(
            Path("configs/client/rtc_observation_time_h50_v1.yaml")
        ),
        simulation_time_reader=lambda: clock.tick * 20_000,
        delay_sampler=lambda: delay,
        policy=policy,
        policy_alignment=alignment,
        scheduler=scheduler,
        bootstrap_policy=lambda obs: _actions(),
        decision_interval_ticks=interval,
        minimum_decision_tick=0,
        monotonic_ns=itertools.count(0, 10).__next__,
        gamma=0.9,
    )
    runtime.bootstrap(lambda: _observation(0))

    def execute(action):
        clock.tick += 1
        return PhysicalStepResult(reward=1.0)

    return runtime, clock, contexts, execute, _observation


def test_runtime_records_rtc_index_and_positive_duration_even_at_zero_delay():
    from latency_meta_mdp.policy_execution import ImmediateLaunchScheduler

    runtime, clock, contexts, execute, observe = _runtime(0, ImmediateLaunchScheduler())
    for tick in range(4):
        runtime.step(formal_tick=tick, observe=lambda: observe(clock.tick), execute=execute)
    assert [t.duration_ticks for t in runtime.transitions] == [1, 1, 1]
    assert runtime.policy_calls == 4
    assert all(c.estimated_delay_ticks >= 0 for c in contexts)
    assert runtime.summary()["protocol_id"] == "rtc_observation_time_h50_v1"


def test_runtime_fixed_scheduler_and_coverage_shield_use_expiration_time():
    from latency_meta_mdp.policy_execution import FixedCursorScheduler

    runtime, clock, _, execute, observe = _runtime(20, FixedCursorScheduler(25))
    for tick in range(71):
        runtime.step(formal_tick=tick, observe=lambda: observe(clock.tick), execute=execute)
    assert [r["launch_formal_tick"] for r in runtime.request_ledger()] == [25, 50]
    assert [e["installed_index"] for e in runtime.events if e["stage"] == "chunk_install"] == [
        20,
        20,
    ]
    assert runtime.summary()["starvation_ticks"] == 0
    runtime, clock, _, execute, observe = _runtime(20, lambda s, o, b: False, interval=7)
    for tick in range(51):
        runtime.step(formal_tick=tick, observe=lambda: observe(clock.tick), execute=execute)
    assert runtime.request_ledger()[0]["launch_formal_tick"] == 30
    assert runtime.summary()["shield_interventions"] == 1
    assert runtime.summary()["starvation_ticks"] == 0


def test_runtime_rejects_return_indexed_checkpoint_contract():
    with pytest.raises(ValueError, match="observation_time"):
        _runtime(4, lambda s, o, b: True, alignment="return_time")


def test_client_rejects_observation_timestamp_drift():
    client, _, _ = _client(0)
    with pytest.raises(ValueError, match="observation"):
        client.run_boundary(
            formal_tick=0,
            observation=SimpleNamespace(formal_tick=1),
            infer=_infer,
            decide_launch=lambda state: True,
            execute=lambda action: None,
        )


def test_runtime_ledger_records_estimate_separately_from_realized_delay():
    from latency_meta_mdp.policy_execution import ImmediateLaunchScheduler

    runtime, clock, _, execute, observe = _runtime(4, ImmediateLaunchScheduler())
    for tick in range(5):
        runtime.step(formal_tick=tick, observe=lambda: observe(clock.tick), execute=execute)
    first = runtime.request_ledger()[0]
    assert first["estimated_delay_ticks"] == 20
    assert first["realized_delay_ticks"] == 4
    assert first["installed_index"] == 4
    assert first["source_formal_tick"] == 0
    assert runtime.request_ledger()[1]["realized_delay_ticks"] is None


def test_zero_delay_request_after_same_boundary_arrival_installs_latest_plan():
    client, clock, harness = _client(4)
    delays = iter([4, 0])
    harness._delay_sampler = lambda: next(delays)
    output = []
    for tick in range(5):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick,
            observation=None,
            infer=lambda ctx: replace(_infer(ctx), actions=_actions(0.1 * (ctx.request_id + 1))),
            decide_launch=lambda state: True,
            execute=lambda action: output.append(action.copy()),
        )
    np.testing.assert_allclose(output[-1][0], 0.2)
    assert not harness.pending
    assert client.delay_history.delays == (4, 0)
