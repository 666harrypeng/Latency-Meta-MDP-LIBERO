import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.data.forecast.samples import DecodedForecast
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.runtime.policy_execution import (
    LogicalPolicyRuntime,
    PhysicalStepResult,
    PolicyObservation,
)
from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config


def obs(tick):
    return PolicyObservation(
        tick, np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), np.zeros(16)
    )


@pytest.mark.parametrize("conditioned", [True, False])
@pytest.mark.parametrize("cutoff", [20, 30, 31])
def test_meta_prepares_on_wait_and_reuses_one_packet_on_launch(conditioned, cutoff):
    tick = [0]
    prepared, decoded, decisions, transitions = [], [], [], []

    class Packet:
        available = True

        def __init__(self, context):
            self.context = context

        def decode(self, context):
            assert context.origin_tick == self.context.origin_tick
            assert context.buffer_version == self.context.buffer_version
            decoded.append(context.origin_tick)
            return DecodedForecast(
                context.origin_tick,
                context.origin_tick + context.estimated_delay_ticks,
                context.estimated_delay_ticks,
                context.buffer_version,
                np.zeros((2, 224, 224, 3), np.uint8),
                np.zeros(16),
            )

    class Provider:
        def observe(self, observation, previous):
            pass

        def prepare(self, context):
            assert not hasattr(context, "request_id")
            assert not hasattr(context, "realized_delay_ticks")
            packet = Packet(context)
            prepared.append(packet)
            return packet

        def predict(self, context):
            pytest.fail("shared Meta path must not predict again on Launch")

    def scheduler(state, observation, packet):
        assert packet is prepared[-1]
        decisions.append(state.formal_tick)
        return state.formal_tick == 18

    def policy(observation, context):
        assert observation.formal_tick == 18
        assert (context.forecast is not None) == conditioned
        return np.zeros((50, 7))

    engine = LogicalPolicyRuntime(
        action_contract=load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
        ),
        client_config=replace(
            load_rtc_client_config(Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")),
            initial_delay_ticks=(4,),
        ),
        simulation_time_reader=lambda: tick[0] * 20000,
        delay_sampler=lambda: 7,
        policy=policy,
        bootstrap_policy=lambda observation: np.zeros((50, 7)),
        scheduler=scheduler,
        forecast_provider=Provider(),
        scheduler_uses_forecast=True,
        policy_uses_forecast=conditioned,
        transition_sink=transitions.append,
        decision_interval_ticks=4,
        policy_alignment="observation_time",
        shield_latest_launch=False,
        gamma=0.99,
        monotonic_ns=itertools.count(0, 10).__next__,
    )
    engine.bootstrap(lambda: obs(0))

    def execute(action):
        tick[0] += 1
        return PhysicalStepResult(float(tick[0] == 31), tick[0] == 31)

    while tick[0] < cutoff:
        engine.step(formal_tick=tick[0], observe=lambda: obs(tick[0]), execute=execute)
    if not engine.terminated:
        engine.truncate(formal_tick=cutoff)
    assert decisions == ([10, 14, 18] if cutoff == 20 else [10, 14, 18, 25, 29])
    assert len(prepared) == len(decisions) and decoded == ([18] if conditioned else [])
    assert [p.context.estimated_delay_ticks for p in prepared] == (
        [4, 4, 4] if cutoff == 20 else [4, 4, 4, 7, 7]
    )
    assert [t.duration_ticks for t in transitions] == (
        [4, 4, 2] if cutoff == 20 else [4, 4, 7, 4, cutoff - 29]
    )
    assert transitions[-1].terminated == (cutoff == 31)
    assert transitions[-1].truncated == (cutoff != 31)
    assert transitions[-1].bootstrap_discount == (
        0 if cutoff == 31 else 0.99 ** transitions[-1].duration_ticks
    )
    assert transitions[-1].reward == (0.99 if cutoff == 31 else 0)
    assert not engine.transitions  # Sink owns records, not a second in-memory collection.
    summary = engine.summary()
    assert summary["forecast_calls"] == len(decisions) and summary["policy_calls"] == 1
    assert summary["forecast_decodes"] == int(conditioned)
    stages = [e["stage"] for e in engine.events if e["formal_tick"] == 18]
    assert stages.index("forecast_prepare") < stages.index("meta_decision")
    row = engine.request_ledger()[0]
    assert row["forecast_prepared_for_meta"] is True
    assert row["forecast_start_ns"] <= row["meta_decision_ns"] <= row["policy_launch_ns"]
    assert row["realized_delay_ticks"] == (None if cutoff == 20 else 7)
    assert row["estimated_delay_ticks"] == 4
