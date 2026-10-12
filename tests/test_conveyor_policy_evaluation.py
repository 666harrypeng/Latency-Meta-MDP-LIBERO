from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.conveyor.outcomes import ParcelLedger
from latency_meta_mdp.runtime.policy_execution import PolicyObservation
from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config


class ConveyorFixture:
    """Replace physics only; use the actual parcel ledger and policy executor."""

    def __init__(self):
        self.action_contract = load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml")
        )
        self.executor = self
        self.env = self
        self.ledger = SimpleNamespace(time_us=0)
        self.world = SimpleNamespace(ledger=ParcelLedger(supply_ticks=60, drain_ticks=10))
        self.tick = 0
        self.boundary_reward = 0.0
        self.done = self.closed = False

    def initialize(self):
        self.world.ledger.spawn(0, 0)
        self.world.ledger.spawn(1, 0)
        return self.observe()

    def observe(self):
        return PolicyObservation(
            self.tick,
            np.zeros((8, 8, 3), np.uint8),
            np.zeros((8, 8, 3), np.uint8),
            np.zeros(16, np.float32),
        )

    def step_formal(self, action):
        self.tick += 1
        self.ledger.time_us += 20000
        self.boundary_reward = float(self.tick in (10, 60))
        if self.tick == 10:
            self.world.ledger.finish(0, "success", self.tick)
        if self.tick == 60:
            self.world.ledger.finish(1, "success", self.tick)
        self.done = self.world.ledger.advance(self.tick)
        return self.observe()

    def close(self):
        self.closed = True


def run(runtime, **kwargs):
    from latency_meta_mdp.runtime.conveyor_evaluation import run_conveyor_policy_episode

    return run_conveyor_policy_episode(
        runtime=runtime,
        policy=lambda observation, context: np.zeros((50, 7)),
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: 4,
        policy_alignment="observation_time",
        **kwargs,
    )


def test_parcel_delivery_does_not_terminate_episode_or_repeat_reward():
    runtime = ConveyorFixture()
    frames = []
    report = run(
        runtime, maximum_steps=100, record_observation=lambda obs: frames.append(obs.formal_tick)
    )
    assert report["total_reward"] == 2
    assert report["executed_steps"] == 60
    assert report["parcels"]["successes"] == 2
    assert report["terminated"] and not report["truncated"]
    assert report["policy_calls"] > 0
    assert report["starvation_ticks"] == 0
    assert frames == list(range(61))
    assert runtime.closed


def test_evaluation_cutoff_preserves_unresolved_parcels():
    report = run(ConveyorFixture(), maximum_steps=30)
    assert report["total_reward"] == 1
    assert report["truncated"] and not report["terminated"]
    assert report["parcels"]["active"] == 1
    assert report["parcels"]["timeouts"] == 0


def test_evaluation_closes_owned_simulator_on_recording_failure():
    runtime = ConveyorFixture()

    def fail(observation):
        raise RuntimeError("recording failed")

    with pytest.raises(RuntimeError, match="recording failed"):
        run(runtime, maximum_steps=100, record_observation=fail)
    assert runtime.closed


def test_five_tick_diagnostic_replans_at_actual_five_tick_intervals():
    from latency_meta_mdp.runtime.policy_execution import FixedCursorScheduler

    report = run(
        ConveyorFixture(),
        maximum_steps=30,
        scheduler=FixedCursorScheduler(5),
        minimum_decision_tick=0,
    )
    launches = [row["launch_formal_tick"] for row in report["request_ledger"]]
    assert launches == [5, 10, 15, 20, 25]


@pytest.mark.parametrize("protocol", ["rtc", "sharp"])
def test_conditioned_conveyor_uses_native_bootstrap_and_counts_forecasts(protocol):
    from latency_meta_mdp.data.forecast.samples import DecodedForecast
    from latency_meta_mdp.runtime.action_chunk_client import load_action_chunk_client_config
    from latency_meta_mdp.runtime.conveyor_evaluation import run_conveyor_policy_episode
    from latency_meta_mdp.runtime.rtc_protocol import RollingDelayHistory

    class Provider:
        def observe(self, observation, previous_action):
            self.latest_tick = observation.formal_tick

        def predict(self, context):
            assert context.origin_tick == self.latest_tick
            return DecodedForecast(
                context.origin_tick,
                context.origin_tick + context.estimated_delay_ticks,
                context.estimated_delay_ticks,
                context.buffer_version,
                np.zeros((2, 224, 224, 3), np.uint8),
                np.zeros(16, np.float32),
            )

    def actor(observation, context):
        assert context.forecast.available
        return np.zeros((50, 7))

    rtc = protocol == "rtc"
    client = (
        load_rtc_client_config(Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml"))
        if rtc
        else load_action_chunk_client_config(
            Path("configs/runtime/client/sharp_return_time_h50_e25_v1.yaml")
        )
    )
    report = run_conveyor_policy_episode(
        runtime=ConveyorFixture(),
        policy=actor,
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        forecast_provider=Provider(),
        forecast_delay_history=(
            None if rtc else RollingDelayHistory(capacity=10, initial_delays=(4,))
        ),
        client_config=client,
        delay_sampler=lambda: 4,
        maximum_steps=100,
        policy_alignment="observation_time" if rtc else None,
    )
    assert report["bootstrap_calls"] == 1
    assert report["forecast_calls"] == report["forecast_decodes"] == report["policy_calls"] > 0
    assert report["parcels"]["successes"] == 2
    assert report["starvation_ticks"] == 0
    assert all(row["forecast_available"] for row in report["request_ledger"])
