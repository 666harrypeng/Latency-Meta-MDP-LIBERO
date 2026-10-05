import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.data.forecast.samples import DecodedForecast
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.runtime.action_chunk_client import load_action_chunk_client_config
from latency_meta_mdp.runtime.policy_execution import (
    FixedCursorScheduler,
    LogicalPolicyRuntime,
    PhysicalStepResult,
    PolicyObservation,
)
from latency_meta_mdp.runtime.rtc_protocol import RollingDelayHistory, RtcInferenceContext


def observation(tick):
    return PolicyObservation(
        tick, np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8), np.zeros(16)
    )


@pytest.mark.parametrize("delay", [0, 2, 12])
def test_sharp_forecast_keeps_index_zero_e25_and_only_completed_delays(delay):
    clock = SimpleNamespace(tick=0)
    seen, executed = [], []

    class Provider:
        def observe(self, obs, previous):
            pass

        def predict(self, context):
            seen.append(context)
            assert not hasattr(context, "realized_delay_ticks")
            return DecodedForecast(
                context.origin_tick,
                context.origin_tick + context.estimated_delay_ticks,
                context.estimated_delay_ticks,
                context.buffer_version,
                np.zeros((2, 224, 224, 3), np.uint8),
                np.zeros(16),
            )

    def policy(obs, context):
        assert isinstance(context, RtcInferenceContext)
        assert context.forecast is not None
        return np.repeat((np.arange(50) / 100 + 0.1)[:, None], 7, axis=1)

    engine = LogicalPolicyRuntime(
        action_contract=load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
        ),
        client_config=load_action_chunk_client_config(
            Path("configs/runtime/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        simulation_time_reader=lambda: clock.tick * 20000,
        delay_sampler=lambda: delay,
        policy=policy,
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        scheduler=FixedCursorScheduler(25),
        forecast_provider=Provider(),
        forecast_delay_history=RollingDelayHistory(capacity=32, initial_delays=(4,) * 5),
        monotonic_ns=itertools.count(0, 10).__next__,
    )
    engine.bootstrap(lambda: observation(0))

    def execute(action):
        executed.append(action.copy())
        clock.tick += 1
        return PhysicalStepResult(0, clock.tick == 76)

    while clock.tick < 76:
        engine.step(
            formal_tick=clock.tick, observe=lambda: observation(clock.tick), execute=execute
        )
    assert [c.origin_tick for c in seen][:2] == [25, 50 + delay]
    assert [c.estimated_delay_ticks for c in seen][:2] == [4, max(4, delay)]
    np.testing.assert_allclose(executed[25 + delay], 0.1)
    np.testing.assert_allclose(executed[26 + delay], 0.11)
    assert all(e["installed_index"] == 0 for e in engine.events if e["stage"] == "chunk_install")


@pytest.mark.parametrize("forecast_only", [False, True])
def test_sharp_bridge_changes_inputs_without_guidance_or_prefix_stitching(forecast_only):
    from latency_meta_mdp.runtime.sharp_forecast_policy import SharpForecastPolicy

    class Native:
        def __call__(self, obs, packet):
            self.obs, self.packet = obs, packet
            return {"actions": np.full((50, 7), 0.5)}

    native = Native()
    forecast = DecodedForecast(25, 32, 7, 0, np.full((2, 224, 224, 3), 42, np.uint8), np.ones(16))
    context = RtcInferenceContext(
        0, 25, observation(25), 0, np.zeros((50, 7)), np.arange(50) < 25, 7, forecast
    )
    actor = SharpForecastPolicy(native, forecast_only=forecast_only)
    result = actor(observation(25), context)
    np.testing.assert_array_equal(result["actions"], 0.5)
    assert native.obs.formal_tick == (32 if forecast_only else 25)
    if forecast_only:
        np.testing.assert_array_equal(native.obs.image, 42)
        np.testing.assert_array_equal(native.obs.state, 1)
        assert native.packet is None
    else:
        assert set(native.packet) == {"rgb", "proprio", "query_ticks"}
    # An unavailable forecast uses the real current observation, never padding RGB.
    from dataclasses import replace

    missing = DecodedForecast(25, 32, 7, 0, None, None)
    actor(observation(25), replace(context, forecast=missing))
    assert native.obs.formal_tick == 25


def test_openpi_sharp_bridge_bootstrap_accepts_observation_without_condition():
    from latency_meta_mdp.runtime.policy_execution import InProcessOpenpiPolicy

    seen = []
    bridge = object.__new__(InProcessOpenpiPolicy)
    bridge.noise_rng = np.random.default_rng(0)
    bridge.belief_input_key = "forecast"
    bridge.policy = SimpleNamespace(
        infer=lambda inputs, **kwargs: seen.append(inputs) or {"actions": np.zeros((50, 7))}
    )
    bridge(observation(0))
    assert "forecast" not in seen[0]
    assert "actions" not in seen[0]
    assert set(seen[0]) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
