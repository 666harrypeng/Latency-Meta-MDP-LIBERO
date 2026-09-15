import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.data.forecast.samples import DecodedForecast
from latency_meta_mdp.runtime.policy_execution import (
    LogicalPolicyRuntime,
    PhysicalStepResult,
    PolicyObservation,
)
from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext, load_rtc_client_config


def actions(base):
    value = np.zeros((50, 7))
    value[:, 0] = base + np.arange(50) / 100
    value[:, -1] = -1
    return value


def observation(h):
    return PolicyObservation(
        h, np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8), np.zeros(16)
    )


def forecast(context, available=True):
    return DecodedForecast(
        context.origin_tick,
        context.origin_tick + context.estimated_delay_ticks,
        context.estimated_delay_ticks,
        context.buffer_version,
        np.full((2, 224, 224, 3), 120, np.uint8) if available else None,
        np.arange(16, dtype=np.float32) if available else None,
    )


def context(q, *, available=True):
    c = RtcInferenceContext(3, 25, observation(25), 7, actions(0), np.arange(50) < 25, q)
    return replace(c, forecast=forecast(c, available))


class Native:
    uses_forecast = False

    def __init__(self):
        self.calls = []

    def __call__(self, obs, info=None):
        self.calls.append((obs, info))
        return {"actions": actions(0 if info is None else 0.5)}


@pytest.mark.parametrize("q", [0, 4, 8, 20])
def test_future_observation_and_guidance_share_future_origin(q):
    from latency_meta_mdp.runtime.planned_handoff_policy import PlannedHandoffPolicy

    c = context(q)
    native = Native()
    result = PlannedHandoffPolicy(native, mode="forecast")(c.observation, c)
    obs, info = native.calls[0]
    assert obs.formal_tick == info.origin_tick == 25 + q
    np.testing.assert_array_equal(obs.image, c.forecast.rgb[0])
    np.testing.assert_array_equal(obs.wrist_image, c.forecast.rgb[1])
    np.testing.assert_array_equal(obs.state, c.forecast.proprio)
    assert obs.prompt == c.observation.prompt and info.forecast is None
    assert info.estimated_delay_ticks == 0
    np.testing.assert_array_equal(info.previous_actions[: 25 - q], c.previous_actions[q:25])
    np.testing.assert_array_equal(info.previous_action_mask, np.arange(50) < 25 - q)
    np.testing.assert_array_equal(result["actions"][:q], c.previous_actions[:q])
    np.testing.assert_array_equal(result["actions"][q:], actions(0.5)[: 50 - q])
    assert result["handoff"]["policy_input_tick"] == 25 + q
    assert result["handoff"]["planned_handoff_tick"] == 25 + q
    assert result["handoff"]["forecast_used"] is True


@pytest.mark.parametrize("mode,available", [("current", True), ("forecast", False)])
def test_current_control_and_missing_forecast_keep_source_time_suffix(mode, available):
    from latency_meta_mdp.runtime.planned_handoff_policy import PlannedHandoffPolicy

    c = context(8, available=available)
    if mode == "current":
        c = replace(c, forecast=None)
    native = Native()
    result = PlannedHandoffPolicy(native, mode=mode)(c.observation, c)
    obs, info = native.calls[0]
    assert obs.formal_tick == info.origin_tick == 25
    assert info.estimated_delay_ticks == 8 and info.forecast is None
    np.testing.assert_array_equal(result["actions"][:8], c.previous_actions[:8])
    np.testing.assert_array_equal(result["actions"][8:], actions(0.5)[8:])
    assert result["handoff"]["forecast_used"] is False


def test_bootstrap_remains_native_and_invalid_context_never_calls_policy():
    from latency_meta_mdp.runtime.planned_handoff_policy import PlannedHandoffPolicy

    native = Native()
    policy = PlannedHandoffPolicy(native, mode="forecast")
    np.testing.assert_array_equal(policy(observation(0))["actions"], actions(0))
    for bad in [
        replace(context(8), previous_action_mask=np.arange(50) < 7),
        replace(context(8), forecast=None),
    ]:
        with pytest.raises(ValueError):
            policy(bad.observation, bad)
    with pytest.raises(ValueError):
        policy(observation(26), context(8))
    assert len(native.calls) == 1


def test_no_overlap_after_handoff_uses_native_sampling_without_fake_guidance():
    from latency_meta_mdp.runtime.planned_handoff_policy import PlannedHandoffPolicy

    c = replace(context(20), previous_action_mask=np.arange(50) < 20)
    native = Native()
    result = PlannedHandoffPolicy(native, mode="forecast")(c.observation, c)
    assert native.calls[0][1] is None
    assert native.calls[0][0].formal_tick == 45
    np.testing.assert_array_equal(result["actions"][:20], c.previous_actions[:20])
    assert result["handoff"]["forecast_used"] is True


@pytest.mark.parametrize("delay", [0, 4, 8, 12, 20])
def test_real_client_preserves_prefix_until_handoff_and_skips_only_late_future_actions(delay):
    from latency_meta_mdp.envs.control import load_action_contract
    from latency_meta_mdp.runtime.planned_handoff_policy import PlannedHandoffPolicy

    class Provider:
        def observe(self, obs, previous):
            pass

        def predict(self, c):
            return forecast(c)

    ticks = [0]
    native = Native()
    engine = LogicalPolicyRuntime(
        action_contract=load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
        ),
        client_config=replace(
            load_rtc_client_config(Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")),
            initial_delay_ticks=(8,),
        ),
        simulation_time_reader=lambda: ticks[0] * 20000,
        delay_sampler=lambda: delay,
        policy=PlannedHandoffPolicy(native, mode="forecast"),
        bootstrap_policy=native,
        scheduler=lambda state, obs, belief: obs.formal_tick == 10,
        forecast_provider=Provider(),
        policy_alignment="observation_time",
        minimum_decision_tick=0,
        shield_latest_launch=False,
        monotonic_ns=itertools.count(0, 10).__next__,
    )
    engine.bootstrap(lambda: observation(0))
    executed = []

    def execute(action):
        executed.append(action.copy())
        ticks[0] += 1
        return PhysicalStepResult(0, ticks[0] == 35)

    while not engine.terminated:
        engine.step(formal_tick=ticks[0], observe=lambda: observation(ticks[0]), execute=execute)
    first_future = max(18, 10 + delay)
    np.testing.assert_array_equal(np.array(executed)[:first_future], actions(0)[:first_future])
    np.testing.assert_array_equal(
        np.array(executed)[first_future:], actions(0.5)[first_future - 18 : 35 - 18]
    )
    ledger = engine.request_ledger()[0]
    assert ledger["arrival_formal_tick"] == 10 + delay
    assert ledger["planned_handoff_tick"] == 18
    assert ledger["policy_input_tick"] == 18
    assert ledger["handoff_forecast_used"] is True
    assert engine.summary()["starvation_ticks"] == 0
