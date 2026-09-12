from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.action_chunk_client import load_action_chunk_client_config
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.policy_execution import PolicyObservation


class Runtime:
    def __init__(self, succeed_at=None):
        self.action_contract = load_action_contract(
            Path("configs/control/panda_osc_pose_delta_v1.yaml")
        )
        self.executor = self
        self.ledger = SimpleNamespace(time_us=0)
        self.tracker = SimpleNamespace(status=SimpleNamespace(value="running"))
        self.tick = 0
        self.succeed_at = succeed_at
        self.actions = []
        self.closed = False

    def initialize(self):
        return self.snapshot()

    def snapshot(self):
        return SimpleNamespace(formal_tick_index=self.tick)

    def step_formal(self, action):
        self.actions.append(action.copy())
        self.tick += 1
        self.ledger.time_us += 20_000
        if self.tick == self.succeed_at:
            self.tracker.status.value = "success"
        return self.snapshot()

    def close(self):
        self.closed = True


def observe(snapshot):
    return PolicyObservation(
        snapshot.formal_tick_index,
        np.zeros((8, 8, 3), np.uint8),
        np.zeros((8, 8, 3), np.uint8),
        np.zeros(16, np.float32),
    )


def test_evaluation_executes_closed_loop_until_task_success(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    runtime = Runtime(succeed_at=60)
    observed_ticks = []

    def policy(observation, belief):
        assert belief is None
        observed_ticks.append(observation.formal_tick)
        return {"actions": np.full((50, 7), 1.2)}

    report = evaluation.run_native_policy_episode(
        runtime=runtime,
        policy=policy,
        client_config=load_action_chunk_client_config(
            Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        delay_sampler=lambda: 4,
        maximum_steps=100,
    )
    assert report["success"] and not report["truncated"]
    assert report["executed_steps"] == 60 and report["completion_time_seconds"] == 1.2
    assert observed_ticks == [0, 25, 54]
    assert report["starvation_ticks"] == 0
    assert report["action_limit_projections"] > 0
    assert np.max(runtime.actions) == 1.0
    assert runtime.closed


def test_evaluation_time_limit_is_not_success(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    runtime = Runtime()
    report = evaluation.run_native_policy_episode(
        runtime=runtime,
        policy=lambda observation, belief: np.zeros((50, 7)),
        client_config=load_action_chunk_client_config(
            Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        delay_sampler=lambda: 0,
        maximum_steps=5,
    )
    assert report["truncated"] and not report["success"]
    assert report["task_status"] == "time_limit" and report["completion_time_seconds"] is None
    assert report["elapsed_simulation_seconds"] == 0.1
    assert runtime.closed


@pytest.mark.parametrize("succeed_at, expected", [(None, False), (30, True)])
def test_declared_task_deadline_is_terminal_and_preserves_final_tick_success(
    monkeypatch, succeed_at, expected
):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    transitions = []
    result = evaluation.run_native_policy_episode(
        runtime=Runtime(succeed_at=succeed_at),
        policy=lambda observation, belief: np.zeros((50, 7)),
        client_config=load_action_chunk_client_config(
            Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        delay_sampler=lambda: 10,
        maximum_steps=30,
        task_horizon_terminal=True,
        transition_sink=transitions.append,
    )
    assert result["success"] is expected
    assert result["terminated"] and not result["truncated"]
    assert result["executed_steps"] == 30
    assert transitions[-1].terminated and transitions[-1].bootstrap_discount == 0
    assert transitions[-1].undiscounted_reward == float(expected)


def test_evaluation_closes_simulator_on_invalid_policy_output(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    runtime = Runtime()
    with pytest.raises(ValueError, match="H50"):
        evaluation.run_native_policy_episode(
            runtime=runtime,
            policy=lambda observation, belief: np.zeros((49, 7)),
            client_config=load_action_chunk_client_config(
                Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
            ),
            delay_sampler=lambda: 0,
            maximum_steps=5,
        )
    assert runtime.closed


def test_recording_has_both_initial_and_terminal_formal_boundaries(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    recorded = []
    report = evaluation.run_native_policy_episode(
        runtime=Runtime(succeed_at=3),
        policy=lambda observation, belief: np.zeros((50, 7)),
        client_config=load_action_chunk_client_config(
            Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        delay_sampler=lambda: 0,
        record_observation=lambda observation: recorded.append(observation.formal_tick),
    )
    assert recorded == [0, 1, 2, 3]
    assert report["recorded_frames"] == 4
    assert report["success"]


def test_conditioned_episode_uses_native_bootstrap_and_actual_buffer(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    observed, queried, native = [], [], []

    class Belief:
        def observe(self, observation, previous_action):
            observed.append(observation.formal_tick)

        def __call__(self, observation, controls):
            assert controls.shape == (20, 7)
            queried.append(observation.formal_tick)
            return observation.formal_tick

    def bootstrap(observation):
        native.append(observation.formal_tick)
        return np.zeros((50, 7))

    def conditioned(observation, belief):
        assert belief == observation.formal_tick
        return np.ones((50, 7))

    result = evaluation.run_native_policy_episode(
        runtime=Runtime(succeed_at=60),
        policy=conditioned,
        bootstrap_policy=bootstrap,
        belief_provider_factory=lambda runtime, snapshot: Belief(),
        client_config=load_action_chunk_client_config(
            Path("configs/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        delay_sampler=lambda: 4,
    )
    assert result["success"] and result["belief_calls"] == 2
    assert native == [0] and queried == [25, 54]
    assert observed == list(range(60))


def test_forecast_evaluation_records_real_history_and_request_context(monkeypatch):
    from latency_meta_mdp import policy_evaluation as evaluation
    from latency_meta_mdp.policy_forecast import DecodedForecast
    from latency_meta_mdp.rtc_protocol import load_rtc_client_config

    monkeypatch.setattr(evaluation, "policy_observation_from_snapshot", observe)
    runtime = Runtime(succeed_at=70)

    class Provider:
        def __init__(self):
            self.ticks = []
            self.queries = []

        def observe(self, observation, previous_action):
            self.ticks.append(observation.formal_tick)
            if observation.formal_tick > 0:
                np.testing.assert_array_equal(previous_action, runtime.actions[-1])

        def predict(self, context):
            assert self.ticks[-1] == context.origin_tick
            self.queries.append(context.origin_tick)
            return DecodedForecast(
                context.origin_tick, context.origin_tick + context.estimated_delay_ticks,
                context.estimated_delay_ticks, context.buffer_version, None, None,
            )

    provider = Provider()

    def policy(observation, context):
        assert context.forecast.source_tick == observation.formal_tick
        return {"actions": np.zeros((50, 7))}

    result = evaluation.run_native_policy_episode(
        runtime=runtime, policy=policy,
        bootstrap_policy=lambda observation: {"actions": np.zeros((50, 7))},
        forecast_provider=provider,
        client_config=load_rtc_client_config(Path("configs/client/rtc_observation_time_h50_v1.yaml")),
        delay_sampler=lambda: 4, maximum_steps=100, policy_alignment="observation_time",
    )
    assert result["success"] and runtime.closed
    assert result["forecast_calls"] == len(provider.queries) > 0
    assert provider.ticks == list(range(70))


def test_rtc_native_bridge_can_bootstrap_with_one_observation():
    from latency_meta_mdp.policy_execution import InProcessRtcOpenpiPolicy

    bridge = InProcessRtcOpenpiPolicy.__new__(InProcessRtcOpenpiPolicy)
    bridge.uses_forecast = False
    bridge.noise_rng = np.random.default_rng(3)
    calls = []

    def infer(inputs, *, noise):
        calls.append(inputs)
        assert noise.shape == (50, 32)
        return {"actions": np.zeros((50, 7))}

    bridge.policy = SimpleNamespace(infer=infer)
    observation = observe(Runtime().snapshot())
    assert bridge(observation)["actions"].shape == (50, 7)
    assert len(calls) == 1
    bridge.uses_forecast = True
    with pytest.raises(ValueError, match="native bootstrap"):
        bridge(observation)
    assert len(calls) == 1
