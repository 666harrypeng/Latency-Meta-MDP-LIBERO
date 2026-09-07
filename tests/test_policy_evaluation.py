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
