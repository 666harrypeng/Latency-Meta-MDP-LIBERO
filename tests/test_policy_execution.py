import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _observation(tick):
    from latency_meta_mdp.runtime.policy_execution import PolicyObservation

    return PolicyObservation(
        formal_tick=tick,
        image=np.zeros((8, 8, 3), np.uint8),
        wrist_image=np.zeros((8, 8, 3), np.uint8),
        state=np.zeros(16, np.float32),
    )


def _runtime(*, scheduler, interval=1, shield=False, belief=None, probabilities=None):
    from latency_meta_mdp.envs.control import load_action_contract
    from latency_meta_mdp.runtime.action_chunk_client import load_action_chunk_client_config
    from latency_meta_mdp.runtime.policy_execution import LogicalPolicyRuntime, PhysicalStepResult

    clock = SimpleNamespace(tick=0)

    def policy(obs, info):
        actions = np.zeros((50, 7))
        actions[:, -1] = -1
        return actions

    runtime = LogicalPolicyRuntime(
        action_contract=load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
        ),
        client_config=load_action_chunk_client_config(
            Path("configs/runtime/client/sharp_return_time_h50_e25_v1.yaml")
        ),
        simulation_time_reader=lambda: clock.tick * 20_000,
        delay_sampler=lambda: 3,
        policy=policy,
        bootstrap_policy=lambda obs: policy(obs, None),
        scheduler=scheduler,
        belief_provider=belief,
        policy_uses_belief=belief is not None,
        decision_interval_ticks=interval,
        minimum_decision_tick=0,
        shield_latest_launch=shield,
        monotonic_ns=itertools.count(0, 10).__next__,
        gamma=0.9,
        latency_probabilities=probabilities,
    )
    runtime.bootstrap(lambda: _observation(0))

    def execute(action):
        clock.tick += 1
        return PhysicalStepResult(reward=float(clock.tick == 8), terminated=clock.tick == 8)

    return runtime, clock, execute


def test_known_latency_law_is_available_without_future_belief():
    law = np.zeros(20, np.float32)
    law[2] = 1
    seen = []

    def scheduler(state, observation, belief):
        seen.append(state.latency_probabilities)
        assert belief is None
        return False

    runtime, clock, execute = _runtime(scheduler=scheduler, probabilities=law)
    runtime.step(formal_tick=0, observe=lambda: _observation(0), execute=execute)
    np.testing.assert_array_equal(seen[0], law)
    assert not seen[0].flags.writeable


def test_logical_runtime_records_variable_duration_transitions_and_request_stages():
    from latency_meta_mdp.runtime.policy_execution import ImmediateLaunchScheduler

    runtime, clock, execute = _runtime(scheduler=ImmediateLaunchScheduler())
    while clock.tick < 8:
        runtime.step(
            formal_tick=clock.tick, observe=lambda: _observation(clock.tick), execute=execute
        )
    assert runtime.terminated
    assert [t.duration_ticks for t in runtime.transitions] == [3, 3, 2]
    assert runtime.transitions[-1].bootstrap_discount == 0
    assert runtime.transitions[-1].reward == 0.9
    assert runtime.summary()["policy_calls"] == 3
    assert runtime.summary()["bootstrap_calls"] == 1
    stages = [e["stage"] for e in runtime.events]
    assert "observation" in stages and "meta_decision" in stages
    assert stages.count("policy_launch") == stages.count("policy_return") == 3
    assert runtime.summary()["latency_mode"] == "controlled_logical_policy_delay"
    assert runtime.summary()["concurrent_deployment_verified"] is False
    requests = runtime.request_ledger()
    assert len(requests) == 3
    assert requests[0]["arrival_formal_tick"] == 3
    assert requests[0]["policy_return_ns"] >= requests[0]["policy_launch_ns"]
    assert requests[0]["chunk_install_ns"] >= requests[0]["policy_return_ns"]
    assert requests[2]["chunk_install_ns"] is None
    assert requests[2]["realized_delay_ticks"] is None


def test_fixed_scheduler_wait_does_not_pay_for_unused_belief_refresh():
    from latency_meta_mdp.runtime.policy_execution import FixedCursorScheduler

    class Belief:
        calls = 0

        def observe(self, observation, previous_action):
            pass

        def __call__(self, observation, controls):
            self.calls += 1
            return {"source_tick": observation.formal_tick}

    belief = Belief()
    runtime, clock, execute = _runtime(scheduler=FixedCursorScheduler(4), belief=belief)
    while clock.tick < 8:
        runtime.step(
            formal_tick=clock.tick, observe=lambda: _observation(clock.tick), execute=execute
        )
    assert belief.calls == 1
    assert runtime.summary()["belief_calls"] == 1


def test_snapshot_packing_excludes_privileged_state():
    from latency_meta_mdp.runtime.policy_execution import policy_observation_from_snapshot

    snapshot = SimpleNamespace(
        formal_tick_index=2,
        robot_qpos=np.zeros(7),
        robot_qvel=np.ones(7),
        robot_gripper_qpos=np.array([0.02, -0.02]),
        robot_gripper_qvel=np.array([0.01, -0.01]),
        cameras={
            k: SimpleNamespace(rgb=np.zeros((8, 8, 3), np.uint8))
            for k in ("agentview", "robot0_eye_in_hand")
        },
        object_qpos=np.ones(7) * 999,
        phase="privileged",
    )
    obs = policy_observation_from_snapshot(snapshot)
    np.testing.assert_allclose(obs.state, [0] * 7 + [1] * 7 + [0.04, 0.02])
    assert not hasattr(obs, "object_qpos") and not hasattr(obs, "phase")
    assert set(obs.to_policy_inputs()) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    snapshot.cameras["agentview"].source_formal_tick = 1
    with pytest.raises(ValueError, match="camera"):
        policy_observation_from_snapshot(snapshot)


def test_latent_threshold_reference_updates_on_actual_shielded_launches():
    from latency_meta_mdp.runtime.policy_execution import LatentChangeScheduler

    scheduler = LatentChangeScheduler(
        feature_reader=lambda obs: np.full(2, obs.formal_tick, dtype=float), threshold=2
    )
    assert scheduler(None, _observation(0), None)
    scheduler.on_launch(None, _observation(0), None)
    assert not scheduler(None, _observation(1), None)
    # The runtime can force this launch; the reference must still move.
    scheduler.on_launch(None, _observation(1), None)
    assert not scheduler(None, _observation(2), None)
    assert scheduler(None, _observation(3), None)
