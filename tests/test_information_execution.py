from pathlib import Path

import numpy as np
import pytest
from test_action_conditioned_jepa_rollout import _normalization
from test_conveyor_policy_evaluation import ConveyorFixture
from test_information_perception import Decoder, Encoder, Predictor

from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
from latency_meta_mdp.meta.information.behaviors import RuleScheduler
from latency_meta_mdp.meta.information.perception import PerceptionAccess
from latency_meta_mdp.meta.information.records import StageRecorder, decision_view
from latency_meta_mdp.runtime.information_execution import run_information_episode
from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config


@pytest.mark.parametrize("delay", [0, 4, 20])
def test_fixed_anchor_matches_existing_rtc_and_conserves_reward(delay):
    recorder = StageRecorder("test")
    encoder = Encoder()
    access = PerceptionAccess(
        FrozenForecastEngine(Predictor(), Decoder(), device="cpu"),
        encoder,
        _normalization(),
        recorder,
    )
    runtime = ConveyorFixture()

    def policy(obs, context):
        # Unique tick-indexed controls reveal any wrong returned-chunk cursor.
        return np.tile(np.arange(50)[:, None] / 100 + context.origin_tick / 1000, (1, 7))

    report = run_information_episode(
        runtime=runtime,
        policy=policy,
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        perception=access,
        scheduler=RuleScheduler("periodic", period=25),
        recorder=recorder,
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: delay,
        maximum_steps=100,
    )
    # Existing RTC E25's observation-time clock launches at 25, 50 for all delays.
    assert [r["origin_tick"] for r in report["requests"]] == [25, 50]
    assert report["starvation_ticks"] == 0
    assert report["task_reward"] == pytest.approx(0.2)
    assert report["parcels"]["successes"] == 2
    assert recorder.prefix_reward == 0.1
    assert sum(r["task_reward"] for r in decision_view(recorder.rows)) == pytest.approx(0.1)
    assert runtime.closed
    assert encoder.batches == [6, 6]
    assert all(r["packet"].stage != "O" or r["forced"] for r in recorder.rows)
    from latency_meta_mdp.runtime.conveyor_evaluation import run_conveyor_policy_episode

    old = run_conveyor_policy_episode(
        runtime=ConveyorFixture(),
        policy=policy,
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: delay,
        maximum_steps=100,
        policy_alignment="observation_time",
    )
    np.testing.assert_array_equal(report["executed_actions"], old["executed_actions"])
    assert [r["origin_tick"] for r in report["requests"]] == [
        r["launch_formal_tick"] for r in old["request_ledger"]
    ]


def test_rule_actions_respect_masks_and_probability():
    from latency_meta_mdp.meta.information.schema import ActorPacket, legal_actions

    for kind in ("periodic", "active", "lazy", "exploration"):
        scheduler = RuleScheduler(kind, seed=123)
        for remaining in (20, 21, 24, 40):
            packet = ActorPacket(
                10, "G", plan_age=50 - remaining, buffer_mask=np.arange(50) < remaining
            )
            mask = legal_actions("G", remaining)
            decision = scheduler.decide(packet, mask)
            assert mask[decision.action] and 0 < decision.probability <= 1


def test_policy_outputs_use_existing_controller_projection():
    recorder = StageRecorder("projection")
    access = PerceptionAccess(
        FrozenForecastEngine(Predictor(), Decoder(), device="cpu"),
        Encoder(),
        _normalization(),
        recorder,
    )
    report = run_information_episode(
        runtime=ConveyorFixture(),
        policy=lambda obs, ctx: np.full((50, 7), 3.0),
        bootstrap_policy=lambda obs: np.full((50, 7), 2.0),
        perception=access,
        scheduler=RuleScheduler("periodic"),
        recorder=recorder,
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: 4,
        maximum_steps=100,
    )
    assert report["action_limit_projections"] == 3 * 50 * 7
    np.testing.assert_array_equal(report["executed_actions"], np.ones((60, 7)))


def test_active_requests_at_every_available_boundary_and_exploration_balances_gate():
    from latency_meta_mdp.meta.information.schema import ActorPacket, legal_actions

    packet = ActorPacket(10, "G", plan_age=0)
    mask = legal_actions("G", 50)
    assert RuleScheduler("active").decide(packet, mask).action == 5
    scheduler = RuleScheduler("exploration", seed=5)
    for _ in range(20):
        decision = scheduler.decide(packet, mask)
        assert decision.probability == pytest.approx(0.5 if decision.action == 5 else 0.1)
