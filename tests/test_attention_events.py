from pathlib import Path

import numpy as np
from test_action_conditioned_jepa_rollout import _normalization
from test_conveyor_policy_evaluation import ConveyorFixture
from test_information_perception import Decoder, Encoder, Predictor


def test_runtime_publishes_paid_history_before_decision_and_records_forced_stages():
    from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
    from latency_meta_mdp.meta.information.behaviors import Decision
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec
    from latency_meta_mdp.meta.information.perception import PerceptionAccess
    from latency_meta_mdp.meta.information.records import StageRecorder
    from latency_meta_mdp.runtime.information_execution import run_information_episode
    from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config

    spec = ActionSpec()
    recorder = StageRecorder("events", action_spec=spec)
    encoder = Encoder()
    perception = PerceptionAccess(
        FrozenForecastEngine(Predictor(), Decoder(), device="cpu"),
        encoder,
        _normalization(),
        recorder,
    )

    class Scheduler:
        component = "meta_q"

        def reset(self, *, episode_id):
            self.events = []
            self.snapshots = []

        def ingest(self, event):
            self.events.append(event)

        def decide(self, packet, mask):
            self.snapshots.append((packet, tuple(self.events)))
            return Decision(spec.id_for("observe" if packet.stage == "G" else "launch"), 1.0)

    scheduler = Scheduler()
    result = run_information_episode(
        runtime=ConveyorFixture(),
        policy=lambda obs, ctx: np.zeros((50, 7)),
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        perception=perception,
        scheduler=scheduler,
        recorder=recorder,
        client_config=load_rtc_client_config(
            Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        ),
        delay_sampler=lambda: 4,
        maximum_steps=100,
        action_spec=spec,
    )
    first = next(e for e in scheduler.events if e.kind == "observe")
    assert first.represented_ticks == (2, 6, 10)
    assert first.payload["visual"].shape == (3, 2, 196, 384)
    assert encoder.batches[0] == 6
    first_p, events = next(s for s in scheduler.snapshots if s[0].stage == "P")
    assert {"observe", "predict"}.issubset({e.kind for e in events})
    assert all(e.available_at < first_p.decision_key for e in events)
    assert all(
        e.available_at < packet.decision_key for packet, es in scheduler.snapshots for e in es
    )
    assert len([e for e in scheduler.events if e.kind == "execute"]) == result["executed_steps"]
    assert any(e.kind == "arrival" for e in scheduler.events)
    assert len(recorder.events) == len(scheduler.events)
    assert all(r["forced"] for r in recorder.rows if r["packet"].stage == "O")
