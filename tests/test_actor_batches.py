import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from attention_fixtures import small_config

from latency_meta_mdp.meta.information.actor_batches import on_policy_auxiliary
from latency_meta_mdp.meta.information.actor_deployment import ActorScheduler
from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, contexts_from_raw
from latency_meta_mdp.meta.information.memory import collate_contexts
from latency_meta_mdp.meta.information.memory_schema import AttentionPacket, PublicEvent
from latency_meta_mdp.meta.information.records import StageRecorder


def test_live_actor_history_roundtrips_after_forced_event_and_rejects_stale_probability():
    config = small_config()
    model = AttentionStageQ(config)
    with torch.no_grad():
        for head in model.heads:
            head.weight.normal_(std=0.03)
    scheduler = ActorScheduler(model, deterministic=False, seed=1)
    scheduler.reset(episode_id="roundtrip")
    recorder = StageRecorder("roundtrip", action_spec=model.action_spec)
    packet = AttentionPacket(10, "G", decision_key=(10, 1))
    legal = model.action_spec.legal("G", packet.remaining, forecast_fresh=False)
    decision = scheduler.decide(packet, legal)
    recorder.start(packet, decision.action, forced=False, probability=decision.probability)
    event = PublicEvent(
        "obs",
        "observe",
        (10, 2),
        (2, 6, 10),
        dict(visual=np.ones((3, 2, 196, 384), np.float32), proprio=np.zeros((3, 16))),
    )
    scheduler.ingest(event)
    recorder.events.append(event)
    recorder.start(
        replace(packet, stage="O", decision_key=(10, 3)), 9, forced=True, probability=1.0
    )
    packet = replace(packet, tick=11, decision_key=(11, 1))
    decision = scheduler.decide(packet, legal)
    recorder.start(packet, decision.action, forced=False, probability=decision.probability)
    recorder.add_reward(11, 1)
    recorder.finish(12, terminated=True)
    rows = contexts_from_raw(
        dict(episode_id="roundtrip", rows=recorder.rows, events=recorder.events),
        model.action_spec,
        config=config["memory"],
    )
    replay = DecisionReplay([rows])
    with torch.no_grad():
        logits = model(
            collate_contexts([r["context"] for r in rows], config=config["memory"])
        ).numpy()
    scores = dict(old_logits=logits, old_values=np.array([0.05, 0.03], np.float32))
    ep = dict(behavior="actor", deterministic=False)
    summary = dict(scheduler_diagnostics=scheduler.diagnostics)
    auxiliary = on_policy_auxiliary(replay, scores, [ep], [summary], actor_hash=scheduler.version)
    np.testing.assert_allclose(auxiliary["returns"], [0.1, 0.1])
    np.testing.assert_allclose(auxiliary["raw_advantage"], [0.05, 0.07])
    stale = copy.deepcopy(summary)
    stale["scheduler_diagnostics"][1]["logits"][0] += 1.0
    with pytest.raises(ValueError, match="roundtrip"):
        on_policy_auxiliary(replay, scores, [ep], [stale], actor_hash=scheduler.version)
    with pytest.raises(ValueError, match="provenance"):
        on_policy_auxiliary(replay, scores, [ep], [summary], actor_hash="older-policy")
