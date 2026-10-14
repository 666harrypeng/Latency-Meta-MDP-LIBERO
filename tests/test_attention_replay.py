from dataclasses import replace

import numpy as np
import pytest

from latency_meta_mdp.meta.information.memory_schema import ActionSpec, AttentionPacket, PublicEvent
from latency_meta_mdp.meta.information.records import StageRecorder


def test_versioned_episode_roundtrip_retains_forced_events_and_conserves_rewards(tmp_path):
    from latency_meta_mdp.meta.information.dataset import load_episode, save_episode

    recorder = StageRecorder("one", action_spec=ActionSpec())
    packet = AttentionPacket(10, "G", decision_key=(10, 1))
    recorder.start(packet, 8, forced=False, probability=0.5)
    recorder.events.append(
        PublicEvent(
            "obs",
            "observe",
            (10, 2),
            (2, 6, 10),
            dict(visual=np.zeros((3, 2, 196, 384)), proprio=np.zeros((3, 16))),
        )
    )
    recorder.start(
        replace(packet, stage="O", decision_key=(10, 3), previous_action=8),
        9,
        forced=True,
        probability=1.0,
    )
    recorder.start(
        replace(packet, stage="P", decision_key=(10, 4), previous_action=9),
        10,
        forced=False,
        probability=1.0,
    )
    recorder.add_reward(10, 1)
    recorder.finish(11, terminated=True)
    binding = {"schema_version": 2}
    save_episode(tmp_path / "episode", recorder, report={}, binding=binding)
    raw = load_episode(tmp_path / "episode", binding=binding, raw=True)
    assert len(raw["rows"]) == 3 and len(raw["events"]) == 1
    assert raw["rows"][2]["packet"].previous_action == 9
    rows = load_episode(tmp_path / "episode", binding=binding)
    assert [r["action"] for r in rows] == [8, 10]
    assert sum(r["task_reward"] for r in rows) == pytest.approx(0.1)


def test_old_replay_observation_first_becomes_visible_after_observe(tmp_path):
    from latency_meta_mdp.meta.information.attention_replay import contexts_from_raw
    from latency_meta_mdp.meta.information.schema import ActorPacket

    current = np.zeros((2, 196, 384), np.float32)
    packets = [
        ActorPacket(10, "G"),
        ActorPacket(10, "O", current_tokens=current, current_tick=10),
        ActorPacket(
            10,
            "P",
            current_tokens=current,
            current_tick=10,
            forecast_tokens=current,
            forecast_tick=10,
            forecast_buffer_version=0,
        ),
    ]
    recorder = StageRecorder("old")
    for i, p in enumerate(packets):
        recorder.start(p, i + 5, forced=i == 1, probability=1.0)
    recorder.add_reward(10, 1)
    recorder.finish(11, terminated=True)
    result = contexts_from_raw(dict(episode_id="old", rows=recorder.rows, events=[]), ActionSpec())
    assert [r["action"] for r in result] == [8, 10]
    assert all(x is None for x in result[0]["context"].visual_refs)
    assert len([x for x in result[1]["context"].visual_refs if x is not None]) == 1
    assert result[1]["context"].forecast_ref.represented_ticks == (30,)
    assert result[0]["next_context"] is result[1]["context"]


def test_raw_mmap_retains_events_and_detects_source_mutation(tmp_path):
    import json

    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import (
        load_packed_raw,
        prepare_packed_episode,
    )

    recorder = StageRecorder("packed", action_spec=ActionSpec())
    recorder.start(AttentionPacket(10, "G", decision_key=(10, 1)), 8, forced=False, probability=1.0)
    recorder.events.append(
        PublicEvent(
            "obs",
            "observe",
            (10, 2),
            (2, 6, 10),
            dict(visual=np.zeros((3, 2, 196, 384)), proprio=np.zeros((3, 16))),
        )
    )
    recorder.add_reward(10, 1)
    recorder.finish(11, terminated=True)
    source = tmp_path / "source"
    binding = {"schema_version": 2}
    save_episode(source, recorder, report={}, binding=binding)
    entry = prepare_packed_episode(
        source,
        tmp_path / "cache",
        binding=binding,
        expected_receipt=sha256(source / "complete.json"),
        raw=True,
    )
    raw = load_packed_raw(entry)
    assert raw["episode_id"] == "packed"
    assert raw["events"][0].represented_ticks == (2, 6, 10)
    assert not raw["events"][0].payload["visual"].flags.writeable
    assert json.loads((entry.path / "complete.json").read_text())["format_version"] == 2
    (source / "audit.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        load_packed_raw(entry)
