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


def test_v2_only_replay_does_not_require_unused_legacy_sleep_vocabulary(tmp_path):
    import json

    from attention_fixtures import small_config

    from latency_meta_mdp.io.paths import repository_root
    from latency_meta_mdp.meta.information.attention_replay import prepare_attention_replay
    from latency_meta_mdp.meta.information.cohort import complete_attempt
    from latency_meta_mdp.meta.information.config import identity_hash, sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler

    config = small_config()
    config["actions"]["sleep_ticks"] = [1, 2, 3, 4, 6, 8, 12]
    spec = ActionSpec(**config["actions"])
    binding = dict(
        schema_version=2,
        assets=config["assets"],
        mode="bound",
        scope="starter_training",
        code_hash="fixture",
        config_hash=identity_hash(config),
        actions=config["actions"],
    )
    ep = dict(
        episode_id="v2-only",
        scene=10000,
        regime="zero",
        behavior="lazy",
        scheduler_seed=1,
        policy_seed=0,
        split="train",
    )
    manifest = dict(binding=binding, episodes=[ep])
    (tmp_path / "cohort.json").write_text(json.dumps(manifest))
    attempt = tmp_path / "episodes/v2-only/attempt-000"
    attempt.mkdir(parents=True)
    rec = StageRecorder("v2-only", action_spec=spec)
    rec.start(
        AttentionPacket(10, "G", decision_key=(10, 1), action_spec=spec),
        0,
        forced=False,
        probability=1.0,
    )
    rec.add_reward(10, 0)
    rec.finish(11, terminated=True)
    _, delay = conveyor_delay_sampler(
        repository_root(), regime="zero", scene_seed=10000, policy_seed=0
    )
    inner = dict(
        assets=config["assets"]["identity"],
        mode="bound",
        scope="starter_training",
        scene=10000,
        code_hash="fixture",
        scheduler_seed=1,
        behavior="lazy",
        policy_seed=0,
        delay=delay,
        actions=config["actions"],
        pattern=None,
    )
    save_episode(attempt / "episode", rec, report={}, binding=inner)
    (attempt / "summary.json").write_text(
        json.dumps(
            dict(
                terminated=True,
                truncated=False,
                starvation_ticks=0,
                scope="starter_training",
                task_reward=0,
                parcels=dict(spawned=10, successes=0, misses=10, timeouts=0, active=0),
            )
        )
    )
    complete_attempt(attempt, binding=dict(cohort=binding, episode=ep))
    replay, receipt = prepare_attention_replay(
        config,
        [
            dict(
                path=str(tmp_path),
                manifest_sha256=sha256(tmp_path / "cohort.json"),
                group="startup",
            )
        ],
        root=tmp_path,
        cache_dir=tmp_path / "cache",
    )
    assert len(replay) == 1 and replay[0]["action"] == 0
    assert receipt["action_mapping"] is None
