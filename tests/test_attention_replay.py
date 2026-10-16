from dataclasses import replace

import numpy as np
import pytest

from latency_meta_mdp.meta.information.memory_schema import ActionSpec, AttentionPacket, PublicEvent
from latency_meta_mdp.meta.information.records import StageRecorder


def _two_packed_episodes(tmp_path):
    from latency_meta_mdp.meta.information.attention_replay import AttentionEpisodes, DecisionReplay
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.memory import DEFAULT_MEMORY
    from latency_meta_mdp.meta.information.packed_replay import prepare_packed_episode

    entries = []
    for i in range(2):
        recorder = StageRecorder(str(i), action_spec=ActionSpec())
        for tick in (0, 1):
            recorder.start(
                AttentionPacket(tick, "G", decision_key=(tick, 1)),
                0,
                forced=False,
                probability=1.0,
            )
        recorder.finish(2, terminated=True)
        source = tmp_path / str(i)
        save_episode(source, recorder, report={}, binding={})
        entries.append(
            prepare_packed_episode(
                source, tmp_path / "cache", binding={},
                expected_receipt=sha256(source / "complete.json"), raw=True,
            )
        )
    replay = DecisionReplay(AttentionEpisodes(entries, ActionSpec(), DEFAULT_MEMORY))
    # Isolate steady-state verification from the already tested cold decoder.
    expected = [replay[i] for i in range(len(replay))]
    return replay, expected


def test_batch_checks_unique_sources_concurrently_and_preserves_order(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from latency_meta_mdp.meta.information.packed_replay import PackedEntry

    replay, expected = _two_packed_episodes(tmp_path)
    original, barrier, checked = PackedEntry.verify_unchanged, Barrier(2), []

    def verify(entry):
        barrier.wait(timeout=5)
        original(entry)
        checked.append(entry.path)

    monkeypatch.setattr(PackedEntry, "verify_unchanged", verify)
    order = [2, 0, 2, 1]
    with ThreadPoolExecutor(max_workers=2) as executor:
        with replay.batch(order, executor=executor) as rows:
            assert all(row is expected[i] for row, i in zip(rows, order, strict=True))
    assert len(checked) == 4  # each unique episode, before and after the read


@pytest.mark.parametrize("location", ["source", "path"])
def test_batch_rejects_source_or_cache_changes_during_read(tmp_path, location):
    from concurrent.futures import ThreadPoolExecutor

    replay, _ = _two_packed_episodes(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(ValueError, match="changed"):
            with replay.batch([0, 2], executor=executor):
                directory = getattr(replay.episodes.entries[0], location)
                (directory / "records.json").write_text("{}")


@pytest.mark.parametrize("subset", [False, True])
def test_input_pipeline_checks_sources_after_collation(tmp_path, monkeypatch, subset):
    from latency_meta_mdp.meta.information import actor_inputs
    from latency_meta_mdp.meta.information.attention_replay import AttentionEpisodes, DecisionReplay
    from latency_meta_mdp.meta.information.memory import DEFAULT_MEMORY

    replay, _ = _two_packed_episodes(tmp_path)
    if subset:
        episodes = replay.episodes
        replay = DecisionReplay(
            AttentionEpisodes(
                [episodes.entries[i] for i in [1, 0]],
                episodes.action_spec,
                episodes.config,
                cache_size=episodes.cache_size,
            )
        )
        assert replay[0]["context"].decision.episode_id.endswith("/1")
    collate = actor_inputs.collate_contexts

    def changed(*args, **kwargs):
        batch = collate(*args, **kwargs)
        (replay.episodes.entries[0].source / "audit.json").write_text("changed")
        return batch

    monkeypatch.setattr(actor_inputs, "collate_contexts", changed)
    with actor_inputs.InputPipeline(replay, DEFAULT_MEMORY, microbatch=2) as pipeline:
        with pytest.raises(ValueError, match="changed"):
            list(pipeline.batches(np.array([0, 2]), successors=True))


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


@pytest.mark.parametrize("exact", [True, False])
def test_compact_raw_cache_requires_lossless_roundtrip_and_preserves_source(tmp_path, exact):
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import (
        compact_packed_episode,
        load_packed_raw,
        prepare_packed_episode,
    )

    visual = np.random.default_rng(8).normal(size=(3, 2, 196, 384)).astype(np.float16)
    visual = visual.astype(np.float32)
    if not exact:
        visual[-1, -1, -1, -1] = np.float32(0.12345678)
    recorder = StageRecorder("compact", action_spec=ActionSpec())
    recorder.events.append(
        PublicEvent(
            "obs", "observe", (10, 0), (2, 6, 10), dict(visual=visual, proprio=np.zeros((3, 16)))
        )
    )
    recorder.start(AttentionPacket(10, "G", decision_key=(10, 1)), 8, forced=False, probability=1.0)
    recorder.finish(11, terminated=True)
    source, cache = tmp_path / "source", tmp_path / "cache"
    save_episode(source, recorder, report={}, binding={"schema_version": 2})
    original = prepare_packed_episode(
        source,
        cache,
        binding={"schema_version": 2},
        expected_receipt=sha256(source / "complete.json"),
        raw=True,
    )
    digest = sha256(original.path / "visual.npy")
    if exact:
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from threading import Barrier
        from unittest.mock import patch

        from latency_meta_mdp.meta.information.packed_replay import CacheBudget
        from latency_meta_mdp.meta.information.replay_retention import tree_bytes

        budget = CacheBudget(cache, tree_bytes(cache) + 1)
        assert compact_packed_episode(original, cache, storage_budget=budget) == original
        barrier, rename = Barrier(2), Path.rename

        def publish(path, destination):
            barrier.wait(timeout=10)
            return rename(path, destination)

        with patch.object(Path, "rename", publish), ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(compact_packed_episode, original, cache) for _ in range(2)]
            results = [future.result() for future in futures]
        assert results[0] == results[1]
        compact = results[0]
    else:
        compact = compact_packed_episode(original, cache)
    assert (compact.path != original.path) == exact
    assert sha256(original.path / "visual.npy") == digest
    pool = np.load(compact.path / "visual.npy", mmap_mode="r")
    assert pool.dtype == (np.float16 if exact else np.float32)
    restored = load_packed_raw(compact)["events"][0].payload["visual"]
    np.testing.assert_array_equal(np.asarray(restored, dtype=np.float32), visual)
    assert all(not frame.flags.writeable for frame in restored)
    assert compact_packed_episode(original, cache) == compact
    (source / "audit.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        compact.verify_unchanged()


def test_raw_cache_reuses_observation_and_prediction_features_losslessly(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import (
        load_packed_raw,
        prepare_packed_episode,
    )

    visual = np.random.default_rng(2).normal(size=(3, 2, 196, 384)).astype(np.float32)
    recorder = StageRecorder("dedup", action_spec=ActionSpec())
    packet = AttentionPacket(
        10,
        "G",
        decision_key=(10, 4),
        current_tick=10,
        current_tokens=visual[-1],
        forecast_tokens=visual[0],
        forecast_tick=10,
        forecast_buffer_version=0,
    )
    recorder.start(packet, 8, forced=False, probability=1.0)
    recorder.events.extend(
        [
            PublicEvent(
                "obs",
                "observe",
                (10, 1),
                (2, 6, 10),
                dict(visual=visual, proprio=np.zeros((3, 16))),
            ),
            PublicEvent(
                "pred",
                "predict",
                (10, 2),
                (30,),
                dict(
                    visual=visual[0],
                    proprio=np.zeros(16),
                    source_tick=10,
                    query_ticks=20,
                    buffer_version=0,
                    prefix_hash="prefix",
                ),
            ),
        ]
    )
    recorder.finish(11, terminated=True)
    source = tmp_path / "source"
    save_episode(source, recorder, report={}, binding={})
    entry = prepare_packed_episode(
        source,
        tmp_path / "cache",
        binding={},
        expected_receipt=sha256(source / "complete.json"),
        raw=True,
    )
    assert np.load(entry.path / "visual.npy", mmap_mode="r").shape[0] == 3
    decoded = load_packed_raw(entry)
    np.testing.assert_array_equal(decoded["events"][0].payload["visual"], visual)
    np.testing.assert_array_equal(decoded["events"][1].payload["visual"], visual[0])
    np.testing.assert_array_equal(decoded["rows"][0]["packet"].current_tokens, visual[-1])
    np.testing.assert_array_equal(decoded["rows"][0]["packet"].forecast_tokens, visual[0])


def test_compact_cache_deduplicates_frames_without_merging_observation_events(tmp_path):
    import torch
    from attention_fixtures import small_config

    from latency_meta_mdp.meta.information.actor_inputs import transfer_batch
    from latency_meta_mdp.meta.information.attention_replay import contexts_from_raw
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.memory import collate_contexts
    from latency_meta_mdp.meta.information.packed_replay import (
        compact_packed_episode,
        load_packed_raw,
        prepare_packed_episode,
    )

    frames = np.arange(4, dtype=np.float32)[:, None, None, None] * np.ones((4, 2, 196, 384))
    frames[0, 0, 0, 0] = -0.0
    rec = StageRecorder("overlap", action_spec=ActionSpec())
    rec.start(AttentionPacket(9, "G", decision_key=(9, 0)), 0, forced=False, probability=1.0)
    for tick, selection in [(10, [0, 1, 2]), (14, [1, 2, 3])]:
        rec.events.append(
            PublicEvent(
                str(tick),
                "observe",
                (tick, 0),
                (tick - 8, tick - 4, tick),
                dict(visual=frames[selection], proprio=np.full((3, 16), tick)),
            )
        )
        rec.start(
            AttentionPacket(
                tick,
                "G",
                decision_key=(tick, 1),
                current_tokens=frames[selection[-1]],
                current_tick=tick,
            ),
            0,
            forced=False,
            probability=1.0,
        )
    rec.events.append(
        PublicEvent(
            "forecast",
            "predict",
            (14, 2),
            (34,),
            dict(
                visual=frames[3],
                proprio=np.zeros(16),
                source_tick=14,
                query_ticks=20,
                buffer_version=0,
                prefix_hash="prefix",
            ),
        )
    )
    rec.start(
        AttentionPacket(
            14,
            "P",
            decision_key=(14, 3),
            forecast_tokens=frames[3],
            forecast_tick=14,
            forecast_buffer_version=0,
        ),
        10,
        forced=False,
        probability=1.0,
    )
    rec.finish(15, terminated=True)
    source, cache = tmp_path / "source", tmp_path / "cache"
    save_episode(source, rec, report={}, binding={})
    original = prepare_packed_episode(
        source, cache, binding={}, expected_receipt=sha256(source / "complete.json"), raw=True
    )
    compact = compact_packed_episode(original, cache)
    assert np.load(compact.path / "visual.npy", mmap_mode="r").shape[0] == 4
    decoded = load_packed_raw(compact)
    assert [e.available_at for e in decoded["events"]] == [(10, 0), (14, 0), (14, 2)]
    np.testing.assert_array_equal(decoded["events"][-1].payload["visual"], frames[3])
    np.testing.assert_array_equal(decoded["rows"][-1]["packet"].forecast_tokens, frames[3])
    np.testing.assert_array_equal(decoded["rows"][-2]["packet"].current_tokens, frames[3])
    for event, selection in zip(decoded["events"][:2], ([0, 1, 2], [1, 2, 3]), strict=True):
        restored = np.asarray(event.payload["visual"], dtype=np.float32)
        assert np.array_equal(
            restored.view(np.uint32), frames[selection].astype(np.float32).view(np.uint32)
        )
        assert all(not frame.flags.writeable for frame in event.payload["visual"])
    rows = [
        contexts_from_raw(load_packed_raw(entry), ActionSpec(), config=small_config()["memory"])
        for entry in (original, compact)
    ]
    batches = [
        transfer_batch(
            collate_contexts(
                [r["context"] for r in data], config=small_config()["memory"], compact_visual=True
            ),
            "cpu",
        )
        for data in rows
    ]
    for key in batches[0].tensors:
        torch.testing.assert_close(batches[0].tensors[key], batches[1].tensors[key], rtol=0, atol=0)


def test_evaluation_trace_keeps_decisions_and_actions_without_replay_features(tmp_path):
    import json

    from latency_meta_mdp.meta.information.dataset import save_evaluation_trace

    rec = StageRecorder("evaluation", action_spec=ActionSpec())
    visual = np.ones((2, 196, 384), np.float32)
    rec.start(
        AttentionPacket(10, "G", decision_key=(10, 1), current_tokens=visual, current_tick=10),
        8,
        forced=False,
        probability=1.0,
    )
    rec.events.append(
        PublicEvent(
            "obs", "observe", (10, 2), (10,), dict(visual=visual[None], proprio=np.zeros((1, 16)))
        )
    )
    rec.finish(11, terminated=True)
    report = {"scope": "evaluation", "executed_actions": [[1, 2, 3]], "task_reward": 0.0}
    save_evaluation_trace(tmp_path / "trace", rec, report=report, binding={"scope": "evaluation"})
    trace = json.loads((tmp_path / "trace/records.json").read_text())
    assert trace["format"] == "evaluation_trace_v1"
    assert trace["rows"][0]["packet"]["current_tokens_present"]
    assert trace["rows"][0]["packet"]["current_tick"] == 10
    assert trace["rows"][0]["action"] == 8
    assert trace["events"][0]["represented_ticks"] == [10]
    assert trace["events"][0]["payload"]["visual_shape"] == [1, 2, 196, 384]
    assert json.loads((tmp_path / "trace/audit.json").read_text()) == report
    assert set(p.name for p in (tmp_path / "trace").iterdir()) == {"records.json", "audit.json"}


def test_packed_cache_checks_budget_before_writing_visual_array(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import CacheBudget, prepare_packed_episode

    recorder = StageRecorder("budget", action_spec=ActionSpec())
    recorder.start(AttentionPacket(10, "G", decision_key=(10, 1)), 8, forced=False, probability=1.0)
    recorder.events.append(
        PublicEvent(
            "obs",
            "observe",
            (10, 2),
            (10,),
            dict(visual=np.zeros((1, 2, 196, 384)), proprio=np.zeros((1, 16))),
        )
    )
    recorder.finish(11, terminated=True)
    source, cache = tmp_path / "source", tmp_path / "cache"
    save_episode(source, recorder, report={}, binding={})
    with pytest.raises(RuntimeError, match="cache storage ceiling"):
        prepare_packed_episode(
            source,
            cache,
            binding={},
            expected_receipt=sha256(source / "complete.json"),
            raw=True,
            storage_budget=CacheBudget(cache, 100),
        )
    assert not list(cache.rglob("visual.npy"))
    assert not list(cache.rglob(".pending-*"))


@pytest.mark.parametrize("subset", [False, True])
def test_v2_only_replay_does_not_require_unused_legacy_sleep_vocabulary(tmp_path, subset):
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
    if subset:
        manifest["episodes"].append(dict(ep, episode_id="retired-episode"))
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
                **({"episode_ids": ["v2-only"]} if subset else {}),
            )
        ],
        root=tmp_path,
        cache_dir=tmp_path / "cache",
    )
    assert len(replay) == 1 and replay[0]["action"] == 0
    assert receipt["action_mapping"] is None


def test_episode_cache_evicts_and_reconstructs_identical_contexts(tmp_path):
    from attention_fixtures import small_config

    from latency_meta_mdp.meta.information.attention_replay import AttentionEpisodes
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import prepare_packed_episode

    rec = StageRecorder("bounded", action_spec=ActionSpec())
    rec.start(AttentionPacket(10, "G", decision_key=(10, 1)), 0, forced=False, probability=1.0)
    rec.finish(11, terminated=True)
    source = tmp_path / "source"
    save_episode(source, rec, report={}, binding={"schema_version": 2})
    entry = prepare_packed_episode(
        source,
        tmp_path / "cache",
        binding={"schema_version": 2},
        expected_receipt=sha256(source / "complete.json"),
        raw=True,
    )
    episodes = AttentionEpisodes(
        [entry, entry], ActionSpec(), small_config()["memory"], cache_size=1
    )
    first = episodes[0][0]["context"].decision
    episodes[1]
    assert len(episodes.loaded) == 1
    assert episodes[0][0]["context"].decision == first
    assert len(episodes.loaded) == 1


def test_spawn_keeps_admitted_lengths_without_rereading_every_episode(tmp_path):
    import pickle

    from attention_fixtures import small_config

    from latency_meta_mdp.meta.information.attention_replay import AttentionEpisodes
    from latency_meta_mdp.meta.information.config import sha256
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.packed_replay import prepare_packed_episode

    recorder = StageRecorder("spawn", action_spec=ActionSpec())
    recorder.start(AttentionPacket(10, "G", decision_key=(10, 1)), 8, forced=False, probability=1.0)
    recorder.finish(11, terminated=True)
    source = tmp_path / "source"
    save_episode(source, recorder, report={}, binding={})
    entry = prepare_packed_episode(
        source,
        tmp_path / "cache",
        binding={},
        expected_receipt=sha256(source / "complete.json"),
        raw=True,
    )
    episodes = AttentionEpisodes([entry], ActionSpec(), small_config()["memory"])
    serialized = pickle.dumps(episodes)
    # Spawn needs only the previously admitted lengths; actual access must still
    # reject changed source contents before using any cached context.
    (entry.path / "records.json").unlink()
    restored = pickle.loads(serialized)
    assert restored.lengths == [1] and not restored.loaded
    with pytest.raises((ValueError, FileNotFoundError)):
        restored[0]
