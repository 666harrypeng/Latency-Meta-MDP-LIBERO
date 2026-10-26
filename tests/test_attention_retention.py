import json
from pathlib import Path

import pytest

from latency_meta_mdp.meta.information.config import sha256


def collection(root, index):
    from latency_meta_mdp.meta.information.cohort import complete_attempt

    directory = root / f"cycle-{index:04d}" / "collection"
    binding = {"scope": "online_training"}
    episodes = []
    for regime in ("zero", "nominal", "family"):
        episode = dict(episode_id=f"{index}-{regime}", regime=regime, split="train")
        episodes.append(episode)
        attempt = directory / "episodes" / episode["episode_id"] / "attempt-000"
        (attempt / "episode").mkdir(parents=True)
        (attempt / "episode/visual.npz").write_bytes(bytes([index]) * 10)
        (attempt / "episode/records.json").write_text("[]")
        (attempt / "episode/complete.json").write_text(json.dumps(episode))
        (attempt / "summary.json").write_text(json.dumps({"task_reward": index % 2}))
        complete_attempt(attempt, binding={"cohort": binding, "episode": episode})
    (directory / "cohort.json").write_text(json.dumps({"binding": binding, "episodes": episodes}))
    return dict(
        path=str(directory), manifest_sha256=sha256(directory / "cohort.json"), group="latest"
    )


def limits(**changes):
    return dict(
        dict(
            recent_batches=2,
            history_episodes=3,
            max_online_bytes=10000,
            max_cache_bytes=10000,
            max_run_bytes=1000000,
            min_free_bytes=0,
            episode_cache_size=2,
        ),
        **changes,
    )


def test_recent_window_and_seeded_history_survive_restart_and_eviction(tmp_path):
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    retention = BoundedReplay(tmp_path, limits(), seed=4)
    results = []
    for index in range(1, 7):
        sources = retention.select(index, collection(tmp_path, index))
        results.append(sources)
        retention.cleanup(index)
    state = json.loads((tmp_path / "replay-state/0006.json").read_text())
    assert [b[0]["batch"] for b in state["recent"]] == [5, 6]
    assert len(state["history"]) == 3
    assert {r["regime"] for r in state["history"]} == {"zero", "nominal", "family"}
    assert (
        len({r["key"] for b in state["recent"] for r in b} | {r["key"] for r in state["history"]})
        == 9
    )
    assert (
        len(list(tmp_path.glob("cycle-*/collection/episodes/*/attempt-*/episode/visual.npz"))) <= 12
    )
    assert len(list(tmp_path.glob("cycle-*/collection/episodes/*/attempt-*/summary.json"))) == 18
    restored = BoundedReplay(tmp_path, limits(), seed=4)
    assert restored.select(6, collection_reference(tmp_path, 6)) == results[-1]
    # Replaying old journal entries must not prune newer active data.
    restored.cleanup(1)
    for source in results[-1]:
        for ep in source["episode_ids"]:
            assert (
                Path(source["path"]) / "episodes" / ep / "attempt-000/episode/visual.npz"
            ).exists()


def collection_reference(root, index):
    p = root / f"cycle-{index:04d}/collection"
    return dict(path=str(p), manifest_sha256=sha256(p / "cohort.json"), group="latest")


def test_retirement_removes_both_cache_formats_and_preserves_recovery_inputs(tmp_path):
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    retention = BoundedReplay(tmp_path, limits(history_episodes=0), seed=4)
    formats = ("mmap-raw-v2", "mmap-raw-f16-v1", "mmap-raw-f16-v2", "mmap-raw-direct-v5")
    for index in range(1, 6):
        retention.select(index, collection(tmp_path, index))
        records = json.loads((retention.inventory_dir / f"{index:04d}.json").read_text())["records"]
        for record in records:
            for name in formats:
                path = retention.cache / name / record["packed_key"]
                path.mkdir(parents=True)
                (path / "visual.npy").write_bytes(b"x")
        retention.cleanup(index)
    protected = set()
    for index in (4, 5):
        state = json.loads((retention.state_dir / f"{index:04d}.json").read_text())
        protected.update(r["packed_key"] for batch in state["recent"] for r in batch)
    for name in formats:
        assert {p.name for p in (retention.cache / name).iterdir()} == protected


def test_journal_distinguishes_retired_payload_from_missing_evidence(tmp_path):
    from latency_meta_mdp.meta.information.campaign import CampaignJournal
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    journal = CampaignJournal(tmp_path, binding={}, allow_retired=True)
    retention = BoundedReplay(tmp_path, limits(), seed=4)
    source = collection(tmp_path, 1)
    payload = Path(source["path"]) / "episodes/1-zero/attempt-000/episode/visual.npz"
    summary = payload.parent.parent / "summary.json"
    result = dict(artifacts={str(payload): sha256(payload), str(summary): sha256(summary)})
    journal.run("old", lambda: result)
    for index in range(1, 12):
        src = source if index == 1 else collection(tmp_path, index)
        retention.select(index, src)
        retention.cleanup(index)
    # At least some old data is deliberately retired; all listed hashes are pinned.
    retired = json.loads((tmp_path / "replay-retired.json").read_text())
    removed = next(iter(retired["files"]))
    old = dict(artifacts={removed: retired["files"][removed]})
    journal.state["phases"]["retired"] = {"result": old}
    assert journal.run("retired", lambda: pytest.fail("must not recollect")) == old
    summary.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        journal.run("old", lambda: result)


def test_retention_rejects_external_collection_before_deletion(tmp_path):
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    source = collection(tmp_path / "foreign", 1)
    retention = BoundedReplay(tmp_path / "owned", limits(), seed=4)
    with pytest.raises(ValueError, match="owned"):
        retention.select(1, source)
    assert list((tmp_path / "foreign").rglob("visual.npz"))


def test_retirement_rechecks_reappearing_payload_before_unlink(tmp_path):
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    retention = BoundedReplay(tmp_path, limits(history_episodes=0), seed=4)
    for index in range(1, 5):
        retention.select(index, collection(tmp_path, index))
        retention.cleanup(index)
    retired = json.loads((tmp_path / "replay-retired.json").read_text())
    payload = Path(next(iter(retired["files"])))
    payload.write_bytes(b"unexpected replacement")
    with pytest.raises(ValueError, match="binding"):
        retention.cleanup(4)
    assert payload.read_bytes() == b"unexpected replacement"


def test_campaign_resume_after_eviction_and_interrupted_fit(tmp_path):
    from latency_meta_mdp.meta.information.attention_campaign import attention_cycle_loop
    from latency_meta_mdp.meta.information.campaign import CampaignJournal, _cohort_artifacts
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay

    recipe = dict(
        startup_sources=[dict(path="immutable-startup", group="startup")],
        reward_regression_policy="report_only",
        cycles=[dict(generation={"name": f"g{i}"}) for i in range(7)],
    )
    collected, fitted, fail = [], [], [True]

    def collect(index, definition, parent):
        source = collection(tmp_path, index)
        collected.append(index)
        return dict(source=source, artifacts=_cohort_artifacts(Path(source["path"])))

    def fit(index, definition, parent, sources):
        assert sources[0] == recipe["startup_sources"][0]
        for source in sources[1:]:
            for ep in source["episode_ids"]:
                assert (
                    Path(source["path"]) / "episodes" / ep / "attempt-000/episode/visual.npz"
                ).is_file()
        if index == 6 and fail[0]:
            fail[0] = False
            raise RuntimeError("interrupted fit")
        fitted.append(index)
        checkpoint = tmp_path / f"checkpoint-{index}"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(bytes([index]))
        return dict(
            latest=str(checkpoint),
            candidates=[],
            optimizer_updates=2,
            validation_episodes=9,
            artifacts={str(checkpoint / "state.pt"): sha256(checkpoint / "state.pt")},
        )

    def run():
        return attention_cycle_loop(
            journal=CampaignJournal(tmp_path, binding=recipe, allow_retired=True),
            recipe=recipe,
            initial=tmp_path / "initial",
            initial_metrics=None,
            collect=collect,
            fit=fit,
            retention=BoundedReplay(tmp_path, limits(history_episodes=0), seed=4),
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        run()
    assert json.loads((tmp_path / "replay-retired.json").read_text())["files"]
    report = run()
    assert report["status"] == "complete"
    assert collected == fitted == list(range(1, 8))
    assert run() == report


def test_replay_byte_limit_shortens_window_and_stops_oversized_batch(tmp_path):
    from latency_meta_mdp.meta.information.replay_retention import BoundedReplay, tree_bytes

    first = collection(tmp_path, 1)
    one_batch = sum(tree_bytes(p) for p in Path(first["path"]).glob("episodes/*/attempt-*"))
    policy = limits(recent_batches=20, history_episodes=0, max_online_bytes=one_batch + 100)
    retention = BoundedReplay(tmp_path, policy, seed=4)
    retention.select(1, first)
    sources = retention.select(2, collection(tmp_path, 2))
    assert len(sources) == 1 and sources[0]["group"] == "latest"
    other = tmp_path / "undersized"
    source = collection(other, 1)
    with pytest.raises(RuntimeError, match="latest batch"):
        BoundedReplay(other, limits(max_online_bytes=1), seed=4).select(1, source)
