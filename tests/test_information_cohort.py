import json

import pytest

from latency_meta_mdp.meta.information.cohort import build_manifest, next_attempt, verify_complete
from latency_meta_mdp.meta.information.config import sha256


def test_optional_starter_covers_each_scene_regime_and_permission_pattern(tmp_path):
    from test_optional_prediction import optional_config

    config = optional_config()
    config["splits"]["train"] = list(range(100, 116))
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=config["excluded_scenes"],
        sources=[],
    )
    episodes = build_manifest(config, audit, root=tmp_path, code_hash="x")["episodes"]
    assert len(episodes) == len({e["episode_id"] for e in episodes}) == 192
    for scene in config["splits"]["train"]:
        for regime in ("zero", "nominal", "family"):
            group = [e for e in episodes if (e["scene"], e["regime"]) == (scene, regime)]
            assert {e["pattern_id"] for e in group} == {
                "direct-e25",
                "forecast-e25",
                "forecast-e75",
                "mixed",
            }
    assert all(e["split"] == "train" for e in episodes)
    assert any(e["pattern"].get("period") == 75 for e in episodes)


def test_starter_shards_are_bound_and_resume_same_assignment(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import cohort, config, preflight

    cfg = dict(mode="bound", assets={}, splits=dict(train=list(range(16))), excluded_scenes=[])
    audit = dict(
        local_inventory_pass=True, candidate_splits=cfg["splits"], excluded_scenes=[], sources=[]
    )
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    monkeypatch.setattr(config, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        preflight, "preflight", lambda *a, **k: dict(compute_processes="", code_hash="x")
    )
    shards = []
    for i in range(4):
        out = tmp_path / str(i)
        out.mkdir()
        args = dict(root=tmp_path, audit_path=path, shard_index=i, num_shards=4)
        _, shard = cohort.prepare_cohort(None, out, **args)
        assert cohort.prepare_cohort(None, out, **args)[1] == shard
        shards.extend(e["episode_id"] for e in shard["episodes"])
    assert len(shards) == len(set(shards)) == 128


def test_cohort_has_complete_paired_mixture_and_rejects_stale_audit(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    config = {
        "mode": "bound",
        "assets": {"identity": {}},
        "splits": {"train": list(range(100, 120)), "validation": [200], "test": [300]},
        "excluded_scenes": [0],
    }
    audit = {
        "local_inventory_pass": True,
        "candidate_splits": config["splits"],
        "excluded_scenes": [0],
        "sources": [{"path": "source.json", "sha256": sha256(source)}],
    }
    manifest = build_manifest(config, audit, root=tmp_path, code_hash="x")
    assert len(manifest["episodes"]) == 128
    assert len({e["episode_id"] for e in manifest["episodes"]}) == 128
    assert {e["scene"] for e in manifest["episodes"]} == set(range(100, 116))
    source.write_text("changed")
    with pytest.raises(ValueError, match="audit"):
        build_manifest(config, audit, root=tmp_path, code_hash="x")


def test_resume_preserves_incomplete_attempt_and_checks_complete_files(tmp_path):
    first = next_attempt(tmp_path, "episode")
    (first / "interrupted.log").write_text("incomplete")
    second = next_attempt(tmp_path, "episode")
    assert first != second and (first / "interrupted.log").exists()
    (second / "data").write_text("done")
    (second / "complete.json").write_text(
        json.dumps({"binding": "x", "files": {"data": sha256(second / "data")}})
    )
    assert verify_complete(second, binding="x")
    (second / "data").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        verify_complete(second, binding="x")


@pytest.mark.parametrize("staged", [False, True, "cleanup_failure"])
def test_collection_resume_skips_only_verified_complete_attempts(tmp_path, monkeypatch, staged):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from latency_meta_mdp.envs.conveyor import config as scene_config
    from latency_meta_mdp.meta.information import cohort, collection, services

    episodes = [
        dict(
            episode_id=f"{i}-nominal-lazy",
            scene=i,
            regime="nominal",
            behavior="lazy",
            scheduler_seed=i,
        )
        for i in range(2)
    ]
    manifest = {"binding": {"code_hash": "test"}, "episodes": episodes}
    monkeypatch.setattr(
        cohort, "prepare_cohort", lambda *a, **k: ({"task_config": "test"}, manifest)
    )
    monkeypatch.setattr(
        scene_config,
        "load_conveyor_spec",
        lambda *a: SimpleNamespace(supply_ticks=10, drain_ticks=2),
    )

    @contextmanager
    def frozen(*a, **k):
        yield object()

    monkeypatch.setattr(services, "frozen_services", frozen)
    if staged == "cleanup_failure":

        def failed_cleanup(path):
            raise OSError("temporary cleanup failure")

        monkeypatch.setattr(cohort.shutil, "rmtree", failed_cleanup)
    calls = []
    scratch = tmp_path / "scratch" if staged else None

    def run(root, config, path, service, **kwargs):
        calls.append(kwargs["seed"])
        assert path.is_relative_to(scratch if staged else tmp_path / "episodes")
        (path / "data").write_text("ok")
        return dict(terminated=True, truncated=False, starvation_ticks=0)

    monkeypatch.setattr(collection, "run_episode", run)
    cohort.collect_cohort(
        None, tmp_path, root=tmp_path, audit_path=None, limit=1, scratch_dir=scratch
    )
    assert calls == [0]
    cohort.collect_cohort(
        None, tmp_path, root=tmp_path, audit_path=None, limit=1, scratch_dir=scratch
    )
    assert calls == [0, 1]
    cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None, scratch_dir=scratch)
    assert calls == [0, 1]
    if staged:
        assert bool(list(scratch.iterdir())) == (staged == "cleanup_failure")
        assert not list((tmp_path / "episodes").rglob("failure.json"))
    (tmp_path / "episodes/0-nominal-lazy/attempt-000/data").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None, scratch_dir=scratch)


def test_staged_attempt_publishes_receipt_only_after_verified_copy(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import cohort

    source, destination = tmp_path / "scratch", tmp_path / "persistent"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_bytes(b"complete trajectory")
    cohort.complete_attempt(source, binding={"episode": 1})
    copied = cohort.shutil.copyfile

    def copy(left, right):
        assert not (destination / "complete.json").exists()
        return copied(left, right)

    monkeypatch.setattr(cohort.shutil, "copyfile", copy)
    cohort.publish_attempt(source, destination, binding={"episode": 1})
    assert cohort.verify_complete(destination, binding={"episode": 1})
    assert (source / "payload").exists()  # Caller cleans scratch only after publication.
    with pytest.raises(FileExistsError):
        cohort.publish_attempt(source, destination, binding={"episode": 1})


def test_staged_attempt_copy_failure_never_publishes_complete(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import cohort

    source, destination = tmp_path / "scratch", tmp_path / "persistent"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_bytes(b"complete trajectory")
    cohort.complete_attempt(source, binding={"episode": 1})

    def corrupt(left, right):
        right.write_bytes(b"corrupt copy")

    monkeypatch.setattr(cohort.shutil, "copyfile", corrupt)
    with pytest.raises(ValueError, match="hash"):
        cohort.publish_attempt(source, destination, binding={"episode": 1})
    assert not (destination / "complete.json").exists()
    assert cohort.verify_complete(source, binding={"episode": 1})


def test_collect_cli_forwards_runtime_scratch(tmp_path, monkeypatch):
    import runpy
    import sys

    from latency_meta_mdp.meta.information import cohort, config

    monkeypatch.setattr(config, "load_config", lambda _: {"schema_version": 3})
    seen = []
    monkeypatch.setattr(cohort, "collect_cohort", lambda *a, **kw: seen.append(kw) or {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_conveyor_meta.py",
            "collect",
            "--config",
            "config",
            "--output-dir",
            str(tmp_path / "out"),
            "--scene-audit",
            "audit",
            "--rollout-scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )
    runpy.run_path("scripts/run_conveyor_meta.py", run_name="__main__")
    assert seen[0]["scratch_dir"] == tmp_path / "scratch"
