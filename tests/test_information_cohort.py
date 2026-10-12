import json

import pytest

from latency_meta_mdp.meta.information.cohort import build_manifest, next_attempt, verify_complete
from latency_meta_mdp.meta.information.config import sha256


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


def test_collection_resume_skips_only_verified_complete_attempts(tmp_path, monkeypatch):
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
    calls = []

    def run(root, config, path, service, **kwargs):
        calls.append(kwargs["seed"])
        (path / "data").write_text("ok")
        return dict(terminated=True, truncated=False, starvation_ticks=0)

    monkeypatch.setattr(collection, "run_episode", run)
    cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None, limit=1)
    assert calls == [0]
    cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None, limit=1)
    assert calls == [0, 1]
    cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None)
    assert calls == [0, 1]
    (tmp_path / "episodes/0-nominal-lazy/attempt-000/data").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        cohort.collect_cohort(None, tmp_path, root=tmp_path, audit_path=None)
