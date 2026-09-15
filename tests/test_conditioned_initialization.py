import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_local_clean_reuse_and_remote_step_are_explicit(tmp_path, monkeypatch):
    import huggingface_hub

    from latency_meta_mdp.policy.initialization import conditioned_initialization

    clean_work = tmp_path / "clean"
    bundle = clean_work / "bundles/data-revision"
    (bundle / "dataset").mkdir(parents=True)
    (bundle / "preparation").mkdir()
    norm = b'{"norm_stats": {}}'
    (bundle / "preparation/preparation.json").write_text(
        json.dumps({"norm_stats_sha256": hashlib.sha256(norm).hexdigest()})
    )
    (bundle / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format_id": "structured_pi05_training_bundle_v1",
                "files": {},
                "level": 3,
                "dataset_root": "dataset",
                "preparation_root": "preparation",
            }
        )
    )
    job = {
        "clean": {
            "level": 3,
            "dataset_revision": "data-revision",
            "dataset_repo": "data/repo",
            "run_name": "clean-run",
        }
    }
    profile = SimpleNamespace(levels={3: SimpleNamespace(config_name="clean-config")})
    schedule = SimpleNamespace(num_train_steps=3000)

    def checkpoint(path):
        (path / "params").mkdir(parents=True)
        (path / "assets").mkdir()
        (path / "assets/norm_stats.json").write_bytes(norm)

    local = clean_work / "checkpoints/clean-config/clean-run/3000"
    checkpoint(local)
    (local.parent / "verified_run-step3000.json").write_text("{}")
    actual, weights, receipt = conditioned_initialization(
        job, profile, schedule, work=tmp_path, clean_work=clean_work
    )
    assert actual == bundle and weights == local and receipt["step"] == 3000
    calls = []

    def download(**kw):
        calls.append(kw)
        if kw.get("repo_type") != "dataset":
            checkpoint(Path(kw["local_dir"]) / "6000")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    job["initialization"] = {"repo_id": "model/repo", "revision": "fixed-revision", "step": 6000}
    actual, weights, receipt = conditioned_initialization(
        job, profile, schedule, work=tmp_path, clean_work=clean_work
    )
    assert weights.name == "6000" and receipt["step"] == 6000
    assert calls[1]["allow_patterns"] == [
        "6000/params/**",
        "6000/assets/**",
        "6000/_CHECKPOINT_METADATA",
    ]
    assert all(c["token"] is False for c in calls)
    del job["initialization"]
    (local / "assets/norm_stats.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="normalization"):
        conditioned_initialization(job, profile, schedule, work=tmp_path, clean_work=clean_work)
