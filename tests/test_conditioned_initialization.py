import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("credential", [None, "unit-test-token"])
def test_local_clean_reuse_and_remote_step_are_explicit(tmp_path, monkeypatch, credential):
    import huggingface_hub
    from huggingface_hub.utils import _headers

    monkeypatch.setattr(_headers, "get_token", lambda: credential)
    monkeypatch.setattr(_headers.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)

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
    expected = None if credential is None else f"Bearer {credential}"
    assert all(
        _headers.build_hf_headers(token=c.get("token")).get("authorization") == expected
        for c in calls
    )
    del job["initialization"]
    (local / "assets/norm_stats.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="normalization"):
        conditioned_initialization(job, profile, schedule, work=tmp_path, clean_work=clean_work)


def test_task_initialization_reuses_verified_bundle_without_legacy_level_keys(
    tmp_path, monkeypatch
):
    import huggingface_hub

    from latency_meta_mdp.policy.initialization import conditioned_initialization

    norm = b'{"norm_stats": {}}'
    inputs = SimpleNamespace(
        bundle_root=tmp_path / "bundle",
        preparation={"norm_stats_sha256": hashlib.sha256(norm).hexdigest()},
        task_parameters={"config_name": "conveyor-config"},
    )
    job = {
        "clean": {"schema_version": 2, "task_id": "conveyor_sort", "run_name": "clean"},
        "initialization": {"repo_id": "owner/full", "revision": "a" * 40, "step": 15237},
    }
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / "15237"
        (checkpoint / "params").mkdir(parents=True)
        (checkpoint / "assets").mkdir()
        (checkpoint / "assets/norm_stats.json").write_bytes(norm)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    bundle, checkpoint, receipt = conditioned_initialization(
        job,
        SimpleNamespace(),
        SimpleNamespace(num_train_steps=91416),
        work=tmp_path / "conditioned",
        clean_work=tmp_path / "clean",
        clean_inputs=inputs,
    )
    assert bundle == inputs.bundle_root
    assert checkpoint.name == "15237" and receipt["step"] == 15237
    assert len(calls) == 1 and calls[0]["repo_id"] == "owner/full"


def test_task_initialization_reuses_local_step(tmp_path, monkeypatch):
    import huggingface_hub

    from latency_meta_mdp.policy.initialization import conditioned_initialization

    norm = b'{"norm_stats": {}}'
    checkpoint = tmp_path / "saved/5079"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    (checkpoint / "assets/norm_stats.json").write_bytes(norm)
    inputs = SimpleNamespace(
        bundle_root=tmp_path / "bundle",
        preparation={"norm_stats_sha256": hashlib.sha256(norm).hexdigest()},
        task_parameters={},
    )
    job = {
        "clean": {"schema_version": 2},
        "initialization": {
            "repo_id": "owner/full",
            "revision": "a" * 40,
            "step": 5079,
            "local_path": str(checkpoint),
        },
    }

    def unexpected_download(**kwargs):
        pytest.fail("local initialization must not download checkpoint")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    kwargs = dict(work=tmp_path / "conditioned", clean_work=tmp_path / "clean", clean_inputs=inputs)
    _, actual, receipt = conditioned_initialization(job, None, None, **kwargs)
    assert actual == checkpoint and receipt["local_path"] == str(checkpoint)
    job["initialization"]["step"] = 10158
    with pytest.raises(ValueError, match="step"):
        conditioned_initialization(job, None, None, **kwargs)
