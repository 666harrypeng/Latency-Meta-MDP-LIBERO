import json

import pytest
import yaml


def test_job_resolves_profile_and_replicated_batch(tmp_path):
    from latency_meta_mdp.policy.config import load_training_job

    data = dict(
        schema_version=1,
        level=2,
        profile="profile.yaml",
        dataset_repo="owner/data",
        dataset_revision="a" * 40,
        run_name="l2-clean",
        device_count=8,
        batch_size=256,
        publish_repo="owner/model",
    )
    p = tmp_path / "job.yaml"
    p.write_text(yaml.safe_dump(data))
    job = load_training_job(p)
    assert job["profile"] == tmp_path / "profile.yaml"
    assert job["batch_size"] // job["device_count"] == 32
    data["batch_size"] = 255
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_training_job(p)


def test_published_checkpoints_exclude_optimizer_and_logs(tmp_path):
    from latency_meta_mdp.io.policy_publish import inference_checkpoint_files

    for path in [
        "1000/params/manifest.ocdbt",
        "1000/assets/local/data/norm_stats.json",
        "1000/train_state/optimizer",
        "1000/_CHECKPOINT_METADATA",
        "run.log",
    ]:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    files = inference_checkpoint_files(tmp_path, (1000,))
    assert set(files) == {
        "1000/params/manifest.ocdbt",
        "1000/assets/local/data/norm_stats.json",
        "1000/_CHECKPOINT_METADATA",
    }
    (tmp_path / "1000/assets/local/data/norm_stats.json").unlink()
    with pytest.raises(ValueError):
        inference_checkpoint_files(tmp_path, (1000,))


def test_bundle_paths_cannot_escape_and_truncated_files_are_rejected(tmp_path):
    from latency_meta_mdp.data.bundles import verify_bundle_files

    (tmp_path / "data").write_bytes(b"1234")
    manifest = {
        "schema_version": 1,
        "format_id": "structured_pi05_training_bundle_v1",
        "files": {"data": 4},
        "level": 2,
        "dataset_root": "dataset",
        "preparation_root": "preparation",
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    assert verify_bundle_files(tmp_path)["level"] == 2
    manifest["files"]["data"] = 5
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_bundle_files(tmp_path)
    manifest["files"] = {"../secret": 1}
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_bundle_files(tmp_path)


def test_publisher_uses_public_inference_allowlist(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from latency_meta_mdp.io.policy_publish import publish_checkpoints

    for name in (
        "1000/params/weights",
        "1000/assets/data/norm_stats.json",
        "1000/train_state/optimizer",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    notices = tmp_path / "notices"
    notices.mkdir()
    for name in ("LICENSE", "NOTICE", "GEMMA_TERMS.txt"):
        (notices / name).write_text("upstream notice")
    calls = {}

    class API:
        def create_repo(self, repo_id, **kwargs):
            calls["create"] = (repo_id, kwargs)

        def create_commit(self, repo_id, operations, commit_message):
            calls["operations"] = {operation.path_in_repo: operation for operation in operations}
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    assert (
        publish_checkpoints(root=tmp_path, steps=(1000,), repo_id="owner/model", notices=notices)
        == "b" * 40
    )
    assert calls["create"][1]["private"] is False
    assert set(calls["operations"]) == {
        "1000/params/weights",
        "1000/assets/data/norm_stats.json",
        "README.md",
        "LICENSE",
        "NOTICE",
        "GEMMA_TERMS.txt",
    }
    assert calls["operations"]["README.md"].path_or_fileobj == b""
