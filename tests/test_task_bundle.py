import json
from pathlib import Path

import pytest

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.policy.profile import load_sft_profile


def preparation(tmp_path):
    root = tmp_path / "prepared"
    data = root / "datasets/local/conveyor"
    (data / "data").mkdir(parents=True)
    (data / "data/episode.parquet").write_bytes(b"fixture")
    norm = root / "assets/local/conveyor/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text(
        json.dumps({"norm_stats": {"state": {"mean": [0] * 16}, "actions": {"mean": [0] * 7}}})
    )
    metadata = dict(
        task_id="conveyor_sort",
        variant="surface",
        split="train",
        repo_id="local/conveyor",
        action_contract="panda_osc_pose_delta_conveyor_v2",
        state_dim=16,
        action_dim=7,
        action_target_contract="masked_h50_real_actions_v1",
        frame_count=1024,
        purpose="development_smoke",
        source_manifest_sha256="b" * 64,
        file_sizes={"data/episode.parquet": 7},
    )
    (data / "metamdp_dataset.json").write_text(json.dumps(metadata))
    patches = {
        p.name: sha256_file(p)
        for p in (
            Path("patches/openpi") / n
            for n in (
                "0001-filter-incomplete-action-chunks.patch",
                "0002-save-completed-step-checkpoints.patch",
                "0003-mask-action-tails.patch",
            )
        )
    }
    prep = dict(
        status="completed",
        purpose="development_smoke",
        task_id=metadata["task_id"],
        action_contract=metadata["action_contract"],
        source_count=1024,
        source_manifest_sha256="b" * 64,
        dataset_manifest="datasets/local/conveyor/metamdp_dataset.json",
        dataset_manifest_sha256=sha256_file(data / "metamdp_dataset.json"),
        norm_stats="assets/local/conveyor/norm_stats.json",
        norm_stats_sha256=sha256_file(norm),
        profile_sha256=sha256_file(Path("configs/contracts/policy/pi05_state16_h50.yaml")),
        openpi_revision=load_sft_profile(
            Path("configs/contracts/policy/pi05_state16_h50.yaml")
        ).openpi_revision,
        patches=patches,
    )
    (root / "preparation.json").write_text(json.dumps(prep))
    return root


def test_task_bundle_roundtrip_and_smoke_training_guard(tmp_path):
    from latency_meta_mdp.data.task_bundle import package_task_bundle, write_local_training_job
    from latency_meta_mdp.policy.clean_data import require_training_source, resolve_clean_inputs
    from latency_meta_mdp.policy.config import load_training_job

    source = preparation(tmp_path)
    bundle = package_task_bundle(source, tmp_path / "bundle")
    job_file = write_local_training_job(bundle, tmp_path / "job.yaml")
    inputs = resolve_clean_inputs(load_training_job(job_file), tmp_path / "run")
    assert inputs.task_parameters["task_id"] == "conveyor_sort"
    assert inputs.preparation["source_count"] == 1024
    require_training_source(inputs, checking=True)
    with pytest.raises(ValueError, match="check-only"):
        require_training_source(inputs, checking=False)
    original = source / "assets/local/conveyor/norm_stats.json"
    assert (
        original.stat().st_ino
        != (inputs.preparation_root / inputs.preparation["norm_stats"]).stat().st_ino
    )
    original.write_text("changed")
    resolve_clean_inputs(load_training_job(job_file), tmp_path / "run")
    (inputs.preparation_root / inputs.preparation["norm_stats"]).write_text("tampered")
    with pytest.raises(ValueError):
        resolve_clean_inputs(load_training_job(job_file), tmp_path / "run")


def test_formal_bundle_passes_training_source_guard(tmp_path):
    from latency_meta_mdp.data.task_bundle import package_task_bundle, write_local_training_job
    from latency_meta_mdp.policy.clean_data import require_training_source, resolve_clean_inputs
    from latency_meta_mdp.policy.config import load_training_job

    root = preparation(tmp_path)
    path = root / "datasets/local/conveyor/metamdp_dataset.json"
    metadata = json.loads(path.read_text())
    metadata["purpose"] = "training_source"
    path.write_text(json.dumps(metadata))
    prep_path = root / "preparation.json"
    prep = json.loads(prep_path.read_text())
    prep.update(purpose="training_source", dataset_manifest_sha256=sha256_file(path))
    prep_path.write_text(json.dumps(prep))
    bundle = package_task_bundle(root, tmp_path / "bundle")
    inputs = resolve_clean_inputs(
        load_training_job(write_local_training_job(bundle, tmp_path / "job.yaml")), tmp_path / "run"
    )
    require_training_source(inputs, checking=False)


def test_packaging_rejects_changed_norm(tmp_path):
    from latency_meta_mdp.data.task_bundle import package_task_bundle

    root = preparation(tmp_path)
    (root / "assets/local/conveyor/norm_stats.json").write_text("changed")
    with pytest.raises(ValueError, match="normalization"):
        package_task_bundle(root, tmp_path / "bundle")


def test_launch_rejects_preparation_from_another_openpi_revision(tmp_path):
    from latency_meta_mdp.data.task_bundle import package_task_bundle, write_local_training_job
    from latency_meta_mdp.policy.clean_data import resolve_clean_inputs
    from latency_meta_mdp.policy.config import load_training_job

    root = preparation(tmp_path)
    path = root / "preparation.json"
    prep = json.loads(path.read_text())
    prep["openpi_revision"] = "f" * 40
    path.write_text(json.dumps(prep))
    bundle = package_task_bundle(root, tmp_path / "bundle")
    job = load_training_job(write_local_training_job(bundle, tmp_path / "job.yaml"))
    with pytest.raises(ValueError, match="OpenPI revision"):
        resolve_clean_inputs(job, tmp_path / "run")
