
import pytest
import yaml


def task_job(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=2,
                task_id="conveyor_sort",
                variant="surface",
                action_contract="panda_osc_pose_delta_conveyor_v2",
                profile="configs/contracts/policy/pi05_state16_h50.yaml",
                training="configs/training/policy/conveyor_clean.yaml",
                dataset=dict(kind="local", path="bundle", manifest_sha256="a" * 64),
                run_name="conveyor-clean-check",
                device_count=8,
                batch_size=256,
                publish_repo=None,
            )
        )
    )
    return path


def test_task_job_uses_real_task_identity_and_source_epoch_schedule(tmp_path):
    from latency_meta_mdp.policy.config import load_training_job, resolve_policy_profile
    from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule

    job = load_training_job(task_job(tmp_path))
    assert "level" not in job and job["dataset"]["path"] == tmp_path / "bundle"
    profile = resolve_policy_profile(job, source_count=15539)
    request = SFTLaunchRequest(
        None, job["run_name"], "formal", False, 8, 256, task_id="conveyor_sort"
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    assert schedule.expected_checkpoint_steps == (61, 122, 183)
    assert schedule.batch_size == 256
    assert 3 <= schedule.num_train_steps * 256 / 15539 < 3.02


def test_task_job_rejects_unpinned_data_and_missing_source_count(tmp_path):
    from latency_meta_mdp.policy.config import load_training_job, resolve_policy_profile

    path = task_job(tmp_path)
    job = load_training_job(path)
    with pytest.raises(ValueError, match="source_count"):
        resolve_policy_profile(job)
    raw = yaml.safe_load(path.read_text())
    raw["dataset"]["manifest_sha256"] = "latest"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_training_job(path)


def test_task_job_requires_epoch_recipe(tmp_path):
    from latency_meta_mdp.policy.config import load_training_job

    path = task_job(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["training"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="training recipe"):
        load_training_job(path)
