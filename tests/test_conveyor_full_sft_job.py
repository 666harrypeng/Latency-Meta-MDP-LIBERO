from pathlib import Path
from types import SimpleNamespace

import pytest

from latency_meta_mdp.policy.config import load_training_job, resolve_policy_profile
from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule


@pytest.mark.parametrize("batch", [32, 64, 128])
def test_full_conveyor_job_keeps_five_epoch_milestones_when_batch_changes(batch):
    job = load_training_job(Path("configs/experiments/conveyor_sort/clean_fsdp.yaml"))
    job["batch_size"] = batch
    profile = resolve_policy_profile(job, source_count=390041)
    request = SFTLaunchRequest(
        None, job["run_name"], "formal", False, 8, batch, task_id="conveyor_sort"
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    assert profile.full_parameter and profile.ema_decay == 0.999
    assert profile.fsdp_devices == 2
    for epoch, step in zip((5, 10, 15), schedule.expected_checkpoint_steps, strict=True):
        assert 0 <= step * batch - epoch * 390041 < 3 * batch
    smoke = resolve_sft_schedule(
        profile=profile,
        request=SFTLaunchRequest(
            None, job["run_name"], "smoke", False, 8, batch, task_id="conveyor_sort"
        ),
    )
    assert smoke.num_train_steps == 100
    assert smoke.decay_steps == schedule.decay_steps


def test_clean_cli_passes_smoke_mode_to_launcher(monkeypatch):
    from latency_meta_mdp.policy import train_clean

    captured = []
    monkeypatch.setattr(train_clean, "run", captured.append)
    monkeypatch.setattr(
        "sys.argv",
        ["train_clean_policy", "--config", "job.yaml", "--output-dir", "pilot", "--mode", "smoke"],
    )
    train_clean.main()
    assert captured[0].mode == "smoke"


def test_smoke_cannot_publish_into_formal_checkpoint_repo():
    from latency_meta_mdp.policy.train_clean import run

    with pytest.raises(ValueError, match="smoke"):
        run(SimpleNamespace(mode="smoke", publish_only=True))
