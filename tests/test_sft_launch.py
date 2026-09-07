from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from latency_meta_mdp.sft_asset_lock import load_sft_asset_lock
from latency_meta_mdp.sft_launch import (
    SFTLaunchRequest,
    resolve_sft_schedule,
    stage_level_dataset,
    stage_level_norm_stats,
)
from latency_meta_mdp.sft_profile import load_sft_profile

_PROFILE_PATH = Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml")
_LOCK_PATH = Path("configs/data/franka_moving_ball_sft_assets_v1.yaml")


def test_stage_level_norm_stats_downloads_locked_assets_to_openpi_path(
    tmp_path: Path,
) -> None:
    profile = load_sft_profile(_PROFILE_PATH)
    lock = load_sft_asset_lock(_LOCK_PATH, profile=profile)
    row = lock.level(1)
    source = Path("outputs/sft/norm_stats/franka-moving-ball-h50-32cea2f/L1")
    calls = []

    def download(*, repo_id: str, revision: str, path_in_repo: str) -> Path:
        calls.append((repo_id, revision, path_in_repo))
        return source / Path(path_in_repo).name

    staged = stage_level_norm_stats(
        profile=profile,
        asset_lock=lock,
        level=1,
        assets_root=tmp_path / "assets",
        download_asset=download,
    )

    assert staged == (
        tmp_path / "assets" / profile.levels[1].config_name / row.repo_id / "norm_stats.json"
    )
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == row.norm_stats_sha256
    assert calls == [
        (
            row.repo_id,
            row.asset_revision,
            "metamdp_assets/openpi/manifest.json",
        ),
        (
            row.repo_id,
            row.asset_revision,
            "metamdp_assets/openpi/norm_stats.json",
        ),
    ]
    assert (
        stage_level_norm_stats(
            profile=profile,
            asset_lock=lock,
            level=1,
            assets_root=tmp_path / "assets",
            download_asset=download,
        )
        == staged
    )
    assert len(calls) == 4


def test_stage_level_norm_stats_rejects_remote_hash_mismatch(tmp_path: Path) -> None:
    profile = load_sft_profile(_PROFILE_PATH)
    lock = load_sft_asset_lock(_LOCK_PATH, profile=profile)
    source = tmp_path / "remote"
    source.mkdir()
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (source / "norm_stats.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        stage_level_norm_stats(
            profile=profile,
            asset_lock=lock,
            level=1,
            assets_root=tmp_path / "assets",
            download_asset=lambda **kwargs: source / Path(kwargs["path_in_repo"]).name,
        )

    assert not (tmp_path / "assets").exists()


def test_stage_level_dataset_pins_the_full_asset_revision(tmp_path: Path) -> None:
    profile = load_sft_profile(_PROFILE_PATH)
    lock = load_sft_asset_lock(_LOCK_PATH, profile=profile)
    row = lock.level(1)
    source_manifest = (
        Path("outputs/derived/lerobot/panda-ball-formal-franka-h50-0ac4f2b")
        / row.repo_id
        / "metamdp_dataset.json"
    )
    calls = []

    def download(*, repo_id: str, revision: str, local_dir: Path) -> Path:
        calls.append((repo_id, revision, local_dir))
        local_dir.mkdir(parents=True)
        (local_dir / "metamdp_dataset.json").write_bytes(source_manifest.read_bytes())
        return local_dir

    staged = stage_level_dataset(
        profile=profile,
        asset_lock=lock,
        level=1,
        dataset_root=tmp_path / "datasets",
        download_dataset=download,
    )

    assert staged == tmp_path / "datasets" / row.repo_id
    assert hashlib.sha256((staged / "metamdp_dataset.json").read_bytes()).hexdigest() == (
        row.dataset_manifest_sha256
    )
    assert calls[0][:2] == (row.repo_id, row.asset_revision)
    assert calls[0][2].parent == staged.parent
    assert calls[0][2].name.startswith(f".{staged.name}.building-")


def test_launch_request_preserves_existing_modes_and_experiment_identity() -> None:
    smoke = SFTLaunchRequest(
        level=1,
        experiment_name="l1-clean-h50-smoke-v1",
        mode="smoke",
        resume=False,
        device_count=1,
        batch_size_override=256,
    )
    formal = SFTLaunchRequest(
        level=3,
        experiment_name="l3-clean-h50-full-v1",
        mode="formal",
        resume=True,
        device_count=1,
    )
    smoke_resume = SFTLaunchRequest(
        level=1,
        experiment_name="l1-clean-h50-smoke-v1",
        mode="smoke",
        resume=True,
        device_count=1,
    )

    profile = load_sft_profile(_PROFILE_PATH)
    smoke_schedule = resolve_sft_schedule(profile=profile, request=smoke)
    smoke_resume_schedule = resolve_sft_schedule(profile=profile, request=smoke_resume)
    formal_schedule = resolve_sft_schedule(profile=profile, request=formal)

    assert smoke_schedule.num_train_steps == 100
    assert smoke_schedule.expected_checkpoint_steps == (100,)
    assert smoke.batch_size_override == 256
    assert smoke_resume_schedule.num_train_steps == 120
    assert smoke_resume_schedule.expected_checkpoint_steps == (100, 120)
    assert formal_schedule.num_train_steps == 3_999
    assert formal_schedule.rolling_save_interval == 250
    assert formal_schedule.milestone_interval == 1_333
    assert formal_schedule.expected_checkpoint_steps == (1_333, 2_666, 3_999)

    with pytest.raises(ValueError, match="experiment"):
        SFTLaunchRequest(1, "../bad", "smoke", False, 1)
    with pytest.raises(ValueError, match="device count"):
        SFTLaunchRequest(1, "valid", "formal", False, 0)


@pytest.mark.parametrize("devices,batch,steps", [(2, 64, 11997), (4, 128, 6000), (4, 192, 3999)])
def test_data_parallel_schedule_preserves_sample_budget(devices, batch, steps):
    profile = load_sft_profile(Path("configs/policy/pi05_structured_state16_h50_v1.yaml"))
    request = SFTLaunchRequest(3, "l3-ddp", "formal", False, devices, batch)
    schedule = resolve_sft_schedule(profile=profile, request=request)
    assert schedule.batch_size == batch
    assert schedule.num_train_steps == schedule.decay_steps == steps
    assert schedule.warmup_steps * batch == profile.warmup_steps * profile.batch_size
    assert 0 <= steps * batch - profile.num_train_steps * profile.batch_size < 3 * batch
    assert schedule.expected_checkpoint_steps == tuple(
        schedule.milestone_interval * i for i in (1, 2, 3)
    )
    assert schedule.rolling_save_interval * batch >= profile.save_interval * profile.batch_size


def test_data_parallel_schedule_rejects_sharding_and_uneven_batches():
    profile = load_sft_profile(_PROFILE_PATH)
    with pytest.raises(ValueError, match="divisible"):
        resolve_sft_schedule(
            profile=profile, request=SFTLaunchRequest(3, "uneven", "smoke", False, 4, 127)
        )
    with pytest.raises(ValueError, match="replicated"):
        resolve_sft_schedule(
            profile=dataclasses.replace(profile, fsdp_devices=2),
            request=SFTLaunchRequest(3, "sharded", "formal", False, 4),
        )
