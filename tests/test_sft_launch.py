from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from latency_meta_mdp.sft_asset_lock import load_sft_asset_lock
from latency_meta_mdp.sft_launch import (
    SFTLaunchRequest,
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


def test_launch_request_locks_single_h200_modes_and_experiment_identity() -> None:
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

    assert smoke.num_train_steps == 100
    assert smoke.expected_checkpoint_steps == (100,)
    assert smoke.batch_size_override == 256
    assert smoke_resume.num_train_steps == 120
    assert smoke_resume.expected_checkpoint_steps == (100, 120)
    assert formal.num_train_steps == 12000
    assert formal.expected_checkpoint_steps == (4000, 8000, 12000)

    with pytest.raises(ValueError, match="experiment"):
        SFTLaunchRequest(1, "../bad", "smoke", False, 1)
    with pytest.raises(ValueError, match="one H200"):
        SFTLaunchRequest(1, "valid", "formal", False, 2)
    with pytest.raises(ValueError, match="smoke"):
        SFTLaunchRequest(
            1,
            "valid",
            "formal",
            False,
            1,
            batch_size_override=256,
        )
