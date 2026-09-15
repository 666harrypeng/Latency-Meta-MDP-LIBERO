"""Historical asset-lock staging for the retired clean dataset chain."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.policy.sft_asset_lock import SFTAssetLock
from latency_meta_mdp.policy.profile import SFTProfile

AssetDownloader = Callable[..., Path]
DatasetDownloader = Callable[..., Path]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def stage_level_dataset(
    *,
    profile: SFTProfile,
    asset_lock: SFTAssetLock,
    level: int,
    dataset_root: Path,
    download_dataset: DatasetDownloader,
) -> Path:
    """Materialize one complete dataset at its immutable asset revision."""

    if level not in (1, 2, 3):
        raise ValueError("SFT dataset level must be 1, 2, or 3")
    if not callable(download_dataset):
        raise TypeError("download_dataset must be callable")
    row = asset_lock.level(level)
    if row.repo_id != profile.levels[level].repo_id:
        raise ValueError("SFT profile and asset lock repo ids disagree")
    target = dataset_root.resolve() / row.repo_id
    dataset_manifest = target / "metamdp_dataset.json"
    if target.exists():
        if (
            not dataset_manifest.is_file()
            or sha256_file(dataset_manifest) != row.dataset_manifest_sha256
        ):
            raise ValueError("existing staged dataset has the wrong manifest hash")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    try:
        downloaded = Path(
            download_dataset(
                repo_id=row.repo_id,
                revision=row.asset_revision,
                local_dir=staging,
            )
        ).resolve()
        if downloaded != staging.resolve():
            raise ValueError("dataset downloader returned an unexpected directory")
        staged_manifest = staging / "metamdp_dataset.json"
        if (
            not staged_manifest.is_file()
            or sha256_file(staged_manifest) != row.dataset_manifest_sha256
        ):
            raise ValueError("downloaded SFT dataset manifest hash mismatch")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def stage_level_norm_stats(
    *,
    profile: SFTProfile,
    asset_lock: SFTAssetLock,
    level: int,
    assets_root: Path,
    download_asset: AssetDownloader,
) -> Path:
    """Download, verify, and atomically stage one pinned norm-stat file."""

    if level not in (1, 2, 3):
        raise ValueError("SFT asset level must be 1, 2, or 3")
    if not callable(download_asset):
        raise TypeError("download_asset must be callable")
    row = asset_lock.level(level)
    if row.repo_id != profile.levels[level].repo_id:
        raise ValueError("SFT profile and asset lock repo ids disagree")
    remote_manifest = Path(
        download_asset(
            repo_id=row.repo_id,
            revision=row.asset_revision,
            path_in_repo="metamdp_assets/openpi/manifest.json",
        )
    ).resolve()
    remote_stats = Path(
        download_asset(
            repo_id=row.repo_id,
            revision=row.asset_revision,
            path_in_repo="metamdp_assets/openpi/norm_stats.json",
        )
    ).resolve()
    if (
        not remote_manifest.is_file()
        or sha256_file(remote_manifest) != row.norm_manifest_sha256
        or not remote_stats.is_file()
        or sha256_file(remote_stats) != row.norm_stats_sha256
    ):
        raise ValueError("downloaded SFT normalization asset hash mismatch")
    manifest = _load_json(remote_manifest)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format_id") != "metamdp_openpi_norm_stats_v1"
        or manifest.get("eligible") is not True
        or manifest.get("implementation_dirty") is not False
        or manifest.get("profile_id") != profile.profile_id
        or manifest.get("level") != level
        or manifest.get("repo_id") != row.repo_id
        or manifest.get("data_revision") != row.data_revision
        or manifest.get("artifacts") != {"norm_stats.json": row.norm_stats_sha256}
    ):
        raise ValueError("downloaded SFT normalization manifest is inconsistent")

    target = (
        assets_root.resolve() / profile.levels[level].config_name / row.repo_id / "norm_stats.json"
    )
    if target.exists():
        if sha256_file(target) != row.norm_stats_sha256:
            raise ValueError("existing staged norm stats have the wrong hash")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    try:
        shutil.copyfile(remote_stats, staging)
        if sha256_file(staging) != row.norm_stats_sha256:
            raise ValueError("staged norm stats changed during copy")
        staging.rename(target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return target
