"""Validated local assets and launch contract for one clean π0.5 SFT run."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.sft_asset_lock import SFTAssetLock
from latency_meta_mdp.sft_profile import SFTProfile

_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


@dataclass(frozen=True)
class SFTLaunchRequest:
    level: int
    experiment_name: str
    mode: str
    resume: bool
    device_count: int
    batch_size_override: int | None = None

    def __post_init__(self) -> None:
        if self.level not in (1, 2, 3):
            raise ValueError("SFT launch level must be 1, 2, or 3")
        if _EXPERIMENT_NAME.fullmatch(self.experiment_name) is None:
            raise ValueError("SFT experiment name is invalid")
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("SFT launch mode must be smoke or formal")
        if type(self.device_count) is not int or self.device_count <= 0:
            raise ValueError("SFT device count must be a positive integer")
        if self.batch_size_override is not None:
            if type(self.batch_size_override) is not int or self.batch_size_override <= 0:
                raise ValueError("global batch-size override must be a positive integer")


@dataclass(frozen=True)
class SFTSchedule:
    num_train_steps: int
    rolling_save_interval: int
    milestone_interval: int
    expected_checkpoint_steps: tuple[int, ...]
    batch_size: int
    warmup_steps: int
    decay_steps: int


def resolve_sft_schedule(*, profile: SFTProfile, request: SFTLaunchRequest) -> SFTSchedule:
    """Keep sample exposure when a replicated-data-parallel run changes its batch.

    Each formal milestone rounds up to a whole global batch. The full run can
    therefore exceed the profile budget by fewer than three global batches.
    Equal sample exposure does not imply identical optimizer trajectories.
    """

    if profile.fsdp_devices != 1:
        raise ValueError("SFT currently requires replicated data parallelism (fsdp_devices=1)")
    batch = request.batch_size_override or profile.batch_size
    if batch % request.device_count:
        raise ValueError("global batch size must be divisible by the SFT device count")

    def scaled(steps: int) -> int:
        return (steps * profile.batch_size + batch - 1) // batch

    milestone = scaled(profile.keep_period)
    decay = 3 * milestone
    warmup = scaled(profile.warmup_steps)

    if request.mode == "smoke":
        if request.resume:
            return SFTSchedule(120, 20, 20, (100, 120), batch, warmup, decay)
        return SFTSchedule(100, 100, 100, (100,), batch, warmup, decay)
    if scaled(profile.save_interval) >= milestone:
        raise ValueError("global batch is too large to preserve distinct checkpoint intervals")
    return SFTSchedule(
        num_train_steps=decay,
        rolling_save_interval=scaled(profile.save_interval),
        milestone_interval=milestone,
        expected_checkpoint_steps=tuple(milestone * i for i in (1, 2, 3)),
        batch_size=batch,
        warmup_steps=warmup,
        decay_steps=decay,
    )


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
