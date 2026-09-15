"""Immutable Hugging Face dataset and OpenPI normalization-asset revisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from latency_meta_mdp.policy.profile import SFTProfile

_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SFTLevelAssetLock:
    repo_id: str
    data_revision: str
    asset_revision: str
    dataset_manifest_sha256: str
    norm_stats_sha256: str
    norm_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.repo_id.count("/") != 1 or any(
            not component for component in self.repo_id.split("/")
        ):
            raise ValueError("SFT asset repo id is invalid")
        if (
            _GIT_SHA1.fullmatch(self.data_revision) is None
            or _GIT_SHA1.fullmatch(self.asset_revision) is None
        ):
            raise ValueError("SFT asset revisions must be full lowercase Git SHAs")
        if self.data_revision == self.asset_revision:
            raise ValueError("data and asset revisions must differ")
        for value in (
            self.dataset_manifest_sha256,
            self.norm_stats_sha256,
            self.norm_manifest_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("SFT asset digests must be full lowercase SHA256 values")


@dataclass(frozen=True)
class SFTAssetLock:
    schema_version: int
    lock_id: str
    profile_id: str
    levels: Mapping[int, SFTLevelAssetLock]

    def __post_init__(self) -> None:
        levels = dict(self.levels)
        object.__setattr__(self, "levels", MappingProxyType(levels))
        if self.schema_version != 1 or self.lock_id != "franka_moving_ball_sft_assets_v1":
            raise ValueError("unsupported SFT asset-lock schema or identifier")
        if not self.profile_id:
            raise ValueError("SFT asset lock profile_id cannot be empty")
        if set(levels) != {1, 2, 3}:
            raise ValueError("SFT asset lock must define L1, L2, and L3")
        if len({row.repo_id for row in levels.values()}) != 3:
            raise ValueError("SFT asset repo ids must be unique")

    def level(self, level: int) -> SFTLevelAssetLock:
        try:
            return self.levels[level]
        except KeyError as exc:
            raise ValueError("SFT asset level must be 1, 2, or 3") from exc


def load_sft_asset_lock(path: Path, *, profile: SFTProfile) -> SFTAssetLock:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "lock_id",
        "profile_id",
        "levels",
    }:
        raise ValueError("SFT asset-lock fields are invalid")
    raw_levels = value["levels"]
    if not isinstance(raw_levels, dict) or any(
        not isinstance(row, dict) for row in raw_levels.values()
    ):
        raise ValueError("SFT asset-lock levels are invalid")
    expected_fields = set(SFTLevelAssetLock.__dataclass_fields__)
    if any(set(row) != expected_fields for row in raw_levels.values()):
        raise ValueError("SFT level asset-lock fields are invalid")
    levels = {level: SFTLevelAssetLock(**row) for level, row in raw_levels.items()}
    lock = SFTAssetLock(
        schema_version=value["schema_version"],
        lock_id=value["lock_id"],
        profile_id=value["profile_id"],
        levels=levels,
    )
    if lock.profile_id != profile.profile_id:
        raise ValueError("SFT asset lock and profile identifiers disagree")
    for level, row in lock.levels.items():
        if row.repo_id != profile.levels[level].repo_id:
            raise ValueError("SFT asset repo does not match the profile")
    return lock
