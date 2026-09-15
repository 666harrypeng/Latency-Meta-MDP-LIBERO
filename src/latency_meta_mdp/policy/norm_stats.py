"""Immutable OpenPI normalization assets for one formal SFT level."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.policy.profile import load_sft_profile

_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_STATE_DIM = 8
_ACTION_DIM = 7


@dataclass(frozen=True)
class NormStatsComputation:
    source_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count <= 0
        ):
            raise ValueError("norm-stat source count must be a positive integer")


NormStatsBackend = Callable[..., NormStatsComputation]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _single_level_row(rows: Any, *, level: int, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} level inventory is invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("level") == level]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one row for L{level}")
    return matches[0]


def _validate_stat_vector(value: Any, *, name: str, width: int) -> None:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain {width} finite values")


def _validate_norm_stats(path: Path) -> None:
    payload = _load_json(path)
    stats = payload.get("norm_stats")
    if not isinstance(stats, dict) or set(stats) != {"state", "actions"}:
        raise ValueError("norm stats must contain state and actions")
    for name, width in (("state", _STATE_DIM), ("actions", _ACTION_DIM)):
        row = stats[name]
        if not isinstance(row, dict) or set(row) != {"mean", "std", "q01", "q99"}:
            raise ValueError(f"{name} norm stats fields are invalid")
        for field in ("mean", "std", "q01", "q99"):
            _validate_stat_vector(
                row[field],
                name=f"{name}.{field}",
                width=width,
            )
        if any(float(value) < 0.0 for value in row["std"]):
            raise ValueError(f"{name}.std cannot be negative")
        if any(
            float(lower) > float(upper) for lower, upper in zip(row["q01"], row["q99"], strict=True)
        ):
            raise ValueError(f"{name} quantile bounds are invalid")


def compute_level_sft_norm_stats(
    *,
    project_root: Path,
    derived_manifest: Path,
    certification_manifest: Path,
    profile_path: Path,
    patch_path: Path,
    level: int,
    data_revision: str,
    output_dir: Path,
    compute_backend: NormStatsBackend,
) -> Path:
    """Compute and atomically publish one level's certified norm stats."""

    if level not in (1, 2, 3):
        raise ValueError("norm stats level must be 1, 2, or 3")
    if _GIT_SHA1.fullmatch(data_revision) is None:
        raise ValueError("data revision must be a full lowercase Git SHA")
    if not callable(compute_backend):
        raise TypeError("compute_backend must be callable")

    project = project_root.resolve()
    derived_path = derived_manifest.resolve()
    certification_path = certification_manifest.resolve()
    profile_file = profile_path.resolve()
    patch_file = patch_path.resolve()
    for path in (derived_path, certification_path, profile_file, patch_file):
        if not path.is_file():
            raise FileNotFoundError(f"norm-stat input does not exist: {path}")

    profile = load_sft_profile(profile_file)
    if sha256_file(patch_file) != profile.openpi_patch_sha256:
        raise ValueError("norm-stat patch does not match the SFT profile")
    derived = _load_json(derived_path)
    certification = _load_json(certification_path)
    profile_hash = sha256_file(profile_file)
    derived_hash = sha256_file(derived_path)
    if (
        derived.get("schema_version") != 1
        or derived.get("format_id") != "metamdp_lerobot_formal_corpus_v1"
        or derived.get("sft_profile_id") != profile.profile_id
        or derived.get("sft_profile_sha256") != profile_hash
        or derived.get("openpi_revision") != profile.openpi_revision
        or derived.get("openpi_patch_sha256") != profile.openpi_patch_sha256
    ):
        raise ValueError("derived LeRobot corpus does not match the SFT profile")
    if (
        certification.get("schema_version") != 1
        or certification.get("format_id") != "metamdp_openpi_formal_certification_v1"
        or certification.get("eligible") is not True
        or certification.get("implementation_dirty") is not False
        or certification.get("derived_manifest_sha256") != derived_hash
        or certification.get("sft_profile_id") != profile.profile_id
        or certification.get("sft_profile_sha256") != profile_hash
        or certification.get("openpi_revision") != profile.openpi_revision
        or certification.get("openpi_patch_sha256") != profile.openpi_patch_sha256
    ):
        raise ValueError("norm stats require an eligible clean formal certification")

    derived_row = _single_level_row(derived.get("datasets"), level=level, label="derived corpus")
    certification_row = _single_level_row(
        certification.get("levels"),
        level=level,
        label="certification",
    )
    expected_repo = profile.levels[level].repo_id
    relative_manifest = derived_row.get("dataset_manifest")
    if not isinstance(relative_manifest, str) or not relative_manifest:
        raise ValueError("derived dataset manifest path is invalid")
    derived_root = derived_path.parent.resolve()
    dataset_manifest = (derived_root / relative_manifest).resolve()
    if not dataset_manifest.is_relative_to(derived_root) or not dataset_manifest.is_file():
        raise ValueError("derived dataset manifest escapes or is missing")
    dataset_manifest_hash = sha256_file(dataset_manifest)
    expected_source_count = derived_row.get("valid_action_chunk_source_count")
    if (
        derived_row.get("repo_id") != expected_repo
        or certification_row.get("repo_id") != expected_repo
        or derived_row.get("dataset_manifest_sha256") != dataset_manifest_hash
        or certification_row.get("dataset_manifest_sha256") != dataset_manifest_hash
        or certification_row.get("source_count") != expected_source_count
        or certification_row.get("norm_source_count") != expected_source_count
        or certification_row.get("no_action_padding") is not True
        or isinstance(expected_source_count, bool)
        or not isinstance(expected_source_count, int)
        or expected_source_count <= 0
    ):
        raise ValueError("derived and certified level data are inconsistent")
    dataset = _load_json(dataset_manifest)
    if (
        dataset.get("repo_id") != expected_repo
        or dataset.get("level") != level
        or dataset.get("valid_action_chunk_source_count") != expected_source_count
    ):
        raise ValueError("dataset manifest does not match the certified level")

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"norm-stat output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        norm_path = building / "norm_stats.json"
        computation = compute_backend(
            level=level,
            repo_id=expected_repo,
            dataset_root=derived_root,
            expected_source_count=expected_source_count,
            output_path=norm_path,
        )
        if not isinstance(computation, NormStatsComputation):
            raise TypeError("compute_backend must return NormStatsComputation")
        if computation.source_count != expected_source_count:
            raise ValueError("norm-stat backend source count does not match certification")
        if not norm_path.is_file():
            raise ValueError("norm-stat backend did not write norm_stats.json")
        _validate_norm_stats(norm_path)

        provenance = collect_implementation_provenance(project)
        manifest = {
            "schema_version": 1,
            "format_id": "metamdp_openpi_norm_stats_v1",
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "profile_id": profile.profile_id,
            "profile_sha256": profile_hash,
            "openpi_revision": profile.openpi_revision,
            "openpi_patch_sha256": profile.openpi_patch_sha256,
            "level": level,
            "repo_id": expected_repo,
            "data_revision": data_revision,
            "source_count": expected_source_count,
            "state_dim": _STATE_DIM,
            "action_dim": _ACTION_DIM,
            "input_sha256": {
                "derived_manifest": derived_hash,
                "certification_manifest": sha256_file(certification_path),
                "dataset_manifest": dataset_manifest_hash,
                "profile": profile_hash,
                "patch": sha256_file(patch_file),
            },
            "artifacts": {"norm_stats.json": sha256_file(norm_path)},
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
