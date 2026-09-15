"""Certification artifact for level-specific LeRobot and OpenPI pilot loading."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.policy.profile import SFTProfile, load_sft_profile

LevelProbe = Callable[..., Mapping[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_dataset_artifacts(dataset_root: Path, dataset: dict[str, Any]) -> None:
    artifacts = dataset.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("derived dataset artifact inventory is invalid")
    for relative_path, expected_hash in artifacts.items():
        path = (dataset_root / relative_path).resolve()
        if not path.is_relative_to(dataset_root) or sha256_file(path) != expected_hash:
            raise ValueError("derived dataset artifact hash mismatch")


def _validate_probe(
    *,
    result: Mapping[str, Any],
    expected_episode_count: int,
    expected_frame_count: int,
    expected_source_count: int,
    profile: SFTProfile,
) -> None:
    norm_batch_sizes = result.get("norm_batch_sizes")
    state_shape = result.get("train_state_shape")
    action_shape = result.get("train_action_shape")
    if not (
        result.get("metadata_fps") == profile.fps
        and result.get("episode_count") == expected_episode_count
        and result.get("frame_count") == expected_frame_count
        and result.get("source_count") == expected_source_count
        and result.get("norm_source_count") == expected_source_count
        and isinstance(norm_batch_sizes, list)
        and all(isinstance(size, int) and size > 0 for size in norm_batch_sizes)
        and sum(norm_batch_sizes) == expected_source_count
        and result.get("no_action_padding") is True
        and isinstance(state_shape, list)
        and len(state_shape) == 2
        and state_shape[0] > 0
        and state_shape[1] == 32
        and isinstance(action_shape, list)
        and action_shape == [state_shape[0], profile.action_horizon, 32]
    ):
        raise ValueError("OpenPI pilot probe does not satisfy the SFT contract")


def _certify_lerobot_run(
    *,
    project_root: Path,
    derived_manifest: Path,
    output_dir: Path,
    profile_path: Path,
    level_probe: LevelProbe,
    expected_derived_format_id: str,
    certification_format_id: str,
) -> Path:
    project = project_root.resolve()
    derived_path = derived_manifest.resolve()
    profile_file = profile_path.resolve()
    derived = _load_json(derived_path)
    profile = load_sft_profile(profile_file)
    if (
        derived.get("schema_version") != 1
        or derived.get("format_id") != expected_derived_format_id
        or derived.get("sft_profile_id") != profile.profile_id
        or derived.get("sft_profile_sha256") != sha256_file(profile_file)
        or derived.get("openpi_revision") != profile.openpi_revision
        or derived.get("openpi_patch_sha256") != profile.openpi_patch_sha256
    ):
        raise ValueError("derived LeRobot run does not match the active SFT profile")
    rows = derived.get("datasets")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("derived LeRobot run must contain three datasets")

    derived_root = derived_path.parent.resolve()
    certified_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: value["level"]):
        level = row.get("level")
        if level not in (1, 2, 3) or row.get("repo_id") != profile.levels[level].repo_id:
            raise ValueError("derived LeRobot level identity is invalid")
        relative_manifest = row.get("dataset_manifest")
        if not isinstance(relative_manifest, str):
            raise ValueError("derived dataset manifest path is invalid")
        dataset_manifest = (derived_root / relative_manifest).resolve()
        if not dataset_manifest.is_relative_to(derived_root) or sha256_file(
            dataset_manifest
        ) != row.get("dataset_manifest_sha256"):
            raise ValueError("derived dataset manifest hash mismatch")
        dataset = _load_json(dataset_manifest)
        _validate_dataset_artifacts(dataset_manifest.parent, dataset)
        expected_episode_count = row["episode_count"]
        expected_frame_count = row["frame_count"]
        expected_source_count = row["valid_action_chunk_source_count"]
        if (
            dataset.get("repo_id") != row["repo_id"]
            or dataset.get("level") != level
            or dataset.get("episode_count") != expected_episode_count
            or dataset.get("frame_count") != expected_frame_count
            or dataset.get("valid_action_chunk_source_count") != expected_source_count
        ):
            raise ValueError("derived dataset counts are inconsistent")
        result = dict(
            level_probe(
                level=level,
                repo_id=row["repo_id"],
                lerobot_home=derived_root,
                profile=profile,
                expected_episode_count=expected_episode_count,
                expected_frame_count=expected_frame_count,
                expected_source_count=expected_source_count,
            )
        )
        _validate_probe(
            result=result,
            expected_episode_count=expected_episode_count,
            expected_frame_count=expected_frame_count,
            expected_source_count=expected_source_count,
            profile=profile,
        )
        certified_rows.append(
            {
                "level": level,
                "repo_id": row["repo_id"],
                "dataset_manifest": relative_manifest,
                "dataset_manifest_sha256": row["dataset_manifest_sha256"],
                **result,
            }
        )

    provenance = collect_implementation_provenance(project)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"certification output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"certification staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": certification_format_id,
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "eligible": not provenance.dirty,
                "derived_manifest_sha256": sha256_file(derived_path),
                "sft_profile_id": profile.profile_id,
                "sft_profile_sha256": sha256_file(profile_file),
                "openpi_revision": profile.openpi_revision,
                "openpi_patch_sha256": profile.openpi_patch_sha256,
                "levels": certified_rows,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"


def certify_lerobot_pilot_run(
    *,
    project_root: Path,
    derived_manifest: Path,
    output_dir: Path,
    profile_path: Path,
    level_probe: LevelProbe,
) -> Path:
    """Certify all three pilot datasets and write one immutable evidence manifest."""

    return _certify_lerobot_run(
        project_root=project_root,
        derived_manifest=derived_manifest,
        output_dir=output_dir,
        profile_path=profile_path,
        level_probe=level_probe,
        expected_derived_format_id="metamdp_lerobot_pilot_run_v1",
        certification_format_id="metamdp_openpi_pilot_certification_v1",
    )


def certify_lerobot_formal_run(
    *,
    project_root: Path,
    derived_manifest: Path,
    output_dir: Path,
    profile_path: Path,
    level_probe: LevelProbe,
) -> Path:
    """Certify all three formal datasets and write immutable loader evidence."""

    return _certify_lerobot_run(
        project_root=project_root,
        derived_manifest=derived_manifest,
        output_dir=output_dir,
        profile_path=profile_path,
        level_probe=level_probe,
        expected_derived_format_id="metamdp_lerobot_formal_corpus_v1",
        certification_format_id="metamdp_openpi_formal_certification_v1",
    )
