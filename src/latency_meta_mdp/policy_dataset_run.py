"""Atomic conversion of a synchronized pilot run into three LeRobot datasets."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.lerobot_conversion import write_lerobot_policy_dataset
from latency_meta_mdp.policy_data import PolicyEpisode, load_policy_episode
from latency_meta_mdp.sft_profile import load_sft_profile


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


def _load_source_episodes(
    source_manifest: Path,
) -> tuple[dict[str, Any], dict[int, list[PolicyEpisode]]]:
    source = _load_json(source_manifest)
    if source.get("schema_version") != 1 or source.get("format_id") != "expert_pilot_run_v1":
        raise ValueError("source must be an expert_pilot_run_v1 manifest")
    if source.get("implementation_dirty") is not False:
        raise ValueError("pilot conversion requires a clean source implementation")
    rows = source.get("episodes")
    artifacts = source.get("artifacts")
    if (
        not isinstance(rows, list)
        or source.get("episode_count") != len(rows)
        or not isinstance(artifacts, dict)
    ):
        raise ValueError("pilot source inventory is invalid")

    source_root = source_manifest.parent.resolve()
    grouped: dict[int, list[PolicyEpisode]] = defaultdict(list)
    episode_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pilot episode rows must be mappings")
        relative_manifest = row.get("episode_manifest")
        if not isinstance(relative_manifest, str) or relative_manifest not in artifacts:
            raise ValueError("pilot episode manifest is absent from the artifact inventory")
        episode_manifest = (source_root / relative_manifest).resolve()
        if not episode_manifest.is_relative_to(source_root):
            raise ValueError("pilot episode manifest escapes the source root")
        if sha256_file(episode_manifest) != artifacts[relative_manifest]:
            raise ValueError("pilot episode manifest hash mismatch")
        episode = load_policy_episode(episode_manifest.parent)
        if (
            episode.episode_id != row.get("episode_id")
            or episode.level != row.get("level")
            or len(episode.actions) != row.get("transition_count")
            or episode.episode_id in episode_ids
        ):
            raise ValueError("pilot episode identity or length is inconsistent")
        episode_ids.add(episode.episode_id)
        grouped[episode.level].append(episode)
    if set(grouped) != {1, 2, 3}:
        raise ValueError("pilot conversion requires L1, L2, and L3 episodes")
    return source, dict(grouped)


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("formal corpus episode path is unsafe")
    return path


def _load_formal_episode_paths(
    source_manifest: Path,
) -> tuple[dict[str, Any], dict[int, list[Path]]]:
    source = _load_json(source_manifest)
    if (
        source.get("schema_version") != 1
        or source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("source must be an eligible clean panda_ball_formal_corpus_v1")
    admitted = source.get("admitted_episode_manifests")
    artifacts = source.get("artifacts")
    if (
        not isinstance(admitted, list)
        or not isinstance(artifacts, dict)
        or source.get("episode_count") != len(admitted)
    ):
        raise ValueError("formal corpus source inventory is invalid")

    source_root = source_manifest.parent.resolve()
    grouped: dict[int, list[Path]] = defaultdict(list)
    identities: set[tuple[int, int]] = set()
    episode_ids: set[str] = set()
    for relative_text in admitted:
        if not isinstance(relative_text, str):
            raise ValueError("formal corpus episode manifest path is invalid")
        relative = _safe_relative(relative_text)
        episode_manifest = (source_root / relative).resolve()
        if not episode_manifest.is_relative_to(source_root) or sha256_file(
            episode_manifest
        ) != artifacts.get(relative_text):
            raise ValueError("formal corpus episode manifest hash mismatch")
        nested = _load_json(episode_manifest)
        nested_artifacts = nested.get("artifacts")
        if nested.get("format_id") != "synchronized_episode_npz_v3" or not isinstance(
            nested_artifacts, dict
        ):
            raise ValueError("formal corpus episode format is invalid")
        for name, digest in nested_artifacts.items():
            nested_relative = _safe_relative(name)
            source_relative = (relative.parent / nested_relative).as_posix()
            if artifacts.get(source_relative) != digest:
                raise ValueError("formal corpus nested artifact inventory is inconsistent")
        metadata = _load_json(episode_manifest.parent / "metadata.json")
        level = metadata.get("level")
        seed = metadata.get("scene_seed")
        episode_id = metadata.get("episode_id")
        if (
            level not in (1, 2, 3)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(episode_id, str)
            or not episode_id
            or (level, seed) in identities
            or episode_id in episode_ids
        ):
            raise ValueError("formal corpus episode identity is invalid or duplicated")
        identities.add((level, seed))
        episode_ids.add(episode_id)
        grouped[level].append(episode_manifest)
    if set(grouped) != {1, 2, 3}:
        raise ValueError("formal corpus conversion requires L1, L2, and L3 episodes")
    expected_per_level = source.get("seed_count_per_level")
    if (
        isinstance(expected_per_level, bool)
        or not isinstance(expected_per_level, int)
        or expected_per_level <= 0
        or any(len(grouped[level]) != expected_per_level for level in (1, 2, 3))
    ):
        raise ValueError("formal corpus per-level episode counts are inconsistent")
    return source, dict(grouped)


def _convert_grouped_to_lerobot(
    *,
    source_path: Path,
    source: dict[str, Any],
    grouped: dict[int, Iterable[PolicyEpisode]],
    output_dir: Path,
    profile_path: Path,
    output_format_id: str,
    dataset_factory: Callable[..., Any] | None,
) -> Path:
    profile_file = profile_path.resolve()
    profile = load_sft_profile(profile_file)
    project_root = profile_file.parents[2]
    patch_path = project_root / "patches/openpi/0001-filter-incomplete-action-chunks.patch"
    if sha256_file(patch_path) != profile.openpi_patch_sha256:
        raise ValueError("SFT profile does not match the project OpenPI patch")

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"derived policy output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"derived policy staging output already exists: {staging}")

    dataset_rows: list[dict[str, Any]] = []
    try:
        staging.mkdir()
        for level in (1, 2, 3):
            repo_id = profile.levels[level].repo_id
            dataset_manifest = write_lerobot_policy_dataset(
                episodes=grouped[level],
                output_dir=staging / repo_id,
                repo_id=repo_id,
                dataset_factory=dataset_factory,
            )
            dataset = _load_json(dataset_manifest)
            dataset_rows.append(
                {
                    "level": level,
                    "repo_id": repo_id,
                    "episode_count": dataset["episode_count"],
                    "frame_count": dataset["frame_count"],
                    "valid_action_chunk_source_count": dataset["valid_action_chunk_source_count"],
                    "dataset_manifest": dataset_manifest.relative_to(staging).as_posix(),
                    "dataset_manifest_sha256": sha256_file(dataset_manifest),
                }
            )
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": output_format_id,
                "source_format_id": source["format_id"],
                "source_run_id": source.get("run_id"),
                "source_implementation_revision": source["implementation_revision"],
                "source_manifest_sha256": sha256_file(source_path),
                "sft_profile_id": profile.profile_id,
                "sft_profile_sha256": sha256_file(profile_file),
                "openpi_revision": profile.openpi_revision,
                "openpi_patch_sha256": profile.openpi_patch_sha256,
                "episode_count": sum(row["episode_count"] for row in dataset_rows),
                "datasets": dataset_rows,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"


def convert_pilot_run_to_lerobot(
    *,
    source_manifest: Path,
    output_dir: Path,
    profile_path: Path,
    dataset_factory: Callable[..., Any] | None = None,
) -> Path:
    """Convert one clean pilot run into an atomic level-specific dataset run."""

    source_path = source_manifest.resolve()
    source, grouped = _load_source_episodes(source_path)
    return _convert_grouped_to_lerobot(
        source_path=source_path,
        source=source,
        grouped=grouped,
        output_dir=output_dir,
        profile_path=profile_path,
        output_format_id="metamdp_lerobot_pilot_run_v1",
        dataset_factory=dataset_factory,
    )


def convert_formal_corpus_to_lerobot(
    *,
    source_manifest: Path,
    output_dir: Path,
    profile_path: Path,
    dataset_factory: Callable[..., Any] | None = None,
) -> Path:
    """Stream one verified formal corpus into three level-specific datasets."""

    source_path = source_manifest.resolve()
    source, grouped_paths = _load_formal_episode_paths(source_path)
    grouped = {
        level: (load_policy_episode(path.parent) for path in grouped_paths[level])
        for level in (1, 2, 3)
    }
    return _convert_grouped_to_lerobot(
        source_path=source_path,
        source=source,
        grouped=grouped,
        output_dir=output_dir,
        profile_path=profile_path,
        output_format_id="metamdp_lerobot_formal_corpus_v1",
        dataset_factory=dataset_factory,
    )
