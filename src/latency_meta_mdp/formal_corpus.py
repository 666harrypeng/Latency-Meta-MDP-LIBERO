"""Audited hard-link materialization of immutable expert collection runs."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file

_SOURCE_FORMATS = {
    "panda_ball_bulk_first_tranche_v1",
    "panda_ball_bulk_range_v1",
}


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


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("formal corpus source path is unsafe")
    return path


def _semantic_signature(source: dict[str, Any], metadata: dict[str, Any]) -> tuple:
    config = metadata.get("config_sha256")
    if not isinstance(config, dict):
        raise ValueError("episode metadata lacks configuration hashes")
    return (
        source.get("collection_id"),
        source.get("bulk_plan_sha256"),
        source.get("record_profile"),
        source.get("camera_width"),
        source.get("camera_height"),
        metadata.get("schema_version"),
        metadata.get("formal_tick_us"),
        metadata.get("physics_dt_us"),
        metadata.get("record_profile"),
        metadata.get("task_id"),
        metadata.get("action_contract_id"),
        metadata.get("action_dim"),
        metadata.get("actuator_dim"),
        metadata.get("expert_id"),
        metadata.get("instruction"),
        tuple(sorted(config.items())),
    )


def materialize_formal_corpus(
    *,
    project_root: Path,
    source_manifests: tuple[Path, ...],
    output_dir: Path,
    levels: tuple[int, ...],
    seed_start: int,
    seed_count: int,
) -> Path:
    if not source_manifests:
        raise ValueError("formal corpus requires source manifests")
    if not levels or tuple(sorted(set(levels))) != levels:
        raise ValueError("formal corpus levels must be sorted and unique")
    if seed_count <= 0:
        raise ValueError("formal corpus seed count must be positive")
    expected = {
        (level, seed)
        for level in levels
        for seed in range(seed_start, seed_start + seed_count)
    }
    episodes: dict[tuple[int, int], dict[str, Any]] = {}
    signatures: dict[int, tuple] = {}
    sources = []
    for manifest_path in source_manifests:
        path = manifest_path.resolve()
        source = _load_json(path)
        if (
            source.get("format_id") not in _SOURCE_FORMATS
            or source.get("eligible") is not True
            or source.get("implementation_dirty") is not False
        ):
            raise ValueError("formal corpus source is not an eligible clean collection")
        root = path.parent
        artifacts = source.get("artifacts")
        admitted = source.get("admitted_episode_manifests")
        if not isinstance(artifacts, dict) or not isinstance(admitted, list):
            raise ValueError("formal corpus source inventory is invalid")
        for relative_text in admitted:
            relative = _safe_relative(relative_text)
            episode_manifest = root / relative
            if sha256_file(episode_manifest) != artifacts.get(relative_text):
                raise ValueError("source episode manifest hash mismatch")
            nested = _load_json(episode_manifest)
            nested_artifacts = nested.get("artifacts")
            if (
                nested.get("format_id") != "synchronized_episode_npz_v3"
                or not isinstance(nested_artifacts, dict)
            ):
                raise ValueError("formal corpus episode format is invalid")
            for name, digest in nested_artifacts.items():
                artifact = episode_manifest.parent / _safe_relative(name)
                if sha256_file(artifact) != digest:
                    raise ValueError("nested episode artifact hash mismatch")
            metadata = _load_json(episode_manifest.parent / "metadata.json")
            identity = (metadata.get("level"), metadata.get("scene_seed"))
            if identity in episodes:
                raise ValueError("formal corpus sources contain duplicate level/seed identities")
            if identity not in expected:
                raise ValueError("formal corpus source identity lies outside expected coverage")
            signature = _semantic_signature(source, metadata)
            level = int(identity[0])
            if level in signatures and signatures[level] != signature:
                raise ValueError("formal corpus episode semantics are incompatible")
            signatures[level] = signature
            episodes[identity] = {
                "episode_manifest": episode_manifest,
                "nested_artifacts": nested_artifacts,
                "source_run_id": source.get("run_id"),
            }
        sources.append(
            {
                "manifest_sha256": sha256_file(path),
                "run_id": source.get("run_id"),
                "implementation_revision": source.get("implementation_revision"),
                "implementation_source_sha256": source.get(
                    "implementation_source_sha256"
                ),
            }
        )
    if set(episodes) != expected:
        missing = sorted(expected - set(episodes))
        raise ValueError(f"formal corpus sources have missing identities: {missing[:5]}")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"formal corpus output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    provenance = collect_implementation_provenance(project_root)
    linked_manifests = []
    artifacts = {}
    try:
        for (level, seed), row in sorted(episodes.items()):
            source_root = row["episode_manifest"].parent
            episode_root = building / "episodes" / f"L{level}" / f"seed_{seed:06d}"
            episode_root.mkdir(parents=True)
            names = ("manifest.json", *sorted(row["nested_artifacts"]))
            for name in names:
                destination = episode_root / name
                os.link(source_root / name, destination)
                artifacts[destination.relative_to(building).as_posix()] = sha256_file(
                    destination
                )
            linked_manifests.append(
                (episode_root / "manifest.json").relative_to(building).as_posix()
            )
        manifest = {
            "schema_version": 1,
            "format_id": "panda_ball_formal_corpus_v1",
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "levels": list(levels),
            "seed_start": seed_start,
            "seed_count_per_level": seed_count,
            "episode_count": len(episodes),
            "source_runs": sources,
            "admitted_episode_manifests": linked_manifests,
            "artifacts": dict(sorted(artifacts.items())),
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
