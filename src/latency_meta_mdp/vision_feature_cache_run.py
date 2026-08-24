"""Audited multi-episode runner for frozen vision feature caches."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.vision_feature_cache import (
    VisionFeatureEncoder,
    write_episode_vision_feature_cache,
)

_SOURCE_FORMAT_ID = "panda_ball_bulk_first_tranche_v1"
_RUN_FORMAT_ID = "vision_feature_cache_run_v1"


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


def _validate_selection(
    *,
    levels: tuple[int, ...],
    seed_start: int,
    seed_count: int,
    boundary_batch_size: int,
) -> tuple[int, ...]:
    normalized = tuple(sorted(set(levels)))
    if not normalized or normalized != levels or any(level not in (1, 2, 3) for level in levels):
        raise ValueError("vision cache levels must be sorted unique values from 1, 2, 3")
    for name, value in (
        ("seed_start", seed_start),
        ("seed_count", seed_count),
        ("boundary_batch_size", boundary_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return normalized


def _validate_source(path: Path) -> tuple[dict[str, Any], Path]:
    source = _load_json(path)
    if (
        source.get("format_id") != _SOURCE_FORMAT_ID
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("vision feature caching requires an eligible clean bulk source")
    admitted = source.get("admitted_episode_manifests")
    artifacts = source.get("artifacts")
    if not isinstance(admitted, list) or not admitted or not isinstance(artifacts, dict):
        raise ValueError("bulk source inventory is invalid")
    return source, path.parent.resolve()


def write_vision_feature_cache_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    encoder: VisionFeatureEncoder,
    output_dir: Path,
    levels: tuple[int, ...],
    seed_start: int,
    seed_count: int,
    boundary_batch_size: int,
) -> Path:
    selected_levels = _validate_selection(
        levels=levels,
        seed_start=seed_start,
        seed_count=seed_count,
        boundary_batch_size=boundary_batch_size,
    )
    source_path = source_bulk_manifest.resolve()
    source, source_root = _validate_source(source_path)
    provenance = collect_implementation_provenance(project_root)
    admitted = set(source["admitted_episode_manifests"])
    source_artifacts = source["artifacts"]
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"vision feature cache run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    episodes: list[dict[str, Any]] = []
    try:
        for level in selected_levels:
            for seed in range(seed_start, seed_start + seed_count):
                source_relative = f"attempts/L{level}/seed_{seed:06d}/manifest.json"
                if source_relative not in admitted:
                    raise ValueError(f"bulk source does not admit {source_relative}")
                source_episode_manifest = source_root / source_relative
                expected_digest = source_artifacts.get(source_relative)
                if (
                    not isinstance(expected_digest, str)
                    or sha256_file(source_episode_manifest) != expected_digest
                ):
                    raise ValueError("bulk source episode-manifest verification failed")
                episode = load_belief_episode(source_episode_manifest.parent)
                if episode.level != level or episode.scene_seed != seed:
                    raise ValueError("selected episode identity disagrees with its source path")
                cache_root = building / f"L{level}" / f"seed_{seed:06d}"
                cache_manifest = write_episode_vision_feature_cache(
                    episode=episode,
                    source_episode_manifest=source_episode_manifest,
                    encoder=encoder,
                    output_dir=cache_root,
                    boundary_batch_size=boundary_batch_size,
                )
                episodes.append(
                    {
                        "level": level,
                        "scene_seed": seed,
                        "episode_id": episode.episode_id,
                        "boundary_count": episode.boundary_count,
                        "source_episode_manifest": source_relative,
                        "cache_manifest": cache_manifest.relative_to(building).as_posix(),
                    }
                )
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        feature_bytes = sum(
            (building / relative).stat().st_size
            for relative in artifacts
            if relative.endswith("features.npy")
        )
        spec = encoder.spec
        manifest = {
            "schema_version": 1,
            "format_id": _RUN_FORMAT_ID,
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "source_bulk_manifest_sha256": sha256_file(source_path),
            "source_bulk_run_id": source["run_id"],
            "encoder_id": spec.encoder_id,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "weights_sha256": spec.weights_sha256,
            "encoder_fingerprint": spec.fingerprint,
            "runtime": asdict(encoder.runtime_info),
            "levels": list(selected_levels),
            "seed_start": seed_start,
            "seed_count": seed_count,
            "boundary_batch_size": boundary_batch_size,
            "maximum_image_batch_size": boundary_batch_size * 2,
            "episode_count": len(episodes),
            "feature_bytes": feature_bytes,
            "episodes": episodes,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"vision feature cache run already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
