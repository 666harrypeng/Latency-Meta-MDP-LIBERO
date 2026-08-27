"""Atomic publication for multi-level Flow Belief ghost diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import (
    collect_implementation_provenance,
    sha256_file,
)

LevelRenderer = Callable[..., Path]


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


def _verify_manifest_artifacts(path: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"manifest artifact inventory is invalid: {path}")
    root = path.parent.resolve()
    for relative, expected_hash in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"manifest artifact entry is invalid: {path}")
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest artifact escapes its root: {relative}") from exc
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"manifest artifact hash mismatch: {relative}")


def _validate_inputs(
    *,
    source_bulk_manifest: Path,
    quality_sample_manifest: Path,
    temporal_config_path: Path,
    selected_levels: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_json(source_bulk_manifest)
    samples = _load_json(quality_sample_manifest)
    if (
        source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or not set(selected_levels).issubset(set(source.get("levels", [])))
    ):
        raise ValueError("ghost rendering requires an eligible formal source corpus")
    if (
        samples.get("format_id") != "flow_belief_quality_sample_run_v1"
        or samples.get("eligible") is not True
        or samples.get("evaluation_split") != "validation"
        or not set(selected_levels).issubset(set(samples.get("levels", [])))
    ):
        raise ValueError("ghost rendering requires eligible validation quality samples")
    input_hashes = samples.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("quality sample input hashes are invalid")
    if input_hashes.get("source_bulk_manifest") != sha256_file(source_bulk_manifest):
        raise ValueError("quality samples do not match the source corpus")
    if input_hashes.get("temporal_config") != sha256_file(temporal_config_path):
        raise ValueError("quality samples do not match the temporal config")
    _verify_manifest_artifacts(quality_sample_manifest, samples)
    return source, samples


def _default_level_renderer(**kwargs: Any) -> Path:
    from latency_meta_mdp.belief.flow.ghost_render_level import (
        render_flow_belief_ghost_level,
    )

    return render_flow_belief_ghost_level(**kwargs)


def render_flow_belief_quality_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    quality_sample_manifest: Path,
    task_config_path: Path,
    control_config_path: Path,
    temporal_config_path: Path,
    ghost_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    level_renderer: LevelRenderer = _default_level_renderer,
) -> Path:
    started = time.perf_counter()
    selected_levels = tuple(sorted(set(levels)))
    if (
        selected_levels != levels
        or not selected_levels
        or any(level not in (1, 2, 3) for level in selected_levels)
    ):
        raise ValueError("ghost levels must be sorted unique values from 1, 2, 3")
    paths = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "quality_sample_manifest": quality_sample_manifest.resolve(),
        "task_config": task_config_path.resolve(),
        "control_config": control_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "ghost_config": ghost_config_path.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"ghost renderer input is missing: {name}={path}")
    _, sample_run = _validate_inputs(
        source_bulk_manifest=paths["source_bulk_manifest"],
        quality_sample_manifest=paths["quality_sample_manifest"],
        temporal_config_path=paths["temporal_config"],
        selected_levels=selected_levels,
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow ghost run already exists: {target}")
    provenance = collect_implementation_provenance(project_root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests: dict[str, str] = {}
    context_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    try:
        for level in selected_levels:
            level_key = f"L{level}"
            sample_relative = sample_run.get("level_manifests", {}).get(level_key)
            if not isinstance(sample_relative, str):
                raise ValueError(f"quality sample level reference is invalid: {level_key}")
            quality_level_manifest = (
                paths["quality_sample_manifest"].parent / sample_relative
            ).resolve()
            manifest_path = level_renderer(
                level=level,
                output_dir=building / level_key,
                project_root=project_root.resolve(),
                source_bulk_manifest=paths["source_bulk_manifest"],
                quality_sample_manifest=paths["quality_sample_manifest"],
                quality_level_manifest=quality_level_manifest,
                task_config_path=paths["task_config"],
                control_config_path=paths["control_config"],
                temporal_config_path=paths["temporal_config"],
                ghost_config_path=paths["ghost_config"],
            )
            manifest = _load_json(manifest_path)
            if (
                manifest.get("format_id") != "level_flow_belief_ghost_v1"
                or manifest.get("eligible") is not True
                or manifest.get("level") != level
                or not isinstance(manifest.get("context_count"), int)
                or not isinstance(manifest.get("invalid_sample_count"), int)
            ):
                raise ValueError(f"ghost level manifest is invalid: {level_key}")
            _verify_manifest_artifacts(manifest_path, manifest)
            level_manifests[level_key] = manifest_path.relative_to(building).as_posix()
            context_counts[level_key] = manifest["context_count"]
            invalid_counts[level_key] = manifest["invalid_sample_count"]
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        blockers = ["dirty_implementation"] if provenance.dirty else []
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_ghost_run_v1",
            "eligible": not blockers,
            "blockers": blockers,
            "levels": list(selected_levels),
            "level_manifests": level_manifests,
            "selected_context_counts": context_counts,
            "invalid_sample_counts": invalid_counts,
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "review_scope": "deterministic_selected_contexts",
            "wall_seconds": time.perf_counter() - started,
            "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        building.rename(target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
