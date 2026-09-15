"""Atomic multi-level publication of Flow Belief rolling samples."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.legacy.belief.common.feature_corpus import load_level_feature_belief_corpus
from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.legacy.belief.flow.rolling_config import load_flow_belief_rolling_config
from latency_meta_mdp.legacy.belief.flow.rolling_samples import export_level_rolling_samples
from latency_meta_mdp.legacy.belief.flow.rolling_selection import select_rolling_seed_windows

LevelProcessor = Callable[..., Path]


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
    for relative, expected in artifacts.items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest artifact escapes its root: {relative}") from exc
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"manifest artifact hash mismatch: {relative}")


def _validate_training_run(
    *,
    paths: dict[str, Path],
    selected_levels: tuple[int, ...],
) -> dict[str, Any]:
    training = _load_json(paths["flow_run_manifest"])
    if (
        training.get("format_id") != "flow_belief_run_v1"
        or training.get("eligible") is not True
        or training.get("checkpoint_scope") != "per_level_only"
        or not set(selected_levels).issubset(set(training.get("levels", [])))
    ):
        raise ValueError("rolling samples require an eligible per-level Flow run")
    expected = {
        "source_bulk_manifest": sha256_file(paths["source_bulk_manifest"]),
        "cache_run_manifest": sha256_file(paths["cache_run_manifest"]),
        "vision_config": sha256_file(paths["vision_config"]),
        "temporal_config": sha256_file(paths["temporal_config"]),
        "latency_law": sha256_file(paths["latency_law"]),
        "flow_config": sha256_file(paths["flow_config"]),
        "split_config": sha256_file(paths["split_config"]),
    }
    if training.get("input_sha256") != expected:
        raise ValueError("rolling sample inputs do not match the Flow training run")
    _verify_manifest_artifacts(paths["flow_run_manifest"], training)
    return training


def _default_level_processor(
    *,
    level: int,
    scene_seed: int,
    output_dir: Path,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    split_config_path: Path,
    rolling_config_path: Path,
    device: str,
) -> Path:
    vision = load_vision_encoder_spec(vision_config_path)
    corpus = load_level_feature_belief_corpus(
        project_root=project_root,
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        expected_spec=vision,
        temporal_config_path=temporal_config_path,
        latency_law_path=latency_law_path,
        split_plan_path=split_config_path,
        level=level,
    )
    rolling_config = load_flow_belief_rolling_config(rolling_config_path)
    selection = select_rolling_seed_windows(
        corpus=corpus,
        scene_seed=scene_seed,
        config=rolling_config,
    )
    return export_level_rolling_samples(
        corpus=corpus,
        selection=selection,
        rolling_config=rolling_config,
        flow_config=load_flow_belief_config(flow_config_path),
        flow_checkpoint_dir=flow_run_manifest.parent / f"L{level}",
        output_dir=output_dir,
        device=device,
    )


def export_flow_belief_rolling_sample_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    split_config_path: Path,
    rolling_config_path: Path,
    output_dir: Path,
    level_seeds: tuple[tuple[int, int], ...],
    device: str,
    level_processor: LevelProcessor = _default_level_processor,
) -> Path:
    started = time.perf_counter()
    levels = tuple(level for level, _ in level_seeds)
    if (
        not level_seeds
        or levels != tuple(sorted(set(levels)))
        or any(
            level not in (1, 2, 3)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            for level, seed in level_seeds
        )
    ):
        raise ValueError("rolling level seeds require sorted unique levels and valid seeds")
    paths = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "flow_run_manifest": flow_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "latency_law": latency_law_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "rolling_config": rolling_config_path.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"rolling sample input is missing: {name}={path}")
    training = _validate_training_run(paths=paths, selected_levels=levels)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow rolling sample run already exists: {target}")
    provenance = collect_implementation_provenance(project_root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests: dict[str, str] = {}
    window_counts: dict[str, int] = {}
    try:
        for level, scene_seed in level_seeds:
            level_key = f"L{level}"
            manifest_path = level_processor(
                level=level,
                scene_seed=scene_seed,
                output_dir=building / level_key / f"seed_{scene_seed:06d}",
                project_root=project_root.resolve(),
                source_bulk_manifest=paths["source_bulk_manifest"],
                cache_run_manifest=paths["cache_run_manifest"],
                flow_run_manifest=paths["flow_run_manifest"],
                vision_config_path=paths["vision_config"],
                temporal_config_path=paths["temporal_config"],
                latency_law_path=paths["latency_law"],
                flow_config_path=paths["flow_config"],
                split_config_path=paths["split_config"],
                rolling_config_path=paths["rolling_config"],
                device=device,
            )
            manifest = _load_json(manifest_path)
            if (
                manifest.get("format_id") != "level_flow_belief_rolling_samples_v1"
                or manifest.get("eligible") is not True
                or manifest.get("level") != level
                or manifest.get("scene_seed") != scene_seed
                or not isinstance(manifest.get("window_count"), int)
            ):
                raise ValueError(f"rolling sample level manifest is invalid: {level_key}")
            _verify_manifest_artifacts(manifest_path, manifest)
            level_manifests[level_key] = manifest_path.relative_to(building).as_posix()
            window_counts[level_key] = manifest["window_count"]
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        blockers = ["dirty_implementation"] if provenance.dirty else []
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_rolling_sample_run_v1",
            "eligible": not blockers and training.get("eligible") is True,
            "blockers": blockers,
            "checkpoint_scope": "per_level_only",
            "levels": list(levels),
            "level_seeds": {f"L{level}": seed for level, seed in level_seeds},
            "level_manifests": level_manifests,
            "window_counts": window_counts,
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
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
