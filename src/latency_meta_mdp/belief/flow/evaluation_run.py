"""Atomic multi-level Flow Belief sample-evaluation runner."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import load_level_feature_belief_corpus
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.evaluation import evaluate_level_flow_belief
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec


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


def evaluate_flow_belief_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    context_limit: int | None = None,
    sample_count: int | None = None,
    step_count: int | None = None,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if not selected_levels or selected_levels != levels or any(
        level not in (1, 2, 3) for level in levels
    ):
        raise ValueError("Flow evaluation levels must be sorted unique values from 1, 2, 3")
    source_path = source_bulk_manifest.resolve()
    cache_path = cache_run_manifest.resolve()
    flow_run_path = flow_run_manifest.resolve()
    vision_path = vision_config_path.resolve()
    temporal_path = temporal_config_path.resolve()
    law_path = latency_law_path.resolve()
    flow_config_path = flow_config_path.resolve()
    for path in (
        source_path,
        cache_path,
        flow_run_path,
        vision_path,
        temporal_path,
        law_path,
        flow_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Flow evaluation input does not exist: {path}")
    training_run = _load_json(flow_run_path)
    if (
        training_run.get("format_id") != "flow_belief_run_v1"
        or training_run.get("checkpoint_scope") != "per_level_only"
        or not set(selected_levels).issubset(set(training_run.get("levels", [])))
    ):
        raise ValueError("Flow evaluation requires a matching Flow training run")
    spec = load_vision_encoder_spec(vision_path)
    config = load_flow_belief_config(flow_config_path)
    provenance = collect_implementation_provenance(project_root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow evaluation run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in selected_levels:
            corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=source_path,
                cache_run_manifest=cache_path,
                expected_spec=spec,
                temporal_config_path=temporal_path,
                latency_law_path=law_path,
                level=level,
            )
            checkpoint_dir = flow_run_path.parent / f"L{level}"
            level_manifest = evaluate_level_flow_belief(
                corpus=corpus,
                config=config,
                checkpoint_dir=checkpoint_dir,
                output_dir=building / f"L{level}",
                device=device,
                context_limit=context_limit,
                sample_count=sample_count,
                step_count=step_count,
            )
            level_manifests[f"L{level}"] = level_manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_evaluation_run_v1",
            "eligible": not provenance.dirty and training_run.get("eligible") is True,
            "blockers": (
                []
                if not provenance.dirty and training_run.get("eligible") is True
                else ["dirty_implementation_or_training_run"]
            ),
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "flow_run_manifest_sha256": sha256_file(flow_run_path),
            "source_bulk_manifest_sha256": sha256_file(source_path),
            "cache_run_manifest_sha256": sha256_file(cache_path),
            "vision_config_sha256": sha256_file(vision_path),
            "temporal_config_sha256": sha256_file(temporal_path),
            "latency_law_sha256": sha256_file(law_path),
            "flow_config_sha256": sha256_file(flow_config_path),
            "levels": list(selected_levels),
            "level_manifests": level_manifests,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Flow evaluation run already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
