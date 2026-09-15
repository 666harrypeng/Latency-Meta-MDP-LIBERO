"""Atomic per-level Flow Belief training runner."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.legacy.belief.common.feature_corpus import load_level_feature_belief_corpus
from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.legacy.belief.flow.training import train_level_flow_belief


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_flow_belief_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    split_config_path: Path | None = None,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if (
        not selected_levels
        or selected_levels != levels
        or any(level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("Flow Belief levels must be sorted unique values from 1, 2, 3")
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "latency_law": latency_law_path.resolve(),
        "flow_config": flow_config_path.resolve(),
    }
    if split_config_path is not None:
        inputs["split_config"] = split_config_path.resolve()
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Flow Belief input does not exist: {path}")
    spec = load_vision_encoder_spec(inputs["vision_config"])
    config = load_flow_belief_config(inputs["flow_config"])
    provenance = collect_implementation_provenance(project_root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow Belief run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in selected_levels:
            corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=spec,
                temporal_config_path=inputs["temporal_config"],
                latency_law_path=inputs["latency_law"],
                split_plan_path=inputs.get("split_config"),
                level=level,
            )
            level_manifest = train_level_flow_belief(
                corpus=corpus,
                config=config,
                output_dir=building / f"L{level}",
                device=device,
            )
            level_manifests[f"L{level}"] = level_manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_run_v1",
            "checkpoint_scope": "per_level_only",
            "initialization": "random_flow_models",
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
            "encoder_id": spec.encoder_id,
            "encoder_fingerprint": spec.fingerprint,
            "model_id": config.model_id,
            "levels": list(selected_levels),
            "level_manifests": level_manifests,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Flow Belief run already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
