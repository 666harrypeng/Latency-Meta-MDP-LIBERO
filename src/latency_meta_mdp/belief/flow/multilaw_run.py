"""Atomic level-specific training for multi-law Flow Belief v2."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import (
    FeatureBeliefCorpus,
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.multilaw_config import (
    load_multilaw_flow_training_config,
)
from latency_meta_mdp.belief.flow.training import train_level_flow_belief
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


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


def _limit_contexts(
    *,
    corpus: FeatureBeliefCorpus,
    training_limit: int | None,
    validation_limit: int | None,
) -> FeatureBeliefCorpus:
    references = dict(corpus.sample_references)
    if training_limit is not None:
        references[ProbeSplit.TRAIN] = references[ProbeSplit.TRAIN][:training_limit]
    if validation_limit is not None:
        references[ProbeSplit.VALIDATION] = references[ProbeSplit.VALIDATION][:validation_limit]
    return dataclasses.replace(corpus, sample_references=references)


def train_multilaw_flow_belief_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    multilaw_data_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    flow_config_path: Path,
    multilaw_config_path: Path,
    split_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    max_epochs: int | None = None,
    training_context_limit: int | None = None,
    validation_context_limit: int | None = None,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if (
        not selected_levels
        or selected_levels != levels
        or any(level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("multi-law Flow levels must be sorted unique values from 1, 2, 3")
    for value in (max_epochs, training_context_limit, validation_context_limit):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("multi-law Flow optional limits must be positive")
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "multilaw_data_manifest": multilaw_data_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "nominal_law": nominal_law_path.resolve(),
        "family_config": family_config_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "multilaw_config": multilaw_config_path.resolve(),
        "split_config": split_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"multi-law Flow input does not exist: {path}")
    data_artifact = _load_json(inputs["multilaw_data_manifest"])
    if (
        data_artifact.get("format_id") != "multilaw_flow_belief_data_artifact_v1"
        or data_artifact.get("eligible") is not True
        or not set(selected_levels).issubset(set(data_artifact.get("levels", [])))
    ):
        raise ValueError("multi-law Flow training requires eligible derived data")
    expected_hashes = data_artifact.get("input_sha256")
    expected_mapping = {
        "source_bulk_manifest": "source_bulk_manifest",
        "cache_run_manifest": "cache_run_manifest",
        "vision_config": "vision_config",
        "temporal_config": "temporal_config",
        "nominal_law": "nominal_law",
        "family_config": "family_config",
        "split_config": "split_config",
        "multilaw_config": "multilaw_config",
    }
    if not isinstance(expected_hashes, dict) or any(
        expected_hashes.get(expected_name) != sha256_file(inputs[input_name])
        for input_name, expected_name in expected_mapping.items()
    ):
        raise ValueError("multi-law Flow data inputs do not match the training request")
    flow_config = load_flow_belief_config(inputs["flow_config"])
    multilaw = load_multilaw_flow_training_config(inputs["multilaw_config"])
    if data_artifact.get("latency_law_family_id") != multilaw.latency_law_family_id:
        raise ValueError("multi-law Flow family identifier is inconsistent")
    spec = load_vision_encoder_spec(inputs["vision_config"])
    training_config = dataclasses.replace(
        flow_config,
        max_epochs=max_epochs or flow_config.max_epochs,
        early_stopping_patience=(
            min(multilaw.early_stopping_patience, max_epochs)
            if max_epochs is not None
            else multilaw.early_stopping_patience
        ),
    )
    provenance = collect_implementation_provenance(project_root)
    complete_run = (
        max_epochs is None and training_context_limit is None and validation_context_limit is None
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"multi-law Flow output already exists: {target}")
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
                latency_law_path=inputs["nominal_law"],
                latency_law_family_path=inputs["family_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            corpus = _limit_contexts(
                corpus=corpus,
                training_limit=training_context_limit,
                validation_limit=validation_context_limit,
            )
            level_manifest = train_level_flow_belief(
                corpus=corpus,
                config=training_config,
                output_dir=building / f"L{level}",
                device=device,
                delay_query_uniform_mix=multilaw.tail_query_uniform_mix,
            )
            level_manifests[f"L{level}"] = level_manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 2,
                "format_id": "flow_belief_multilaw_run_v2",
                "checkpoint_scope": "per_level_only",
                "initialization": "random_multilaw_flow_models",
                "eligible": not provenance.dirty and complete_run,
                "blockers": (
                    []
                    if not provenance.dirty and complete_run
                    else ["dirty_implementation_or_bounded_smoke"]
                ),
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "model_id": multilaw.training_id,
                "latency_law_family_id": multilaw.latency_law_family_id,
                "delay_query_uniform_mix": multilaw.tail_query_uniform_mix,
                "levels": list(selected_levels),
                "level_manifests": level_manifests,
                "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
