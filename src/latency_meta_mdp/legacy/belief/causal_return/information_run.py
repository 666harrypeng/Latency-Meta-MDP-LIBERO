"""Atomic multi-level runners for causal-return information-state estimation."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from pathlib import Path

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.legacy.belief.causal_return.contracts import (
    load_information_state_config,
)
from latency_meta_mdp.legacy.belief.causal_return.information_data import (
    InformationStateCorpus,
    load_information_state_corpus,
)
from latency_meta_mdp.legacy.belief.causal_return.information_evaluation import (
    evaluate_level_information_state,
)
from latency_meta_mdp.legacy.belief.causal_return.information_training import (
    train_level_information_state,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_levels(levels: tuple[int, ...]) -> None:
    if (
        not levels
        or levels != tuple(sorted(set(levels)))
        or any(level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("information-state levels must be sorted unique values from 1, 2, 3")


def _evenly_spaced_references(references: tuple, limit: int | None) -> tuple:
    if limit is None or limit >= len(references):
        return references
    if limit == 1:
        return (references[len(references) // 2],)
    indices = [round(index * (len(references) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise ValueError("information-state bounded selection produced duplicate references")
    return tuple(references[index] for index in indices)


def limit_information_state_corpus(
    corpus: InformationStateCorpus,
    *,
    training_context_limit: int | None,
    validation_context_limit: int | None,
) -> InformationStateCorpus:
    references = dict(corpus.sample_references)
    references[ProbeSplit.TRAIN] = _evenly_spaced_references(
        references[ProbeSplit.TRAIN],
        training_context_limit,
    )
    references[ProbeSplit.VALIDATION] = _evenly_spaced_references(
        references[ProbeSplit.VALIDATION],
        validation_context_limit,
    )
    return dataclasses.replace(corpus, sample_references=references)


def _resolve_inputs(
    *,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    information_config_path: Path,
) -> dict[str, Path]:
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "information_config": information_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"information-state input does not exist: {path}")
    return inputs


def train_information_state_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    information_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    max_epochs: int | None = None,
    training_context_limit: int | None = None,
    validation_context_limit: int | None = None,
) -> Path:
    _validate_levels(levels)
    for value in (max_epochs, training_context_limit, validation_context_limit):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("information-state optional limits must be positive integers")
    root = project_root.resolve()
    inputs = _resolve_inputs(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        vision_config_path=vision_config_path,
        temporal_config_path=temporal_config_path,
        split_config_path=split_config_path,
        information_config_path=information_config_path,
    )
    spec = load_vision_encoder_spec(inputs["vision_config"])
    base_config = load_information_state_config(inputs["information_config"])
    config = dataclasses.replace(
        base_config,
        max_epochs=max_epochs or base_config.max_epochs,
        early_stopping_patience=(
            min(base_config.early_stopping_patience, max_epochs)
            if max_epochs is not None
            else base_config.early_stopping_patience
        ),
    )
    complete = (
        max_epochs is None and training_context_limit is None and validation_context_limit is None
    )
    provenance = collect_implementation_provenance(root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"information-state training run exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in levels:
            corpus = load_information_state_corpus(
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=spec,
                temporal_config_path=inputs["temporal_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            corpus = limit_information_state_corpus(
                corpus,
                training_context_limit=training_context_limit,
                validation_context_limit=validation_context_limit,
            )
            manifest = train_level_information_state(
                corpus=corpus,
                config=config,
                output_dir=building / f"L{level}",
                device=device,
            )
            level_manifests[f"L{level}"] = manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        eligible = complete and not provenance.dirty
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_information_state_training_run",
                "eligible": eligible,
                "blockers": [] if eligible else ["dirty_or_bounded_review_run"],
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "checkpoint_scope": "per_level_only",
                "levels": list(levels),
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


def evaluate_information_state_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    training_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    information_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
) -> Path:
    _validate_levels(levels)
    root = project_root.resolve()
    inputs = _resolve_inputs(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        vision_config_path=vision_config_path,
        temporal_config_path=temporal_config_path,
        split_config_path=split_config_path,
        information_config_path=information_config_path,
    )
    training_path = training_run_manifest.resolve()
    training = _load_json(training_path)
    if training.get("format_id") != "causal_return_information_state_training_run" or not set(
        levels
    ).issubset(set(training.get("levels", []))):
        raise ValueError("information-state evaluation requires a matching training run")
    spec = load_vision_encoder_spec(inputs["vision_config"])
    provenance = collect_implementation_provenance(root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"information-state evaluation run exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in levels:
            corpus = load_information_state_corpus(
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=spec,
                temporal_config_path=inputs["temporal_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            checkpoint_dir = training_path.parent / f"L{level}"
            manifest = evaluate_level_information_state(
                corpus=corpus,
                checkpoint_dir=checkpoint_dir,
                output_dir=building / f"L{level}",
                device=device,
            )
            level_manifests[f"L{level}"] = manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        eligible = not provenance.dirty and training.get("eligible") is True
        blockers = []
        if provenance.dirty:
            blockers.append("implementation_dirty")
        if training.get("eligible") is not True:
            blockers.append("training_run_ineligible")
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_information_state_evaluation_run",
                "eligible": eligible,
                "blockers": blockers,
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "levels": list(levels),
                "level_manifests": level_manifests,
                "training_run_manifest_sha256": sha256_file(training_path),
                "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
