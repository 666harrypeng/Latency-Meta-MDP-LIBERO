"""Atomic multi-level runners for motion-aware causal-return histories."""

from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

from latency_meta_mdp.artifacts import (
    ImplementationProvenance,
    collect_implementation_provenance,
    sha256_file,
)
from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    load_motion_aware_history_config,
)
from latency_meta_mdp.belief.causal_return.motion_aware_data import (
    MotionAwareHistoryCorpus,
    load_motion_aware_history_corpus,
)
from latency_meta_mdp.belief.causal_return.motion_aware_evaluation import (
    evaluate_level_motion_aware_history,
)
from latency_meta_mdp.belief.causal_return.motion_aware_training import (
    MotionAwareTrainingProvenance,
    _fsync_directory,
    _rename_directory_no_replace,
    _write_json,
    train_level_motion_aware_history,
)
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_levels(levels: tuple[int, ...]) -> None:
    if not levels or levels != tuple(sorted(set(levels))) or any(
        level not in (1, 2, 3) for level in levels
    ):
        raise ValueError("motion-aware levels must be sorted unique values from 1, 2, 3")


def _evenly_spaced_references(references: tuple, limit: int | None) -> tuple:
    if limit is None or limit >= len(references):
        return references
    if limit == 1:
        return (references[len(references) // 2],)
    indices = [round(index * (len(references) - 1) / (limit - 1)) for index in range(limit)]
    if len(set(indices)) != limit:
        raise ValueError("motion-aware bounded selection produced duplicate references")
    return tuple(references[index] for index in indices)


def limit_motion_aware_history_corpus(
    corpus: MotionAwareHistoryCorpus,
    *,
    training_context_limit: int | None,
    validation_context_limit: int | None,
) -> MotionAwareHistoryCorpus:
    references = dict(corpus.sample_references)
    references[ProbeSplit.TRAIN] = _evenly_spaced_references(
        references[ProbeSplit.TRAIN], training_context_limit
    )
    references[ProbeSplit.VALIDATION] = _evenly_spaced_references(
        references[ProbeSplit.VALIDATION], validation_context_limit
    )
    return dataclasses.replace(corpus, sample_references=references)


def _resolve_inputs(
    *,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    motion_aware_config_path: Path,
) -> dict[str, Path]:
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "motion_aware_config": motion_aware_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"motion-aware input does not exist: {path}")
    return inputs


def _input_hashes(inputs: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in inputs.items()}


def _input_paths(inputs: dict[str, Path]) -> dict[str, str]:
    return {name: str(path) for name, path in inputs.items()}


def _resolve_run_artifact(*, run_manifest_path: Path, relative: str) -> Path:
    root = run_manifest_path.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("motion-aware declared artifact escapes its run root") from error
    return path


def _verify_declared_artifacts(*, run_manifest_path: Path, run: dict) -> None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("motion-aware run artifact inventory is invalid")
    for relative, digest in artifacts.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("motion-aware run artifact inventory is invalid")
        path = _resolve_run_artifact(
            run_manifest_path=run_manifest_path,
            relative=relative,
        )
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"motion-aware run artifact hash mismatch: {relative}")


def _level_manifest_path(*, run_manifest_path: Path, run: dict, level: int) -> Path:
    relative = run.get("level_manifests", {}).get(f"L{level}")
    if not isinstance(relative, str):
        raise ValueError("motion-aware level manifest is missing")
    if run.get("artifacts", {}).get(relative) is None:
        raise ValueError("motion-aware level manifest is not bound by the run artifact map")
    return _resolve_run_artifact(
        run_manifest_path=run_manifest_path,
        relative=relative,
    )


def train_motion_aware_history_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    motion_aware_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    max_epochs: int | None = None,
    training_context_limit: int | None = None,
    validation_context_limit: int | None = None,
    corpus_loader: Callable = load_motion_aware_history_corpus,
    level_trainer: Callable = train_level_motion_aware_history,
    provenance_collector: Callable[[Path], ImplementationProvenance] = (
        collect_implementation_provenance
    ),
) -> Path:
    """Train selected levels and publish one hash-bound training run."""

    _validate_levels(levels)
    for value in (max_epochs, training_context_limit, validation_context_limit):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("motion-aware optional limits must be positive integers")
    root = project_root.resolve()
    inputs = _resolve_inputs(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        vision_config_path=vision_config_path,
        temporal_config_path=temporal_config_path,
        split_config_path=split_config_path,
        motion_aware_config_path=motion_aware_config_path,
    )
    hashes = _input_hashes(inputs)
    vision_spec = load_vision_encoder_spec(inputs["vision_config"])
    base_config = load_motion_aware_history_config(inputs["motion_aware_config"])
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
        max_epochs is None
        and training_context_limit is None
        and validation_context_limit is None
    )
    implementation = provenance_collector(root)
    training_provenance = MotionAwareTrainingProvenance(
        implementation_revision=implementation.revision,
        implementation_source_sha256=implementation.source_sha256,
        implementation_dirty=implementation.dirty,
        bounded_review=not complete,
        input_sha256=hashes,
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"motion-aware training run exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in levels:
            corpus = corpus_loader(
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=vision_spec,
                temporal_config_path=inputs["temporal_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            corpus = limit_motion_aware_history_corpus(
                corpus,
                training_context_limit=training_context_limit,
                validation_context_limit=validation_context_limit,
            )
            manifest = level_trainer(
                corpus=corpus,
                config=config,
                output_dir=building / f"L{level}",
                device=device,
                provenance=training_provenance,
            )
            level_manifests[f"L{level}"] = manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        artifact_eligible = training_provenance.artifact_eligible
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_motion_aware_history_training_run",
                "scientific_gate_pass": False,
                "artifact_eligible": artifact_eligible,
                "provenance_blockers": (
                    [] if artifact_eligible else ["dirty_or_bounded_review_run"]
                ),
                "implementation_revision": implementation.revision,
                "implementation_source_sha256": implementation.source_sha256,
                "implementation_dirty": implementation.dirty,
                "bounded_review": not complete,
                "checkpoint_scope": "per_level_only",
                "levels": list(levels),
                "level_manifests": level_manifests,
                "input_paths": _input_paths(inputs),
                "input_sha256": hashes,
                "artifacts": artifacts,
            },
        )
        _fsync_directory(building)
        _rename_directory_no_replace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def _load_baseline_level(
    *,
    baseline_path: Path,
    baseline: dict,
    level: int,
) -> dict[str, np.ndarray]:
    level_manifest_path = _level_manifest_path(
        run_manifest_path=baseline_path,
        run=baseline,
        level=level,
    )
    level_manifest = _load_json(level_manifest_path)
    if (
        level_manifest.get("format_id") != "causal_return_information_state_evaluation"
        or level_manifest.get("level") != level
    ):
        raise ValueError("motion-aware baseline level identity is invalid")
    digest = level_manifest.get("artifacts", {}).get("predictions.npz")
    predictions_path = level_manifest_path.parent / "predictions.npz"
    if not isinstance(digest, str) or sha256_file(predictions_path) != digest:
        raise ValueError("motion-aware baseline predictions hash mismatch")
    with np.load(predictions_path, allow_pickle=False) as source:
        return {
            "episode_id": np.array(source["episode_id"], copy=True),
            "source_tick": np.array(source["source_tick"], copy=True),
            "source_phase": np.array(source["source_phase"], copy=True),
            "object_state_target": np.array(source["object_state_target"], copy=True),
            "object_state_prediction": np.array(source["object_state_mean"], copy=True),
        }


def evaluate_motion_aware_history_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    training_run_manifest: Path,
    baseline_evaluation_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    split_config_path: Path,
    motion_aware_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    corpus_loader: Callable = load_motion_aware_history_corpus,
    level_evaluator: Callable = evaluate_level_motion_aware_history,
    provenance_collector: Callable[[Path], ImplementationProvenance] = (
        collect_implementation_provenance
    ),
) -> Path:
    """Evaluate selected levels against an explicit matched historical run."""

    _validate_levels(levels)
    root = project_root.resolve()
    inputs = _resolve_inputs(
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        vision_config_path=vision_config_path,
        temporal_config_path=temporal_config_path,
        split_config_path=split_config_path,
        motion_aware_config_path=motion_aware_config_path,
    )
    hashes = _input_hashes(inputs)
    training_path = training_run_manifest.resolve()
    training = _load_json(training_path)
    if (
        training.get("format_id") != "causal_return_motion_aware_history_training_run"
        or not set(levels).issubset(set(training.get("levels", [])))
        or training.get("input_sha256") != hashes
    ):
        raise ValueError("motion-aware evaluation requires a matching training run")
    _verify_declared_artifacts(run_manifest_path=training_path, run=training)
    baseline_path = baseline_evaluation_manifest.resolve()
    baseline = _load_json(baseline_path)
    if (
        baseline.get("format_id") != "causal_return_information_state_evaluation_run"
        or not set(levels).issubset(set(baseline.get("levels", [])))
        or baseline.get("input_sha256", {}).get("source_bulk_manifest")
        != hashes["source_bulk_manifest"]
    ):
        raise ValueError("motion-aware evaluation requires a matched historical baseline")
    _verify_declared_artifacts(run_manifest_path=baseline_path, run=baseline)
    vision_spec = load_vision_encoder_spec(inputs["vision_config"])
    implementation = provenance_collector(root)
    evaluation_eligible = (
        not implementation.dirty and training.get("artifact_eligible") is True
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"motion-aware evaluation run exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    scientific_passes = []
    try:
        for level in levels:
            corpus = corpus_loader(
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=vision_spec,
                temporal_config_path=inputs["temporal_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            baseline_arrays = _load_baseline_level(
                baseline_path=baseline_path,
                baseline=baseline,
                level=level,
            )
            checkpoint_manifest_path = _level_manifest_path(
                run_manifest_path=training_path,
                run=training,
                level=level,
            )
            manifest = level_evaluator(
                corpus=corpus,
                checkpoint_dir=checkpoint_manifest_path.parent,
                baseline=baseline_arrays,
                baseline_source_sha256=hashes["source_bulk_manifest"],
                output_dir=building / f"L{level}",
                device=device,
                evaluation_artifact_eligible=evaluation_eligible,
            )
            level_value = _load_json(manifest)
            scientific_passes.append(level_value.get("scientific_gate_pass") is True)
            level_manifests[f"L{level}"] = manifest.relative_to(building).as_posix()
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        scientific_gate_pass = all(scientific_passes)
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_motion_aware_history_evaluation_run",
                "scientific_gate_pass": scientific_gate_pass,
                "artifact_eligible": evaluation_eligible,
                "provenance_blockers": (
                    [] if evaluation_eligible else ["implementation_or_training_ineligible"]
                ),
                "implementation_revision": implementation.revision,
                "implementation_source_sha256": implementation.source_sha256,
                "implementation_dirty": implementation.dirty,
                "levels": list(levels),
                "level_manifests": level_manifests,
                "input_paths": _input_paths(inputs),
                "training_run_manifest_sha256": sha256_file(training_path),
                "training_run_manifest_path": str(training_path),
                "baseline_run_manifest_sha256": sha256_file(baseline_path),
                "baseline_run_manifest_path": str(baseline_path),
                "input_sha256": hashes,
                "artifacts": artifacts,
            },
        )
        _fsync_directory(building)
        _rename_directory_no_replace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
