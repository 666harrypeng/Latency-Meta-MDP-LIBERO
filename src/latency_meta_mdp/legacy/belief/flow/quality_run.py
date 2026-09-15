"""Atomic multi-level Flow Belief quality-sample export."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latency_meta_mdp.data.vision.contracts import VisionEncoderSpec, load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.legacy.belief.common.feature_corpus import (
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.legacy.belief.flow.config import (
    FlowBeliefConfig,
    load_flow_belief_config,
)
from latency_meta_mdp.legacy.belief.flow.quality_config import (
    FlowBeliefQualitySampleConfig,
    load_flow_belief_quality_sample_config,
)
from latency_meta_mdp.legacy.belief.flow.quality_samples import (
    export_level_quality_samples,
    load_flow_belief_normalization,
)
from latency_meta_mdp.legacy.belief.flow.quality_selection import (
    load_formal_flow_summary,
    score_validation_contexts,
    select_quality_contexts,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

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


def _verify_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"manifest artifact inventory is invalid: {manifest_path}")
    root = manifest_path.parent.resolve()
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"manifest artifact entry is invalid: {manifest_path}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest artifact escapes its root: {relative}") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"manifest artifact hash mismatch: {relative}")


def _validate_run_inputs(
    *,
    source_path: Path,
    cache_path: Path,
    flow_run_path: Path,
    evaluation_run_path: Path,
    vision_path: Path,
    temporal_path: Path,
    latency_path: Path,
    flow_config_path: Path,
    split_path: Path,
    selected_levels: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = _load_json(flow_run_path)
    evaluation = _load_json(evaluation_run_path)
    if (
        training.get("format_id") != "flow_belief_run_v1"
        or training.get("eligible") is not True
        or training.get("checkpoint_scope") != "per_level_only"
        or not set(selected_levels).issubset(set(training.get("levels", [])))
    ):
        raise ValueError("quality sample export requires an eligible Flow training run")
    if (
        evaluation.get("format_id") != "flow_belief_evaluation_run_v1"
        or evaluation.get("eligible") is not True
        or evaluation.get("evaluation_split") != ProbeSplit.VALIDATION.value
        or not set(selected_levels).issubset(set(evaluation.get("levels", [])))
        or evaluation.get("flow_run_manifest_sha256") != sha256_file(flow_run_path)
    ):
        raise ValueError("quality sample export requires a matching validation run")
    expected_hashes = {
        "source_bulk_manifest": sha256_file(source_path),
        "cache_run_manifest": sha256_file(cache_path),
        "vision_config": sha256_file(vision_path),
        "temporal_config": sha256_file(temporal_path),
        "latency_law": sha256_file(latency_path),
        "flow_config": sha256_file(flow_config_path),
        "split_config": sha256_file(split_path),
    }
    if training.get("input_sha256") != expected_hashes:
        raise ValueError("Flow training inputs do not match the quality inputs")
    evaluation_hashes = {
        "source_bulk_manifest_sha256": expected_hashes["source_bulk_manifest"],
        "cache_run_manifest_sha256": expected_hashes["cache_run_manifest"],
        "vision_config_sha256": expected_hashes["vision_config"],
        "temporal_config_sha256": expected_hashes["temporal_config"],
        "latency_law_sha256": expected_hashes["latency_law"],
        "flow_config_sha256": expected_hashes["flow_config"],
        "split_config_sha256": expected_hashes["split_config"],
    }
    if any(evaluation.get(name) != value for name, value in evaluation_hashes.items()):
        raise ValueError("Flow evaluation inputs do not match the quality inputs")
    _verify_artifacts(flow_run_path, training)
    _verify_artifacts(evaluation_run_path, evaluation)
    return training, evaluation


def _default_level_processor(
    *,
    level: int,
    output_dir: Path,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    evaluation_run_manifest: Path,
    vision_spec: VisionEncoderSpec,
    temporal_config_path: Path,
    latency_law_path: Path,
    split_config_path: Path,
    flow_config: FlowBeliefConfig,
    quality_config: FlowBeliefQualitySampleConfig,
    device: str,
) -> Path:
    corpus = load_level_feature_belief_corpus(
        project_root=project_root,
        source_bulk_manifest=source_bulk_manifest,
        cache_run_manifest=cache_run_manifest,
        expected_spec=vision_spec,
        temporal_config_path=temporal_config_path,
        latency_law_path=latency_law_path,
        split_plan_path=split_config_path,
        level=level,
    )
    validation_count = len(corpus.sample_references[ProbeSplit.VALIDATION])
    evaluation_root = evaluation_run_manifest.parent
    flow_root = flow_run_manifest.parent
    summary = load_formal_flow_summary(
        evaluation_root / f"L{level}/summary_arrays.npz",
        expected_context_count=validation_count,
    )
    normalization = load_flow_belief_normalization(flow_root / f"L{level}/normalization.npz")
    scored = score_validation_contexts(
        corpus=corpus,
        summary=summary,
        normalization=normalization,
    )
    selections = select_quality_contexts(scored=scored, config=quality_config)
    return export_level_quality_samples(
        corpus=corpus,
        config=quality_config,
        flow_config=flow_config,
        flow_checkpoint_dir=flow_root / f"L{level}",
        evaluation_summary=summary,
        selections=selections,
        output_dir=output_dir,
        device=device,
    )


def export_flow_belief_quality_sample_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    evaluation_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    split_config_path: Path,
    quality_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    level_processor: LevelProcessor = _default_level_processor,
) -> Path:
    started = time.perf_counter()
    selected_levels = tuple(sorted(set(levels)))
    if (
        selected_levels != levels
        or not selected_levels
        or any(level not in (1, 2, 3) for level in selected_levels)
    ):
        raise ValueError("quality sample levels must be sorted unique values from 1, 2, 3")
    paths = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "flow_run_manifest": flow_run_manifest.resolve(),
        "evaluation_run_manifest": evaluation_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "latency_law": latency_law_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "quality_config": quality_config_path.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"quality sample input is missing: {name}={path}")
    training, evaluation = _validate_run_inputs(
        source_path=paths["source_bulk_manifest"],
        cache_path=paths["cache_run_manifest"],
        flow_run_path=paths["flow_run_manifest"],
        evaluation_run_path=paths["evaluation_run_manifest"],
        vision_path=paths["vision_config"],
        temporal_path=paths["temporal_config"],
        latency_path=paths["latency_law"],
        flow_config_path=paths["flow_config"],
        split_path=paths["split_config"],
        selected_levels=selected_levels,
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow quality sample run already exists: {target}")
    provenance = collect_implementation_provenance(project_root)
    vision_spec = load_vision_encoder_spec(paths["vision_config"])
    flow_config = load_flow_belief_config(paths["flow_config"])
    quality_config = load_flow_belief_quality_sample_config(paths["quality_config"])
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests: dict[str, str] = {}
    selected_counts: dict[str, int] = {}
    role_counts: dict[str, dict[str, int]] = {}
    try:
        for level in selected_levels:
            manifest_path = level_processor(
                level=level,
                output_dir=building / f"L{level}",
                project_root=project_root.resolve(),
                source_bulk_manifest=paths["source_bulk_manifest"],
                cache_run_manifest=paths["cache_run_manifest"],
                flow_run_manifest=paths["flow_run_manifest"],
                evaluation_run_manifest=paths["evaluation_run_manifest"],
                vision_spec=vision_spec,
                temporal_config_path=paths["temporal_config"],
                latency_law_path=paths["latency_law"],
                split_config_path=paths["split_config"],
                flow_config=flow_config,
                quality_config=quality_config,
                device=device,
            )
            manifest = _load_json(manifest_path)
            if (
                manifest.get("format_id") != "level_flow_belief_quality_samples_v1"
                or manifest.get("eligible") is not True
                or manifest.get("level") != level
                or not isinstance(manifest.get("context_count"), int)
                or not isinstance(manifest.get("role_counts"), dict)
            ):
                raise ValueError(f"quality sample level manifest is invalid: L{level}")
            level_key = f"L{level}"
            level_manifests[level_key] = manifest_path.relative_to(building).as_posix()
            selected_counts[level_key] = manifest["context_count"]
            role_counts[level_key] = manifest["role_counts"]
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        eligible = (
            not provenance.dirty
            and training.get("eligible") is True
            and evaluation.get("eligible") is True
        )
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_quality_sample_run_v1",
            "eligible": eligible,
            "blockers": [] if eligible else ["dirty_implementation_or_input_run"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "evaluation_split": quality_config.evaluation_split.value,
            "levels": list(selected_levels),
            "level_manifests": level_manifests,
            "selected_context_counts": selected_counts,
            "role_counts": role_counts,
            "sample_count": quality_config.sample_count,
            "display_delay_ticks": list(quality_config.display_delay_ticks),
            "solver": quality_config.solver,
            "solver_step_count": quality_config.solver_step_count,
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
