"""Evaluate multi-law Flow Beliefs under nominal and shifted latency laws."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import (
    FeatureBeliefCorpus,
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.evaluation import evaluate_level_flow_belief
from latency_meta_mdp.belief.flow.multilaw_config import (
    load_multilaw_flow_training_config,
)
from latency_meta_mdp.belief.flow.multilaw_evaluation_config import (
    load_multilaw_flow_evaluation_config,
)
from latency_meta_mdp.latency_law_family import load_episode_latency_law_family
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit

_REGIMES = (
    "nominal",
    "in_family",
    "shifted_fast",
    "shifted_slow",
    "shifted_wide",
)


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


def _replace_corpus_law(
    *,
    corpus: FeatureBeliefCorpus,
    probabilities: np.ndarray,
) -> FeatureBeliefCorpus:
    probability = np.array(probabilities, dtype=np.float64, copy=True)
    if (
        probability.shape != (20,)
        or np.any(probability <= 0.0)
        or not np.isclose(probability.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("replacement latency law is invalid")
    probability.setflags(write=False)
    records = tuple(
        dataclasses.replace(record, latency_probabilities=probability) for record in corpus.records
    )
    return dataclasses.replace(corpus, records=records)


def evaluate_multilaw_flow_belief_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    multilaw_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    flow_config_path: Path,
    multilaw_config_path: Path,
    split_config_path: Path,
    evaluation_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    context_limit: int | None = None,
    sample_count: int | None = None,
    solver_step_count: int | None = None,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if (
        not selected_levels
        or selected_levels != levels
        or any(level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("multi-law evaluation levels are invalid")
    for value in (context_limit, sample_count, solver_step_count):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("multi-law evaluation limits must be positive")
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "multilaw_run_manifest": multilaw_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "nominal_law": nominal_law_path.resolve(),
        "family_config": family_config_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "multilaw_config": multilaw_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "evaluation_config": evaluation_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"multi-law evaluation input does not exist: {path}")
    training_run = _load_json(inputs["multilaw_run_manifest"])
    if training_run.get("format_id") != "flow_belief_multilaw_run_v2" or not set(
        selected_levels
    ).issubset(set(training_run.get("levels", []))):
        raise ValueError("multi-law evaluation requires a matching training run")
    expected_hashes = training_run.get("input_sha256")
    mapping = {
        "source_bulk_manifest": "source_bulk_manifest",
        "cache_run_manifest": "cache_run_manifest",
        "vision_config": "vision_config",
        "temporal_config": "temporal_config",
        "nominal_law": "nominal_law",
        "family_config": "family_config",
        "flow_config": "flow_config",
        "multilaw_config": "multilaw_config",
        "split_config": "split_config",
    }
    if not isinstance(expected_hashes, dict) or any(
        expected_hashes.get(expected_name) != sha256_file(inputs[input_name])
        for input_name, expected_name in mapping.items()
    ):
        raise ValueError("multi-law evaluation inputs do not match training")
    flow_config = load_flow_belief_config(inputs["flow_config"])
    multilaw = load_multilaw_flow_training_config(inputs["multilaw_config"])
    evaluation_spec = load_multilaw_flow_evaluation_config(inputs["evaluation_config"])
    family = load_episode_latency_law_family(inputs["family_config"])
    vision_spec = load_vision_encoder_spec(inputs["vision_config"])
    provenance = collect_implementation_provenance(project_root)
    complete_run = (
        context_limit is None
        and sample_count is None
        and solver_step_count is None
        and training_run.get("eligible") is True
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"multi-law evaluation output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    try:
        for level in selected_levels:
            nominal_corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=vision_spec,
                temporal_config_path=inputs["temporal_config"],
                latency_law_path=inputs["nominal_law"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            in_family_corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=vision_spec,
                temporal_config_path=inputs["temporal_config"],
                latency_law_path=inputs["nominal_law"],
                latency_law_family_path=inputs["family_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            regimes: dict[str, tuple[FeatureBeliefCorpus, dict[str, Any]]] = {
                "nominal": (
                    nominal_corpus,
                    {"law_id": nominal_corpus.latency_law.law_id},
                ),
                "in_family": (
                    in_family_corpus,
                    {"law_family_id": multilaw.latency_law_family_id},
                ),
            }
            for name, shifted_spec in evaluation_spec.shifted_laws.items():
                shifted = family.build_shifted_law(
                    name=name,
                    mean_logit_offset=shifted_spec.mean_logit_offset,
                    log_concentration_offset=shifted_spec.log_concentration_offset,
                    uniform_floor=shifted_spec.uniform_floor,
                )
                regimes[f"shifted_{name}"] = (
                    _replace_corpus_law(
                        corpus=in_family_corpus,
                        probabilities=shifted.probabilities,
                    ),
                    {
                        "shape_alpha": shifted.shape_alpha,
                        "shape_beta": shifted.shape_beta,
                        "uniform_floor": shifted.uniform_floor,
                        "effective_mean_seconds": shifted.effective_mean_seconds,
                        "probability_sha256": shifted.probability_sha256,
                        "probabilities": shifted.probabilities.tolist(),
                    },
                )
            if tuple(regimes) != _REGIMES:
                raise RuntimeError("multi-law evaluation regime order is invalid")
            level_dir = building / f"L{level}"
            level_dir.mkdir()
            regime_manifests = {}
            regime_metadata = {}
            checkpoint_dir = inputs["multilaw_run_manifest"].parent / f"L{level}"
            for regime_name, (corpus, metadata) in regimes.items():
                regime_manifest = evaluate_level_flow_belief(
                    corpus=corpus,
                    config=flow_config,
                    checkpoint_dir=checkpoint_dir,
                    output_dir=level_dir / regime_name,
                    device=device,
                    context_limit=context_limit,
                    sample_count=sample_count,
                    step_count=solver_step_count,
                    evaluation_split=ProbeSplit.VALIDATION,
                )
                regime_manifests[regime_name] = regime_manifest.relative_to(level_dir).as_posix()
                regime_metadata[regime_name] = metadata
            _write_json(
                level_dir / "summary.json",
                {
                    "level": level,
                    "regimes": list(_REGIMES),
                    "regime_manifests": regime_manifests,
                    "regime_metadata": regime_metadata,
                },
            )
            artifacts = {
                path.relative_to(level_dir).as_posix(): sha256_file(path)
                for path in sorted(level_dir.rglob("*"))
                if path.is_file()
            }
            _write_json(
                level_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "format_id": "level_flow_belief_multilaw_evaluation_v1",
                    "level": level,
                    "regimes": list(_REGIMES),
                    "artifacts": artifacts,
                },
            )
            level_manifests[f"L{level}"] = f"L{level}/manifest.json"
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "flow_belief_multilaw_evaluation_run_v1",
                "eligible": not provenance.dirty and complete_run,
                "blockers": (
                    []
                    if not provenance.dirty and complete_run
                    else ["dirty_implementation_or_bounded_smoke"]
                ),
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "training_run_sha256": sha256_file(inputs["multilaw_run_manifest"]),
                "levels": list(selected_levels),
                "regimes": list(_REGIMES),
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
