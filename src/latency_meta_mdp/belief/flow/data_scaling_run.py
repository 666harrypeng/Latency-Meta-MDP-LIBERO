"""Nested-corpus scaling runs for Flow belief data sufficiency."""

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
from latency_meta_mdp.belief.flow.data_sufficiency import (
    decide_data_scaling,
    load_flow_data_sufficiency_config,
    subset_feature_belief_corpus,
)
from latency_meta_mdp.belief.flow.evaluation import evaluate_level_flow_belief
from latency_meta_mdp.belief.flow.training import train_level_flow_belief
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit

_DECISION_METRICS = (
    "overall_object_position_rmse_mm",
    "overall_object_velocity_rmse_mm_s",
    "pre_handoff_object_position_rmse_mm",
    "pre_handoff_object_velocity_rmse_mm_s",
    "worst_decile_object_position_rmse_mm",
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_scaling_metrics(
    *,
    checkpoint_dir: Path,
    evaluation_dir: Path,
    latency_probabilities: np.ndarray,
) -> dict[str, float]:
    checkpoint = checkpoint_dir.resolve()
    evaluation = evaluation_dir.resolve()
    metrics = _load_json(evaluation / "metrics.json")
    with np.load(checkpoint / "normalization.npz", allow_pickle=False) as source:
        target_std = np.array(source["target_std"], dtype=np.float64, copy=True)
    with np.load(evaluation / "summary_arrays.npz", allow_pickle=False) as source:
        sample_mean = np.array(source["sample_mean_normalized"], dtype=np.float64, copy=True)
        target = np.array(source["target_normalized"], dtype=np.float64, copy=True)
    weights = np.asarray(latency_probabilities, dtype=np.float64)
    if (
        sample_mean.ndim != 3
        or sample_mean.shape != target.shape
        or sample_mean.shape[1:] != (20, 22)
        or target_std.shape != (22,)
        or weights.shape != (20,)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("Flow scaling artifacts have incompatible shapes")
    physical_error = (sample_mean - target) * target_std
    object_squared_error = np.mean(np.square(physical_error[..., 16:19]), axis=-1)
    per_context_object_rmse_mm = (
        np.sqrt(np.sum(weights[None] * object_squared_error, axis=1)) * 1_000.0
    )
    result = {
        "overall_object_position_rmse_mm": float(
            metrics["overall"]["physical"]["object_position"]["rmse_display"]
        ),
        "overall_object_velocity_rmse_mm_s": float(
            metrics["overall"]["physical"]["object_linear_velocity"]["rmse_display"]
        ),
        "pre_handoff_object_position_rmse_mm": float(
            metrics["pre_handoff"]["physical"]["object_position"]["rmse_display"]
        ),
        "pre_handoff_object_velocity_rmse_mm_s": float(
            metrics["pre_handoff"]["physical"]["object_linear_velocity"]["rmse_display"]
        ),
        "worst_decile_object_position_rmse_mm": float(np.quantile(per_context_object_rmse_mm, 0.9)),
        "delay_1_object_position_rmse_mm": float(
            metrics["per_delay"]["1"]["physical"]["object_position"]["rmse_display"]
        ),
        "delay_1_object_velocity_rmse_mm_s": float(
            metrics["per_delay"]["1"]["physical"]["object_linear_velocity"]["rmse_display"]
        ),
        "delay_20_object_position_rmse_mm": float(
            metrics["per_delay"]["20"]["physical"]["object_position"]["rmse_display"]
        ),
        "delay_20_object_velocity_rmse_mm_s": float(
            metrics["per_delay"]["20"]["physical"]["object_linear_velocity"]["rmse_display"]
        ),
        "coverage_68": float(metrics["overall"]["distribution"]["coverage_68"]),
        "coverage_95": float(metrics["overall"]["distribution"]["coverage_95"]),
        "energy_score": float(metrics["overall"]["distribution"]["energy_score"]),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError("Flow scaling metrics must be finite and positive")
    return result


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


def _validate_optional_positive(values: dict[str, int | None]) -> None:
    if any(
        value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0)
        for value in values.values()
    ):
        raise ValueError("Flow scaling optional limits must be positive integers")


def run_flow_belief_data_scaling(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    physical_audit_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    split_config_path: Path,
    audit_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    subset_sizes: tuple[int, ...] | None = None,
    model_seeds: tuple[int, ...] | None = None,
    max_epochs: int | None = None,
    training_context_limit: int | None = None,
    validation_context_limit: int | None = None,
    evaluation_sample_count: int | None = None,
    solver_step_count: int | None = None,
) -> Path:
    selected_levels = tuple(sorted(set(levels)))
    if (
        not selected_levels
        or selected_levels != levels
        or any(level not in (1, 2, 3) for level in levels)
    ):
        raise ValueError("Flow scaling levels must be sorted unique values from 1, 2, 3")
    _validate_optional_positive(
        {
            "max_epochs": max_epochs,
            "training_context_limit": training_context_limit,
            "validation_context_limit": validation_context_limit,
            "evaluation_sample_count": evaluation_sample_count,
            "solver_step_count": solver_step_count,
        }
    )
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "physical_audit_manifest": physical_audit_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "latency_law": latency_law_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "audit_config": audit_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Flow scaling input does not exist: {path}")
    physical_audit = _load_json(inputs["physical_audit_manifest"])
    if (
        physical_audit.get("format_id") != "flow_belief_physical_data_audit_v1"
        or physical_audit.get("eligible") is not True
        or not set(selected_levels).issubset(
            {int(value.removeprefix("L")) for value in physical_audit.get("episode_counts", {})}
        )
    ):
        raise ValueError("Flow scaling requires an eligible matching physical-data audit")
    expected_inputs = physical_audit.get("input_sha256")
    if (
        not isinstance(expected_inputs, dict)
        or expected_inputs.get("source_bulk_manifest")
        != sha256_file(inputs["source_bulk_manifest"])
        or expected_inputs.get("split_config") != sha256_file(inputs["split_config"])
        or expected_inputs.get("audit_config") != sha256_file(inputs["audit_config"])
    ):
        raise ValueError("Flow scaling physical-audit inputs do not match")
    audit_config = load_flow_data_sufficiency_config(inputs["audit_config"])
    selected_sizes = tuple(subset_sizes or audit_config.subset_sizes)
    selected_seeds = tuple(model_seeds or audit_config.model_seeds)
    if (
        not selected_sizes
        or selected_sizes != tuple(sorted(set(selected_sizes)))
        or not set(selected_sizes) <= set(audit_config.subset_sizes)
        or not selected_seeds
        or selected_seeds != tuple(sorted(set(selected_seeds)))
        or not set(selected_seeds) <= set(audit_config.model_seeds)
    ):
        raise ValueError("Flow scaling subset sizes or model seeds are invalid")
    flow_config = load_flow_belief_config(inputs["flow_config"])
    vision_spec = load_vision_encoder_spec(inputs["vision_config"])
    latency_law = load_latency_law(inputs["latency_law"])
    provenance = collect_implementation_provenance(project_root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow scaling output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    level_decisions = {}
    complete_grid = (
        selected_sizes == audit_config.subset_sizes
        and selected_seeds == audit_config.model_seeds
        and max_epochs is None
        and training_context_limit is None
        and validation_context_limit is None
        and evaluation_sample_count is None
        and solver_step_count is None
    )
    try:
        for level in selected_levels:
            full_corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=vision_spec,
                temporal_config_path=inputs["temporal_config"],
                latency_law_path=inputs["latency_law"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            audit_level_dir = inputs["physical_audit_manifest"].parent / f"L{level}"
            nested = _load_json(audit_level_dir / "nested_subsets.json")
            metrics_by_size_seed: dict[int, dict[int, dict[str, float]]] = {}
            level_dir = building / f"L{level}"
            level_dir.mkdir()
            for size in selected_sizes:
                episode_ids = tuple(nested.get(str(size), ()))
                if len(episode_ids) != size:
                    raise ValueError("Flow scaling nested subset size is invalid")
                subset_corpus = subset_feature_belief_corpus(
                    corpus=full_corpus,
                    training_episode_ids=episode_ids,
                )
                subset_corpus = _limit_contexts(
                    corpus=subset_corpus,
                    training_limit=training_context_limit,
                    validation_limit=validation_context_limit,
                )
                metrics_by_size_seed[size] = {}
                for model_seed in selected_seeds:
                    cell_dir = level_dir / f"subset_{size:04d}" / f"seed_{model_seed}"
                    checkpoint_dir = cell_dir / "checkpoint"
                    evaluation_dir = cell_dir / "evaluation"
                    training_config = dataclasses.replace(
                        flow_config,
                        random_seed=model_seed,
                        max_epochs=max_epochs or flow_config.max_epochs,
                        early_stopping_patience=(
                            min(flow_config.early_stopping_patience, max_epochs)
                            if max_epochs is not None
                            else flow_config.early_stopping_patience
                        ),
                    )
                    evaluation_config = dataclasses.replace(
                        training_config,
                        evaluation_sample_count=(
                            evaluation_sample_count or training_config.evaluation_sample_count
                        ),
                        solver_step_count=(solver_step_count or training_config.solver_step_count),
                    )
                    print(
                        f"[scaling][L{level}] subset={size} seed={model_seed} start",
                        flush=True,
                    )
                    train_level_flow_belief(
                        corpus=subset_corpus,
                        config=training_config,
                        output_dir=checkpoint_dir,
                        device=device,
                    )
                    evaluate_level_flow_belief(
                        corpus=subset_corpus,
                        config=evaluation_config,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=evaluation_dir,
                        device=device,
                        evaluation_split=ProbeSplit.VALIDATION,
                    )
                    cell_metrics = extract_scaling_metrics(
                        checkpoint_dir=checkpoint_dir,
                        evaluation_dir=evaluation_dir,
                        latency_probabilities=latency_law.probabilities,
                    )
                    _write_json(cell_dir / "scaling_metrics.json", cell_metrics)
                    metrics_by_size_seed[size][model_seed] = cell_metrics
                    print(
                        f"[scaling][L{level}] subset={size} seed={model_seed} done",
                        flush=True,
                    )
            decision = None
            if complete_grid:
                decision_input = {
                    size: {
                        seed: {name: row[name] for name in _DECISION_METRICS}
                        for seed, row in by_seed.items()
                    }
                    for size, by_seed in metrics_by_size_seed.items()
                }
                decision = decide_data_scaling(
                    metrics_by_size_seed=decision_input,
                    improvement_trigger=audit_config.scaling_improvement_trigger,
                    seed_disagreement_trigger=(audit_config.model_seed_disagreement_trigger),
                )
                level_decisions[f"L{level}"] = decision["decision"]
            summary = {
                "level": level,
                "complete_grid": complete_grid,
                "subset_sizes": list(selected_sizes),
                "model_seeds": list(selected_seeds),
                "metrics_by_size_seed": {
                    str(size): {str(seed): row for seed, row in by_seed.items()}
                    for size, by_seed in metrics_by_size_seed.items()
                },
                "decision": decision,
            }
            _write_json(level_dir / "summary.json", summary)
            level_artifacts = {
                path.relative_to(level_dir).as_posix(): sha256_file(path)
                for path in sorted(level_dir.rglob("*"))
                if path.is_file()
            }
            _write_json(
                level_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "format_id": "level_flow_belief_data_scaling_v1",
                    "level": level,
                    "complete_grid": complete_grid,
                    "artifacts": level_artifacts,
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
                "format_id": "flow_belief_data_scaling_run_v1",
                "eligible": not provenance.dirty and complete_grid,
                "blockers": (
                    []
                    if not provenance.dirty and complete_grid
                    else ["dirty_implementation_or_incomplete_scaling_grid"]
                ),
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "levels": list(selected_levels),
                "subset_sizes": list(selected_sizes),
                "model_seeds": list(selected_seeds),
                "complete_grid": complete_grid,
                "level_decisions": level_decisions,
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
