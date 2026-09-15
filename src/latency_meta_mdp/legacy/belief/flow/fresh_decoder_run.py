"""Atomic orchestration for frozen-Encoder fresh-Decoder probes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.legacy.belief.common.feature_corpus import load_level_feature_belief_corpus
from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.legacy.belief.flow.context_sampling import (
    load_verified_flow_quality_level,
)
from latency_meta_mdp.legacy.belief.flow.fresh_decoder_config import (
    load_fresh_decoder_probe_config,
)
from latency_meta_mdp.legacy.belief.flow.fresh_decoder_evaluation import (
    evaluate_cached_vector_field,
)
from latency_meta_mdp.legacy.belief.flow.fresh_decoder_probe import (
    FrozenEncoderFreshDecoder,
    cache_frozen_belief_splits,
)
from latency_meta_mdp.legacy.belief.flow.fresh_decoder_training import train_fresh_vector_field
from latency_meta_mdp.legacy.belief.flow.vector_field import ConditionalStateVectorField
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

_ADMISSION_GROUPS = ("object_position", "robot_joint_position")


def _rmse(metrics: dict[str, Any], group: str) -> float:
    try:
        value = float(metrics["overall"]["physical"][group]["rmse_display"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"fresh Decoder metrics are missing {group} RMSE") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"fresh Decoder {group} RMSE must be finite and positive")
    return value


def summarize_fresh_decoder_metrics(
    *,
    joint_metrics: dict[str, Any],
    fresh_metrics: dict[int, dict[str, Any]],
    rmse_ratio_max: float,
) -> dict[str, Any]:
    seeds = tuple(sorted(fresh_metrics))
    if (
        len(seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or not 1.0 <= rmse_ratio_max <= 2.0
    ):
        raise ValueError("fresh Decoder summary requires three seeds and a valid gate")
    joint = {group: _rmse(joint_metrics, group) for group in _ADMISSION_GROUPS}
    per_seed = {
        str(seed): {
            group: _rmse(fresh_metrics[seed], group) / joint[group] for group in _ADMISSION_GROUPS
        }
        for seed in seeds
    }
    median = {
        group: float(np.median([per_seed[str(seed)][group] for seed in seeds]))
        for group in _ADMISSION_GROUPS
    }
    failed = [group for group in _ADMISSION_GROUPS if median[group] > rmse_ratio_max]
    return {
        "decoder_seeds": list(seeds),
        "rmse_ratio_max": rmse_ratio_max,
        "joint_rmse": joint,
        "per_seed_rmse_ratios": per_seed,
        "median_rmse_ratios": median,
        "failed_groups": failed,
        "admitted": not failed,
    }


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verify_training_run_inputs(
    *,
    training_run: dict[str, Any],
    inputs: dict[str, Path],
    levels: tuple[int, ...],
) -> None:
    if (
        training_run.get("format_id") != "flow_belief_run_v1"
        or training_run.get("checkpoint_scope") != "per_level_only"
        or training_run.get("eligible") is not True
        or not set(levels).issubset(set(training_run.get("levels", [])))
    ):
        raise ValueError("fresh Decoder probe requires an eligible matching Flow run")
    expected_hashes = training_run.get("input_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("fresh Decoder source run has no verified input hashes")
    mapping = {
        "source_bulk_manifest": "source_bulk_manifest",
        "cache_run_manifest": "cache_run_manifest",
        "vision_config": "vision_config",
        "temporal_config": "temporal_config",
        "latency_law": "latency_law",
        "flow_config": "flow_config",
    }
    if "split_config" in inputs:
        mapping["split_config"] = "split_config"
    for input_name, hash_name in mapping.items():
        if expected_hashes.get(hash_name) != sha256_file(inputs[input_name]):
            raise ValueError(f"fresh Decoder source run input hash mismatch: {input_name}")


def run_fresh_decoder_probe(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    flow_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    latency_law_path: Path,
    flow_config_path: Path,
    probe_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    device: str,
    split_config_path: Path | None = None,
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
        raise ValueError("fresh Decoder levels must be sorted unique values from 1, 2, 3")
    optional_positive = {
        "training_context_limit": training_context_limit,
        "validation_context_limit": validation_context_limit,
        "evaluation_sample_count": evaluation_sample_count,
        "solver_step_count": solver_step_count,
    }
    if any(
        value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0)
        for value in optional_positive.values()
    ):
        raise ValueError("fresh Decoder optional limits must be positive integers")
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "flow_run_manifest": flow_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "latency_law": latency_law_path.resolve(),
        "flow_config": flow_config_path.resolve(),
        "probe_config": probe_config_path.resolve(),
    }
    if split_config_path is not None:
        inputs["split_config"] = split_config_path.resolve()
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"fresh Decoder input does not exist: {path}")
    training_run = _load_json(inputs["flow_run_manifest"])
    _verify_training_run_inputs(
        training_run=training_run,
        inputs=inputs,
        levels=selected_levels,
    )
    flow_config = load_flow_belief_config(inputs["flow_config"])
    probe_config = load_fresh_decoder_probe_config(inputs["probe_config"])
    training_config = dataclasses.replace(
        flow_config,
        max_epochs=probe_config.max_epochs,
        early_stopping_patience=probe_config.early_stopping_patience,
    )
    evaluation_config = dataclasses.replace(
        flow_config,
        evaluation_sample_count=(evaluation_sample_count or flow_config.evaluation_sample_count),
        solver_step_count=solver_step_count or flow_config.solver_step_count,
    )
    spec = load_vision_encoder_spec(inputs["vision_config"])
    provenance = collect_implementation_provenance(project_root)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"fresh Decoder run already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    level_manifests = {}
    level_admission = {}
    try:
        for level in selected_levels:
            checkpoint_dir = inputs["flow_run_manifest"].parent / f"L{level}"
            loaded = load_verified_flow_quality_level(
                checkpoint_dir=checkpoint_dir,
                flow_config=flow_config,
                expected_level=level,
                device=device,
            )
            probe = FrozenEncoderFreshDecoder.from_source_model(
                source_model=loaded.model,
                config=flow_config,
                decoder_seed=probe_config.decoder_seeds[0],
            )
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
            limits = {}
            if training_context_limit is not None:
                limits[ProbeSplit.TRAIN] = training_context_limit
            if validation_context_limit is not None:
                limits[ProbeSplit.VALIDATION] = validation_context_limit
            cached = cache_frozen_belief_splits(
                corpus=corpus,
                probe=probe,
                normalization=loaded.normalization,
                splits=(ProbeSplit.TRAIN, ProbeSplit.VALIDATION),
                batch_size=flow_config.batch_size,
                device=device,
                context_limits=limits,
            )
            level_dir = building / f"L{level}"
            level_dir.mkdir()
            joint_metrics = evaluate_cached_vector_field(
                vector_field=loaded.model.vector_field,
                cached=cached[ProbeSplit.VALIDATION],
                normalization=loaded.normalization,
                config=evaluation_config,
                level=level,
                device=device,
            )
            _write_json(level_dir / "joint_decoder_metrics.json", joint_metrics)
            fresh_metrics = {}
            for decoder_seed in probe_config.decoder_seeds:
                seed_dir = level_dir / f"seed_{decoder_seed:06d}"
                train_fresh_vector_field(
                    cached=cached,
                    config=training_config,
                    level=level,
                    decoder_seed=decoder_seed,
                    output_dir=seed_dir,
                    device=device,
                )
                vector_field = ConditionalStateVectorField(flow_config).to(device)
                vector_field.load_state_dict(
                    load_safetensors(seed_dir / "vector_field.safetensors"),
                    strict=True,
                )
                metrics = evaluate_cached_vector_field(
                    vector_field=vector_field,
                    cached=cached[ProbeSplit.VALIDATION],
                    normalization=loaded.normalization,
                    config=evaluation_config,
                    level=level,
                    device=device,
                )
                fresh_metrics[decoder_seed] = metrics
                _write_json(seed_dir / "evaluation_metrics.json", metrics)
            summary = summarize_fresh_decoder_metrics(
                joint_metrics=joint_metrics,
                fresh_metrics=fresh_metrics,
                rmse_ratio_max=probe_config.rmse_ratio_max,
            )
            _write_json(level_dir / "summary.json", summary)
            level_admission[f"L{level}"] = summary["admitted"]
            level_artifacts = {
                path.relative_to(level_dir).as_posix(): sha256_file(path)
                for path in sorted(level_dir.rglob("*"))
                if path.is_file()
            }
            level_manifest = {
                "schema_version": 1,
                "format_id": "level_flow_belief_fresh_decoder_probe_v1",
                "level": level,
                "source_model_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
                "source_encoder_sha256": sha256_file(checkpoint_dir / "encoder.safetensors"),
                "cached_belief_sha256": {
                    split.value: _array_sha256(cached[split].belief_tokens)
                    for split in (ProbeSplit.TRAIN, ProbeSplit.VALIDATION)
                },
                "training_context_count": len(cached[ProbeSplit.TRAIN].belief_tokens),
                "validation_context_count": len(cached[ProbeSplit.VALIDATION].belief_tokens),
                "decoder_seeds": list(probe_config.decoder_seeds),
                "admitted": summary["admitted"],
                "artifacts": level_artifacts,
            }
            _write_json(level_dir / "manifest.json", level_manifest)
            level_manifests[f"L{level}"] = f"L{level}/manifest.json"
            del cached, corpus, probe, loaded
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_fresh_decoder_probe_run_v1",
            "eligible": not provenance.dirty and training_run.get("eligible") is True,
            "blockers": (
                []
                if not provenance.dirty and training_run.get("eligible") is True
                else ["dirty_implementation_or_source_run"]
            ),
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "source_flow_run_sha256": sha256_file(inputs["flow_run_manifest"]),
            "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
            "levels": list(selected_levels),
            "decoder_seeds": list(probe_config.decoder_seeds),
            "level_manifests": level_manifests,
            "level_admission": level_admission,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
