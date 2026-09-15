"""Verified deterministic Flow samples for selected quality contexts."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.legacy.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.legacy.belief.flow.context_sampling import (
    load_flow_belief_normalization,
    load_verified_flow_quality_level,
    sample_flow_validation_contexts,
)
from latency_meta_mdp.legacy.belief.flow.quality_config import FlowBeliefQualitySampleConfig
from latency_meta_mdp.legacy.belief.flow.quality_selection import FormalFlowSummary
from latency_meta_mdp.legacy.belief.flow.quality_types import (
    QualitySampleBundle,
    QualitySelection,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

__all__ = [
    "export_level_quality_samples",
    "load_flow_belief_normalization",
    "load_verified_flow_quality_level",
]


def _load_json(path: Path) -> dict[str, Any]:
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


def _selection_rows(selections: tuple[QualitySelection, ...]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(selection) for selection in selections]


def _selected_metrics(bundle: QualitySampleBundle) -> dict[str, Any]:
    sample_mean = bundle.physical_samples.mean(axis=2)
    error = sample_mean - bundle.physical_targets
    group_errors = {
        "robot_joint_position_rad": np.sqrt(np.mean(np.square(error[..., :7]), axis=-1)),
        "object_position_m": np.linalg.norm(error[..., 16:19], axis=-1),
        "object_linear_velocity_m_s": np.linalg.norm(error[..., 19:22], axis=-1),
    }
    quantiles = (0.5, 0.9, 0.95)
    result: dict[str, Any] = {
        "error_quantiles": {},
        "signed_object_position_bias_m": np.mean(error[..., 16:19], axis=(0, 1)).tolist(),
        "nonfinite_sample_count": int(
            np.size(bundle.physical_samples)
            - np.count_nonzero(np.isfinite(bundle.physical_samples))
        ),
    }
    for name, values in group_errors.items():
        result["error_quantiles"][name] = {
            f"p{int(quantile * 100)}": float(np.quantile(values, quantile))
            for quantile in quantiles
        }
    return result


def export_level_quality_samples(
    *,
    corpus: FeatureBeliefCorpus,
    config: FlowBeliefQualitySampleConfig,
    flow_config: FlowBeliefConfig,
    flow_checkpoint_dir: Path,
    evaluation_summary: FormalFlowSummary,
    selections: tuple[QualitySelection, ...],
    output_dir: Path,
    device: str,
) -> Path:
    started = time.perf_counter()
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow quality level output already exists: {target}")
    if not selections or any(selection.identity.level != corpus.level for selection in selections):
        raise ValueError("Flow quality selections do not match the corpus level")
    offsets = tuple(selection.identity.validation_offset for selection in selections)
    if offsets != tuple(sorted(set(offsets))):
        raise ValueError("Flow quality selection offsets must be sorted and unique")
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    if evaluation_summary.context_count != len(references):
        raise ValueError("Flow quality summary and validation corpus counts disagree")
    for selection in selections:
        context = corpus.materialize(
            ProbeSplit.VALIDATION,
            selection.identity.validation_offset,
        )
        if (
            context.episode_id != selection.identity.episode_id
            or context.scene_seed != selection.identity.scene_seed
            or context.source_tick != selection.identity.source_tick
        ):
            raise ValueError("Flow quality selection identity does not match the corpus")
    loaded = load_verified_flow_quality_level(
        checkpoint_dir=flow_checkpoint_dir,
        flow_config=flow_config,
        expected_level=corpus.level,
        device=device,
    )
    sampling_started = time.perf_counter()
    sampled = sample_flow_validation_contexts(
        corpus=corpus,
        loaded=loaded,
        flow_config=flow_config,
        validation_offsets=offsets,
        delay_ticks=tuple(range(1, 21)),
        sample_count=config.sample_count,
        solver=config.solver,
        solver_step_count=config.solver_step_count,
        device=device,
    )
    sampling_wall_seconds = time.perf_counter() - sampling_started
    reproduced = sampled.normalized_samples
    noise_sha256_by_offset = sampled.noise_sha256_by_offset
    expected_mean = evaluation_summary.sample_mean_normalized[list(offsets)]
    expected_std = evaluation_summary.sample_std_normalized[list(offsets)]
    reproduced_mean = reproduced.mean(axis=2)
    reproduced_std = reproduced.std(axis=2)
    mean_allclose = bool(
        np.allclose(
            reproduced_mean,
            expected_mean,
            atol=config.summary_parity_atol,
            rtol=config.summary_parity_rtol,
        )
    )
    std_allclose = bool(
        np.allclose(
            reproduced_std,
            expected_std,
            atol=config.summary_parity_atol,
            rtol=config.summary_parity_rtol,
        )
    )
    if not mean_allclose or not std_allclose:
        raise ValueError("Flow quality reproduced samples do not match the formal summary")
    display_rows = np.asarray(config.display_delay_ticks, dtype=np.int64) - 1
    contexts = [corpus.materialize(ProbeSplit.VALIDATION, offset) for offset in offsets]
    normalized_targets = evaluation_summary.target_normalized[
        np.ix_(np.asarray(offsets), display_rows)
    ]
    normalized_samples = reproduced[:, display_rows]
    target_mean = np.asarray(loaded.normalization.target_mean, dtype=np.float32)
    target_std = np.asarray(loaded.normalization.target_std, dtype=np.float32)
    physical_samples = normalized_samples * target_std + target_mean
    physical_targets = normalized_targets * target_std + target_mean
    bundle = QualitySampleBundle(
        selections=selections,
        display_delay_ticks=np.asarray(config.display_delay_ticks),
        latency_probabilities=np.stack(
            [context.latency_probabilities[display_rows] for context in contexts]
        ),
        normalized_samples=normalized_samples,
        normalized_targets=normalized_targets,
        physical_samples=physical_samples,
        physical_targets=physical_targets,
        interaction_mode=evaluation_summary.interaction_mode[
            np.ix_(np.asarray(offsets), display_rows)
        ],
        absorbing=evaluation_summary.absorbing[np.ix_(np.asarray(offsets), display_rows)],
    )
    parity = {
        "mean_allclose": mean_allclose,
        "std_allclose": std_allclose,
        "mean_max_abs_error": float(np.max(np.abs(reproduced_mean - expected_mean))),
        "std_max_abs_error": float(np.max(np.abs(reproduced_std - expected_std))),
        "mean_physical_max_abs_error": float(
            np.max(np.abs(reproduced_mean - expected_mean) * target_std)
        ),
        "std_physical_max_abs_error": float(
            np.max(np.abs(reproduced_std - expected_std) * target_std)
        ),
        "atol": config.summary_parity_atol,
        "rtol": config.summary_parity_rtol,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "selection.json", _selection_rows(selections))
        _write_json(building / "parity.json", parity)
        _write_json(building / "metrics.json", _selected_metrics(bundle))
        np.savez(
            building / "samples.npz",
            validation_offsets=np.asarray(offsets, dtype=np.int64),
            display_delay_ticks=bundle.display_delay_ticks,
            latency_probabilities=bundle.latency_probabilities,
            normalized_samples=bundle.normalized_samples,
            normalized_targets=bundle.normalized_targets,
            physical_samples=bundle.physical_samples,
            physical_targets=bundle.physical_targets,
            interaction_mode=bundle.interaction_mode,
            absorbing=bundle.absorbing,
        )
        artifacts = {
            name: sha256_file(building / name)
            for name in ("selection.json", "parity.json", "metrics.json", "samples.npz")
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_quality_samples_v1",
            "eligible": True,
            "level": corpus.level,
            "context_count": len(selections),
            "sample_count": config.sample_count,
            "role_counts": dict(
                sorted(Counter(role for row in selections for role in row.roles).items())
            ),
            "display_delay_ticks": list(config.display_delay_ticks),
            "solver": config.solver,
            "solver_step_count": config.solver_step_count,
            "sampling_wall_seconds": sampling_wall_seconds,
            "seconds_per_selected_context": sampling_wall_seconds / len(selections),
            "wall_seconds": time.perf_counter() - started,
            "generated_noise_sha256": {
                str(offset): noise_sha256_by_offset[offset] for offset in offsets
            },
            "checkpoint_model_sha256": sha256_file(flow_checkpoint_dir / "model.safetensors"),
            "checkpoint_encoder_sha256": sha256_file(flow_checkpoint_dir / "encoder.safetensors"),
            "checkpoint_normalization_sha256": sha256_file(
                flow_checkpoint_dir / "normalization.npz"
            ),
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        building.rename(target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
