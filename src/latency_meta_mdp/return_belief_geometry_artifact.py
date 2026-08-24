"""Atomic evidence for return-belief geometry and action compatibility."""

from __future__ import annotations

import html
import json
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import (
    collect_implementation_provenance,
    sha256_file,
)
from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.belief_data_artifact import load_belief_data_view_config
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.return_belief_geometry import (
    RETURN_STATE_DIM,
    RETURN_STATE_NAMES,
    ActionCompatibilityMetrics,
    ReturnBeliefAuditConfig,
    StateGeometryMetrics,
    build_return_contexts,
    build_return_state_stream,
    fit_state_normalization,
    load_return_belief_audit_config,
    measure_action_compatibility,
    measure_state_geometry,
)

_PHASE_ORDER = ("pregrasp", "approach", "close", "lift")
_STATE_METRICS = tuple(field.name for field in fields(StateGeometryMetrics))
_ACTION_METRICS = tuple(field.name for field in fields(ActionCompatibilityMetrics))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _quantile_label(value: float) -> str:
    return f"q{round(value * 1000):04d}"


def _metric_summary(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    quantiles: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    return {
        name: {
            _quantile_label(probability): float(value)
            for probability, value in zip(
                quantiles,
                np.quantile([row[name] for row in rows], quantiles),
                strict=True,
            )
        }
        for name in metric_names
    }


def _group_summary(
    *,
    state_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    quantiles: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "state_context_count": len(state_rows),
        "action_context_count": len(action_rows),
        "state_metrics": _metric_summary(state_rows, _STATE_METRICS, quantiles),
        "action_metrics": _metric_summary(action_rows, _ACTION_METRICS, quantiles),
    }


def _rows_to_arrays(
    *,
    prefix: str,
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_episode_id": np.asarray(
            [row["episode_id"] for row in rows], dtype=np.str_
        ),
        f"{prefix}_level": np.asarray(
            [row["level"] for row in rows], dtype=np.int8
        ),
        f"{prefix}_source_tick": np.asarray(
            [row["source_tick"] for row in rows], dtype=np.int64
        ),
        f"{prefix}_source_phase": np.asarray(
            [row["source_phase"] for row in rows], dtype=np.str_
        ),
        **{
            f"{prefix}_metric_{name}": np.asarray(
                [row[name] for row in rows], dtype=np.float64
            )
            for name in metric_names
        },
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    metric_specs = (
        ("object_position_rms_spread", "Object position spread"),
        ("affine_residual_fraction", "Non-affine state fraction"),
        ("phase_crossing_probability", "Phase crossing probability"),
        ("translation_rms_deviation", "Action translation disagreement"),
        ("prefix_opposite_direction_rate", "Opposite-prefix rate"),
    )
    levels = sorted(summary["levels"], key=int)
    width = 980
    row_height = 76
    height = 90 + row_height * len(metric_specs)
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for metric, _label in metric_specs:
        source_key = "action_metrics" if metric in _ACTION_METRICS else "state_metrics"
        for level in levels:
            quantiles = summary["levels"][level][source_key].get(metric)
            values[metric][level] = 0.0 if not quantiles else quantiles["q0900"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}"
        ".label{font-size:14px}.level{font-size:12px;font-weight:700}.bar{fill:#2878c8}"
        ".track{fill:#e8edf4}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="30" y="42">Return-Belief Geometry</text>',
        '<text class="label" x="30" y="65">p90 by level; descriptive evidence, '
        'not an architecture verdict</text>',
    ]
    for row_index, (metric, label) in enumerate(metric_specs):
        y = 100 + row_index * row_height
        maximum = max(values[metric].values(), default=0.0)
        scale = 1.0 if maximum <= 1e-15 else maximum
        lines.append(
            f'<g data-metric="{html.escape(metric)}"><text class="label" x="30" y="{y}">'
            f"{html.escape(label)}</text>"
        )
        for level_index, level in enumerate(levels):
            bar_y = y + 10 + level_index * 17
            value = values[metric][level]
            bar_width = 560.0 * value / scale
            lines.extend(
                (
                    f'<text class="level" x="320" y="{bar_y + 11}">L{level}</text>',
                    f'<rect class="track" x="350" y="{bar_y}" width="560" height="12" rx="3"/>',
                    f'<rect class="bar" x="350" y="{bar_y}" '
                    f'width="{bar_width:.3f}" height="12" rx="3"/>',
                    f'<text class="label" x="920" y="{bar_y + 11}">{value:.6g}</text>',
                )
            )
        lines.append("</g>")
    lines.append("</svg>\n")
    return "\n".join(lines)


def _validate_source(source_path: Path) -> tuple[dict[str, Any], Path]:
    source = _load_json(source_path)
    if (
        source.get("format_id") != "panda_ball_bulk_first_tranche_v1"
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("analysis requires an eligible clean bulk source")
    admitted = source.get("admitted_episode_manifests")
    artifacts = source.get("artifacts")
    if not isinstance(admitted, list) or not admitted or not isinstance(artifacts, dict):
        raise ValueError("bulk source inventory is invalid")
    source_root = source_path.parent.resolve()
    for relative, digest in artifacts.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("bulk source artifact inventory is invalid")
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError("bulk source artifact path is invalid")
        if sha256_file(path) != digest:
            raise ValueError(f"bulk source artifact hash mismatch: {relative}")
    return source, source_root


def _build_summary(
    *,
    config: ReturnBeliefAuditConfig,
    source: dict[str, Any],
    source_path: Path,
    audit_config_path: Path,
    view_config_path: Path,
    latency_law_path: Path,
    provenance,
    law,
    normalization,
    episode_ids: dict[int, set[str]],
    state_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in sorted(episode_ids):
        level_state = [row for row in state_rows if row["level"] == level]
        level_action = [row for row in action_rows if row["level"] == level]
        phases = {}
        for phase in _PHASE_ORDER:
            phase_state = [row for row in level_state if row["source_phase"] == phase]
            if not phase_state:
                continue
            phase_action = [row for row in level_action if row["source_phase"] == phase]
            phases[phase] = _group_summary(
                state_rows=phase_state,
                action_rows=phase_action,
                quantiles=config.summary_quantiles,
            )
        levels[str(level)] = {
            "episode_count": len(episode_ids[level]),
            "episode_ids": sorted(episode_ids[level]),
            **_group_summary(
                state_rows=level_state,
                action_rows=level_action,
                quantiles=config.summary_quantiles,
            ),
            "phases": phases,
        }
    return {
        "schema_version": 1,
        "format_id": "return_belief_geometry_summary_v1",
        "analysis_id": config.analysis_id,
        "implementation_revision": provenance.revision,
        "implementation_source_sha256": provenance.source_sha256,
        "implementation_dirty": provenance.dirty,
        "source_bulk_manifest": source_path.as_posix(),
        "source_bulk_manifest_sha256": sha256_file(source_path),
        "source_run_id": source.get("run_id"),
        "audit_config_sha256": sha256_file(audit_config_path),
        "view_config_sha256": sha256_file(view_config_path),
        "latency_law_config_sha256": sha256_file(latency_law_path),
        "episode_count": sum(len(values) for values in episode_ids.values()),
        "state_schema": {
            "schema_id": "franka_joint_ball_return_state_v1",
            "dimension": RETURN_STATE_DIM,
            "names": list(RETURN_STATE_NAMES),
            "object_orientation_masked": True,
            "object_angular_velocity_masked": True,
        },
        "action_schema": {
            "contract_id": "panda_osc_pose_delta_v1",
            "dimension": 7,
            "prediction_horizon": 50,
            "prefix_ticks": config.action_prefix_ticks,
            "translation_rotation_units": "normalized_osc_command",
            "gripper_units": "normalized_command",
        },
        "state_normalization": {
            "mean": normalization.mean.tolist(),
            "scale": normalization.scale.tolist(),
            "scale_floor": config.state_scale_floor,
        },
        "latency_law": {
            "law_id": law.law_id,
            "delay_ticks": list(law.delay_ticks),
            "probabilities": law.probabilities.tolist(),
            "central_probability_mass": config.central_probability_mass,
        },
        "metric_semantics": {
            "automatic_density_verdict": False,
            "state_metric_names": list(_STATE_METRICS),
            "action_metric_names": list(_ACTION_METRICS),
            "quantiles": list(config.summary_quantiles),
        },
        "levels": levels,
    }


def write_return_belief_geometry_artifact(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    audit_config_path: Path,
    view_config_path: Path,
    latency_law_path: Path,
    output_dir: Path,
) -> Path:
    """Analyze one certified bulk source and atomically publish local evidence."""

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"return-belief analysis output already exists: {target}")
    project = project_root.resolve()
    source_path = source_bulk_manifest.resolve()
    audit_path = audit_config_path.resolve()
    view_path = view_config_path.resolve()
    law_path = latency_law_path.resolve()
    source, source_root = _validate_source(source_path)
    config = load_return_belief_audit_config(audit_path)
    view_config = load_belief_data_view_config(view_path)
    law = load_latency_law(law_path)
    admitted = source["admitted_episode_manifests"]
    episodes = []
    state_streams = []
    episode_ids: dict[int, set[str]] = defaultdict(set)
    for relative in admitted:
        if not isinstance(relative, str):
            raise ValueError("admitted episode paths must be strings")
        manifest_path = (source_root / relative).resolve()
        if not manifest_path.is_relative_to(source_root):
            raise ValueError("admitted episode path escapes the source root")
        episode = load_belief_episode(manifest_path.parent)
        episodes.append(episode)
        state_streams.append(build_return_state_stream(episode))
        episode_ids[episode.level].add(episode.episode_id)
    normalization = fit_state_normalization(
        tuple(state_streams), floor=config.state_scale_floor
    )
    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for episode in episodes:
        contexts = build_return_contexts(
            episode=episode,
            temporal_contract=view_config.temporal_contract,
            latency_law=law,
        )
        for context in contexts:
            identity = {
                "episode_id": context.episode_id,
                "level": context.level,
                "source_tick": context.source_tick,
                "source_phase": context.source_phase,
            }
            state_metrics = measure_state_geometry(
                context,
                normalization=normalization,
                central_probability_mass=config.central_probability_mass,
            )
            state_rows.append({**identity, **asdict(state_metrics)})
            action_metrics = measure_action_compatibility(
                context, prefix_ticks=config.action_prefix_ticks
            )
            if action_metrics is not None:
                action_rows.append({**identity, **asdict(action_metrics)})
    provenance = collect_implementation_provenance(project)
    summary = _build_summary(
        config=config,
        source=source,
        source_path=source_path,
        audit_config_path=audit_path,
        view_config_path=view_path,
        latency_law_path=law_path,
        provenance=provenance,
        law=law,
        normalization=normalization,
        episode_ids=episode_ids,
        state_rows=state_rows,
        action_rows=action_rows,
    )
    arrays = {
        **_rows_to_arrays(
            prefix="state", rows=state_rows, metric_names=_STATE_METRICS
        ),
        **_rows_to_arrays(
            prefix="action", rows=action_rows, metric_names=_ACTION_METRICS
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"return-belief staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_json(staging / "summary.json", summary)
        with (staging / "context_metrics.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _write_text(staging / "summary.svg", _summary_svg(summary))
        artifact_names = ("context_metrics.npz", "summary.json", "summary.svg")
        artifact_hashes = {
            name: sha256_file(staging / name) for name in artifact_names
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "return_belief_geometry_artifact_v1",
                "analysis_id": config.analysis_id,
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "eligible": not provenance.dirty,
                "source_bulk_manifest_sha256": sha256_file(source_path),
                "artifacts": artifact_hashes,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
