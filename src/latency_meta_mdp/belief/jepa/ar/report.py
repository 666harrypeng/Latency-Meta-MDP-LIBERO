"""Reproducible fold-level summaries for L3 temporal-configuration selection."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPORAL_CONFIG_IDS = (
    "dense_20ms_history_100ms",
    "stride2_40ms_history_120ms",
    "stride4_80ms_history_160ms",
    "stride5_100ms_history_200ms",
)

_CONFIG_STRIDES = dict(zip(TEMPORAL_CONFIG_IDS, (1, 2, 4, 5), strict=True))
_ASSIGNED_DELAYS = {
    1: list(range(1, 21)),
    2: [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20],
    4: [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16, 20, 20, 20],
    5: [5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 20, 20, 20],
}
_COMMON_INPUT_KEYS = (
    "cache_manifest",
    "model_config",
    "selection_manifest",
    "source_manifest",
    "split_manifest",
    "training_config",
)
_SCALAR_METRICS = {
    "native_latent_rmse": ("native", "source_mean_latent_rmse"),
    "native_proprio_rmse": ("native", "source_mean_proprio_rmse"),
    "assigned_model_latent_rmse": (
        "deployed_d20",
        "source_mean_assigned_model_latent_rmse",
    ),
    "assigned_model_proprio_rmse": (
        "deployed_d20",
        "source_mean_assigned_model_proprio_rmse",
    ),
    "deployed_latent_rmse": ("deployed_d20", "source_mean_deployed_latent_rmse"),
    "deployed_proprio_rmse": ("deployed_d20", "source_mean_deployed_proprio_rmse"),
    "temporal_quantization_latent_rmse": (
        "deployed_d20",
        "source_mean_temporal_quantization_latent_rmse",
    ),
    "temporal_quantization_proprio_rmse": (
        "deployed_d20",
        "source_mean_temporal_quantization_proprio_rmse",
    ),
}
_DEPLOYED_CURVES = {
    "assigned_model_latent_rmse": "per_delay_assigned_model_latent_rmse",
    "assigned_model_proprio_rmse": "per_delay_assigned_model_proprio_rmse",
    "deployed_latent_rmse": "per_delay_deployed_latent_rmse",
    "deployed_proprio_rmse": "per_delay_deployed_proprio_rmse",
    "temporal_quantization_latent_rmse": "per_delay_temporal_quantization_latent_rmse",
    "temporal_quantization_proprio_rmse": "per_delay_temporal_quantization_proprio_rmse",
}
_NATIVE_CURVES = {
    "latent_predictive_skill": "per_anchor_latent_predictive_skill",
    "latent_rmse": "per_anchor_latent_rmse",
    "persistence_latent_rmse": "per_anchor_persistence_latent_rmse",
    "persistence_proprio_rmse": "per_anchor_persistence_proprio_rmse",
    "proprio_rmse": "per_anchor_proprio_rmse",
}


@dataclass(frozen=True)
class TemporalSelectionReportPaths:
    summary_json: Path
    deployed_d20_csv: Path
    native_anchors_csv: Path


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_curve(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} has an invalid length")
    return [_finite_float(item, label=label) for item in value]


def _statistics(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "fold_values": values,
    }


def _load_run(path: Path, *, config_id: str, fold_index: int, model_seed: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid temporal-selection manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid temporal-selection manifest: {path}")
    if (
        value.get("schema_version") != 1
        or value.get("format_id") != "action_conditioned_jepa_fold_run_v1"
        or value.get("qualification_only") is not False
    ):
        raise ValueError(f"manifest is not a completed selection run: {path}")
    preflight = value.get("preflight")
    progress = value.get("progress")
    final = value.get("final_development")
    provenance = value.get("input_sha256")
    if not all(isinstance(item, dict) for item in (preflight, progress, final, provenance)):
        raise ValueError(f"manifest fields are incomplete: {path}")
    assert isinstance(preflight, dict)
    assert isinstance(progress, dict)
    assert isinstance(final, dict)
    assert isinstance(provenance, dict)
    if (
        preflight.get("temporal_config_id") != config_id
        or preflight.get("fold_index") != fold_index
        or preflight.get("model_seed") != model_seed
        or progress.get("temporal_config_id") != config_id
        or progress.get("fold_index") != fold_index
        or progress.get("model_seed") != model_seed
    ):
        raise ValueError(f"manifest identity does not match its selection slot: {path}")
    if (
        progress.get("completed_epochs") != preflight.get("max_epochs")
        or progress.get("optimizer_steps") != preflight.get("total_optimizer_steps")
        or progress.get("examples_seen")
        != preflight.get("logical_global_batch_size") * preflight.get("total_optimizer_steps")
    ):
        raise ValueError(f"manifest progress is not terminal: {path}")
    native = final.get("native")
    deployed = final.get("deployed_d20")
    if not isinstance(native, dict) or not isinstance(deployed, dict):
        raise ValueError(f"final development metrics are incomplete: {path}")
    stride = _CONFIG_STRIDES[config_id]
    native_ticks = list(range(stride, 21, stride))
    if native.get("native_delay_ticks") != native_ticks:
        raise ValueError(f"native delay inventory is invalid: {path}")
    if deployed.get("dense_delay_ticks") != list(range(1, 21)):
        raise ValueError(f"deployed D20 inventory is invalid: {path}")
    if deployed.get("assigned_native_delay_ticks") != _ASSIGNED_DELAYS[stride]:
        raise ValueError(f"deployed delay assignment is invalid: {path}")
    source_count = preflight.get("development_context_count")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
        or native.get("source_count") != source_count
        or deployed.get("source_count") != source_count
    ):
        raise ValueError(f"development source count is invalid: {path}")
    for _, (section, key) in _SCALAR_METRICS.items():
        _finite_float(final[section].get(key), label=f"{path}:{section}.{key}")
    for key in _DEPLOYED_CURVES.values():
        _finite_curve(deployed.get(key), length=20, label=f"{path}:deployed_d20.{key}")
    for key in _NATIVE_CURVES.values():
        _finite_curve(native.get(key), length=len(native_ticks), label=f"{path}:native.{key}")
    if set(provenance) != {*_COMMON_INPUT_KEYS, "temporal_config"} or any(
        not isinstance(provenance[key], str) or not provenance[key] for key in provenance
    ):
        raise ValueError(f"input provenance is invalid: {path}")
    return value


def build_temporal_selection_report(root: Path, *, model_seed: int) -> dict[str, Any]:
    """Load exactly four folds per config and aggregate folds with equal weight."""

    runs: dict[str, list[dict[str, Any]]] = {}
    missing = []
    for config_id in TEMPORAL_CONFIG_IDS:
        config_runs = []
        for fold_index in range(4):
            path = root / config_id / f"fold-{fold_index}" / f"seed-{model_seed}" / "manifest.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            config_runs.append(
                _load_run(
                    path,
                    config_id=config_id,
                    fold_index=fold_index,
                    model_seed=model_seed,
                )
            )
        runs[config_id] = config_runs
    if missing or any(len(values) != 4 for values in runs.values()):
        raise ValueError("temporal selection requires a complete four-fold inventory")

    reference = runs[TEMPORAL_CONFIG_IDS[0]][0]["input_sha256"]
    common_provenance = {key: reference[key] for key in _COMMON_INPUT_KEYS}
    for config_runs in runs.values():
        for run in config_runs:
            if any(
                run["input_sha256"][key] != common_provenance[key] for key in _COMMON_INPUT_KEYS
            ):
                raise ValueError("temporal selection runs do not share common input provenance")
    for fold_index in range(4):
        counts = {
            runs[config_id][fold_index]["preflight"]["development_context_count"]
            for config_id in TEMPORAL_CONFIG_IDS
        }
        if len(counts) != 1:
            raise ValueError("temporal configs do not share fold development contexts")

    configurations: dict[str, Any] = {}
    for config_id, config_runs in runs.items():
        native_ticks = config_runs[0]["final_development"]["native"]["native_delay_ticks"]
        assigned_ticks = config_runs[0]["final_development"]["deployed_d20"][
            "assigned_native_delay_ticks"
        ]
        scalar_metrics = {}
        for name, (section, key) in _SCALAR_METRICS.items():
            scalar_metrics[name] = _statistics(
                [float(run["final_development"][section][key]) for run in config_runs]
            )
        deployed_curves = {}
        for name, key in _DEPLOYED_CURVES.items():
            deployed_curves[name] = [
                _statistics(
                    [
                        float(run["final_development"]["deployed_d20"][key][delay_index])
                        for run in config_runs
                    ]
                )
                for delay_index in range(20)
            ]
        native_curves = {}
        for name, key in _NATIVE_CURVES.items():
            native_curves[name] = [
                _statistics(
                    [
                        float(run["final_development"]["native"][key][anchor_index])
                        for run in config_runs
                    ]
                )
                for anchor_index in range(len(native_ticks))
            ]
        configurations[config_id] = {
            "fold_indices": [0, 1, 2, 3],
            "development_source_counts": [
                run["preflight"]["development_context_count"] for run in config_runs
            ],
            "temporal_config_sha256": config_runs[0]["input_sha256"]["temporal_config"],
            "native_delay_ticks": native_ticks,
            "dense_delay_ticks": list(range(1, 21)),
            "assigned_native_delay_ticks": assigned_ticks,
            "scalar_metrics": scalar_metrics,
            "native_anchor_metrics": native_curves,
            "deployed_d20_metrics": deployed_curves,
        }
    return {
        "schema_version": 1,
        "format_id": "action_conditioned_jepa_temporal_selection_report_v1",
        "model_seed": model_seed,
        "fold_count": 4,
        "fold_aggregation": "equal_weight_sample_mean_and_sample_std",
        "config_order": list(TEMPORAL_CONFIG_IDS),
        "common_input_sha256": common_provenance,
        "configurations": configurations,
    }


def _write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_temporal_selection_report(
    root: Path,
    output_dir: Path,
    *,
    model_seed: int,
) -> TemporalSelectionReportPaths:
    """Materialize one JSON summary and two long-form curve tables."""

    report = build_temporal_selection_report(root, model_seed=model_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    deployed_rows = []
    native_rows = []
    for config_id in report["config_order"]:
        config = report["configurations"][config_id]
        for index, delay_tick in enumerate(config["dense_delay_ticks"]):
            row = {
                "config_id": config_id,
                "delay_tick": delay_tick,
                "delay_ms": delay_tick * 20,
                "assigned_native_delay_tick": config["assigned_native_delay_ticks"][index],
            }
            for name, values in config["deployed_d20_metrics"].items():
                row[f"{name}_mean"] = values[index]["mean"]
                row[f"{name}_sample_std"] = values[index]["sample_std"]
            deployed_rows.append(row)
        for index, delay_tick in enumerate(config["native_delay_ticks"]):
            row = {
                "config_id": config_id,
                "delay_tick": delay_tick,
                "delay_ms": delay_tick * 20,
            }
            for name, values in config["native_anchor_metrics"].items():
                row[f"{name}_mean"] = values[index]["mean"]
                row[f"{name}_sample_std"] = values[index]["sample_std"]
            native_rows.append(row)

    deployed_path = output_dir / "deployed_d20.csv"
    native_path = output_dir / "native_anchors.csv"
    _write_csv(deployed_path, fieldnames=list(deployed_rows[0]), rows=deployed_rows)
    _write_csv(native_path, fieldnames=list(native_rows[0]), rows=native_rows)
    return TemporalSelectionReportPaths(
        summary_json=summary_path,
        deployed_d20_csv=deployed_path,
        native_anchors_csv=native_path,
    )
