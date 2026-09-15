"""Physical-unit metrics for the diagnostic temporal state probe."""

from __future__ import annotations

from typing import Any

import numpy as np

from latency_meta_mdp.legacy.vision_probe_data import PROBE_TARGET_DIM

_GROUPS = (
    ("object_position", slice(0, 3), "mm"),
    ("object_linear_velocity", slice(3, 6), "mm/s"),
    ("relative_position", slice(6, 9), "mm"),
)


def _validate_targets(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != PROBE_TARGET_DIM
        or len(array) == 0
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite non-empty [N, 9] array")
    return array


def state_regression_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    predicted = _validate_targets(prediction, name="prediction")
    expected = _validate_targets(target, name="target")
    if predicted.shape != expected.shape:
        raise ValueError("probe prediction and target shapes disagree")
    metrics: dict[str, Any] = {"sample_count": len(expected)}
    for name, columns, display_unit in _GROUPS:
        error = predicted[:, columns] - expected[:, columns]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        mae = float(np.mean(np.abs(error)))
        centered = expected[:, columns] - expected[:, columns].mean(axis=0, keepdims=True)
        total = float(np.sum(np.square(centered)))
        residual = float(np.sum(np.square(error)))
        metrics[name] = {
            "rmse_si": rmse,
            "mae_si": mae,
            "rmse_display": rmse * 1000.0,
            "mae_display": mae * 1000.0,
            "display_unit": display_unit,
            "r2": None if total <= 0.0 else 1.0 - residual / total,
        }
    return metrics


def train_mean_baseline(
    *,
    training_targets: np.ndarray,
    evaluation_count: int,
) -> np.ndarray:
    training = _validate_targets(training_targets, name="training_targets")
    if (
        isinstance(evaluation_count, bool)
        or not isinstance(evaluation_count, int)
        or evaluation_count <= 0
    ):
        raise ValueError("evaluation_count must be a positive integer")
    return np.repeat(training.mean(axis=0, keepdims=True), evaluation_count, axis=0)
