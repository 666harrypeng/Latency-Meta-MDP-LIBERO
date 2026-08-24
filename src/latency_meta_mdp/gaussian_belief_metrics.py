"""Weighted physical and calibration metrics for Gaussian return beliefs."""

from __future__ import annotations

from typing import Any

import numpy as np

_GROUPS = (
    ("robot_joint_position", slice(0, 7), "rad", 1.0),
    ("robot_joint_velocity", slice(7, 14), "rad/s", 1.0),
    ("gripper_width", slice(14, 15), "mm", 1000.0),
    ("gripper_width_velocity", slice(15, 16), "mm/s", 1000.0),
    ("object_position", slice(16, 19), "mm", 1000.0),
    ("object_linear_velocity", slice(19, 22), "mm/s", 1000.0),
)


def _validate(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(mean, dtype=np.float64)
    uncertainty = np.asarray(std, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64)
    if (
        predicted.ndim != 3
        or predicted.shape[-1] != 22
        or uncertainty.shape != predicted.shape
        or expected.shape != predicted.shape
        or probability.shape != predicted.shape[:2]
        or np.any(uncertainty <= 0.0)
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(uncertainty))
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(probability))
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("Gaussian metric arrays do not satisfy the weighted prediction contract")
    return predicted, uncertainty, expected, probability


def gaussian_prediction_metrics(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    predicted, uncertainty, expected, probability = _validate(
        mean=mean,
        std=std,
        target=target,
        weights=weights,
    )
    result: dict[str, Any] = {
        "sample_count": predicted.shape[0],
        "query_count": predicted.shape[1],
    }
    for name, columns, unit, multiplier in _GROUPS:
        error = predicted[..., columns] - expected[..., columns]
        sigma = uncertainty[..., columns]
        dimension = error.shape[-1]
        denominator = float(np.sum(probability) * dimension)
        expanded_weight = probability[..., None]
        mse = float(np.sum(expanded_weight * np.square(error)) / denominator)
        mae = float(np.sum(expanded_weight * np.abs(error)) / denominator)
        standardized = error / sigma
        standardized_mean = float(
            np.sum(expanded_weight * standardized) / denominator
        )
        centered = standardized - standardized_mean
        variance = float(np.sum(expanded_weight * np.square(centered)) / denominator)
        scale = max(variance, 1e-12) ** 0.5
        skew = float(
            np.sum(expanded_weight * np.power(centered / scale, 3)) / denominator
        )
        excess_kurtosis = float(
            np.sum(expanded_weight * np.power(centered / scale, 4)) / denominator - 3.0
        )
        result[name] = {
            "rmse_si": mse**0.5,
            "mae_si": mae,
            "rmse_display": mse**0.5 * multiplier,
            "mae_display": mae * multiplier,
            "display_unit": unit,
            "coverage_1sigma": float(
                np.sum(expanded_weight * (np.abs(error) <= sigma)) / denominator
            ),
            "coverage_2sigma": float(
                np.sum(expanded_weight * (np.abs(error) <= 2.0 * sigma)) / denominator
            ),
            "standardized_residual_mean": standardized_mean,
            "standardized_residual_variance": variance,
            "standardized_residual_skew": skew,
            "standardized_residual_excess_kurtosis": excess_kurtosis,
        }
    return result


def weighted_normalized_gaussian_nll(
    *,
    mean: np.ndarray,
    log_std: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    predicted = np.asarray(mean, dtype=np.float64)
    logarithm = np.asarray(log_std, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64)
    if (
        predicted.shape != logarithm.shape
        or predicted.shape != expected.shape
        or predicted.ndim != 3
        or predicted.shape[-1] != 22
        or probability.shape != predicted.shape[:2]
    ):
        raise ValueError("normalized Gaussian NLL arrays have invalid shapes")
    residual = (expected - predicted) * np.exp(-logarithm)
    element = 0.5 * (
        np.square(residual) + 2.0 * logarithm + np.log(2.0 * np.pi)
    )
    query_nll = element.mean(axis=-1)
    return float(np.mean(np.sum(probability * query_nll, axis=-1)))
