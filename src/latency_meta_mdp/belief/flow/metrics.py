"""Sample-based proper scores and calibration metrics for Flow Beliefs."""

from __future__ import annotations

from typing import Any

import numpy as np

_PHYSICAL_GROUPS = (
    ("robot_joint_position", slice(0, 7), "rad", 1.0),
    ("robot_joint_velocity", slice(7, 14), "rad/s", 1.0),
    ("gripper_width", slice(14, 15), "mm", 1000.0),
    ("gripper_width_velocity", slice(15, 16), "mm/s", 1000.0),
    ("object_position", slice(16, 19), "mm", 1000.0),
    ("object_linear_velocity", slice(19, 22), "mm/s", 1000.0),
)


def _validate(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = np.asarray(samples, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    probability = np.asarray(weights, dtype=np.float64)
    if (
        draws.ndim != 4
        or draws.shape[-1] != 22
        or expected.shape != draws.shape[:2] + (22,)
        or probability.shape != draws.shape[:2]
        or not np.all(np.isfinite(draws))
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(probability))
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("Flow sample metrics require finite [N,R,S,22] weighted arrays")
    return draws, expected, probability


def energy_score(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    draws, expected, probability = _validate(
        samples=samples,
        target=target,
        weights=weights,
    )
    context_scores = []
    for start in range(0, len(draws), 4):
        stop = min(start + 4, len(draws))
        chunk = draws[start:stop]
        target_distance = np.linalg.norm(
            chunk - expected[start:stop, :, None, :],
            axis=-1,
        ).mean(axis=-1)
        pairwise_distance = np.linalg.norm(
            chunk[:, :, :, None, :] - chunk[:, :, None, :, :],
            axis=-1,
        ).mean(axis=(-1, -2))
        per_query = target_distance - 0.5 * pairwise_distance
        context_scores.append(
            np.sum(probability[start:stop] * per_query, axis=-1)
        )
    return float(np.mean(np.concatenate(context_scores, axis=0)))


def sample_distribution_metrics(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    draws, expected, probability = _validate(
        samples=samples,
        target=target,
        weights=weights,
    )
    lower_68, upper_68 = np.quantile(draws, (0.16, 0.84), axis=2)
    lower_95, upper_95 = np.quantile(draws, (0.025, 0.975), axis=2)
    expanded_weight = probability[..., None]
    denominator = float(np.sum(probability) * draws.shape[-1])
    coverage_68 = np.sum(
        expanded_weight * ((expected >= lower_68) & (expected <= upper_68))
    ) / denominator
    coverage_95 = np.sum(
        expanded_weight * ((expected >= lower_95) & (expected <= upper_95))
    ) / denominator
    dispersion = np.std(draws, axis=2)
    dispersion_mean = float(np.sum(expanded_weight * dispersion) / denominator)
    collapsed_fraction = float(
        np.sum(expanded_weight * (dispersion < 1e-5)) / denominator
    )
    return {
        "sample_count": draws.shape[2],
        "energy_score": energy_score(
            samples=draws,
            target=expected,
            weights=probability,
        ),
        "coverage_68": float(coverage_68),
        "coverage_95": float(coverage_95),
        "normalized_dispersion_mean": dispersion_mean,
        "collapsed_dimension_fraction": collapsed_fraction,
        "collapsed": dispersion_mean < 1e-4 or collapsed_fraction > 0.95,
    }


def sample_mean_physical_metrics(
    *,
    normalized_samples: np.ndarray,
    normalized_target: np.ndarray,
    weights: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, Any]:
    draws, expected, probability = _validate(
        samples=normalized_samples,
        target=normalized_target,
        weights=weights,
    )
    offset = np.asarray(target_mean, dtype=np.float64)
    scale = np.asarray(target_std, dtype=np.float64)
    if offset.shape != (22,) or scale.shape != (22,) or np.any(scale <= 0.0):
        raise ValueError("Flow physical target normalization must contain positive 22D arrays")
    sample_mean = draws.mean(axis=2) * scale + offset
    physical_target = expected * scale + offset
    lower_68, upper_68 = np.quantile(draws, (0.16, 0.84), axis=2)
    lower_95, upper_95 = np.quantile(draws, (0.025, 0.975), axis=2)
    lower_68 = lower_68 * scale + offset
    upper_68 = upper_68 * scale + offset
    lower_95 = lower_95 * scale + offset
    upper_95 = upper_95 * scale + offset
    expanded_weight = probability[..., None]
    result: dict[str, Any] = {}
    for name, columns, unit, multiplier in _PHYSICAL_GROUPS:
        error = sample_mean[..., columns] - physical_target[..., columns]
        dimension = error.shape[-1]
        denominator = float(np.sum(probability) * dimension)
        mse = float(np.sum(expanded_weight * np.square(error)) / denominator)
        mae = float(np.sum(expanded_weight * np.abs(error)) / denominator)
        target_group = physical_target[..., columns]
        result[name] = {
            "rmse_si": mse**0.5,
            "mae_si": mae,
            "rmse_display": mse**0.5 * multiplier,
            "mae_display": mae * multiplier,
            "display_unit": unit,
            "coverage_68": float(
                np.sum(
                    expanded_weight
                    * (
                        (target_group >= lower_68[..., columns])
                        & (target_group <= upper_68[..., columns])
                    )
                )
                / denominator
            ),
            "coverage_95": float(
                np.sum(
                    expanded_weight
                    * (
                        (target_group >= lower_95[..., columns])
                        & (target_group <= upper_95[..., columns])
                    )
                )
                / denominator
            ),
        }
    return result
