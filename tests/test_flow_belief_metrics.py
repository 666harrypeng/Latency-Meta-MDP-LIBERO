from __future__ import annotations

import numpy as np
import pytest


def test_energy_score_is_zero_for_perfect_deterministic_samples() -> None:
    from latency_meta_mdp.belief.flow.metrics import energy_score

    target = np.ones((2, 3, 22), dtype=np.float64)
    samples = np.repeat(target[:, :, None, :], 4, axis=2)
    weights = np.full((2, 3), 1.0 / 3.0)

    assert energy_score(samples=samples, target=target, weights=weights) == pytest.approx(0.0)


def test_energy_score_matches_two_sample_hand_calculation() -> None:
    from latency_meta_mdp.belief.flow.metrics import energy_score

    target = np.zeros((1, 1, 22), dtype=np.float64)
    target[..., 0] = 1.0
    samples = np.zeros((1, 1, 2, 22), dtype=np.float64)
    samples[..., 1, 0] = 2.0
    weights = np.ones((1, 1), dtype=np.float64)

    assert energy_score(samples=samples, target=target, weights=weights) == pytest.approx(0.5)


def test_sample_distribution_metrics_detect_coverage_and_collapse() -> None:
    from latency_meta_mdp.belief.flow.metrics import sample_distribution_metrics

    target = np.zeros((1, 1, 22), dtype=np.float64)
    diverse = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    samples = np.repeat(diverse[None, None, :, None], 22, axis=3)
    weights = np.ones((1, 1), dtype=np.float64)

    metrics = sample_distribution_metrics(
        samples=samples,
        target=target,
        weights=weights,
    )
    collapsed = sample_distribution_metrics(
        samples=np.zeros((1, 1, 32, 22), dtype=np.float64),
        target=target,
        weights=weights,
    )

    assert metrics["coverage_68"] == pytest.approx(1.0)
    assert metrics["coverage_95"] == pytest.approx(1.0)
    assert metrics["collapsed"] is False
    assert collapsed["collapsed"] is True


def test_sample_mean_physical_metrics_report_object_error_in_millimeters() -> None:
    from latency_meta_mdp.belief.flow.metrics import sample_mean_physical_metrics

    target = np.zeros((1, 1, 22), dtype=np.float64)
    samples = np.zeros((1, 1, 4, 22), dtype=np.float64)
    samples[..., 16:19] = 0.002

    metrics = sample_mean_physical_metrics(
        normalized_samples=samples,
        normalized_target=target,
        weights=np.ones((1, 1), dtype=np.float64),
        target_mean=np.zeros(22, dtype=np.float64),
        target_std=np.ones(22, dtype=np.float64),
    )

    assert metrics["object_position"]["rmse_display"] == pytest.approx(2.0)
    assert metrics["object_position"]["display_unit"] == "mm"
