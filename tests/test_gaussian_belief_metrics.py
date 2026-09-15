from __future__ import annotations

import numpy as np
import pytest


def test_gaussian_metrics_use_latency_weights_and_physical_units() -> None:
    from latency_meta_mdp.legacy.gaussian_belief_metrics import gaussian_prediction_metrics

    target = np.zeros((2, 2, 22), dtype=np.float64)
    mean = target.copy()
    mean[..., 16:19] += 0.002
    mean[..., 19:22] += 0.010
    std = np.full_like(target, 0.020)
    weights = np.asarray([[0.25, 0.75], [0.6, 0.4]])

    metrics = gaussian_prediction_metrics(
        mean=mean,
        std=std,
        target=target,
        weights=weights,
    )

    assert metrics["object_position"]["rmse_display"] == pytest.approx(2.0)
    assert metrics["object_position"]["display_unit"] == "mm"
    assert metrics["object_linear_velocity"]["rmse_display"] == pytest.approx(10.0)
    assert metrics["object_linear_velocity"]["display_unit"] == "mm/s"
    assert metrics["object_position"]["coverage_1sigma"] == pytest.approx(1.0)
    assert metrics["object_position"]["coverage_2sigma"] == pytest.approx(1.0)
