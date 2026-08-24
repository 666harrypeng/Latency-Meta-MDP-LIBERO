from __future__ import annotations

import numpy as np
import pytest


def test_state_probe_metrics_preserve_physical_units_and_group_dimensions() -> None:
    from latency_meta_mdp.vision_probe_metrics import state_regression_metrics

    target = np.stack((np.zeros(9), np.ones(9)), axis=0)
    prediction = target.copy()
    prediction[:, 0:3] += 0.001
    prediction[:, 3:6] += 0.010
    prediction[:, 6:9] -= 0.002

    metrics = state_regression_metrics(prediction=prediction, target=target)

    assert metrics["sample_count"] == 2
    assert metrics["object_position"]["rmse_si"] == pytest.approx(0.001)
    assert metrics["object_position"]["rmse_display"] == pytest.approx(1.0)
    assert metrics["object_position"]["display_unit"] == "mm"
    assert metrics["object_linear_velocity"]["rmse_si"] == pytest.approx(0.010)
    assert metrics["object_linear_velocity"]["rmse_display"] == pytest.approx(10.0)
    assert metrics["object_linear_velocity"]["display_unit"] == "mm/s"
    assert metrics["relative_position"]["rmse_si"] == pytest.approx(0.002)
    assert metrics["relative_position"]["rmse_display"] == pytest.approx(2.0)


def test_train_mean_baseline_uses_only_training_targets() -> None:
    from latency_meta_mdp.vision_probe_metrics import train_mean_baseline

    training = np.stack((np.zeros(9), np.full(9, 2.0)), axis=0)

    baseline = train_mean_baseline(training_targets=training, evaluation_count=3)

    np.testing.assert_array_equal(baseline, np.ones((3, 9)))
