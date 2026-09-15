from __future__ import annotations

from pathlib import Path

import pytest

from latency_meta_mdp.legacy.belief.flow.data_scaling_run import extract_scaling_metrics
from latency_meta_mdp.runtime.latency_law import load_latency_law


def test_extract_scaling_metrics_matches_formal_l1_evaluation() -> None:
    checkpoint = Path("outputs/analysis/flow_belief/dinov3-formal-180-20-f6caf55/L1")
    evaluation = Path("outputs/analysis/flow_belief_evaluation/dinov3-formal-180-20-f6caf55/L1")
    if not checkpoint.is_dir() or not evaluation.is_dir():
        pytest.skip("scaling metric extraction requires formal Flow artifacts")

    metrics = extract_scaling_metrics(
        checkpoint_dir=checkpoint,
        evaluation_dir=evaluation,
        latency_probabilities=load_latency_law(
            Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml")
        ).probabilities,
    )

    assert metrics["overall_object_position_rmse_mm"] == pytest.approx(1.14734385)
    assert metrics["overall_object_velocity_rmse_mm_s"] == pytest.approx(4.14559766)
    assert metrics["pre_handoff_object_position_rmse_mm"] == pytest.approx(1.07904848)
    assert metrics["pre_handoff_object_velocity_rmse_mm_s"] == pytest.approx(1.44496149)
    assert (
        metrics["worst_decile_object_position_rmse_mm"] > metrics["overall_object_position_rmse_mm"]
    )
