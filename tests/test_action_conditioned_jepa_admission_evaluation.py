from __future__ import annotations

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaEvaluationBatch,
)
from latency_meta_mdp.belief.jepa.contracts import (
    FutureLatentRollout,
    LaunchContextBatch,
)


class _FixedProprioRollout(torch.nn.Module):
    def __init__(self, proprio: torch.Tensor) -> None:
        super().__init__()
        self.proprio = proprio

    def rollout_native(self, context: LaunchContextBatch) -> FutureLatentRollout:
        return FutureLatentRollout(
            native_delay_ticks=torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64),
            future_visual_latents=torch.zeros(
                context.batch_size,
                5,
                2,
                196,
                384,
                dtype=torch.float16,
            ),
            future_proprio=self.proprio,
        )


def _batch(*, target: torch.Tensor, absorbing: torch.Tensor | None = None):
    batch_size = target.shape[0]
    current = torch.zeros(batch_size, 16, dtype=torch.float32)
    current[:, 7:14] = 1.0
    current[:, 15] = 0.1
    if absorbing is None:
        absorbing = torch.zeros(batch_size, 5, dtype=torch.bool)
    return TemporalJepaEvaluationBatch(
        indices=tuple(
            SharedJepaSampleIndex(
                level=3,
                split="validation",
                episode_id=f"episode-{index}",
                source_tick=10,
                boundary_disposition="recorded_complete",
            )
            for index in range(batch_size)
        ),
        temporal_config_id="stride4_80ms_history_160ms",
        context=LaunchContextBatch(
            vision_history=torch.zeros(batch_size, 3, 2, 196, 384, dtype=torch.float16),
            proprio_history=torch.zeros(batch_size, 3, 16, dtype=torch.float32),
            executed_controls=torch.zeros(batch_size, 2, 4, 7, dtype=torch.float32),
            executable_controls=torch.zeros(batch_size, 5, 4, 7, dtype=torch.float32),
        ),
        current_proprio_physical=current,
        target_visual_latents=torch.zeros(
            batch_size,
            5,
            2,
            196,
            384,
            dtype=torch.float16,
        ),
        target_proprio_physical=target,
        target_absorbing=absorbing,
    )


def _constant_velocity_target(batch_size: int = 1) -> torch.Tensor:
    seconds = torch.tensor([0.08, 0.16, 0.24, 0.32, 0.40], dtype=torch.float32)
    target = torch.zeros(batch_size, 5, 16, dtype=torch.float32)
    target[:, :, :7] = seconds[None, :, None]
    target[:, :, 7:14] = 1.0
    target[:, :, 14] = 0.1 * seconds[None, :]
    target[:, :, 15] = 0.1
    return target


def test_j2_uses_seconds_for_constant_velocity_and_preserves_direction() -> None:
    """Catches multiplying velocity by raw tick indices or losing displacement direction."""

    from latency_meta_mdp.belief.jepa.ar.evaluation import (
        evaluate_future_proprio_batch,
    )

    target = _constant_velocity_target()
    metrics = evaluate_future_proprio_batch(
        model=_FixedProprioRollout(target),
        batch=_batch(target=target),
    )

    np.testing.assert_allclose(metrics.qpos_rmse, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.qvel_rmse, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.gripper_width_mae, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.gripper_velocity_mae, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.constant_velocity_qpos_rmse, 0.0, atol=1e-7)
    np.testing.assert_allclose(metrics.constant_velocity_gripper_width_mae, 0.0, atol=1e-7)
    np.testing.assert_allclose(metrics.persistence_qpos_rmse[0], [0.08, 0.16, 0.24, 0.32, 0.40])
    np.testing.assert_allclose(metrics.qpos_displacement_cosine, 1.0, atol=1e-7)
    np.testing.assert_array_equal(metrics.qpos_direction_valid, True)
    np.testing.assert_array_equal(metrics.gripper_direction_correct, True)
    np.testing.assert_array_equal(metrics.gripper_direction_valid, True)


def test_j2_reports_physical_components_without_mixing_units() -> None:
    """Catches collapsing quantities with four different physical units into one RMSE."""

    from latency_meta_mdp.belief.jepa.ar.evaluation import (
        evaluate_future_proprio_batch,
    )

    target = _constant_velocity_target()
    predicted = target.clone()
    predicted[:, :, :7] += 1.0
    predicted[:, :, 7:14] += 2.0
    predicted[:, :, 14] += 3.0
    predicted[:, :, 15] += 4.0
    metrics = evaluate_future_proprio_batch(
        model=_FixedProprioRollout(predicted),
        batch=_batch(target=target),
    )

    np.testing.assert_allclose(metrics.qpos_rmse, 1.0)
    np.testing.assert_allclose(metrics.qvel_rmse, 2.0)
    np.testing.assert_allclose(metrics.gripper_width_mae, 3.0)
    np.testing.assert_allclose(metrics.gripper_velocity_mae, 4.0)
    assert not hasattr(metrics, "proprio_rmse")


def test_j2_primary_summary_excludes_absorbing_targets() -> None:
    """Catches repeated terminal states making dynamic prediction quality look artificially good."""

    from latency_meta_mdp.belief.jepa.ar.evaluation import (
        evaluate_future_proprio_batch,
        summarize_future_proprio,
    )

    target = _constant_velocity_target()
    predicted = target.clone()
    predicted[:, -1, :7] += 100.0
    absorbing = torch.tensor([[False, False, False, False, True]])
    summary = summarize_future_proprio(
        (
            evaluate_future_proprio_batch(
                model=_FixedProprioRollout(predicted),
                batch=_batch(target=target, absorbing=absorbing),
            ),
        )
    )

    assert summary["dynamic_value_count"] == 4
    assert summary["absorbing_value_count"] == 1
    assert summary["source_mean_qpos_rmse_rad"] == 0.0
    assert summary["absorbing_source_mean_qpos_rmse_rad"] == 100.0
    np.testing.assert_allclose(
        summary["per_anchor_qpos_rmse_rad"][:4],
        0.0,
        atol=0.0,
    )
    assert summary["per_anchor_qpos_rmse_rad"][4] is None


def test_j2_counts_collapsed_qpos_prediction_as_zero_direction_skill() -> None:
    """Catches excluding a zero-motion prediction when the target actually moves."""

    from latency_meta_mdp.belief.jepa.ar.evaluation import (
        evaluate_future_proprio_batch,
    )

    target = _constant_velocity_target()
    predicted = target.clone()
    predicted[:, :, :7] = 0.0
    metrics = evaluate_future_proprio_batch(
        model=_FixedProprioRollout(predicted),
        batch=_batch(target=target),
    )

    np.testing.assert_array_equal(metrics.qpos_direction_valid, True)
    np.testing.assert_allclose(metrics.qpos_displacement_cosine, 0.0, atol=0.0)


def test_j2_can_score_one_precomputed_rollout_without_calling_model_again() -> None:
    """Catches forcing J2 and J3 to execute separate AR5 predictor rollouts."""

    from latency_meta_mdp.belief.jepa.ar.evaluation import (
        evaluate_future_proprio_rollout,
    )

    target = _constant_velocity_target()
    batch = _batch(target=target)
    rollout = _FixedProprioRollout(target).rollout_native(batch.context)
    metrics = evaluate_future_proprio_rollout(rollout=rollout, batch=batch)

    np.testing.assert_allclose(metrics.qpos_rmse, 0.0, atol=0.0)
    assert metrics.native_delay_ticks == (4, 8, 12, 16, 20)
