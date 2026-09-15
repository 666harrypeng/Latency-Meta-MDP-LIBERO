from __future__ import annotations

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaDeployedEvaluationBatch,
    TemporalJepaEvaluationBatch,
)
from latency_meta_mdp.belief.jepa.contracts import (
    FutureLatentRollout,
    LaunchContextBatch,
)


class _FixedRolloutModel(torch.nn.Module):
    def __init__(self, visual: torch.Tensor, proprio: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        self.proprio = proprio

    def rollout_native(self, context: LaunchContextBatch) -> FutureLatentRollout:
        stride = context.model_stride_ticks
        return FutureLatentRollout(
            native_delay_ticks=torch.arange(stride, 21, stride, dtype=torch.int64),
            future_visual_latents=self.visual,
            future_proprio=self.proprio,
        )


def _evaluation_batch() -> TemporalJepaEvaluationBatch:
    anchors = torch.arange(1, 6, dtype=torch.float32).reshape(1, 5, 1, 1, 1)
    visual = anchors.expand(1, 5, 2, 196, 384).to(torch.float16).contiguous()
    proprio = anchors.reshape(1, 5, 1).expand(1, 5, 16).contiguous()
    return TemporalJepaEvaluationBatch(
        indices=(
            SharedJepaSampleIndex(
                level=3,
                split="train",
                episode_id="development",
                source_tick=10,
                boundary_disposition="recorded_complete",
            ),
        ),
        temporal_config_id="stride4_80ms_history_160ms",
        context=LaunchContextBatch(
            vision_history=torch.zeros(1, 3, 2, 196, 384, dtype=torch.float16),
            proprio_history=torch.zeros(1, 3, 16, dtype=torch.float32),
            executed_controls=torch.zeros(1, 2, 4, 7, dtype=torch.float32),
            executable_controls=torch.zeros(1, 5, 4, 7, dtype=torch.float32),
        ),
        current_proprio_physical=torch.zeros(1, 16, dtype=torch.float32),
        target_visual_latents=visual,
        target_proprio_physical=proprio,
        target_absorbing=torch.zeros(1, 5, dtype=torch.bool),
    )


def test_free_rollout_metrics_separate_model_error_from_persistence() -> None:
    """Catches selecting dense grids because nearby ground-truth latents are easier to copy."""

    from latency_meta_mdp.belief.jepa.ar.metrics import (
        evaluate_temporal_batch,
    )

    batch = _evaluation_batch()
    model = _FixedRolloutModel(
        visual=(batch.target_visual_latents.float() + 0.5).to(torch.float16),
        proprio=batch.target_proprio_physical + 0.5,
    )
    metrics = evaluate_temporal_batch(model=model, batch=batch)

    np.testing.assert_allclose(metrics.latent_rmse, 0.5, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(metrics.proprio_rmse, 0.5, atol=0.0, rtol=0.0)
    expected_persistence = np.arange(1, 6)[None, :]
    np.testing.assert_allclose(metrics.persistence_latent_rmse, expected_persistence, atol=0.0)
    np.testing.assert_allclose(metrics.persistence_proprio_rmse, expected_persistence, atol=0.0)
    np.testing.assert_allclose(
        metrics.latent_predictive_skill,
        1.0 - 0.25 / np.square(expected_persistence),
        atol=1e-7,
    )
    assert metrics.native_delay_ticks == (4, 8, 12, 16, 20)
    assert metrics.source_count == 1


def test_intrinsic_metrics_can_score_one_precomputed_rollout() -> None:
    """Catches requiring a second AR5 call when final qualification also computes J2/J3."""

    from latency_meta_mdp.belief.jepa.ar.metrics import (
        evaluate_temporal_rollout,
    )

    batch = _evaluation_batch()
    rollout = _FixedRolloutModel(
        visual=batch.target_visual_latents,
        proprio=batch.target_proprio_physical,
    ).rollout_native(batch.context)
    metrics = evaluate_temporal_rollout(rollout=rollout, batch=batch)

    np.testing.assert_allclose(metrics.latent_rmse, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.proprio_rmse, 0.0, atol=0.0)


def test_temporal_evaluation_summary_aggregates_sources_before_anchors() -> None:
    """Catches giving a denser temporal grid more aggregate statistical weight."""

    from latency_meta_mdp.belief.jepa.ar.metrics import (
        evaluate_temporal_batch,
        summarize_temporal_evaluation,
    )

    batch = _evaluation_batch()
    model = _FixedRolloutModel(
        visual=(batch.target_visual_latents.float() + 0.5).to(torch.float16),
        proprio=batch.target_proprio_physical + 0.5,
    )
    metrics = evaluate_temporal_batch(model=model, batch=batch)
    summary = summarize_temporal_evaluation((metrics, metrics))

    assert summary["source_count"] == 2
    assert summary["native_delay_ticks"] == [4, 8, 12, 16, 20]
    assert summary["source_mean_latent_rmse"] == 0.5
    assert summary["source_mean_proprio_rmse"] == 0.5


def test_monitor_index_selection_balances_every_development_episode() -> None:
    """Catches monitoring long episodes more heavily than short episodes."""

    from latency_meta_mdp.belief.jepa.ar.metrics import (
        select_monitor_indices,
    )

    indices = tuple(
        SharedJepaSampleIndex(
            level=3,
            split="train",
            episode_id=episode_id,
            source_tick=tick,
            boundary_disposition="recorded_complete",
        )
        for episode_id, stop in (("short", 6), ("long", 10))
        for tick in range(stop)
    )
    selected = select_monitor_indices(indices=indices, contexts_per_episode=4)

    assert len(selected) == 8
    assert sum(value.episode_id == "short" for value in selected) == 4
    assert sum(value.episode_id == "long" for value in selected) == 4
    assert selected == select_monitor_indices(indices=indices, contexts_per_episode=4)


def test_deployed_d20_metrics_separate_temporal_quantization_from_model_error() -> None:
    """Catches attributing nearest-anchor approximation error to the JEPA predictor."""

    from latency_meta_mdp.belief.jepa.ar.metrics import (
        evaluate_deployed_temporal_batch,
        summarize_deployed_temporal_evaluation,
    )

    native_delays = torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64)
    anchors = native_delays.to(torch.float32).reshape(1, 5, 1, 1, 1)
    native_visual = anchors.expand(1, 5, 2, 196, 384).to(torch.float16).contiguous()
    native_proprio = anchors.reshape(1, 5, 1).expand(1, 5, 16).contiguous()
    dense = torch.arange(1, 21, dtype=torch.float32).reshape(1, 20, 1, 1, 1)
    dense_visual = dense.expand(1, 20, 2, 196, 384).to(torch.float16).contiguous()
    dense_proprio = dense.reshape(1, 20, 1).expand(1, 20, 16).contiguous()
    batch = TemporalJepaDeployedEvaluationBatch(
        indices=_evaluation_batch().indices,
        temporal_config_id="stride4_80ms_history_160ms",
        context=_evaluation_batch().context,
        native_delay_ticks=native_delays,
        dense_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
        current_proprio_physical=torch.zeros(1, 16, dtype=torch.float32),
        dense_target_visual_latents=dense_visual,
        dense_target_proprio_physical=dense_proprio,
        dense_target_absorbing=torch.zeros(1, 20, dtype=torch.bool),
    )
    model = _FixedRolloutModel(visual=native_visual, proprio=native_proprio)

    metrics = evaluate_deployed_temporal_batch(model=model, batch=batch)
    expected_assignment = np.array(
        [4, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16, 20, 20, 20]
    )
    expected_error = np.abs(expected_assignment - np.arange(1, 21))[None, :]
    np.testing.assert_array_equal(metrics.assigned_native_delay_ticks, expected_assignment)
    np.testing.assert_allclose(metrics.temporal_quantization_latent_rmse, expected_error)
    np.testing.assert_allclose(metrics.deployed_latent_rmse, expected_error)
    np.testing.assert_allclose(metrics.assigned_model_latent_rmse, 0.0, atol=0.0)
    np.testing.assert_allclose(metrics.temporal_quantization_proprio_rmse, expected_error)
    np.testing.assert_allclose(metrics.deployed_proprio_rmse, expected_error)
    summary = summarize_deployed_temporal_evaluation((metrics, metrics))
    assert summary["source_count"] == 2
    assert summary["dense_delay_ticks"] == list(range(1, 21))
    np.testing.assert_allclose(summary["per_delay_deployed_latent_rmse"], expected_error[0])
    np.testing.assert_allclose(
        summary["per_delay_temporal_quantization_latent_rmse"],
        expected_error[0],
    )
