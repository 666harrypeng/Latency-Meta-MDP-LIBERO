"""Matched native endpoints and coordinate metrics for direct versus AR evaluation."""

from __future__ import annotations

import numpy as np
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import (
    ForecastQuery,
    FutureLatentPrediction,
    LaunchContextBatch,
)
from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
    DirectPredictionSample,
)
from latency_meta_mdp.belief.action_conditioned_jepa.rollout import ActionConditionedJepaPredictor


@torch.no_grad()
def predict_legacy_endpoint(
    model: ActionConditionedJepaPredictor, query: ForecastQuery
) -> FutureLatentPrediction:
    """Stop after q/4 AR calls; never compute unrequested later endpoints.

    As with DirectJepaPredictor.predict_at, the caller owns autocast and timing.
    Only equal, native q within a batch is accepted; no off-grid interpolation.
    """
    if (model.config.history_ticks, model.config.model_stride_ticks) != (3, 4):
        raise ValueError("comparison requires the admitted stride4 native model")
    q = int(query.query_ticks[0])
    if q not in (4, 8, 12, 16, 20) or not torch.all(query.query_ticks == q):
        raise ValueError("legacy comparison requires a single native query per batch")
    context = LaunchContextBatch(
        vision_history=query.vision_history,
        proprio_history=query.proprio_history,
        executed_controls=query.executed_controls,
        executable_controls=query.executable_controls.masked_fill(
            ~query.control_mask[..., None], 0
        ).reshape(-1, 5, 4, 7),
    )
    visual, proprio = model.rollout_endpoint_for_loss(context, horizon=q // 4)
    return FutureLatentPrediction(
        source_ticks=query.source_ticks.clone(),
        target_ticks=query.source_ticks + query.query_ticks,
        visual_latents=visual.half(),
        proprio=proprio.float() * model.proprio_scale + model.proprio_mean,
    )


def prediction_mse_by_example(
    prediction: FutureLatentPrediction, sample: DirectPredictionSample
) -> dict[str, np.ndarray]:
    """Return per-example MSE; aggregate before taking sqrt for coordinate RMSE.

    Keep master/episode grouping outside this function, so frames are not treated
    as independent task instances when computing uncertainty.
    """
    if not torch.equal(
        prediction.source_ticks.cpu(), sample.query.source_ticks.cpu()
    ) or not torch.equal(
        prediction.target_ticks.cpu(), (sample.query.source_ticks + sample.query.query_ticks).cpu()
    ):
        raise ValueError("prediction and target timestamps are not paired")
    if (
        prediction.visual_latents.shape != sample.target_visual.shape
        or prediction.proprio.shape != sample.target_proprio.shape
    ):
        raise ValueError("prediction and target shapes are not paired")
    visual = prediction.visual_latents.detach().cpu().double().numpy()
    target_visual = sample.target_visual.detach().cpu().double().numpy()
    proprio = prediction.proprio.detach().cpu().double().numpy()
    target_proprio = sample.target_proprio.detach().cpu().double().numpy()
    if not all(np.isfinite(v).all() for v in (visual, target_visual, proprio, target_proprio)):
        raise ValueError("paired prediction metrics require finite values")
    squared = np.square(proprio - target_proprio)
    return {
        "visual": np.square(visual - target_visual).mean(axis=(1, 2, 3)),
        "qpos": squared[:, :7].mean(axis=1),
        "qvel": squared[:, 7:14].mean(axis=1),
        "gripper_width": squared[:, 14],
        "gripper_width_velocity": squared[:, 15],
    }


@torch.no_grad()
def copy_current_prediction(
    query: ForecastQuery, *, proprio_mean: torch.Tensor, proprio_scale: torch.Tensor
) -> FutureLatentPrediction:
    """Diagnostic persistence baseline, without claiming physical hold dynamics."""
    return FutureLatentPrediction(
        source_ticks=query.source_ticks.clone(),
        target_ticks=query.source_ticks + query.query_ticks,
        visual_latents=query.vision_history[:, -1].clone(),
        proprio=(query.proprio_history[:, -1] * proprio_scale + proprio_mean).float(),
    )
