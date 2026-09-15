"""Streaming intrinsic metrics for free-running temporal JEPA rollouts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.return_belief import (
    upper_tie_nearest_anchor_indices,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaDeployedEvaluationBatch,
    TemporalJepaEvaluationBatch,
)
from latency_meta_mdp.belief.jepa.contracts import FutureLatentRollout


@dataclass(frozen=True)
class TemporalBatchMetrics:
    native_delay_ticks: tuple[int, ...]
    latent_rmse: np.ndarray
    proprio_rmse: np.ndarray
    persistence_latent_rmse: np.ndarray
    persistence_proprio_rmse: np.ndarray
    latent_predictive_skill: np.ndarray

    @property
    def source_count(self) -> int:
        return int(self.latent_rmse.shape[0])


@dataclass(frozen=True)
class DeployedTemporalBatchMetrics:
    dense_delay_ticks: tuple[int, ...]
    assigned_native_delay_ticks: np.ndarray
    deployed_latent_rmse: np.ndarray
    deployed_proprio_rmse: np.ndarray
    temporal_quantization_latent_rmse: np.ndarray
    temporal_quantization_proprio_rmse: np.ndarray
    assigned_model_latent_rmse: np.ndarray
    assigned_model_proprio_rmse: np.ndarray

    @property
    def source_count(self) -> int:
        return int(self.deployed_latent_rmse.shape[0])


def select_monitor_indices(
    *,
    indices: tuple[SharedJepaSampleIndex, ...],
    contexts_per_episode: int,
) -> tuple[SharedJepaSampleIndex, ...]:
    if (
        type(indices) is not tuple
        or not indices
        or any(not isinstance(value, SharedJepaSampleIndex) for value in indices)
    ):
        raise ValueError("indices must contain SharedJepaSampleIndex values")
    if type(contexts_per_episode) is not int or contexts_per_episode <= 0:
        raise ValueError("contexts_per_episode must be a positive integer")
    grouped: dict[str, list[SharedJepaSampleIndex]] = defaultdict(list)
    for index in indices:
        grouped[index.episode_id].append(index)
    selected = []
    for episode_id in sorted(grouped):
        values = tuple(sorted(grouped[episode_id]))
        if len(values) <= contexts_per_episode:
            selected.extend(values)
            continue
        positions = np.linspace(
            0,
            len(values) - 1,
            num=contexts_per_episode,
            dtype=np.int64,
        )
        selected.extend(values[int(position)] for position in positions)
    return tuple(selected)


def _cpu_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(torch.float64).cpu().numpy()


@torch.no_grad()
def evaluate_temporal_batch(
    *,
    model: torch.nn.Module,
    batch: TemporalJepaEvaluationBatch,
) -> TemporalBatchMetrics:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(batch, TemporalJepaEvaluationBatch):
        raise TypeError("batch must be TemporalJepaEvaluationBatch")
    return evaluate_temporal_rollout(
        rollout=model.rollout_native(batch.context),
        batch=batch,
    )


@torch.no_grad()
def evaluate_temporal_rollout(
    *,
    rollout: FutureLatentRollout,
    batch: TemporalJepaEvaluationBatch,
) -> TemporalBatchMetrics:
    if not isinstance(rollout, FutureLatentRollout):
        raise TypeError("rollout must be FutureLatentRollout")
    if not isinstance(batch, TemporalJepaEvaluationBatch):
        raise TypeError("batch must be TemporalJepaEvaluationBatch")
    if (
        rollout.future_visual_latents.shape != batch.target_visual_latents.shape
        or rollout.future_proprio.shape != batch.target_proprio_physical.shape
    ):
        raise ValueError("rollout and evaluation targets have incompatible shapes")
    visual_error = rollout.future_visual_latents.float() - batch.target_visual_latents.float()
    proprio_error = rollout.future_proprio.float() - batch.target_proprio_physical.float()
    latent_mse = visual_error.square().mean(dim=(2, 3, 4))
    proprio_mse = proprio_error.square().mean(dim=2)
    current_visual = batch.context.vision_history[:, -1].float().unsqueeze(1)
    current_proprio = batch.current_proprio_physical.float().unsqueeze(1)
    persistence_latent_mse = (
        (current_visual - batch.target_visual_latents.float()).square().mean(dim=(2, 3, 4))
    )
    persistence_proprio_mse = (
        (current_proprio - batch.target_proprio_physical.float()).square().mean(dim=2)
    )
    skill = torch.where(
        persistence_latent_mse > 1e-12,
        1.0 - latent_mse / persistence_latent_mse,
        torch.where(
            latent_mse <= 1e-12,
            torch.zeros_like(latent_mse),
            -torch.ones_like(latent_mse),
        ),
    )
    values = (
        latent_mse,
        proprio_mse,
        persistence_latent_mse,
        persistence_proprio_mse,
        skill,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("temporal evaluation produced nonfinite metrics")
    return TemporalBatchMetrics(
        native_delay_ticks=tuple(int(value) for value in rollout.native_delay_ticks.tolist()),
        latent_rmse=_cpu_array(torch.sqrt(latent_mse)),
        proprio_rmse=_cpu_array(torch.sqrt(proprio_mse)),
        persistence_latent_rmse=_cpu_array(torch.sqrt(persistence_latent_mse)),
        persistence_proprio_rmse=_cpu_array(torch.sqrt(persistence_proprio_mse)),
        latent_predictive_skill=_cpu_array(skill),
    )


def summarize_temporal_evaluation(
    batches: tuple[TemporalBatchMetrics, ...],
) -> dict[str, Any]:
    if (
        type(batches) is not tuple
        or not batches
        or any(not isinstance(value, TemporalBatchMetrics) for value in batches)
    ):
        raise ValueError("batches must contain TemporalBatchMetrics")
    anchors = batches[0].native_delay_ticks
    if any(value.native_delay_ticks != anchors for value in batches):
        raise ValueError("evaluation batches must share native delay anchors")
    latent = np.concatenate([value.latent_rmse for value in batches], axis=0)
    proprio = np.concatenate([value.proprio_rmse for value in batches], axis=0)
    persistence_latent = np.concatenate(
        [value.persistence_latent_rmse for value in batches],
        axis=0,
    )
    persistence_proprio = np.concatenate(
        [value.persistence_proprio_rmse for value in batches],
        axis=0,
    )
    skill = np.concatenate([value.latent_predictive_skill for value in batches], axis=0)
    return {
        "source_count": int(latent.shape[0]),
        "native_delay_ticks": list(anchors),
        "source_mean_latent_rmse": float(latent.mean(axis=1).mean()),
        "source_mean_proprio_rmse": float(proprio.mean(axis=1).mean()),
        "per_anchor_latent_rmse": latent.mean(axis=0).tolist(),
        "per_anchor_proprio_rmse": proprio.mean(axis=0).tolist(),
        "per_anchor_persistence_latent_rmse": persistence_latent.mean(axis=0).tolist(),
        "per_anchor_persistence_proprio_rmse": persistence_proprio.mean(axis=0).tolist(),
        "per_anchor_latent_predictive_skill": skill.mean(axis=0).tolist(),
    }


@torch.no_grad()
def evaluate_deployed_temporal_batch(
    *,
    model: torch.nn.Module,
    batch: TemporalJepaDeployedEvaluationBatch,
) -> DeployedTemporalBatchMetrics:
    """Evaluate the deployed nearest-anchor approximation on the full D20 grid."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(batch, TemporalJepaDeployedEvaluationBatch):
        raise TypeError("batch must be TemporalJepaDeployedEvaluationBatch")
    rollout = model.rollout_native(batch.context)
    if not torch.equal(rollout.native_delay_ticks, batch.native_delay_ticks):
        raise ValueError("rollout native delays disagree with evaluation anchors")
    if batch.dense_target_visual_latents.shape[:2] != (
        batch.batch_size,
        batch.dense_delay_ticks.numel(),
    ) or batch.dense_target_proprio_physical.shape[:2] != (
        batch.batch_size,
        batch.dense_delay_ticks.numel(),
    ):
        raise ValueError("dense D20 target shapes are incompatible")
    assignment = upper_tie_nearest_anchor_indices(
        dense_delay_ticks=batch.dense_delay_ticks,
        native_delay_ticks=batch.native_delay_ticks,
    )
    native_positions = torch.searchsorted(
        batch.dense_delay_ticks,
        batch.native_delay_ticks,
    )
    if bool((native_positions >= batch.dense_delay_ticks.numel()).any()) or not torch.equal(
        batch.dense_delay_ticks[native_positions],
        batch.native_delay_ticks,
    ):
        raise ValueError("native anchors must be a subset of the dense D20 grid")

    predicted_visual = rollout.future_visual_latents[:, assignment].float()
    predicted_proprio = rollout.future_proprio[:, assignment].float()
    dense_visual = batch.dense_target_visual_latents.float()
    dense_proprio = batch.dense_target_proprio_physical.float()
    native_gt_visual = dense_visual[:, native_positions]
    native_gt_proprio = dense_proprio[:, native_positions]
    assigned_gt_visual = native_gt_visual[:, assignment]
    assigned_gt_proprio = native_gt_proprio[:, assignment]

    def visual_rmse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((left - right).square().mean(dim=(2, 3, 4)))

    def proprio_rmse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((left - right).square().mean(dim=2))

    deployed_latent = visual_rmse(predicted_visual, dense_visual)
    deployed_proprio = proprio_rmse(predicted_proprio, dense_proprio)
    quantization_latent = visual_rmse(assigned_gt_visual, dense_visual)
    quantization_proprio = proprio_rmse(assigned_gt_proprio, dense_proprio)
    assigned_model_latent = visual_rmse(predicted_visual, assigned_gt_visual)
    assigned_model_proprio = proprio_rmse(predicted_proprio, assigned_gt_proprio)
    values = (
        deployed_latent,
        deployed_proprio,
        quantization_latent,
        quantization_proprio,
        assigned_model_latent,
        assigned_model_proprio,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("deployed temporal evaluation produced nonfinite metrics")
    return DeployedTemporalBatchMetrics(
        dense_delay_ticks=tuple(int(value) for value in batch.dense_delay_ticks.tolist()),
        assigned_native_delay_ticks=_cpu_array(batch.native_delay_ticks[assignment]),
        deployed_latent_rmse=_cpu_array(deployed_latent),
        deployed_proprio_rmse=_cpu_array(deployed_proprio),
        temporal_quantization_latent_rmse=_cpu_array(quantization_latent),
        temporal_quantization_proprio_rmse=_cpu_array(quantization_proprio),
        assigned_model_latent_rmse=_cpu_array(assigned_model_latent),
        assigned_model_proprio_rmse=_cpu_array(assigned_model_proprio),
    )


def summarize_deployed_temporal_evaluation(
    batches: tuple[DeployedTemporalBatchMetrics, ...],
) -> dict[str, Any]:
    if (
        type(batches) is not tuple
        or not batches
        or any(not isinstance(value, DeployedTemporalBatchMetrics) for value in batches)
    ):
        raise ValueError("batches must contain DeployedTemporalBatchMetrics")
    delays = batches[0].dense_delay_ticks
    assignments = batches[0].assigned_native_delay_ticks
    if any(
        value.dense_delay_ticks != delays
        or not np.array_equal(value.assigned_native_delay_ticks, assignments)
        for value in batches[1:]
    ):
        raise ValueError("deployed evaluation batches must share delay grids")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([getattr(value, name) for value in batches], axis=0)

    deployed_latent = concatenate("deployed_latent_rmse")
    deployed_proprio = concatenate("deployed_proprio_rmse")
    quantization_latent = concatenate("temporal_quantization_latent_rmse")
    quantization_proprio = concatenate("temporal_quantization_proprio_rmse")
    model_latent = concatenate("assigned_model_latent_rmse")
    model_proprio = concatenate("assigned_model_proprio_rmse")
    return {
        "source_count": int(deployed_latent.shape[0]),
        "dense_delay_ticks": list(delays),
        "assigned_native_delay_ticks": assignments.astype(np.int64).tolist(),
        "source_mean_deployed_latent_rmse": float(deployed_latent.mean(axis=1).mean()),
        "source_mean_deployed_proprio_rmse": float(deployed_proprio.mean(axis=1).mean()),
        "source_mean_temporal_quantization_latent_rmse": float(
            quantization_latent.mean(axis=1).mean()
        ),
        "source_mean_temporal_quantization_proprio_rmse": float(
            quantization_proprio.mean(axis=1).mean()
        ),
        "source_mean_assigned_model_latent_rmse": float(model_latent.mean(axis=1).mean()),
        "source_mean_assigned_model_proprio_rmse": float(model_proprio.mean(axis=1).mean()),
        "per_delay_deployed_latent_rmse": deployed_latent.mean(axis=0).tolist(),
        "per_delay_deployed_proprio_rmse": deployed_proprio.mean(axis=0).tolist(),
        "per_delay_temporal_quantization_latent_rmse": quantization_latent.mean(axis=0).tolist(),
        "per_delay_temporal_quantization_proprio_rmse": quantization_proprio.mean(axis=0).tolist(),
        "per_delay_assigned_model_latent_rmse": model_latent.mean(axis=0).tolist(),
        "per_delay_assigned_model_proprio_rmse": model_proprio.mean(axis=0).tolist(),
    }
