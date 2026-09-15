"""Checkpoint-independent stride-4 Belief qualification composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.evaluation import (
    FutureProprioBatchMetrics,
    evaluate_future_proprio_rollout,
    summarize_future_proprio,
)
from latency_meta_mdp.belief.jepa.ar.metrics import (
    TemporalBatchMetrics,
    evaluate_temporal_rollout,
    summarize_temporal_evaluation,
)
from latency_meta_mdp.belief.jepa.ar.return_belief import (
    assemble_return_latent_belief,
    weighted_future_proprio,
    weighted_future_visual_latents,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaEvaluationBatch,
)
from latency_meta_mdp.belief.jepa.config import JepaTemporalSampling
from latency_meta_mdp.belief.jepa.contracts import FutureLatentRollout
from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
    TemporalSignalEpisode,
)
from latency_meta_mdp.belief.jepa.diagnostics.readout import (
    ObjectStateBatchMetrics,
    ObjectStateNormalization,
    evaluate_object_state_latents,
    gather_object_state_targets,
    summarize_object_state_metrics,
)


@dataclass(frozen=True)
class Stride4QualificationBatch:
    future_latent: TemporalBatchMetrics
    future_proprio: FutureProprioBatchMetrics
    gt_latent_ceiling: ObjectStateBatchMetrics
    predicted_latent: ObjectStateBatchMetrics


def aggregate_metric_by_master(
    *,
    values: np.ndarray,
    valid: np.ndarray,
    indices: tuple[SharedJepaSampleIndex, ...],
    signal_episodes: Mapping[str, TemporalSignalEpisode],
) -> list[dict[str, float | int]]:
    metric = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=np.bool_)
    if (
        metric.ndim != 2
        or metric.shape != mask.shape
        or metric.shape[0] != len(indices)
        or not np.all(np.isfinite(metric))
        or any(not isinstance(value, SharedJepaSampleIndex) for value in indices)
        or set(value.episode_id for value in indices) - set(signal_episodes)
    ):
        raise ValueError("master metric aggregation inputs are invalid")
    contexts_by_episode: dict[str, list[float]] = {}
    for row, index in enumerate(indices):
        selected = metric[row][mask[row]]
        if selected.size:
            contexts_by_episode.setdefault(index.episode_id, []).append(float(selected.mean()))
    episodes_by_master: dict[int, list[float]] = {}
    for episode_id, context_values in contexts_by_episode.items():
        master = signal_episodes[episode_id].record.logical_master_task_index
        episodes_by_master.setdefault(master, []).append(float(np.mean(context_values)))
    return [
        {
            "master_task_index": master,
            "value": float(np.mean(episodes_by_master[master])),
        }
        for master in sorted(episodes_by_master)
    ]


@torch.no_grad()
def evaluate_latency_mixture_invariants(
    *,
    rollout: FutureLatentRollout,
    probabilities_by_law: Mapping[str, torch.Tensor],
    sampling: JepaTemporalSampling,
) -> dict[str, object]:
    if not isinstance(rollout, FutureLatentRollout):
        raise TypeError("rollout must be FutureLatentRollout")
    if not isinstance(probabilities_by_law, Mapping) or not probabilities_by_law:
        raise ValueError("probabilities_by_law cannot be empty")
    names = sorted(probabilities_by_law)
    expected_delay: dict[str, list[float]] = {}
    maximum_mass_error = 0.0
    for name in names:
        if type(name) is not str or not name:
            raise ValueError("latency law names cannot be empty")
        belief = assemble_return_latent_belief(
            rollout,
            probabilities_by_law[name],
            sampling=sampling,
        )
        if (
            belief.future_visual_latents is not rollout.future_visual_latents
            or belief.future_proprio is not rollout.future_proprio
        ):
            raise RuntimeError("latency assembly changed fixed-delay futures")
        direct_proprio = torch.sum(
            belief.delay_probabilities[..., None] * belief.future_proprio,
            dim=1,
        )
        direct_visual = torch.sum(
            belief.delay_probabilities[..., None, None, None] * belief.future_visual_latents,
            dim=1,
        )
        if not torch.allclose(
            weighted_future_proprio(belief), direct_proprio
        ) or not torch.allclose(weighted_future_visual_latents(belief), direct_visual):
            raise RuntimeError("latency-weighted summary disagrees with direct aggregation")
        mass_error = torch.abs(belief.delay_probabilities.sum(dim=1) - 1.0)
        maximum_mass_error = max(maximum_mass_error, float(mass_error.max()))
        expected_delay[name] = (
            torch.sum(
                belief.delay_probabilities * belief.delay_ticks.to(torch.float32)[None],
                dim=1,
            )
            .cpu()
            .tolist()
        )
    return {
        "law_names": names,
        "fixed_future_tensors_shared": True,
        "maximum_probability_mass_error": maximum_mass_error,
        "expected_delay_ticks": expected_delay,
    }


@torch.no_grad()
def evaluate_stride4_qualification_batch(
    *,
    model: torch.nn.Module,
    readout: torch.nn.Module,
    object_normalization: ObjectStateNormalization,
    batch: TemporalJepaEvaluationBatch,
    signal_episodes: Mapping[str, TemporalSignalEpisode],
) -> Stride4QualificationBatch:
    if not isinstance(model, torch.nn.Module) or not isinstance(readout, torch.nn.Module):
        raise TypeError("model and readout must be torch modules")
    if not isinstance(batch, TemporalJepaEvaluationBatch):
        raise TypeError("batch must be TemporalJepaEvaluationBatch")
    rollout = model.rollout_native(batch.context)
    target_object_state = gather_object_state_targets(
        indices=batch.indices,
        native_delay_ticks=(4, 8, 12, 16, 20),
        episodes=signal_episodes,
    ).to(batch.context.device)
    common = {
        "readout": readout,
        "normalization": object_normalization,
        "target_object_state": target_object_state,
        "absorbing": batch.target_absorbing,
    }
    return Stride4QualificationBatch(
        future_latent=evaluate_temporal_rollout(
            rollout=rollout,
            batch=batch,
        ),
        future_proprio=evaluate_future_proprio_rollout(
            rollout=rollout,
            batch=batch,
        ),
        gt_latent_ceiling=evaluate_object_state_latents(
            visual_latents=batch.target_visual_latents,
            **common,
        ),
        predicted_latent=evaluate_object_state_latents(
            visual_latents=rollout.future_visual_latents,
            **common,
        ),
    )


def summarize_stride4_qualification(
    batches: tuple[Stride4QualificationBatch, ...],
    *,
    signal_episodes: Mapping[str, TemporalSignalEpisode],
) -> dict[str, object]:
    if (
        type(batches) is not tuple
        or not batches
        or any(not isinstance(value, Stride4QualificationBatch) for value in batches)
    ):
        raise ValueError("batches must contain Stride4QualificationBatch")
    indices = tuple(index for batch in batches for index in batch.future_proprio.indices)

    def concatenate(group: str, name: str) -> np.ndarray:
        return np.concatenate(
            tuple(getattr(getattr(batch, group), name) for batch in batches),
            axis=0,
        )

    dynamic = concatenate("future_proprio", "dynamic_mask")
    per_master = {
        "j1_latent_rmse": aggregate_metric_by_master(
            values=concatenate("future_latent", "latent_rmse"),
            valid=dynamic,
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j1_latent_predictive_skill": aggregate_metric_by_master(
            values=concatenate("future_latent", "latent_predictive_skill"),
            valid=dynamic,
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j2_qpos_rmse_rad": aggregate_metric_by_master(
            values=concatenate("future_proprio", "qpos_rmse"),
            valid=dynamic,
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j2_qvel_rmse_rad_s": aggregate_metric_by_master(
            values=concatenate("future_proprio", "qvel_rmse"),
            valid=dynamic,
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j2_gripper_width_mae_m": aggregate_metric_by_master(
            values=concatenate("future_proprio", "gripper_width_mae"),
            valid=dynamic,
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j2_qpos_displacement_cosine": aggregate_metric_by_master(
            values=concatenate("future_proprio", "qpos_displacement_cosine"),
            valid=dynamic & concatenate("future_proprio", "qpos_direction_valid"),
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j3_gt_ceiling_position_rmse_m": aggregate_metric_by_master(
            values=concatenate("gt_latent_ceiling", "position_rmse_m"),
            valid=concatenate("gt_latent_ceiling", "dynamic_mask"),
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j3_predicted_position_rmse_m": aggregate_metric_by_master(
            values=concatenate("predicted_latent", "position_rmse_m"),
            valid=concatenate("predicted_latent", "dynamic_mask"),
            indices=indices,
            signal_episodes=signal_episodes,
        ),
        "j3_predicted_velocity_rmse_m_s": aggregate_metric_by_master(
            values=concatenate("predicted_latent", "velocity_rmse_m_s"),
            valid=concatenate("predicted_latent", "dynamic_mask"),
            indices=indices,
            signal_episodes=signal_episodes,
        ),
    }
    return {
        "j1_future_latent": summarize_temporal_evaluation(
            tuple(value.future_latent for value in batches)
        ),
        "j2_future_proprio": summarize_future_proprio(
            tuple(value.future_proprio for value in batches)
        ),
        "j3_gt_latent_ceiling": summarize_object_state_metrics(
            tuple(value.gt_latent_ceiling for value in batches)
        ),
        "j3_predicted_latent": summarize_object_state_metrics(
            tuple(value.predicted_latent for value in batches)
        ),
        "per_master": per_master,
    }
