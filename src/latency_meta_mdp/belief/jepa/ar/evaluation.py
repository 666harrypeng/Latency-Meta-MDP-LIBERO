"""Decision-relevant physical metrics for admitted JEPA future-proprio rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaEvaluationBatch,
)
from latency_meta_mdp.belief.jepa.contracts import FutureLatentRollout

_FORMAL_TICK_SECONDS = 0.02


def _cpu_float(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(torch.float64).cpu().numpy()


def _cpu_bool(value: torch.Tensor) -> np.ndarray:
    return value.detach().to(torch.bool).cpu().numpy()


@dataclass(frozen=True)
class FutureProprioBatchMetrics:
    indices: tuple[SharedJepaSampleIndex, ...]
    native_delay_ticks: tuple[int, ...]
    dynamic_mask: np.ndarray
    absorbing_mask: np.ndarray
    qpos_rmse: np.ndarray
    qvel_rmse: np.ndarray
    gripper_width_mae: np.ndarray
    gripper_velocity_mae: np.ndarray
    persistence_qpos_rmse: np.ndarray
    persistence_qvel_rmse: np.ndarray
    persistence_gripper_width_mae: np.ndarray
    persistence_gripper_velocity_mae: np.ndarray
    constant_velocity_qpos_rmse: np.ndarray
    constant_velocity_qvel_rmse: np.ndarray
    constant_velocity_gripper_width_mae: np.ndarray
    constant_velocity_gripper_velocity_mae: np.ndarray
    qpos_displacement_cosine: np.ndarray
    qpos_direction_valid: np.ndarray
    gripper_direction_correct: np.ndarray
    gripper_direction_valid: np.ndarray


@torch.no_grad()
def evaluate_future_proprio_batch(
    *,
    model: torch.nn.Module,
    batch: TemporalJepaEvaluationBatch,
) -> FutureProprioBatchMetrics:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(batch, TemporalJepaEvaluationBatch):
        raise TypeError("batch must be TemporalJepaEvaluationBatch")
    return evaluate_future_proprio_rollout(
        rollout=model.rollout_native(batch.context),
        batch=batch,
    )


@torch.no_grad()
def evaluate_future_proprio_rollout(
    *,
    rollout: FutureLatentRollout,
    batch: TemporalJepaEvaluationBatch,
) -> FutureProprioBatchMetrics:
    if not isinstance(rollout, FutureLatentRollout):
        raise TypeError("rollout must be FutureLatentRollout")
    if not isinstance(batch, TemporalJepaEvaluationBatch):
        raise TypeError("batch must be TemporalJepaEvaluationBatch")
    if batch.temporal_config_id != "stride4_80ms_history_160ms":
        raise ValueError("L3 admission proprio evaluation requires stride-4")
    if (
        tuple(rollout.native_delay_ticks.tolist()) != (4, 8, 12, 16, 20)
        or rollout.future_proprio.shape != batch.target_proprio_physical.shape
    ):
        raise ValueError("rollout and stride-4 proprio targets are incompatible")
    predicted = rollout.future_proprio.float()
    target = batch.target_proprio_physical.float()
    current = batch.current_proprio_physical.float().unsqueeze(1).expand_as(target)
    delays_seconds = (
        rollout.native_delay_ticks.to(
            device=target.device,
            dtype=torch.float32,
        )
        * _FORMAL_TICK_SECONDS
    )
    constant_velocity = current.clone()
    constant_velocity[:, :, :7] = (
        current[:, :, :7] + delays_seconds[None, :, None] * current[:, :, 7:14]
    )
    constant_velocity[:, :, 14] = current[:, :, 14] + delays_seconds[None, :] * current[:, :, 15]

    def rmse(left: torch.Tensor, right: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return torch.sqrt((left[:, :, start:stop] - right[:, :, start:stop]).square().mean(2))

    def absolute(left: torch.Tensor, right: torch.Tensor, index: int) -> torch.Tensor:
        return torch.abs(left[:, :, index] - right[:, :, index])

    predicted_delta = predicted[:, :, :7] - current[:, :, :7]
    target_delta = target[:, :, :7] - current[:, :, :7]
    predicted_norm = torch.linalg.vector_norm(predicted_delta, dim=2)
    target_norm = torch.linalg.vector_norm(target_delta, dim=2)
    qpos_direction_valid = target_norm > 1e-8
    direction = torch.zeros_like(predicted_norm)
    direction[qpos_direction_valid] = (predicted_delta * target_delta).sum(2)[
        qpos_direction_valid
    ] / (predicted_norm * target_norm).clamp_min(1e-8)[qpos_direction_valid]
    predicted_gripper_delta = predicted[:, :, 14] - current[:, :, 14]
    target_gripper_delta = target[:, :, 14] - current[:, :, 14]
    gripper_direction_valid = torch.abs(target_gripper_delta) > 1e-8
    gripper_direction_correct = gripper_direction_valid & (
        torch.sign(predicted_gripper_delta) == torch.sign(target_gripper_delta)
    )

    float_values = {
        "qpos_rmse": rmse(predicted, target, 0, 7),
        "qvel_rmse": rmse(predicted, target, 7, 14),
        "gripper_width_mae": absolute(predicted, target, 14),
        "gripper_velocity_mae": absolute(predicted, target, 15),
        "persistence_qpos_rmse": rmse(current, target, 0, 7),
        "persistence_qvel_rmse": rmse(current, target, 7, 14),
        "persistence_gripper_width_mae": absolute(current, target, 14),
        "persistence_gripper_velocity_mae": absolute(current, target, 15),
        "constant_velocity_qpos_rmse": rmse(constant_velocity, target, 0, 7),
        "constant_velocity_qvel_rmse": rmse(constant_velocity, target, 7, 14),
        "constant_velocity_gripper_width_mae": absolute(constant_velocity, target, 14),
        "constant_velocity_gripper_velocity_mae": absolute(constant_velocity, target, 15),
        "qpos_displacement_cosine": direction,
    }
    if not all(bool(torch.isfinite(value).all()) for value in float_values.values()):
        raise FloatingPointError("future proprio evaluation produced nonfinite values")
    return FutureProprioBatchMetrics(
        indices=batch.indices,
        native_delay_ticks=(4, 8, 12, 16, 20),
        dynamic_mask=_cpu_bool(~batch.target_absorbing),
        absorbing_mask=_cpu_bool(batch.target_absorbing),
        **{name: _cpu_float(value) for name, value in float_values.items()},
        qpos_direction_valid=_cpu_bool(qpos_direction_valid),
        gripper_direction_correct=_cpu_bool(gripper_direction_correct),
        gripper_direction_valid=_cpu_bool(gripper_direction_valid),
    )


def _source_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    source_values = []
    for row, selected in zip(values, mask, strict=True):
        if np.any(selected):
            source_values.append(float(np.mean(row[selected])))
    return None if not source_values else float(np.mean(source_values))


def _per_anchor(values: np.ndarray, mask: np.ndarray) -> list[float | None]:
    return [
        None if not np.any(mask[:, anchor]) else float(np.mean(values[:, anchor][mask[:, anchor]]))
        for anchor in range(values.shape[1])
    ]


def summarize_future_proprio(
    batches: tuple[FutureProprioBatchMetrics, ...],
) -> dict[str, Any]:
    if (
        type(batches) is not tuple
        or not batches
        or any(not isinstance(value, FutureProprioBatchMetrics) for value in batches)
    ):
        raise ValueError("batches must contain FutureProprioBatchMetrics")
    delays = batches[0].native_delay_ticks
    if any(value.native_delay_ticks != delays for value in batches[1:]):
        raise ValueError("future proprio batches must share native delays")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([getattr(value, name) for value in batches], axis=0)

    dynamic = concatenate("dynamic_mask")
    absorbing = concatenate("absorbing_mask")
    metrics = {
        "qpos_rmse_rad": concatenate("qpos_rmse"),
        "qvel_rmse_rad_s": concatenate("qvel_rmse"),
        "gripper_width_mae_m": concatenate("gripper_width_mae"),
        "gripper_velocity_mae_m_s": concatenate("gripper_velocity_mae"),
        "persistence_qpos_rmse_rad": concatenate("persistence_qpos_rmse"),
        "persistence_qvel_rmse_rad_s": concatenate("persistence_qvel_rmse"),
        "persistence_gripper_width_mae_m": concatenate("persistence_gripper_width_mae"),
        "persistence_gripper_velocity_mae_m_s": concatenate("persistence_gripper_velocity_mae"),
        "constant_velocity_qpos_rmse_rad": concatenate("constant_velocity_qpos_rmse"),
        "constant_velocity_qvel_rmse_rad_s": concatenate("constant_velocity_qvel_rmse"),
        "constant_velocity_gripper_width_mae_m": concatenate("constant_velocity_gripper_width_mae"),
        "constant_velocity_gripper_velocity_mae_m_s": concatenate(
            "constant_velocity_gripper_velocity_mae"
        ),
    }
    summary: dict[str, Any] = {
        "source_count": int(dynamic.shape[0]),
        "native_delay_ticks": list(delays),
        "native_delay_ms": [tick * 20 for tick in delays],
        "dynamic_value_count": int(dynamic.sum()),
        "absorbing_value_count": int(absorbing.sum()),
    }
    for name, values in metrics.items():
        summary[f"source_mean_{name}"] = _source_mean(values, dynamic)
        summary[f"per_anchor_{name}"] = _per_anchor(values, dynamic)
        summary[f"absorbing_source_mean_{name}"] = _source_mean(values, absorbing)
    direction = concatenate("qpos_displacement_cosine")
    direction_mask = dynamic & concatenate("qpos_direction_valid")
    gripper_correct = concatenate("gripper_direction_correct").astype(np.float64)
    gripper_mask = dynamic & concatenate("gripper_direction_valid")
    summary["source_mean_qpos_displacement_cosine"] = _source_mean(
        direction,
        direction_mask,
    )
    summary["per_anchor_qpos_displacement_cosine"] = _per_anchor(
        direction,
        direction_mask,
    )
    summary["source_mean_gripper_direction_accuracy"] = _source_mean(
        gripper_correct,
        gripper_mask,
    )
    summary["per_anchor_gripper_direction_accuracy"] = _per_anchor(
        gripper_correct,
        gripper_mask,
    )
    return summary
