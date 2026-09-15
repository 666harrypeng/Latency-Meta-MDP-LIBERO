"""Metrics and controls for frozen-Encoder latency-law retention probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.legacy.belief.flow.law_retention_config import (
    LawRetentionProbeConfig,
)

LAW_VARIANT_NAMES = (
    "nominal",
    "in_family",
    "shifted_fast",
    "shifted_slow",
    "shifted_wide",
)


@dataclass(frozen=True)
class LinearLawReadoutFit:
    validation_probabilities: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    best_epoch: int
    best_validation_cross_entropy: float
    epochs_completed: int
    training_history: tuple[dict[str, float | int], ...]


def _flatten_probe_values(
    *,
    tokens: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    token_array = np.asarray(tokens, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    if (
        token_array.ndim < 3
        or token_array.shape[1] != 5
        or target_array.shape != token_array.shape[:2] + (20,)
        or not np.all(np.isfinite(token_array))
        or not np.all(np.isfinite(target_array))
        or np.any(target_array < 0.0)
        or not np.allclose(target_array.sum(axis=-1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("law readout requires finite paired token and probability arrays")
    shape = target_array.shape[:2]
    return (
        token_array.reshape(shape[0] * shape[1], -1),
        target_array.reshape(shape[0] * shape[1], 20),
        shape,
    )


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-target * torch.log_softmax(logits, dim=-1), dim=-1))


def fit_linear_law_readout(
    *,
    training_tokens: np.ndarray,
    training_targets: np.ndarray,
    validation_tokens: np.ndarray,
    validation_targets: np.ndarray,
    config: LawRetentionProbeConfig,
    device: str,
    seed: int,
) -> LinearLawReadoutFit:
    if not isinstance(config, LawRetentionProbeConfig):
        raise TypeError("law readout requires a typed probe config")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("law readout seed must be non-negative")
    train_x, train_y, _train_shape = _flatten_probe_values(
        tokens=training_tokens,
        targets=training_targets,
    )
    validation_x, validation_y, validation_shape = _flatten_probe_values(
        tokens=validation_tokens,
        targets=validation_targets,
    )
    input_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    input_std = np.maximum(input_std, 1e-5)
    train_x = (train_x - input_mean) / input_std
    validation_x = (validation_x - input_mean) / input_std

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for law readout but is unavailable")
        torch.cuda.manual_seed_all(seed)
    model = nn.Linear(train_x.shape[1], 20).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_tensor = torch.from_numpy(train_x)
    train_target = torch.from_numpy(train_y)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    validation_target = torch.from_numpy(validation_y).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, config.readout_max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), generator=generator)
        total = 0.0
        for start in range(0, len(permutation), config.readout_batch_size):
            indexes = permutation[start : start + config.readout_batch_size]
            values = train_tensor[indexes].to(device)
            targets = train_target[indexes].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _soft_cross_entropy(model(values), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(indexes)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                _soft_cross_entropy(model(validation_tensor), validation_target).item()
            )
        history.append(
            {
                "epoch": epoch,
                "training_cross_entropy": total / len(train_tensor),
                "validation_cross_entropy": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.readout_patience:
                break
    if best_state is None:
        raise RuntimeError("law readout produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(validation_tensor), dim=-1).cpu().numpy()
    return LinearLawReadoutFit(
        validation_probabilities=probabilities.reshape(*validation_shape, 20),
        input_mean=input_mean,
        input_std=input_std,
        weight=best_state["weight"].numpy(),
        bias=best_state["bias"].numpy(),
        best_epoch=best_epoch,
        best_validation_cross_entropy=best_loss,
        epochs_completed=len(history),
        training_history=tuple(history),
    )


def select_evenly_spaced_offsets(*, total: int, limit: int) -> tuple[int, ...]:
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise ValueError("context selection sizes must be positive integers")
    selected_count = min(total, limit)
    if selected_count == total:
        return tuple(range(total))
    return tuple(
        int(value) for value in np.rint(np.linspace(0, total - 1, selected_count)).astype(np.int64)
    )


def law_invariant_tokens(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim < 3 or values.shape[1] != 5 or not np.all(np.isfinite(values)):
        raise ValueError("law-retention tokens must have five law variants")
    mean = values.mean(axis=1, keepdims=True)
    return np.broadcast_to(mean, values.shape).copy()


def _validate_probability_pair(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    variant_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(predicted, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if (
        variant_names != LAW_VARIANT_NAMES
        or prediction.ndim != 3
        or prediction.shape != expected.shape
        or prediction.shape[1:] != (5, 20)
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(expected))
        or np.any(prediction < 0.0)
        or np.any(expected < 0.0)
        or not np.allclose(prediction.sum(axis=-1), 1.0, atol=1e-6, rtol=0)
        or not np.allclose(expected.sum(axis=-1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("law reconstruction requires normalized [N,5,20] arrays")
    return prediction, expected


def evaluate_law_reconstruction(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    variant_names: tuple[str, ...],
) -> dict[str, float]:
    prediction, expected = _validate_probability_pair(
        predicted=predicted,
        target=target,
        variant_names=variant_names,
    )
    midpoint = 0.5 * (prediction + expected)
    epsilon = np.finfo(np.float64).tiny
    prediction_safe = np.maximum(prediction, epsilon)
    expected_safe = np.maximum(expected, epsilon)
    midpoint_safe = np.maximum(midpoint, epsilon)
    js = 0.5 * np.sum(
        prediction * (np.log(prediction_safe) - np.log(midpoint_safe)),
        axis=-1,
    ) + 0.5 * np.sum(
        expected * (np.log(expected_safe) - np.log(midpoint_safe)),
        axis=-1,
    )
    delay_seconds = np.arange(1, 21, dtype=np.float64) * 0.02
    predicted_mean = np.sum(prediction * delay_seconds, axis=-1)
    target_mean = np.sum(expected * delay_seconds, axis=-1)
    predicted_regions = prediction.reshape(*prediction.shape[:-1], 4, 5).sum(axis=-1)
    target_regions = expected.reshape(*expected.shape[:-1], 4, 5).sum(axis=-1)
    fast = variant_names.index("shifted_fast")
    nominal = variant_names.index("nominal")
    slow = variant_names.index("shifted_slow")
    ordering = (predicted_mean[:, fast] < predicted_mean[:, nominal]) & (
        predicted_mean[:, nominal] < predicted_mean[:, slow]
    )
    return {
        "jensen_shannon_divergence_mean": float(np.mean(js)),
        "probability_l1_mean": float(np.mean(np.sum(np.abs(prediction - expected), axis=-1))),
        "effective_mean_mae_ms": float(np.mean(np.abs(predicted_mean - target_mean)) * 1000.0),
        "region_mass_mae": float(np.mean(np.abs(predicted_regions - target_regions))),
        "fast_nominal_slow_ordering_accuracy": float(np.mean(ordering)),
    }
