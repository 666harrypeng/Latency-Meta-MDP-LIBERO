"""Sample-based evaluation over cached frozen Flow belief tokens."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.fresh_decoder_probe import CachedBeliefSplit
from latency_meta_mdp.belief.flow.metrics import (
    sample_distribution_metrics,
    sample_mean_physical_metrics,
)
from latency_meta_mdp.belief.flow.sampler import sample_flow_belief
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.belief_training_data import InteractionMode


def _metric_bundle(
    *,
    samples: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    normalization: FlowBeliefNormalization,
) -> dict[str, Any]:
    return {
        "distribution": sample_distribution_metrics(
            samples=samples,
            target=targets,
            weights=weights,
        ),
        "physical": sample_mean_physical_metrics(
            normalized_samples=samples,
            normalized_target=targets,
            weights=weights,
            target_mean=normalization.target_mean,
            target_std=normalization.target_std,
        ),
    }


def _subset_metric_bundle(
    *,
    samples: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    normalization: FlowBeliefNormalization,
) -> dict[str, Any] | None:
    masked = weights * mask
    mass = masked.sum(axis=1)
    valid = mass > 0.0
    if not np.any(valid):
        return None
    normalized_weights = masked[valid] / mass[valid, None]
    result = _metric_bundle(
        samples=samples[valid],
        targets=targets[valid],
        weights=normalized_weights,
        normalization=normalization,
    )
    result["retained_probability_mass_mean"] = float(np.mean(mass[valid]))
    return result


def evaluate_cached_vector_field(
    *,
    vector_field: nn.Module,
    cached: CachedBeliefSplit,
    normalization: FlowBeliefNormalization,
    config: FlowBeliefConfig,
    level: int,
    device: str,
) -> dict[str, Any]:
    if len(cached.belief_tokens) == 0:
        raise ValueError("fresh Decoder evaluation requires cached contexts")
    if level not in (1, 2, 3):
        raise ValueError("fresh Decoder evaluation level must be 1, 2, or 3")
    vector_field = vector_field.to(device)
    vector_field.eval()
    delay_ticks = np.arange(1, 21, dtype=np.int64)
    sample_batches = []
    sampling_seconds = 0.0
    with torch.inference_mode():
        for start in range(0, len(cached.belief_tokens), config.batch_size):
            stop = min(start + config.batch_size, len(cached.belief_tokens))
            noises = []
            for offset in range(start, stop):
                rng = np.random.default_rng(config.evaluation_seed + level * 1_000_000 + offset)
                noises.append(
                    rng.standard_normal(
                        (20, config.evaluation_sample_count, 22),
                        dtype=np.float32,
                    )
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            sampled = sample_flow_belief(
                vector_field=vector_field,
                belief_tokens=torch.from_numpy(
                    np.array(cached.belief_tokens[start:stop], copy=True)
                ).to(device),
                delay_ticks=torch.from_numpy(
                    np.broadcast_to(delay_ticks, (stop - start, 20)).copy()
                ).to(device),
                noise=torch.from_numpy(np.stack(noises)).to(device),
                solver=config.solver,
                step_count=config.solver_step_count,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            sampling_seconds += time.perf_counter() - started
            sample_batches.append(sampled.cpu().numpy())
    samples = np.concatenate(sample_batches, axis=0)
    targets = np.asarray(cached.target_states, dtype=np.float32)
    weights = np.asarray(cached.latency_probabilities, dtype=np.float64)
    metrics: dict[str, Any] = {
        "overall": _metric_bundle(
            samples=samples,
            targets=targets,
            weights=weights,
            normalization=normalization,
        ),
        "per_delay": {},
        "sampling": {
            "wall_seconds": sampling_seconds,
            "seconds_per_context": sampling_seconds / len(cached.belief_tokens),
            "sample_count": config.evaluation_sample_count,
            "solver": config.solver,
            "solver_step_count": config.solver_step_count,
        },
    }
    for delay_index, delay in enumerate(delay_ticks):
        metrics["per_delay"][str(int(delay))] = _metric_bundle(
            samples=samples[:, delay_index : delay_index + 1],
            targets=targets[:, delay_index : delay_index + 1],
            weights=np.ones((len(samples), 1), dtype=np.float64),
            normalization=normalization,
        )
    masks = {
        "pre_handoff": (
            ~cached.absorbing
            & np.isin(
                cached.interaction_mode,
                (InteractionMode.FREE, InteractionMode.CONTACT),
            )
        ),
        "post_handoff": (~cached.absorbing & (cached.interaction_mode == InteractionMode.GRASPED)),
        "absorbing": cached.absorbing,
    }
    for name, mask in masks.items():
        bundle = _subset_metric_bundle(
            samples=samples,
            targets=targets,
            weights=weights,
            mask=mask,
            normalization=normalization,
        )
        if bundle is not None:
            metrics[name] = bundle
    return metrics
