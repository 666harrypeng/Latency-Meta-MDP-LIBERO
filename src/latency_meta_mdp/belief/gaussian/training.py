"""Training utilities for the per-level Gaussian return-belief baseline."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.gaussian.config import GaussianBeliefConfig
from latency_meta_mdp.belief.gaussian.metrics import (
    gaussian_prediction_metrics,
    weighted_normalized_gaussian_nll,
)
from latency_meta_mdp.belief.gaussian.model import (
    GaussianBeliefModel,
    diagonal_gaussian_nll,
)
from latency_meta_mdp.belief.gaussian.training_data import (
    GaussianBeliefDataset,
    GaussianBeliefItem,
    GaussianBeliefNormalization,
    build_gaussian_belief_normalization,
)
from latency_meta_mdp.belief_training_data import InteractionMode
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class GaussianBeliefBatch:
    vision_history: torch.Tensor
    proprio_history: torch.Tensor
    remaining_actions: torch.Tensor
    latency_probabilities: torch.Tensor
    delay_ticks: torch.Tensor
    query_probabilities: torch.Tensor
    target_states: torch.Tensor
    interaction_mode: torch.Tensor
    absorbing: torch.Tensor


def _stack(items: list[GaussianBeliefItem], name: str) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([np.asarray(getattr(item, name)) for item in items], axis=0)
    )


def collate_gaussian_belief_items(
    items: list[GaussianBeliefItem],
) -> GaussianBeliefBatch:
    if not items:
        raise ValueError("cannot collate an empty Gaussian belief batch")
    return GaussianBeliefBatch(
        vision_history=_stack(items, "vision_history"),
        proprio_history=_stack(items, "proprio_history"),
        remaining_actions=_stack(items, "remaining_actions"),
        latency_probabilities=_stack(items, "latency_probabilities"),
        delay_ticks=_stack(items, "delay_ticks"),
        query_probabilities=_stack(items, "query_probabilities"),
        target_states=_stack(items, "target_states"),
        interaction_mode=_stack(items, "interaction_mode"),
        absorbing=_stack(items, "absorbing"),
    )


def _loader(
    dataset: GaussianBeliefDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        collate_fn=collate_gaussian_belief_items,
    )


def _to_device(batch: GaussianBeliefBatch, device: str) -> dict[str, torch.Tensor]:
    return {
        "vision_history": batch.vision_history.to(device=device, non_blocking=True),
        "proprio_history": batch.proprio_history.to(device=device, non_blocking=True),
        "remaining_actions": batch.remaining_actions.to(device=device, non_blocking=True),
        "latency_probabilities": batch.latency_probabilities.to(
            device=device, non_blocking=True
        ),
        "delay_ticks": batch.delay_ticks.to(device=device, non_blocking=True),
        "target_states": batch.target_states.to(device=device, non_blocking=True),
        "query_probabilities": batch.query_probabilities.to(
            device=device, non_blocking=True
        ),
    }


def _weighted_nll_torch(
    *,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    standardized = (target - mean) * torch.exp(-log_std)
    element = 0.5 * (
        torch.square(standardized) + 2.0 * log_std + math.log(2.0 * math.pi)
    )
    return torch.mean(torch.sum(weights * element.mean(dim=-1), dim=-1))


def _validation_loss(
    model: GaussianBeliefModel,
    loader: DataLoader,
    *,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            values = _to_device(batch, device)
            mean, log_std, _belief = model(
                vision_history=values["vision_history"],
                proprio_history=values["proprio_history"],
                remaining_actions=values["remaining_actions"],
                latency_probabilities=values["latency_probabilities"],
                delay_ticks=values["delay_ticks"],
            )
            loss = _weighted_nll_torch(
                mean=mean,
                log_std=log_std,
                target=values["target_states"],
                weights=values["query_probabilities"],
            )
            total += float(loss.item()) * len(batch.vision_history)
            count += len(batch.vision_history)
    return total / count


def _subset_metrics(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    normalized_mean: np.ndarray,
    normalized_log_std: np.ndarray,
    normalized_target: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    masked_weights = weights * mask
    mass = masked_weights.sum(axis=1)
    valid = mass > 0.0
    if not np.any(valid):
        return None
    normalized_weights = masked_weights[valid] / mass[valid, None]
    result = gaussian_prediction_metrics(
        mean=mean[valid],
        std=std[valid],
        target=target[valid],
        weights=normalized_weights,
    )
    result["normalized_weighted_nll"] = weighted_normalized_gaussian_nll(
        mean=normalized_mean[valid],
        log_std=normalized_log_std[valid],
        target=normalized_target[valid],
        weights=normalized_weights,
    )
    result["retained_probability_mass_mean"] = float(np.mean(mass[valid]))
    return result


def _evaluate(
    model: GaussianBeliefModel,
    loader: DataLoader,
    *,
    normalization: GaussianBeliefNormalization,
    device: str,
) -> dict[str, Any]:
    model.eval()
    means = []
    log_stds = []
    targets = []
    weights = []
    modes = []
    absorbing = []
    with torch.inference_mode():
        for batch in loader:
            values = _to_device(batch, device)
            mean, log_std, _belief = model(
                vision_history=values["vision_history"],
                proprio_history=values["proprio_history"],
                remaining_actions=values["remaining_actions"],
                latency_probabilities=values["latency_probabilities"],
                delay_ticks=values["delay_ticks"],
            )
            means.append(mean.cpu().numpy())
            log_stds.append(log_std.cpu().numpy())
            targets.append(batch.target_states.numpy())
            weights.append(batch.query_probabilities.numpy())
            modes.append(batch.interaction_mode.numpy())
            absorbing.append(batch.absorbing.numpy())
    normalized_mean = np.concatenate(means, axis=0)
    normalized_log_std = np.concatenate(log_stds, axis=0)
    normalized_target = np.concatenate(targets, axis=0)
    probability = np.concatenate(weights, axis=0).astype(np.float64)
    interaction_mode = np.concatenate(modes, axis=0)
    absorbing_mask = np.concatenate(absorbing, axis=0).astype(bool)
    target_scale = normalization.target_std[None, None, :]
    target_offset = normalization.target_mean[None, None, :]
    physical_mean = normalized_mean * target_scale + target_offset
    physical_target = normalized_target * target_scale + target_offset
    physical_std = np.exp(normalized_log_std) * target_scale
    overall = _subset_metrics(
        mean=physical_mean,
        std=physical_std,
        target=physical_target,
        normalized_mean=normalized_mean,
        normalized_log_std=normalized_log_std,
        normalized_target=normalized_target,
        weights=probability,
        mask=np.ones_like(absorbing_mask, dtype=bool),
    )
    if overall is None:
        raise RuntimeError("Gaussian belief evaluation produced no valid samples")
    result: dict[str, Any] = {"overall": overall}
    subsets = {
        "pre_handoff": (
            ~absorbing_mask
            & np.isin(interaction_mode, (InteractionMode.FREE, InteractionMode.CONTACT))
        ),
        "post_handoff": (
            ~absorbing_mask & (interaction_mode == InteractionMode.GRASPED)
        ),
        "absorbing": absorbing_mask,
    }
    for name, mask in subsets.items():
        metric = _subset_metrics(
            mean=physical_mean,
            std=physical_std,
            target=physical_target,
            normalized_mean=normalized_mean,
            normalized_log_std=normalized_log_std,
            normalized_target=normalized_target,
            weights=probability,
            mask=mask,
        )
        if metric is not None:
            result[name] = metric
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_level_gaussian_belief(
    *,
    corpus: FeatureBeliefCorpus,
    config: GaussianBeliefConfig,
    output_dir: Path,
    device: str,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Gaussian belief output already exists: {target}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Gaussian belief training but is unavailable")
    seed = config.random_seed + corpus.level
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    normalization = build_gaussian_belief_normalization(corpus)
    datasets = {
        split: GaussianBeliefDataset(
            corpus=corpus,
            split=split,
            normalization=normalization,
            config=config,
            exhaustive_queries=split is not ProbeSplit.TRAIN,
        )
        for split in ProbeSplit
    }
    loaders = {
        split: _loader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split is ProbeSplit.TRAIN,
            seed=seed,
        )
        for split, dataset in datasets.items()
    }
    model = GaussianBeliefModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        datasets[ProbeSplit.TRAIN].set_epoch(epoch - 1)
        model.train()
        total = 0.0
        count = 0
        for batch in loaders[ProbeSplit.TRAIN]:
            values = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            mean, log_std, _belief = model(
                vision_history=values["vision_history"],
                proprio_history=values["proprio_history"],
                remaining_actions=values["remaining_actions"],
                latency_probabilities=values["latency_probabilities"],
                delay_ticks=values["delay_ticks"],
            )
            loss = diagonal_gaussian_nll(
                mean=mean,
                log_std=log_std,
                target=values["target_states"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item()) * len(batch.vision_history)
            count += len(batch.vision_history)
        training_loss = total / count
        validation_loss = _validation_loss(
            model,
            loaders[ProbeSplit.VALIDATION],
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_sampled_normalized_nll": training_loss,
                "validation_exhaustive_weighted_normalized_nll": validation_loss,
            }
        )
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Gaussian belief training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    metrics = {
        split.value: _evaluate(
            model,
            loaders[split],
            normalization=normalization,
            device=device,
        )
        for split in (ProbeSplit.VALIDATION, ProbeSplit.HOLDOUT)
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        save_safetensors(
            {name: value.contiguous() for name, value in best_state.items()},
            building / "model.safetensors",
        )
        encoder_state = {
            name.removeprefix("encoder."): value.contiguous()
            for name, value in best_state.items()
            if name.startswith("encoder.")
        }
        save_safetensors(encoder_state, building / "encoder.safetensors")
        np.savez(
            building / "normalization.npz",
            proprio_mean=normalization.proprio_mean,
            proprio_std=normalization.proprio_std,
            action_mean=normalization.action_mean,
            action_std=normalization.action_std,
            target_mean=normalization.target_mean,
            target_std=normalization.target_std,
        )
        _write_json(building / "metrics.json", metrics)
        _write_json(building / "training_history.json", history)
        artifact_names = (
            "model.safetensors",
            "encoder.safetensors",
            "normalization.npz",
            "metrics.json",
            "training_history.json",
        )
        artifacts = {
            name: sha256_file(building / name) for name in artifact_names
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_gaussian_belief_v1",
            "level": corpus.level,
            "checkpoint_scope": "single_level_only",
            "config": asdict(config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "encoder_parameter_count": sum(
                parameter.numel() for parameter in model.encoder.parameters()
            ),
            "episode_counts": {
                split.value: corpus.episode_counts[split] for split in ProbeSplit
            },
            "sample_counts": {
                split.value: corpus.sample_counts[split] for split in ProbeSplit
            },
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_normalized_nll": best_loss,
            "torch_version": torch.__version__,
            "device": device,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Gaussian belief output already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
