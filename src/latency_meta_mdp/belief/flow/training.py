"""Joint Flow Encoder/vector-field training with deterministic validation paths."""

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
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.model import (
    FlowBeliefModel,
    build_flow_matching_batch,
    conditional_flow_matching_loss,
)
from latency_meta_mdp.belief.flow.training_data import (
    FlowBeliefDataset,
    FlowBeliefItem,
    build_flow_belief_normalization,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class FlowBeliefBatch:
    vision_history: torch.Tensor
    proprio_history: torch.Tensor
    remaining_actions: torch.Tensor
    latency_probabilities: torch.Tensor
    delay_ticks: torch.Tensor
    query_probabilities: torch.Tensor
    target_states: torch.Tensor
    noise: torch.Tensor
    flow_time: torch.Tensor
    interaction_mode: torch.Tensor
    absorbing: torch.Tensor


def _stack(items: list[FlowBeliefItem], name: str) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([np.asarray(getattr(item, name)) for item in items], axis=0)
    )


def collate_flow_belief_items(items: list[FlowBeliefItem]) -> FlowBeliefBatch:
    if not items:
        raise ValueError("cannot collate an empty Flow Belief batch")
    return FlowBeliefBatch(
        vision_history=_stack(items, "vision_history"),
        proprio_history=_stack(items, "proprio_history"),
        remaining_actions=_stack(items, "remaining_actions"),
        latency_probabilities=_stack(items, "latency_probabilities"),
        delay_ticks=_stack(items, "delay_ticks"),
        query_probabilities=_stack(items, "query_probabilities"),
        target_states=_stack(items, "target_states"),
        noise=_stack(items, "noise"),
        flow_time=_stack(items, "flow_time"),
        interaction_mode=_stack(items, "interaction_mode"),
        absorbing=_stack(items, "absorbing"),
    )


def _loader(
    dataset: FlowBeliefDataset,
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
        collate_fn=collate_flow_belief_items,
    )


def _to_device(batch: FlowBeliefBatch, device: str) -> dict[str, torch.Tensor]:
    return {
        name: getattr(batch, name).to(device=device, non_blocking=True)
        for name in (
            "vision_history",
            "proprio_history",
            "remaining_actions",
            "latency_probabilities",
            "delay_ticks",
            "query_probabilities",
            "target_states",
            "noise",
            "flow_time",
        )
    }


def _predict_velocity(
    model: FlowBeliefModel,
    values: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    path = build_flow_matching_batch(
        target_state=values["target_states"],
        noise=values["noise"],
        flow_time=values["flow_time"],
    )
    prediction, _belief = model(
        vision_history=values["vision_history"],
        proprio_history=values["proprio_history"],
        remaining_actions=values["remaining_actions"],
        latency_probabilities=values["latency_probabilities"],
        noisy_state=path.noisy_state,
        flow_time=path.flow_time,
        delay_ticks=values["delay_ticks"],
    )
    return prediction, path.target_velocity


def _weighted_flow_mse(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    query_mse = torch.mean(torch.square(prediction - target), dim=-1)
    return torch.mean(torch.sum(weights * query_mse, dim=-1))


def _validation_loss(
    model: FlowBeliefModel,
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
            prediction, target = _predict_velocity(model, values)
            loss = _weighted_flow_mse(
                prediction=prediction,
                target=target,
                weights=values["query_probabilities"],
            )
            total += float(loss.item()) * len(batch.vision_history)
            count += len(batch.vision_history)
    return total / count


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_level_flow_belief(
    *,
    corpus: FeatureBeliefCorpus,
    config: FlowBeliefConfig,
    output_dir: Path,
    device: str,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow Belief output already exists: {target}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Flow Belief training but is unavailable")
    seed = config.random_seed + corpus.level
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    normalization = build_flow_belief_normalization(corpus)
    datasets = {
        split: FlowBeliefDataset(
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
    model = FlowBeliefModel(config).to(device)
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
            prediction, target_velocity = _predict_velocity(model, values)
            loss = conditional_flow_matching_loss(
                predicted_velocity=prediction,
                target_velocity=target_velocity,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm,
            )
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
                "training_sampled_flow_mse": training_loss,
                "validation_fixed_weighted_flow_mse": validation_loss,
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
        raise RuntimeError("Flow Belief training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    metrics = {
        "validation_fixed_flow_mse": _validation_loss(
            model,
            loaders[ProbeSplit.VALIDATION],
            device=device,
        ),
        "holdout_fixed_flow_mse": _validation_loss(
            model,
            loaders[ProbeSplit.HOLDOUT],
            device=device,
        ),
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
            "format_id": "level_flow_belief_v1",
            "level": corpus.level,
            "checkpoint_scope": "single_level_only",
            "initialization": "random_flow_encoder_and_vector_field",
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
            "best_validation_flow_mse": best_loss,
            "torch_version": torch.__version__,
            "device": device,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"Flow Belief output already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
