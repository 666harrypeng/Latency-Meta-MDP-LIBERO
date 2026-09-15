"""Normalization, batches, and objectives for information-state estimation."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import Dataset

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.causal_return.contracts import (
    InformationStateConfig,
    InformationStateSample,
)
from latency_meta_mdp.legacy.belief.causal_return.information_state import InformationStateEstimator
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def _readonly(value, *, dtype=np.float32) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InformationStateNormalization:
    robot_mean: np.ndarray
    robot_std: np.ndarray
    object_mean: np.ndarray
    object_std: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "robot_mean": (16,),
            "robot_std": (16,),
            "object_mean": (6,),
            "object_std": (6,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"information-state normalization {name} is invalid")
            if name.endswith("std") and np.any(value <= 0.0):
                raise ValueError("information-state normalization scales must be positive")
            object.__setattr__(self, name, _readonly(value))


def build_information_state_normalization(corpus) -> InformationStateNormalization:
    robot_values = []
    object_values = []
    references = corpus.sample_references[ProbeSplit.TRAIN]
    if not references:
        raise ValueError("information-state normalization requires training samples")
    state_values = getattr(corpus, "state_values", None)
    for offset in range(len(references)):
        if callable(state_values):
            robot, target = state_values(ProbeSplit.TRAIN, offset)
        else:
            sample = corpus.materialize(ProbeSplit.TRAIN, offset)
            robot, target = sample.robot_history, sample.object_state_target
        robot_values.append(np.asarray(robot, dtype=np.float64))
        object_values.append(np.asarray(target, dtype=np.float64))
    robot = np.concatenate(robot_values, axis=0)
    objects = np.stack(object_values)
    return InformationStateNormalization(
        robot_mean=robot.mean(axis=0),
        robot_std=np.maximum(robot.std(axis=0), 1e-6),
        object_mean=objects.mean(axis=0),
        object_std=np.maximum(objects.std(axis=0), 1e-6),
    )


@dataclass(frozen=True)
class InformationStateItem:
    vision_history: np.ndarray
    robot_history: np.ndarray
    history_time_ms: np.ndarray
    history_valid_mask: np.ndarray
    object_state_target: np.ndarray
    episode_id: str
    scene_seed: int
    source_tick: int
    source_phase: str
    motion_curvature: float
    motion_transition_distance_ticks: int


class InformationStateDataset(Dataset):
    def __init__(self, *, corpus, split: ProbeSplit, normalization: InformationStateNormalization):
        self.corpus = corpus
        self.split = split
        self.normalization = normalization
        self.references = corpus.sample_references[split]

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, offset: int) -> InformationStateItem:
        sample: InformationStateSample = self.corpus.materialize(self.split, offset)
        return InformationStateItem(
            vision_history=np.array(sample.vision_history, copy=True),
            robot_history=(
                (sample.robot_history - self.normalization.robot_mean)
                / self.normalization.robot_std
            ).astype(np.float32),
            history_time_ms=np.array(sample.history_time_ms, copy=True),
            history_valid_mask=np.array(sample.history_valid_mask, copy=True),
            object_state_target=(
                (sample.object_state_target - self.normalization.object_mean)
                / self.normalization.object_std
            ).astype(np.float32),
            episode_id=sample.episode_id,
            scene_seed=sample.scene_seed,
            source_tick=sample.source_tick,
            source_phase=sample.source_phase,
            motion_curvature=sample.motion_curvature,
            motion_transition_distance_ticks=sample.motion_transition_distance_ticks,
        )


@dataclass(frozen=True)
class InformationStateBatch:
    vision_history: torch.Tensor
    robot_history: torch.Tensor
    history_time_ms: torch.Tensor
    history_valid_mask: torch.Tensor
    object_state_target: torch.Tensor
    episode_id: tuple[str, ...]
    scene_seed: tuple[int, ...]
    source_tick: tuple[int, ...]
    source_phase: tuple[str, ...]
    motion_curvature: torch.Tensor
    motion_transition_distance_ticks: torch.Tensor


def collate_information_state_items(items: list[InformationStateItem]) -> InformationStateBatch:
    if not items:
        raise ValueError("information-state batch cannot be empty")

    def stack(name: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(item, name) for item in items]))

    return InformationStateBatch(
        vision_history=stack("vision_history"),
        robot_history=stack("robot_history"),
        history_time_ms=stack("history_time_ms"),
        history_valid_mask=stack("history_valid_mask"),
        object_state_target=stack("object_state_target"),
        episode_id=tuple(item.episode_id for item in items),
        scene_seed=tuple(item.scene_seed for item in items),
        source_tick=tuple(item.source_tick for item in items),
        source_phase=tuple(item.source_phase for item in items),
        motion_curvature=torch.tensor([item.motion_curvature for item in items]),
        motion_transition_distance_ticks=torch.tensor(
            [item.motion_transition_distance_ticks for item in items],
            dtype=torch.int64,
        ),
    )


def heteroscedastic_gaussian_nll(
    *,
    mean: torch.Tensor,
    log_scale: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_scale.shape or mean.shape != target.shape or mean.shape[-1] != 6:
        raise ValueError("information-state NLL tensors have incompatible shapes")
    residual = (target - mean) * torch.exp(-log_scale)
    return torch.mean(0.5 * residual.square() + log_scale)


def _batch_values(batch: InformationStateBatch, *, device: str) -> dict[str, torch.Tensor]:
    return {
        "vision_history": batch.vision_history.to(device=device, non_blocking=True),
        "robot_history": batch.robot_history.to(device=device, non_blocking=True),
        "history_time_ms": batch.history_time_ms.to(device=device, non_blocking=True),
        "history_valid_mask": batch.history_valid_mask.to(device=device, non_blocking=True),
        "object_state_target": batch.object_state_target.to(device=device, non_blocking=True),
    }


def _mean_validation_loss(
    *,
    model: InformationStateEstimator,
    loader,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            values = _batch_values(batch, device=device)
            estimate = model(
                vision_history=values["vision_history"],
                robot_history=values["robot_history"],
                history_time_ms=values["history_time_ms"],
                history_valid_mask=values["history_valid_mask"],
            )
            loss = heteroscedastic_gaussian_nll(
                mean=estimate.object_state_mean,
                log_scale=estimate.object_state_log_scale,
                target=values["object_state_target"],
            )
            batch_count = len(batch.episode_id)
            total += float(loss) * batch_count
            count += batch_count
    if count == 0:
        raise ValueError("information-state validation loader is empty")
    return total / count


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_level_information_state(
    *,
    corpus,
    config: InformationStateConfig,
    output_dir: Path,
    device: str,
) -> Path:
    """Train one level from scratch and publish one atomic checkpoint."""

    if corpus.level not in (1, 2, 3):
        raise ValueError("information-state training level is invalid")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"information-state checkpoint already exists: {target}")
    normalization = build_information_state_normalization(corpus)
    train_dataset = InformationStateDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
    )
    validation_dataset = InformationStateDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=normalization,
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("information-state training requires train and validation samples")
    generator = torch.Generator().manual_seed(config.random_seed + corpus.level)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_information_state_items,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_information_state_items,
    )
    torch.manual_seed(config.random_seed + corpus.level)
    model = InformationStateEstimator(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    history = []
    started = time.perf_counter()
    for epoch in range(config.max_epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            values = _batch_values(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(
                vision_history=values["vision_history"],
                robot_history=values["robot_history"],
                history_time_ms=values["history_time_ms"],
                history_valid_mask=values["history_valid_mask"],
            )
            loss = heteroscedastic_gaussian_nll(
                mean=estimate.object_state_mean,
                log_scale=estimate.object_state_log_scale,
                target=values["object_state_target"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_count = len(batch.episode_id)
            train_total += float(loss.detach()) * batch_count
            train_count += batch_count
        train_loss = train_total / train_count
        validation_loss = _mean_validation_loss(
            model=model,
            loader=validation_loader,
            device=device,
        )
        improved = validation_loss < best_loss - config.early_stopping_min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_loss,
                "validation_nll": validation_loss,
                "best_validation_nll": best_loss,
                "patience": patience,
            }
        )
        print(
            f"[causal-return-state] L{corpus.level} epoch={epoch + 1}/{config.max_epochs} "
            f"train_nll={train_loss:.6f} val_nll={validation_loss:.6f} "
            f"best={best_loss:.6f} patience={patience}/{config.early_stopping_patience}",
            flush=True,
        )
        if patience >= config.early_stopping_patience:
            break
    if best_state is None or best_epoch < 0 or not np.isfinite(best_loss):
        raise RuntimeError("information-state training did not produce a finite checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        save_safetensors(best_state, building / "model.safetensors")
        with (building / "normalization.npz").open("xb") as handle:
            np.savez(
                handle,
                robot_mean=normalization.robot_mean,
                robot_std=normalization.robot_std,
                object_mean=normalization.object_mean,
                object_std=normalization.object_std,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(building / "training_history.json", history)
        artifacts = {
            name: sha256_file(building / name)
            for name in ("model.safetensors", "normalization.npz", "training_history.json")
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_information_state_checkpoint",
                "level": corpus.level,
                "config": dataclasses.asdict(config),
                "training_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "best_epoch": best_epoch,
                "best_validation_nll": best_loss,
                "wall_seconds": time.perf_counter() - started,
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
