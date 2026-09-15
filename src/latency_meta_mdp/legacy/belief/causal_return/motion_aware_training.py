"""Normalization and deterministic supervision for motion-aware histories."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryConfig,
    MotionAwareHistorySample,
)
from latency_meta_mdp.legacy.belief.causal_return.spatiotemporal_history import (
    MotionAwareHistoryEncoder,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

_PROVENANCE_INPUTS = frozenset(
    {
        "source_bulk_manifest",
        "cache_run_manifest",
        "vision_config",
        "temporal_config",
        "split_config",
        "motion_aware_config",
    }
)


def _readonly(value, *, dtype=np.float32) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MotionAwareNormalization:
    """Train-split statistics for deployment inputs and physical readouts."""

    robot_mean: np.ndarray
    robot_std: np.ndarray
    position_mean: np.ndarray
    position_std: np.ndarray
    velocity_mean: np.ndarray
    velocity_std: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "robot_mean": (16,),
            "robot_std": (16,),
            "position_mean": (3,),
            "position_std": (3,),
            "velocity_mean": (3,),
            "velocity_std": (3,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"motion-aware normalization {name} is invalid")
            if name.endswith("std") and np.any(value <= 0.0):
                raise ValueError("motion-aware normalization scales must be positive")
            object.__setattr__(self, name, _readonly(value))


def _is_digest(value: str, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class MotionAwareTrainingProvenance:
    """Code and immutable-input identity attached to one level checkpoint."""

    implementation_revision: str
    implementation_source_sha256: str
    implementation_dirty: bool
    bounded_review: bool
    input_sha256: dict[str, str]

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.implementation_revision, length=40)
            or not _is_digest(self.implementation_source_sha256, length=64)
            or not isinstance(self.implementation_dirty, bool)
            or not isinstance(self.bounded_review, bool)
            or set(self.input_sha256) != _PROVENANCE_INPUTS
            or any(not _is_digest(value, length=64) for value in self.input_sha256.values())
        ):
            raise ValueError("motion-aware training provenance is invalid")
        object.__setattr__(self, "input_sha256", dict(self.input_sha256))

    @property
    def artifact_eligible(self) -> bool:
        return not self.implementation_dirty and not self.bounded_review


def build_motion_aware_normalization(corpus) -> MotionAwareNormalization:
    """Compute normalization from training contexts only."""

    references = corpus.sample_references[ProbeSplit.TRAIN]
    if not references:
        raise ValueError("motion-aware normalization requires training samples")
    robot_values = []
    position_values = []
    velocity_values = []
    state_values = getattr(corpus, "state_values", None)
    for offset in range(len(references)):
        if callable(state_values):
            robot, position, velocity = state_values(ProbeSplit.TRAIN, offset)
        else:
            sample = corpus.materialize(ProbeSplit.TRAIN, offset)
            robot = sample.robot_history
            position = sample.object_position_target
            velocity = sample.object_velocity_target
        robot_values.append(np.asarray(robot, dtype=np.float64))
        position_values.append(np.asarray(position, dtype=np.float64))
        velocity_values.append(np.asarray(velocity, dtype=np.float64))
    robot_array = np.concatenate(robot_values, axis=0)
    position_array = np.stack(position_values)
    velocity_array = np.stack(velocity_values)
    return MotionAwareNormalization(
        robot_mean=robot_array.mean(axis=0),
        robot_std=np.maximum(robot_array.std(axis=0), 1e-6),
        position_mean=position_array.mean(axis=0),
        position_std=np.maximum(position_array.std(axis=0), 1e-6),
        velocity_mean=velocity_array.mean(axis=0),
        velocity_std=np.maximum(velocity_array.std(axis=0), 1e-6),
    )


@dataclass(frozen=True)
class MotionAwareItem:
    vision_history: np.ndarray
    robot_history_normalized: np.ndarray
    history_valid_mask: np.ndarray
    object_position_target: np.ndarray
    object_velocity_target: np.ndarray
    object_position_physical: np.ndarray
    object_velocity_physical: np.ndarray
    episode_id: str
    scene_seed: int
    source_tick: int
    source_phase: str
    transition_offset_ticks: int
    transition_offset_valid: bool
    exact_transition: bool


class MotionAwareDataset(Dataset):
    """Normalize deployment histories and current-state targets per level."""

    def __init__(
        self,
        *,
        corpus,
        split: ProbeSplit,
        normalization: MotionAwareNormalization,
    ) -> None:
        self.corpus = corpus
        self.split = split
        self.normalization = normalization
        self.references = corpus.sample_references[split]
        transition_metadata = getattr(corpus, "transition_metadata", None)
        if callable(transition_metadata):
            metadata = tuple(
                transition_metadata(split, offset) for offset in range(len(self.references))
            )
        else:
            metadata = tuple(
                (
                    sample.transition_offset_ticks,
                    sample.transition_offset_valid,
                )
                for sample in (
                    corpus.materialize(split, offset) for offset in range(len(self.references))
                )
            )
        self.exact_transition = tuple(valid and offset == 0 for offset, valid in metadata)
        valid_count = len(self.exact_transition) - sum(self.exact_transition)
        self.valid_velocity_fraction = (
            valid_count / len(self.exact_transition) if self.exact_transition else 0.0
        )

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, offset: int) -> MotionAwareItem:
        sample: MotionAwareHistorySample = self.corpus.materialize(self.split, offset)
        position = np.asarray(sample.object_position_target, dtype=np.float32)
        velocity = np.asarray(sample.object_velocity_target, dtype=np.float32)
        exact = self.exact_transition[offset]
        return MotionAwareItem(
            vision_history=np.array(sample.vision_history, copy=True),
            robot_history_normalized=(
                (sample.robot_history - self.normalization.robot_mean)
                / self.normalization.robot_std
            ).astype(np.float32),
            history_valid_mask=np.array(sample.history_valid_mask, copy=True),
            object_position_target=(
                (position - self.normalization.position_mean) / self.normalization.position_std
            ).astype(np.float32),
            object_velocity_target=(
                (velocity - self.normalization.velocity_mean) / self.normalization.velocity_std
            ).astype(np.float32),
            object_position_physical=np.array(position, copy=True),
            object_velocity_physical=np.array(velocity, copy=True),
            episode_id=sample.episode_id,
            scene_seed=sample.scene_seed,
            source_tick=sample.source_tick,
            source_phase=sample.source_phase,
            transition_offset_ticks=sample.transition_offset_ticks,
            transition_offset_valid=sample.transition_offset_valid,
            exact_transition=exact,
        )


@dataclass(frozen=True)
class MotionAwareBatch:
    vision_history: torch.Tensor
    robot_history_normalized: torch.Tensor
    history_valid_mask: torch.Tensor
    object_position_target: torch.Tensor
    object_velocity_target: torch.Tensor
    object_position_physical: torch.Tensor
    object_velocity_physical: torch.Tensor
    exact_transition_mask: torch.Tensor
    transition_offset_ticks: torch.Tensor
    transition_offset_valid: torch.Tensor
    episode_id: tuple[str, ...]
    scene_seed: tuple[int, ...]
    source_tick: tuple[int, ...]
    source_phase: tuple[str, ...]


def collate_motion_aware_items(items: list[MotionAwareItem]) -> MotionAwareBatch:
    if not items:
        raise ValueError("motion-aware batch cannot be empty")

    def stack(name: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(item, name) for item in items]))

    return MotionAwareBatch(
        vision_history=stack("vision_history"),
        robot_history_normalized=stack("robot_history_normalized"),
        history_valid_mask=stack("history_valid_mask"),
        object_position_target=stack("object_position_target"),
        object_velocity_target=stack("object_velocity_target"),
        object_position_physical=stack("object_position_physical"),
        object_velocity_physical=stack("object_velocity_physical"),
        exact_transition_mask=torch.tensor(
            [item.exact_transition for item in items], dtype=torch.bool
        ),
        transition_offset_ticks=torch.tensor(
            [item.transition_offset_ticks for item in items], dtype=torch.int64
        ),
        transition_offset_valid=torch.tensor(
            [item.transition_offset_valid for item in items], dtype=torch.bool
        ),
        episode_id=tuple(item.episode_id for item in items),
        scene_seed=tuple(item.scene_seed for item in items),
        source_tick=tuple(item.source_tick for item in items),
        source_phase=tuple(item.source_phase for item in items),
    )


@dataclass(frozen=True)
class MotionAwareLossTerms:
    position_sum: torch.Tensor
    position_count: int
    velocity_sum: torch.Tensor
    velocity_count: int

    def __add__(self, other: MotionAwareLossTerms) -> MotionAwareLossTerms:
        return MotionAwareLossTerms(
            position_sum=self.position_sum + other.position_sum,
            position_count=self.position_count + other.position_count,
            velocity_sum=self.velocity_sum + other.velocity_sum,
            velocity_count=self.velocity_count + other.velocity_count,
        )

    def mean(self) -> torch.Tensor:
        if self.position_count <= 0:
            raise ValueError("motion-aware position loss count must be positive")
        position = self.position_sum / self.position_count
        velocity = (
            self.velocity_sum / self.velocity_count
            if self.velocity_count > 0
            else self.velocity_sum
        )
        return position + velocity

    def unbiased_minibatch_mean(self, *, valid_velocity_fraction: float) -> torch.Tensor:
        if (
            self.position_count <= 0
            or not np.isfinite(valid_velocity_fraction)
            or not 0.0 <= valid_velocity_fraction <= 1.0
        ):
            raise ValueError("motion-aware valid velocity fraction is invalid")
        position = self.position_sum / self.position_count
        if valid_velocity_fraction == 0.0:
            return position + self.velocity_sum
        velocity = self.velocity_sum / (self.position_count * valid_velocity_fraction)
        return position + velocity


def motion_aware_huber_terms(
    *,
    predicted_position: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    exact_transition_mask: torch.Tensor,
    delta: float,
) -> MotionAwareLossTerms:
    """Return additive Huber sums/counts under the exact-transition mask."""

    shape = predicted_position.shape
    if (
        shape != predicted_velocity.shape
        or shape != target_position.shape
        or shape != target_velocity.shape
        or len(shape) != 2
        or shape[1] != 3
        or tuple(exact_transition_mask.shape) != (shape[0],)
        or exact_transition_mask.dtype is not torch.bool
    ):
        raise ValueError("motion-aware Huber tensors have incompatible shapes")
    if isinstance(delta, bool) or not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("motion-aware Huber delta must be positive and finite")
    values = (
        predicted_position,
        predicted_velocity,
        target_position,
        target_velocity,
    )
    if any(not torch.all(torch.isfinite(value)) for value in values):
        raise ValueError("motion-aware Huber tensors must be finite")
    position_sum = F.huber_loss(
        predicted_position,
        target_position,
        delta=delta,
        reduction="sum",
    )
    valid_velocity = ~exact_transition_mask
    if torch.any(valid_velocity):
        velocity_sum = F.huber_loss(
            predicted_velocity[valid_velocity],
            target_velocity[valid_velocity],
            delta=delta,
            reduction="sum",
        )
    else:
        velocity_sum = predicted_velocity.sum() * 0.0
    return MotionAwareLossTerms(
        position_sum=position_sum,
        position_count=predicted_position.numel(),
        velocity_sum=velocity_sum,
        velocity_count=int(torch.count_nonzero(valid_velocity)) * predicted_velocity.shape[1],
    )


def masked_motion_aware_huber(
    *,
    predicted_position: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_velocity: torch.Tensor,
    exact_transition_mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Return the global-mean form used by direct loss-contract callers."""

    return motion_aware_huber_terms(
        predicted_position=predicted_position,
        predicted_velocity=predicted_velocity,
        target_position=target_position,
        target_velocity=target_velocity,
        exact_transition_mask=exact_transition_mask,
        delta=delta,
    ).mean()


def _batch_values(batch: MotionAwareBatch, *, device: str) -> dict[str, torch.Tensor]:
    return {
        "vision_history": batch.vision_history.to(device=device, non_blocking=True),
        "robot_history_normalized": batch.robot_history_normalized.to(
            device=device, non_blocking=True
        ),
        "history_valid_mask": batch.history_valid_mask.to(device=device, non_blocking=True),
        "object_position_target": batch.object_position_target.to(device=device, non_blocking=True),
        "object_velocity_target": batch.object_velocity_target.to(device=device, non_blocking=True),
        "exact_transition_mask": batch.exact_transition_mask.to(device=device, non_blocking=True),
    }


def _mean_validation_loss(
    *,
    model: nn.Module,
    loader,
    config: MotionAwareHistoryConfig,
    device: str,
) -> float:
    model.eval()
    position_sum = 0.0
    position_count = 0
    velocity_sum = 0.0
    velocity_count = 0
    with torch.inference_mode():
        for batch in loader:
            values = _batch_values(batch, device=device)
            estimate = model(
                vision_history=values["vision_history"],
                robot_history_normalized=values["robot_history_normalized"],
                history_valid_mask=values["history_valid_mask"],
            )
            terms = motion_aware_huber_terms(
                predicted_position=estimate.object_position_normalized,
                predicted_velocity=estimate.object_velocity_normalized,
                target_position=values["object_position_target"],
                target_velocity=values["object_velocity_target"],
                exact_transition_mask=values["exact_transition_mask"],
                delta=config.huber_delta,
            )
            position_sum += float(terms.position_sum)
            position_count += terms.position_count
            velocity_sum += float(terms.velocity_sum)
            velocity_count += terms.velocity_count
    if position_count == 0:
        raise ValueError("motion-aware validation loader is empty")
    return position_sum / position_count + (
        velocity_sum / velocity_count if velocity_count > 0 else 0.0
    )


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish one directory without replacing an existing path."""

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"motion-aware training output exists: {target}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"motion-aware training output exists: {target}")
    raise OSError(error, os.strerror(error), str(target))


def train_level_motion_aware_history(
    *,
    corpus,
    config: MotionAwareHistoryConfig,
    output_dir: Path,
    device: str,
    provenance: MotionAwareTrainingProvenance,
    model_factory: Callable[[MotionAwareHistoryConfig], nn.Module] = MotionAwareHistoryEncoder,
) -> Path:
    """Train one level and atomically publish its best validation checkpoint."""

    if corpus.level not in (1, 2, 3):
        raise ValueError("motion-aware training level is invalid")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for motion-aware training but is unavailable")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"motion-aware training output exists: {target}")
    normalization = build_motion_aware_normalization(corpus)
    train_dataset = MotionAwareDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
    )
    validation_dataset = MotionAwareDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=normalization,
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("motion-aware training requires train and validation samples")
    generator = torch.Generator().manual_seed(config.random_seed + corpus.level)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_motion_aware_items,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_motion_aware_items,
    )
    torch.manual_seed(config.random_seed + corpus.level)
    np.random.seed(config.random_seed + corpus.level)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.random_seed + corpus.level)
    model = model_factory(config).to(device)
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
        train_position_sum = 0.0
        train_position_count = 0
        train_velocity_sum = 0.0
        train_velocity_count = 0
        for batch in train_loader:
            values = _batch_values(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(
                vision_history=values["vision_history"],
                robot_history_normalized=values["robot_history_normalized"],
                history_valid_mask=values["history_valid_mask"],
            )
            terms = motion_aware_huber_terms(
                predicted_position=estimate.object_position_normalized,
                predicted_velocity=estimate.object_velocity_normalized,
                target_position=values["object_position_target"],
                target_velocity=values["object_velocity_target"],
                exact_transition_mask=values["exact_transition_mask"],
                delta=config.huber_delta,
            )
            loss = terms.unbiased_minibatch_mean(
                valid_velocity_fraction=train_dataset.valid_velocity_fraction
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            train_position_sum += float(terms.position_sum.detach())
            train_position_count += terms.position_count
            train_velocity_sum += float(terms.velocity_sum.detach())
            train_velocity_count += terms.velocity_count
        train_loss = train_position_sum / train_position_count + (
            train_velocity_sum / train_velocity_count if train_velocity_count > 0 else 0.0
        )
        validation_loss = _mean_validation_loss(
            model=model,
            loader=validation_loader,
            config=config,
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
                "training_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            "[motion-aware-history] "
            f"L{corpus.level} epoch={epoch + 1}/{config.max_epochs} "
            f"train={train_loss:.6f} val={validation_loss:.6f} "
            f"best={best_loss:.6f} patience={patience}/{config.early_stopping_patience}",
            flush=True,
        )
        if patience >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("motion-aware training produced no finite validation checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        save_safetensors(
            {name: value.contiguous() for name, value in best_state.items()},
            building / "model.safetensors",
        )
        _fsync_file(building / "model.safetensors")
        with (building / "normalization.npz").open("xb") as handle:
            np.savez(
                handle,
                robot_mean=normalization.robot_mean,
                robot_std=normalization.robot_std,
                position_mean=normalization.position_mean,
                position_std=normalization.position_std,
                velocity_mean=normalization.velocity_mean,
                velocity_std=normalization.velocity_std,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(building / "training_history.json", history)
        artifacts = {
            name: sha256_file(building / name)
            for name in (
                "model.safetensors",
                "normalization.npz",
                "training_history.json",
            )
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_motion_aware_history_checkpoint",
                "scientific_gate_pass": False,
                "artifact_eligible": provenance.artifact_eligible,
                "level": corpus.level,
                "config": asdict(config),
                "implementation_revision": provenance.implementation_revision,
                "implementation_source_sha256": provenance.implementation_source_sha256,
                "implementation_dirty": provenance.implementation_dirty,
                "bounded_review": provenance.bounded_review,
                "input_sha256": provenance.input_sha256,
                "training_sample_count": len(train_dataset),
                "validation_sample_count": len(validation_dataset),
                "training_episode_count": corpus.episode_counts[ProbeSplit.TRAIN],
                "validation_episode_count": corpus.episode_counts[ProbeSplit.VALIDATION],
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "epochs_completed": len(history),
                "wall_seconds": time.perf_counter() - started,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "artifacts": artifacts,
            },
        )
        _fsync_directory(building)
        _rename_directory_no_replace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
