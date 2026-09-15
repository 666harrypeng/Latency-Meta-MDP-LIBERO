"""Training and evaluation for level-specific frozen-vision diagnostic probes."""

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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.vision_probe_config import VisionProbeConfig
from latency_meta_mdp.legacy.vision_probe_corpus import VisionProbeCorpus
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit
from latency_meta_mdp.legacy.vision_probe_metrics import (
    state_regression_metrics,
    train_mean_baseline,
)
from latency_meta_mdp.legacy.vision_state_probe import TemporalVisionStateProbe


@dataclass(frozen=True)
class ProbeNormalization:
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def _proprio_stream(episode: Any) -> np.ndarray:
    deployment = episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    return np.concatenate(
        (
            deployment.robot_qpos,
            deployment.robot_qvel,
            width[:, None],
            width_velocity[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def _target_stream(episode: Any) -> np.ndarray:
    return np.concatenate(
        (
            episode.supervision.object_pose[:, :3],
            episode.supervision.object_velocity[:, :3],
            episode.supervision.relative_geometry[:, :3],
        ),
        axis=1,
    ).astype(np.float32)


def _training_normalization(corpus: VisionProbeCorpus) -> ProbeNormalization:
    proprio = []
    targets = []
    for record in corpus.records:
        if record.indices[0].split is not ProbeSplit.TRAIN:
            continue
        proprio.append(_proprio_stream(record.episode))
        stream = _target_stream(record.episode)
        targets.append(stream[[index.source_tick for index in record.indices]])
    proprio_values = np.concatenate(proprio, axis=0)
    target_values = np.concatenate(targets, axis=0)
    return ProbeNormalization(
        proprio_mean=proprio_values.mean(axis=0),
        proprio_std=np.maximum(proprio_values.std(axis=0), 1e-6),
        target_mean=target_values.mean(axis=0),
        target_std=np.maximum(target_values.std(axis=0), 1e-6),
    )


class _ProbeDataset(Dataset):
    def __init__(
        self,
        *,
        corpus: VisionProbeCorpus,
        split: ProbeSplit,
        normalization: ProbeNormalization,
    ) -> None:
        self.corpus = corpus
        self.references = corpus.sample_references[split]
        self.normalization = normalization
        self.proprio_streams = tuple(_proprio_stream(record.episode) for record in corpus.records)
        self.target_streams = tuple(_target_stream(record.episode) for record in corpus.records)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, offset: int):
        record_index, sample_index = self.references[offset]
        record = self.corpus.records[record_index]
        index = record.indices[sample_index]
        history = slice(index.history_start_tick, index.source_tick + 1)
        vision = np.array(record.cache.features[history], dtype=np.float16, copy=True)
        proprio = np.array(self.proprio_streams[record_index][history], copy=True)
        proprio = (proprio - self.normalization.proprio_mean) / self.normalization.proprio_std
        target = np.array(self.target_streams[record_index][index.source_tick], copy=True)
        target = (target - self.normalization.target_mean) / self.normalization.target_std
        return (
            torch.from_numpy(vision),
            torch.from_numpy(proprio.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            index.pre_handoff,
        )


def _loader(
    dataset: _ProbeDataset,
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
    )


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for vision, proprio, target, _pre_handoff in loader:
            prediction = model(
                vision.to(device=device, non_blocking=True),
                proprio.to(device=device, non_blocking=True),
            )
            total += float(torch.sum(torch.square(prediction - target.to(device))).item())
            count += target.numel()
    return total / count


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    normalization: ProbeNormalization,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    pre_handoff = []
    with torch.inference_mode():
        for vision, proprio, target, phase in loader:
            prediction = model(
                vision.to(device=device, non_blocking=True),
                proprio.to(device=device, non_blocking=True),
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(target.numpy())
            pre_handoff.append(phase.numpy())
    prediction_normalized = np.concatenate(predictions, axis=0)
    target_normalized = np.concatenate(targets, axis=0)
    return (
        prediction_normalized * normalization.target_std + normalization.target_mean,
        target_normalized * normalization.target_std + normalization.target_mean,
        np.concatenate(pre_handoff, axis=0).astype(bool),
    )


def _improvement(probe: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    result = {}
    for name in ("object_position", "object_linear_velocity", "relative_position"):
        denominator = baseline[name]["rmse_si"]
        result[name] = (
            float("nan") if denominator <= 0.0 else 1.0 - probe[name]["rmse_si"] / denominator
        )
    return result


def _metric_bundle(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    pre_handoff: np.ndarray,
    normalization: ProbeNormalization,
) -> dict[str, Any]:
    baseline_prediction = train_mean_baseline(
        training_targets=np.stack(
            (normalization.target_mean, normalization.target_mean),
            axis=0,
        ),
        evaluation_count=len(target),
    )
    probe = state_regression_metrics(prediction=prediction, target=target)
    baseline = state_regression_metrics(prediction=baseline_prediction, target=target)
    result: dict[str, Any] = {
        "probe": probe,
        "train_mean_baseline": baseline,
        "rmse_improvement_fraction": _improvement(probe, baseline),
    }
    for phase_name, mask in (
        ("pre_handoff", pre_handoff),
        ("post_handoff", ~pre_handoff),
    ):
        if np.any(mask):
            phase_probe = state_regression_metrics(
                prediction=prediction[mask],
                target=target[mask],
            )
            phase_baseline = state_regression_metrics(
                prediction=baseline_prediction[mask],
                target=target[mask],
            )
            result[phase_name] = {
                "probe": phase_probe,
                "train_mean_baseline": phase_baseline,
                "rmse_improvement_fraction": _improvement(
                    phase_probe,
                    phase_baseline,
                ),
            }
    return result


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def train_level_state_probe(
    *,
    corpus: VisionProbeCorpus,
    config: VisionProbeConfig,
    output_dir: Path,
    device: str,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"state probe output already exists: {target}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for state probe training but is unavailable")
    torch.manual_seed(config.random_seed + corpus.level)
    np.random.seed(config.random_seed + corpus.level)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.random_seed + corpus.level)
    normalization = _training_normalization(corpus)
    datasets = {
        split: _ProbeDataset(
            corpus=corpus,
            split=split,
            normalization=normalization,
        )
        for split in ProbeSplit
    }
    loaders = {
        split: _loader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split is ProbeSplit.TRAIN,
            seed=config.random_seed + corpus.level,
        )
        for split, dataset in datasets.items()
    }
    model = TemporalVisionStateProbe(
        history_sample_count=config.history_sample_count,
        patch_projection_dim=config.patch_projection_dim,
        temporal_hidden_dim=config.temporal_hidden_dim,
    ).to(device)
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
        model.train()
        total = 0.0
        count = 0
        for vision, proprio, target_state, _pre_handoff in loaders[ProbeSplit.TRAIN]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                vision.to(device=device, non_blocking=True),
                proprio.to(device=device, non_blocking=True),
            )
            expected = target_state.to(device=device, non_blocking=True)
            loss = torch.mean(torch.square(prediction - expected))
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target_state)
            count += len(target_state)
        training_loss = total / count
        validation_loss = _validation_loss(
            model,
            loaders[ProbeSplit.VALIDATION],
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_normalized_mse": training_loss,
                "validation_normalized_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("state probe training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for split in (ProbeSplit.VALIDATION, ProbeSplit.HOLDOUT):
        prediction, expected, pre_handoff = _evaluate(
            model,
            loaders[split],
            normalization=normalization,
            device=device,
        )
        metrics[split.value] = _metric_bundle(
            prediction=prediction,
            target=expected,
            pre_handoff=pre_handoff,
            normalization=normalization,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        model_path = building / "model.safetensors"
        save_safetensors(
            {name: value.contiguous() for name, value in best_state.items()},
            model_path,
        )
        np.savez(
            building / "normalization.npz",
            proprio_mean=normalization.proprio_mean,
            proprio_std=normalization.proprio_std,
            target_mean=normalization.target_mean,
            target_std=normalization.target_std,
        )
        _write_json(building / "metrics.json", metrics)
        _write_json(building / "training_history.json", history)
        artifacts = {
            name: sha256_file(building / name)
            for name in (
                "model.safetensors",
                "normalization.npz",
                "metrics.json",
                "training_history.json",
            )
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_temporal_state_probe_v1",
            "level": corpus.level,
            "checkpoint_scope": "single_level_only",
            "config": asdict(config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "episode_counts": {split.value: corpus.episode_counts[split] for split in ProbeSplit},
            "sample_counts": {split.value: corpus.sample_counts[split] for split in ProbeSplit},
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_normalized_mse": best_loss,
            "torch_version": torch.__version__,
            "device": device,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        if target.exists():
            raise FileExistsError(f"state probe output already exists: {target}")
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
