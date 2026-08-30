"""Physical-unit metrics for causal-return current-state estimation."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.causal_return.contracts import InformationStateConfig
from latency_meta_mdp.belief.causal_return.information_state import InformationStateEstimator
from latency_meta_mdp.belief.causal_return.information_training import (
    InformationStateDataset,
    InformationStateNormalization,
    collate_information_state_items,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit

_PHASES = ("pregrasp", "approach", "close", "lift")


def _subset_metrics(
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    target: np.ndarray,
    nll: np.ndarray,
    rows: np.ndarray,
) -> dict:
    count = int(np.count_nonzero(rows))
    if count == 0:
        return {
            "context_count": 0,
            "object_position_rmse_m": 0.0,
            "object_position_mae_m": 0.0,
            "object_velocity_rmse_m_s": 0.0,
            "object_velocity_mae_m_s": 0.0,
            "signed_position_bias_m": [0.0, 0.0, 0.0],
            "signed_velocity_bias_m_s": [0.0, 0.0, 0.0],
            "coverage_68": 0.0,
            "coverage_95": 0.0,
            "normalized_nll": 0.0,
            "invalid_value_count": 0,
        }
    selected_mean = mean[rows]
    selected_scale = scale[rows]
    selected_target = target[rows]
    error = selected_mean - selected_target
    position_norm = np.linalg.norm(error[:, :3], axis=-1)
    velocity_norm = np.linalg.norm(error[:, 3:], axis=-1)
    absolute = np.abs(error)
    return {
        "context_count": count,
        "object_position_rmse_m": float(np.sqrt(np.mean(np.square(position_norm)))),
        "object_position_mae_m": float(np.mean(position_norm)),
        "object_velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_norm)))),
        "object_velocity_mae_m_s": float(np.mean(velocity_norm)),
        "signed_position_bias_m": np.mean(error[:, :3], axis=0).tolist(),
        "signed_velocity_bias_m_s": np.mean(error[:, 3:], axis=0).tolist(),
        "coverage_68": float(np.mean(absolute <= selected_scale)),
        "coverage_95": float(np.mean(absolute <= 1.96 * selected_scale)),
        "normalized_nll": float(np.mean(nll[rows])),
        "invalid_value_count": int(
            selected_mean.size
            + selected_scale.size
            + selected_target.size
            - np.count_nonzero(np.isfinite(selected_mean))
            - np.count_nonzero(np.isfinite(selected_scale))
            - np.count_nonzero(np.isfinite(selected_target))
        ),
    }


def compute_information_state_metrics(
    *,
    object_state_mean: np.ndarray,
    object_state_scale: np.ndarray,
    object_state_target: np.ndarray,
    source_phase: np.ndarray,
    motion_curvature: np.ndarray,
    motion_transition_distance_ticks: np.ndarray,
    normalized_nll: np.ndarray,
) -> dict:
    """Report overall, phase, curvature, and transition-near current-state quality."""

    mean = np.asarray(object_state_mean, dtype=np.float64)
    scale = np.asarray(object_state_scale, dtype=np.float64)
    target = np.asarray(object_state_target, dtype=np.float64)
    phase = np.asarray(source_phase).astype(str)
    curvature = np.asarray(motion_curvature, dtype=np.float64)
    transition = np.asarray(motion_transition_distance_ticks, dtype=np.int64)
    nll = np.asarray(normalized_nll, dtype=np.float64)
    count = len(mean)
    if (
        mean.shape != (count, 6)
        or scale.shape != (count, 6)
        or target.shape != (count, 6)
        or phase.shape != (count,)
        or curvature.shape != (count,)
        or transition.shape != (count,)
        or nll.shape != (count,)
        or np.any(scale <= 0.0)
        or any(value not in _PHASES for value in phase)
    ):
        raise ValueError("information-state metric arrays have invalid shapes or values")
    numeric = (mean, scale, target, curvature, nll)
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("information-state metric arrays must be finite")
    all_rows = np.ones(count, dtype=bool)
    positive_curvature = curvature[curvature > 0.0]
    curvature_rows = np.zeros(count, dtype=bool)
    if len(positive_curvature):
        threshold = float(np.quantile(positive_curvature, 0.75))
        curvature_rows = curvature >= threshold
    transition_rows = (transition >= 0) & (transition <= 5)
    return {
        "overall": _subset_metrics(
            mean=mean,
            scale=scale,
            target=target,
            nll=nll,
            rows=all_rows,
        ),
        "phase": {
            name: _subset_metrics(
                mean=mean,
                scale=scale,
                target=target,
                nll=nll,
                rows=phase == name,
            )
            for name in _PHASES
        },
        "high_curvature": _subset_metrics(
            mean=mean,
            scale=scale,
            target=target,
            nll=nll,
            rows=curvature_rows,
        ),
        "transition_near": _subset_metrics(
            mean=mean,
            scale=scale,
            target=target,
            nll=nll,
            rows=transition_rows,
        ),
    }


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_checkpoint(
    *, checkpoint_dir: Path, expected_level: int, device: str
) -> tuple[InformationStateEstimator, InformationStateNormalization, dict]:
    root = checkpoint_dir.resolve()
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("format_id") != "causal_return_information_state_checkpoint"
        or manifest.get("level") != expected_level
    ):
        raise ValueError("information-state checkpoint identity is invalid")
    artifacts = manifest.get("artifacts")
    required = {"model.safetensors", "normalization.npz", "training_history.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError("information-state checkpoint artifact inventory is invalid")
    for name, digest in artifacts.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"information-state checkpoint hash mismatch: {name}")
    config = InformationStateConfig(**manifest["config"])
    model = InformationStateEstimator(config).to(device)
    model.load_state_dict(load_safetensors(root / "model.safetensors", device=device), strict=True)
    model.requires_grad_(False)
    model.eval()
    with np.load(root / "normalization.npz", allow_pickle=False) as source:
        normalization = InformationStateNormalization(
            robot_mean=np.array(source["robot_mean"], copy=True),
            robot_std=np.array(source["robot_std"], copy=True),
            object_mean=np.array(source["object_mean"], copy=True),
            object_std=np.array(source["object_std"], copy=True),
        )
    return model, normalization, manifest


def evaluate_level_information_state(
    *,
    corpus,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str,
) -> Path:
    """Evaluate one frozen estimator on the episode-level validation split."""

    model, normalization, checkpoint_manifest = _load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_level=corpus.level,
        device=device,
    )
    dataset = InformationStateDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=normalization,
    )
    if not dataset:
        raise ValueError("information-state evaluation requires validation samples")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model.config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_information_state_items,
    )
    means = []
    scales = []
    targets = []
    nlls = []
    episode_ids = []
    scene_seeds = []
    source_ticks = []
    phases = []
    curvatures = []
    transitions = []
    with torch.inference_mode():
        for batch in loader:
            vision = batch.vision_history.to(device)
            robot = batch.robot_history.to(device)
            times = batch.history_time_ms.to(device)
            valid = batch.history_valid_mask.to(device)
            target_normalized = batch.object_state_target.to(device)
            estimate = model(
                vision_history=vision,
                robot_history=robot,
                history_time_ms=times,
                history_valid_mask=valid,
            )
            scale_normalized = torch.exp(estimate.object_state_log_scale)
            residual = (target_normalized - estimate.object_state_mean) / scale_normalized
            nll = torch.mean(
                0.5 * residual.square() + estimate.object_state_log_scale,
                dim=-1,
            )
            object_std = torch.from_numpy(np.array(normalization.object_std, copy=True)).to(device)
            object_mean = torch.from_numpy(
                np.array(normalization.object_mean, copy=True)
            ).to(device)
            means.append(
                (estimate.object_state_mean * object_std + object_mean).cpu().numpy()
            )
            scales.append((scale_normalized * object_std).cpu().numpy())
            targets.append((target_normalized * object_std + object_mean).cpu().numpy())
            nlls.append(nll.cpu().numpy())
            episode_ids.extend(batch.episode_id)
            scene_seeds.extend(batch.scene_seed)
            source_ticks.extend(batch.source_tick)
            phases.extend(batch.source_phase)
            curvatures.append(batch.motion_curvature.numpy())
            transitions.append(batch.motion_transition_distance_ticks.numpy())
    mean_array = np.concatenate(means)
    scale_array = np.concatenate(scales)
    target_array = np.concatenate(targets)
    nll_array = np.concatenate(nlls)
    curvature_array = np.concatenate(curvatures)
    transition_array = np.concatenate(transitions)
    phase_array = np.asarray(phases)
    metrics = compute_information_state_metrics(
        object_state_mean=mean_array,
        object_state_scale=scale_array,
        object_state_target=target_array,
        source_phase=phase_array,
        motion_curvature=curvature_array,
        motion_transition_distance_ticks=transition_array,
        normalized_nll=nll_array,
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"information-state evaluation output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        with (building / "predictions.npz").open("xb") as handle:
            np.savez(
                handle,
                episode_id=np.asarray(episode_ids),
                scene_seed=np.asarray(scene_seeds, dtype=np.int64),
                source_tick=np.asarray(source_ticks, dtype=np.int64),
                source_phase=phase_array,
                motion_curvature=curvature_array,
                motion_transition_distance_ticks=transition_array,
                object_state_mean=mean_array,
                object_state_scale=scale_array,
                object_state_target=target_array,
                normalized_nll=nll_array,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(building / "metrics.json", metrics)
        artifacts = {
            name: sha256_file(building / name)
            for name in ("predictions.npz", "metrics.json")
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_information_state_evaluation",
                "level": corpus.level,
                "context_count": len(dataset),
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint_dir.resolve() / "manifest.json"
                ),
                "checkpoint_best_epoch": checkpoint_manifest["best_epoch"],
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
