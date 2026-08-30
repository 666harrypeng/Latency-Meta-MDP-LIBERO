"""Physical metrics and matched baselines for motion-aware histories."""

from __future__ import annotations

import os
import shutil
import statistics
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryConfig,
)
from latency_meta_mdp.belief.causal_return.motion_aware_training import (
    MotionAwareDataset,
    MotionAwareNormalization,
    _fsync_directory,
    _rename_directory_no_replace,
    _write_json,
    collate_motion_aware_items,
)
from latency_meta_mdp.belief.causal_return.spatiotemporal_history import (
    MotionAwareHistoryEncoder,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit

_PHASES = ("pregrasp", "approach", "close", "lift")


def _subset_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
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
            "invalid_value_count": 0,
        }
    selected_prediction = prediction[rows]
    selected_target = target[rows]
    error = selected_prediction - selected_target
    position_norm = np.linalg.norm(error[:, :3], axis=-1)
    velocity_norm = np.linalg.norm(error[:, 3:], axis=-1)
    return {
        "context_count": count,
        "object_position_rmse_m": float(np.sqrt(np.mean(np.square(position_norm)))),
        "object_position_mae_m": float(np.mean(position_norm)),
        "object_velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_norm)))),
        "object_velocity_mae_m_s": float(np.mean(velocity_norm)),
        "signed_position_bias_m": np.mean(error[:, :3], axis=0).tolist(),
        "signed_velocity_bias_m_s": np.mean(error[:, 3:], axis=0).tolist(),
        "invalid_value_count": int(
            selected_prediction.size
            + selected_target.size
            - np.count_nonzero(np.isfinite(selected_prediction))
            - np.count_nonzero(np.isfinite(selected_target))
        ),
    }


def compute_motion_aware_metrics(
    *,
    object_state_prediction: np.ndarray,
    object_state_target: np.ndarray,
    source_phase: np.ndarray,
    transition_offset_ticks: np.ndarray,
    transition_offset_valid: np.ndarray,
) -> dict:
    """Compute SI-unit metrics with signed L3 transition partitions."""

    prediction = np.asarray(object_state_prediction, dtype=np.float64)
    target = np.asarray(object_state_target, dtype=np.float64)
    phase = np.asarray(source_phase).astype(str)
    offset = np.asarray(transition_offset_ticks, dtype=np.int64)
    valid = np.asarray(transition_offset_valid, dtype=np.bool_)
    count = len(prediction)
    if (
        prediction.shape != (count, 6)
        or target.shape != (count, 6)
        or phase.shape != (count,)
        or offset.shape != (count,)
        or valid.shape != (count,)
        or any(value not in _PHASES for value in phase)
    ):
        raise ValueError("motion-aware metric arrays have invalid shapes or phases")
    all_rows = np.ones(count, dtype=bool)
    regimes = {
        "before": valid & (offset >= -5) & (offset <= -1),
        "exact": valid & (offset == 0),
        "after_1": valid & (offset == 1),
        "after_2_to_5": valid & (offset >= 2) & (offset <= 5),
        "far": (~valid) | (np.abs(offset) > 5),
    }
    membership = np.stack(tuple(regimes.values()), axis=1)
    if count and not np.all(np.sum(membership, axis=1) == 1):
        raise ValueError("motion-aware transition regimes must partition every context")
    return {
        "overall": _subset_metrics(
            prediction=prediction,
            target=target,
            rows=all_rows,
        ),
        "phase": {
            name: _subset_metrics(
                prediction=prediction,
                target=target,
                rows=phase == name,
            )
            for name in _PHASES
        },
        "transition": {
            name: _subset_metrics(
                prediction=prediction,
                target=target,
                rows=rows,
            )
            for name, rows in regimes.items()
        },
    }


@dataclass(frozen=True)
class MatchedBaseline:
    episode_id: np.ndarray
    source_tick: np.ndarray
    source_phase: np.ndarray
    candidate_prediction: np.ndarray
    baseline_prediction: np.ndarray
    candidate_target: np.ndarray
    baseline_target: np.ndarray


def _keys(values: Mapping[str, np.ndarray]) -> list[tuple[str, int]]:
    episode_id = np.asarray(values["episode_id"]).astype(str)
    source_tick = np.asarray(values["source_tick"], dtype=np.int64)
    if episode_id.shape != source_tick.shape or episode_id.ndim != 1:
        raise ValueError("matched baseline identity arrays have invalid shapes")
    keys = list(zip(episode_id.tolist(), source_tick.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("matched baseline identity contains duplicate contexts")
    return keys


def match_baseline_contexts(
    *,
    candidate: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    candidate_source_sha256: str,
    baseline_source_sha256: str,
) -> MatchedBaseline:
    """Bind candidate/baseline predictions to identical canonical physical contexts."""

    if (
        len(candidate_source_sha256) != 64
        or candidate_source_sha256 != baseline_source_sha256
    ):
        raise ValueError("matched baseline source manifest identity differs")
    required = {
        "episode_id",
        "source_tick",
        "source_phase",
        "object_state_target",
        "object_state_prediction",
    }
    if not required.issubset(candidate) or not required.issubset(baseline):
        raise ValueError("matched baseline arrays are incomplete")
    candidate_keys = _keys(candidate)
    baseline_keys = _keys(baseline)
    if set(candidate_keys) != set(baseline_keys):
        raise ValueError("matched baseline context identity differs")
    candidate_order = np.asarray(sorted(range(len(candidate_keys)), key=candidate_keys.__getitem__))
    baseline_lookup = {key: index for index, key in enumerate(baseline_keys)}
    sorted_keys = [candidate_keys[index] for index in candidate_order]
    baseline_order = np.asarray([baseline_lookup[key] for key in sorted_keys])
    candidate_target = np.asarray(candidate["object_state_target"])
    baseline_target = np.asarray(baseline["object_state_target"])
    if candidate_target.ndim != 2 or candidate_target.shape[1:] != (6,):
        raise ValueError("matched baseline canonical target has an invalid shape")
    try:
        np.testing.assert_allclose(
            baseline_target[baseline_order],
            candidate_target[candidate_order],
            rtol=1e-6,
            atol=1e-7,
        )
    except AssertionError as error:
        raise ValueError("matched baseline target differs from canonical state") from error
    candidate_phase = np.asarray(candidate["source_phase"]).astype(str)[candidate_order]
    baseline_phase = np.asarray(baseline["source_phase"]).astype(str)[baseline_order]
    if not np.array_equal(candidate_phase, baseline_phase):
        raise ValueError("matched baseline phase identity differs")
    return MatchedBaseline(
        episode_id=np.asarray([key[0] for key in sorted_keys]),
        source_tick=np.asarray([key[1] for key in sorted_keys], dtype=np.int64),
        source_phase=candidate_phase,
        candidate_prediction=np.asarray(candidate["object_state_prediction"])[candidate_order],
        baseline_prediction=np.asarray(baseline["object_state_prediction"])[baseline_order],
        candidate_target=candidate_target[candidate_order],
        baseline_target=baseline_target[baseline_order],
    )


def paired_episode_bootstrap(
    *,
    episode_id: np.ndarray,
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    seed: int,
    resample_count: int,
) -> dict:
    """Bootstrap paired improvement by resampling whole validation episodes."""

    episodes = np.asarray(episode_id).astype(str)
    baseline = np.asarray(baseline_error, dtype=np.float64)
    candidate = np.asarray(candidate_error, dtype=np.float64)
    if (
        episodes.ndim != 1
        or baseline.shape != episodes.shape
        or candidate.shape != episodes.shape
        or len(episodes) == 0
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(candidate))
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(resample_count, bool)
        or not isinstance(resample_count, int)
        or resample_count <= 0
    ):
        raise ValueError("paired episode bootstrap inputs are invalid")
    unique = np.unique(episodes)
    if len(unique) < 2:
        raise ValueError("paired episode bootstrap requires at least two episodes")
    improvement = baseline - candidate
    sums = np.asarray([np.sum(improvement[episodes == name]) for name in unique])
    counts = np.asarray([np.count_nonzero(episodes == name) for name in unique])
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique), size=(resample_count, len(unique)))
    values = np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "seed": seed,
        "resample_count": resample_count,
        "episode_count": len(unique),
        "mean_improvement": float(np.mean(improvement)),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "scientific_improvement_pass": bool(lower > 0.0),
    }


def evaluate_motion_aware_admission(
    *,
    level: int,
    candidate_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
    episode_id: np.ndarray,
    transition_offset_ticks: np.ndarray,
    transition_offset_valid: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    require_formal_episode_count: bool = False,
) -> dict:
    """Apply matched deterministic Module A gates and the L3 paired bootstrap."""

    if level not in (1, 2, 3):
        raise ValueError("motion-aware admission level must be 1, 2, or 3")
    blockers = []

    def no_regression(*, subset: str, metric: str, blocker: str) -> None:
        candidate_subset = (
            candidate_metrics["overall"]
            if subset == "overall"
            else candidate_metrics["transition"][subset]
        )
        baseline_subset = (
            baseline_metrics["overall"]
            if subset == "overall"
            else baseline_metrics["transition"][subset]
        )
        if candidate_subset[metric] > baseline_subset[metric] + 1e-12:
            blockers.append(blocker)

    no_regression(
        subset="overall",
        metric="object_position_rmse_m",
        blocker="overall_position_regression",
    )
    no_regression(
        subset="overall",
        metric="object_velocity_rmse_m_s",
        blocker="overall_velocity_regression",
    )
    bootstrap = None
    if level == 3:
        for subset in ("far", "before"):
            no_regression(
                subset=subset,
                metric="object_position_rmse_m",
                blocker=f"{subset}_position_regression",
            )
            no_regression(
                subset=subset,
                metric="object_velocity_rmse_m_s",
                blocker=f"{subset}_velocity_regression",
            )
        after_velocity = candidate_metrics["transition"]["after_2_to_5"][
            "object_velocity_rmse_m_s"
        ]
        far_velocity = candidate_metrics["transition"]["far"][
            "object_velocity_rmse_m_s"
        ]
        if after_velocity > 1.5 * far_velocity + 1e-12:
            blockers.append("after_to_far_velocity_ratio")
        offset = np.asarray(transition_offset_ticks, dtype=np.int64)
        valid = np.asarray(transition_offset_valid, dtype=np.bool_)
        after = valid & (offset >= 2) & (offset <= 5)
        candidate = np.asarray(candidate_prediction, dtype=np.float64)
        baseline = np.asarray(baseline_prediction, dtype=np.float64)
        expected = np.asarray(target, dtype=np.float64)
        if (
            candidate.shape != baseline.shape
            or candidate.shape != expected.shape
            or candidate.shape != (len(offset), 6)
            or np.count_nonzero(after) == 0
        ):
            raise ValueError("motion-aware L3 admission arrays are invalid")
        candidate_error = np.linalg.norm(candidate[after, 3:] - expected[after, 3:], axis=1)
        baseline_error = np.linalg.norm(baseline[after, 3:] - expected[after, 3:], axis=1)
        bootstrap = paired_episode_bootstrap(
            episode_id=np.asarray(episode_id)[after],
            baseline_error=baseline_error,
            candidate_error=candidate_error,
            seed=20260830,
            resample_count=10_000,
        )
        if require_formal_episode_count and bootstrap["episode_count"] != 20:
            blockers.append("formal_validation_episode_count")
        if not bootstrap["scientific_improvement_pass"]:
            blockers.append("after_velocity_bootstrap")
    invalid = int(candidate_metrics["overall"]["invalid_value_count"])
    if invalid:
        blockers.append("invalid_predictions")
    return {
        "scientific_gate_pass": not blockers,
        "blockers": blockers,
        "paired_after_velocity_bootstrap": bootstrap,
    }


def _load_checkpoint(
    *,
    checkpoint_dir: Path,
    expected_level: int,
    device: str,
    model_factory: Callable[[MotionAwareHistoryConfig], nn.Module],
) -> tuple[nn.Module, MotionAwareNormalization, dict]:
    root = checkpoint_dir.resolve()
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("format_id") != "causal_return_motion_aware_history_checkpoint"
        or manifest.get("level") != expected_level
    ):
        raise ValueError("motion-aware checkpoint identity is invalid")
    artifacts = manifest.get("artifacts")
    required = {"model.safetensors", "normalization.npz", "training_history.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError("motion-aware checkpoint inventory is invalid")
    for name, digest in artifacts.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"motion-aware checkpoint hash mismatch: {name}")
    config = MotionAwareHistoryConfig(**manifest["config"])
    model = model_factory(config).to(device)
    model.load_state_dict(
        load_safetensors(root / "model.safetensors", device=device),
        strict=True,
    )
    model.requires_grad_(False)
    model.eval()
    with np.load(root / "normalization.npz", allow_pickle=False) as source:
        normalization = MotionAwareNormalization(
            robot_mean=np.array(source["robot_mean"], copy=True),
            robot_std=np.array(source["robot_std"], copy=True),
            position_mean=np.array(source["position_mean"], copy=True),
            position_std=np.array(source["position_std"], copy=True),
            velocity_mean=np.array(source["velocity_mean"], copy=True),
            velocity_std=np.array(source["velocity_std"], copy=True),
        )
    return model, normalization, manifest


def _load_json(path: Path) -> dict:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def benchmark_motion_aware_history_encoder(
    *,
    model: nn.Module,
    config: MotionAwareHistoryConfig,
    device: str,
    batch_sizes: tuple[int, ...],
    warmup_count: int,
    trial_count: int,
) -> dict:
    """Benchmark frozen forward latency without changing architecture or inputs."""

    if (
        not batch_sizes
        or batch_sizes != tuple(sorted(set(batch_sizes)))
        or any(size <= 0 for size in batch_sizes)
        or warmup_count <= 0
        or trial_count <= 0
    ):
        raise ValueError("motion-aware runtime benchmark settings are invalid")
    runtime_device = torch.device(device)
    model = model.to(runtime_device).eval()
    results = {}
    for batch in batch_sizes:
        vision = torch.zeros(
            (
                batch,
                config.history_sample_count,
                config.camera_count,
                config.patch_token_count,
                config.vision_feature_dim,
            ),
            dtype=torch.float16,
            device=runtime_device,
        )
        robot = torch.zeros(
            (batch, config.history_sample_count, config.robot_state_dim),
            dtype=torch.float32,
            device=runtime_device,
        )
        valid = torch.ones(
            (batch, config.history_sample_count),
            dtype=torch.bool,
            device=runtime_device,
        )
        with torch.inference_mode():
            for _ in range(warmup_count):
                model(
                    vision_history=vision,
                    robot_history_normalized=robot,
                    history_valid_mask=valid,
                )
            if runtime_device.type == "cuda":
                torch.cuda.synchronize(runtime_device)
                torch.cuda.reset_peak_memory_stats(runtime_device)
            timings = []
            for _ in range(trial_count):
                if runtime_device.type == "cuda":
                    with torch.cuda.device(runtime_device):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        model(
                            vision_history=vision,
                            robot_history_normalized=robot,
                            history_valid_mask=valid,
                        )
                        end.record()
                    torch.cuda.synchronize(runtime_device)
                    timings.append(float(start.elapsed_time(end)))
                else:
                    started = time.perf_counter()
                    model(
                        vision_history=vision,
                        robot_history_normalized=robot,
                        history_valid_mask=valid,
                    )
                    timings.append((time.perf_counter() - started) * 1000.0)
        results[str(batch)] = {
            "p50_ms": statistics.median(timings),
            "p95_ms": _percentile(timings, 0.95),
            "mean_ms": statistics.mean(timings),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(runtime_device))
                if runtime_device.type == "cuda"
                else 0
            ),
        }
    return {
        "device": device,
        "warmup_count": warmup_count,
        "trial_count": trial_count,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batches": results,
    }


def _candidate_metadata_order(
    *,
    matched: MatchedBaseline,
    episode_id: list[str],
    source_tick: list[int],
    transition_offset_ticks: list[int],
    transition_offset_valid: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (name, tick): (offset, valid)
        for name, tick, offset, valid in zip(
            episode_id,
            source_tick,
            transition_offset_ticks,
            transition_offset_valid,
        )
    }
    values = [
        lookup[(name, int(tick))]
        for name, tick in zip(matched.episode_id.tolist(), matched.source_tick.tolist())
    ]
    return (
        np.asarray([value[0] for value in values], dtype=np.int64),
        np.asarray([value[1] for value in values], dtype=np.bool_),
    )


def evaluate_level_motion_aware_history(
    *,
    corpus,
    checkpoint_dir: Path,
    baseline: Mapping[str, np.ndarray],
    baseline_source_sha256: str,
    output_dir: Path,
    device: str,
    model_factory: Callable[[MotionAwareHistoryConfig], nn.Module] = MotionAwareHistoryEncoder,
    runtime_warmup_count: int = 50,
    runtime_trial_count: int = 300,
    runtime_batch_sizes: tuple[int, ...] = (4, 8),
    evaluation_artifact_eligible: bool | None = None,
) -> Path:
    """Evaluate one frozen checkpoint and publish a matched physical gate."""

    model, normalization, checkpoint_manifest = _load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        expected_level=corpus.level,
        device=device,
        model_factory=model_factory,
    )
    dataset = MotionAwareDataset(
        corpus=corpus,
        split=ProbeSplit.VALIDATION,
        normalization=normalization,
    )
    if not dataset:
        raise ValueError("motion-aware evaluation requires validation samples")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=checkpoint_manifest["config"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motion_aware_items,
    )
    predictions = []
    targets = []
    episode_ids = []
    scene_seeds = []
    source_ticks = []
    phases = []
    offsets = []
    transition_valid = []
    with torch.inference_mode():
        for batch in loader:
            estimate = model(
                vision_history=batch.vision_history.to(device),
                robot_history_normalized=batch.robot_history_normalized.to(device),
                history_valid_mask=batch.history_valid_mask.to(device),
            )
            position = (
                estimate.object_position_normalized.cpu().numpy()
                * normalization.position_std
                + normalization.position_mean
            )
            velocity = (
                estimate.object_velocity_normalized.cpu().numpy()
                * normalization.velocity_std
                + normalization.velocity_mean
            )
            predictions.append(np.concatenate((position, velocity), axis=1))
            targets.append(
                np.concatenate(
                    (
                        batch.object_position_physical.numpy(),
                        batch.object_velocity_physical.numpy(),
                    ),
                    axis=1,
                )
            )
            episode_ids.extend(batch.episode_id)
            scene_seeds.extend(batch.scene_seed)
            source_ticks.extend(batch.source_tick)
            phases.extend(batch.source_phase)
            offsets.extend(batch.transition_offset_ticks.tolist())
            transition_valid.extend(batch.transition_offset_valid.tolist())
    candidate_prediction = np.concatenate(predictions)
    candidate_target = np.concatenate(targets)
    candidate = {
        "episode_id": np.asarray(episode_ids),
        "source_tick": np.asarray(source_ticks, dtype=np.int64),
        "source_phase": np.asarray(phases),
        "object_state_target": candidate_target,
        "object_state_prediction": candidate_prediction,
    }
    source_sha = checkpoint_manifest["input_sha256"]["source_bulk_manifest"]
    matched = match_baseline_contexts(
        candidate=candidate,
        baseline=baseline,
        candidate_source_sha256=source_sha,
        baseline_source_sha256=baseline_source_sha256,
    )
    matched_offsets, matched_transition_valid = _candidate_metadata_order(
        matched=matched,
        episode_id=episode_ids,
        source_tick=source_ticks,
        transition_offset_ticks=offsets,
        transition_offset_valid=transition_valid,
    )
    candidate_metrics = compute_motion_aware_metrics(
        object_state_prediction=matched.candidate_prediction,
        object_state_target=matched.candidate_target,
        source_phase=matched.source_phase,
        transition_offset_ticks=matched_offsets,
        transition_offset_valid=matched_transition_valid,
    )
    baseline_metrics = compute_motion_aware_metrics(
        object_state_prediction=matched.baseline_prediction,
        object_state_target=matched.candidate_target,
        source_phase=matched.source_phase,
        transition_offset_ticks=matched_offsets,
        transition_offset_valid=matched_transition_valid,
    )
    admission = evaluate_motion_aware_admission(
        level=corpus.level,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        episode_id=matched.episode_id,
        transition_offset_ticks=matched_offsets,
        transition_offset_valid=matched_transition_valid,
        candidate_prediction=matched.candidate_prediction,
        baseline_prediction=matched.baseline_prediction,
        target=matched.candidate_target,
        require_formal_episode_count=not bool(
            checkpoint_manifest.get("bounded_review", False)
        ),
    )
    runtime = benchmark_motion_aware_history_encoder(
        model=model,
        config=MotionAwareHistoryConfig(**checkpoint_manifest["config"]),
        device=device,
        batch_sizes=runtime_batch_sizes,
        warmup_count=runtime_warmup_count,
        trial_count=runtime_trial_count,
    )
    artifact_eligible = bool(checkpoint_manifest["artifact_eligible"])
    if evaluation_artifact_eligible is not None:
        if not isinstance(evaluation_artifact_eligible, bool):
            raise TypeError("evaluation_artifact_eligible must be a boolean or None")
        artifact_eligible = artifact_eligible and evaluation_artifact_eligible
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"motion-aware evaluation output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        with (building / "predictions.npz").open("xb") as handle:
            np.savez(
                handle,
                episode_id=matched.episode_id,
                scene_seed=np.asarray(
                    [dict(zip(episode_ids, scene_seeds))[name] for name in matched.episode_id],
                    dtype=np.int64,
                ),
                source_tick=matched.source_tick,
                source_phase=matched.source_phase,
                transition_offset_ticks=matched_offsets,
                transition_offset_valid=matched_transition_valid,
                object_state_prediction=matched.candidate_prediction,
                object_state_target=matched.candidate_target,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(building / "metrics.json", candidate_metrics)
        _write_json(
            building / "comparison.json",
            {"baseline_metrics": baseline_metrics, "admission": admission},
        )
        _write_json(building / "runtime.json", runtime)
        artifacts = {
            name: sha256_file(building / name)
            for name in ("predictions.npz", "metrics.json", "comparison.json", "runtime.json")
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "causal_return_motion_aware_history_evaluation",
                "level": corpus.level,
                "context_count": len(matched.episode_id),
                "checkpoint_manifest_sha256": sha256_file(
                    checkpoint_dir.resolve() / "manifest.json"
                ),
                "baseline_source_sha256": baseline_source_sha256,
                "scientific_gate_pass": admission["scientific_gate_pass"],
                "scientific_blockers": admission["blockers"],
                "artifact_eligible": artifact_eligible,
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
