"""Frozen-Flow evaluation against paired action-buffer counterfactuals."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import load_level_feature_belief_corpus
from latency_meta_mdp.belief.flow.buffer_causality import (
    BufferCausalityCandidate,
    load_buffer_causality_config,
)
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.context_sampling import load_verified_flow_quality_level
from latency_meta_mdp.belief.flow.sampler import sample_flow_belief
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BufferCausalityMetrics:
    qpos_error: np.ndarray
    object_position_error: np.ndarray
    object_velocity_error: np.ndarray
    qpos_expert_control_error: np.ndarray
    qpos_conditioning_gain: np.ndarray
    qpos_effect_cosine: np.ndarray
    qpos_effect_cosine_valid: np.ndarray
    qpos_effect_magnitude_ratio: np.ndarray
    gt_object_position_effect: np.ndarray
    predicted_object_position_effect: np.ndarray
    object_effect_is_invariant: np.ndarray
    token_delta_rms: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.qpos_error).shape
        if len(shape) != 3:
            raise ValueError("buffer-causality metrics require context/branch/delay axes")
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name))
            expected = shape[:2] if name == "token_delta_rms" else shape
            if value.shape != expected:
                raise ValueError(f"buffer-causality metric {name} has invalid shape")
            if value.dtype != np.bool_ and not np.all(np.isfinite(value)):
                raise ValueError(f"buffer-causality metric {name} must be finite")
            object.__setattr__(self, name, _readonly(value))


def _rms(value: np.ndarray, *, axis: int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value), axis=axis))


def latency_weighted_context_mean(
    values: np.ndarray,
    latency_probabilities: np.ndarray,
) -> np.ndarray:
    """Weight delays within each context, then weight contexts equally."""

    array = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(latency_probabilities, dtype=np.float64)
    if (
        array.ndim != 3
        or probabilities.shape != (array.shape[0], array.shape[2])
        or not np.all(np.isfinite(array))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-9, rtol=0.0)
    ):
        raise ValueError("latency-weighted metric inputs are invalid")
    per_context = np.sum(array * probabilities[:, None, :], axis=-1)
    return per_context.mean(axis=0)


def _masked_latency_weighted_context_mean(
    values: np.ndarray,
    mask: np.ndarray,
    latency_probabilities: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    probabilities = np.asarray(latency_probabilities, dtype=np.float64)
    weights = probabilities[:, None, :] * valid
    denominator = weights.sum(axis=-1)
    numerator = np.sum(array * weights, axis=-1)
    result = np.zeros(array.shape[1], dtype=np.float64)
    for branch in range(array.shape[1]):
        contexts = denominator[:, branch] > 0.0
        if np.any(contexts):
            result[branch] = np.mean(
                numerator[contexts, branch] / denominator[contexts, branch]
            )
    return result


def compute_buffer_causality_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    belief_tokens: np.ndarray,
    branch_ids: tuple[str, ...],
) -> BufferCausalityMetrics:
    """Compute paired branch effects without averaging away delay or context."""

    samples = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    tokens = np.asarray(belief_tokens, dtype=np.float64)
    if (
        samples.ndim != 5
        or samples.shape[-1] != 22
        or truth.shape != samples.shape[:3] + (22,)
        or tokens.ndim != 4
        or tokens.shape[:2] != samples.shape[:2]
        or tuple(branch_ids)[0] != "expert"
        or len(branch_ids) != samples.shape[1]
    ):
        raise ValueError("buffer-causality prediction arrays have incompatible shapes")
    mean = samples.mean(axis=3)
    error = mean - truth
    qpos_error = _rms(error[..., :7], axis=-1)
    object_position_error = np.linalg.norm(error[..., 16:19], axis=-1)
    object_velocity_error = np.linalg.norm(error[..., 19:22], axis=-1)

    expert_mean = mean[:, :1]
    control_error = expert_mean - truth
    qpos_control_error = _rms(control_error[..., :7], axis=-1)
    qpos_gain = qpos_control_error - qpos_error

    gt_effect = truth - truth[:, :1]
    predicted_effect = mean - expert_mean
    gt_qpos = gt_effect[..., :7]
    predicted_qpos = predicted_effect[..., :7]
    gt_norm = np.linalg.norm(gt_qpos, axis=-1)
    predicted_norm = np.linalg.norm(predicted_qpos, axis=-1)
    cosine_valid = gt_norm > 1e-12
    cosine = np.zeros_like(gt_norm)
    cosine[cosine_valid] = np.sum(gt_qpos * predicted_qpos, axis=-1)[cosine_valid] / (
        gt_norm[cosine_valid] * predicted_norm[cosine_valid] + 1e-12
    )
    magnitude_ratio = np.zeros_like(gt_norm)
    magnitude_ratio[cosine_valid] = predicted_norm[cosine_valid] / gt_norm[cosine_valid]

    gt_object_effect = np.linalg.norm(gt_effect[..., 16:19], axis=-1)
    predicted_object_effect = np.linalg.norm(predicted_effect[..., 16:19], axis=-1)
    object_invariant = gt_object_effect <= 1e-9
    token_delta = tokens - tokens[:, :1]
    token_delta_rms = _rms(token_delta, axis=(-1, -2))
    return BufferCausalityMetrics(
        qpos_error=qpos_error,
        object_position_error=object_position_error,
        object_velocity_error=object_velocity_error,
        qpos_expert_control_error=qpos_control_error,
        qpos_conditioning_gain=qpos_gain,
        qpos_effect_cosine=cosine,
        qpos_effect_cosine_valid=cosine_valid,
        qpos_effect_magnitude_ratio=magnitude_ratio,
        gt_object_position_effect=gt_object_effect,
        predicted_object_position_effect=predicted_object_effect,
        object_effect_is_invariant=object_invariant,
        token_delta_rms=token_delta_rms,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_flow_run_eligibility(*, checkpoint_dir: Path, level: int) -> bool:
    """Verify that a level checkpoint is admitted by its parent multi-law run."""

    checkpoint = checkpoint_dir.resolve()
    run_manifest_path = checkpoint.parent / "manifest.json"
    level_manifest_path = checkpoint / "manifest.json"
    run = _load_json(run_manifest_path)
    relative = f"L{level}/manifest.json"
    artifacts = run.get("artifacts")
    if (
        run.get("format_id") != "flow_belief_multilaw_run_v2"
        or level not in run.get("levels", [])
        or run.get("level_manifests", {}).get(f"L{level}") != relative
        or not isinstance(artifacts, dict)
        or artifacts.get(relative) != sha256_file(level_manifest_path)
    ):
        raise ValueError("level checkpoint is not bound by its parent multi-law run")
    return run.get("eligible") is True and run.get("blockers") == []


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_simulation(
    path: Path,
) -> tuple[dict[str, Any], tuple[BufferCausalityCandidate, ...], dict[str, np.ndarray]]:
    manifest = _load_json(path)
    if manifest.get("format_id") != "flow_belief_buffer_counterfactual_sim_v1":
        raise ValueError("buffer-causality simulator manifest format is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "contexts.json",
        "counterfactuals.npz",
    }:
        raise ValueError("buffer-causality simulator artifact inventory is invalid")
    for name, digest in artifacts.items():
        artifact = path.parent / name
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"buffer-causality simulator artifact hash mismatch: {name}")
    raw_contexts = json.loads((path.parent / "contexts.json").read_text(encoding="utf-8"))
    if not isinstance(raw_contexts, list):
        raise ValueError("buffer-causality contexts must be a list")
    contexts = tuple(BufferCausalityCandidate(**row) for row in raw_contexts)
    with np.load(path.parent / "counterfactuals.npz", allow_pickle=False) as source:
        required = {
            "branch_actions",
            "target_states",
            "handoff_state",
            "replay_max_abs",
            "expert_reference_max_abs",
        }
        if set(source.files) != required:
            raise ValueError("buffer-causality simulator arrays are invalid")
        arrays = {name: np.array(source[name], copy=True) for name in required}
    return manifest, contexts, arrays


def _summary(
    *,
    contexts: tuple[BufferCausalityCandidate, ...],
    branch_ids: tuple[str, ...],
    metrics: BufferCausalityMetrics,
    latency_probabilities: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"branches": {}, "phases": {}}
    weighted_qpos_error = latency_weighted_context_mean(
        metrics.qpos_error,
        latency_probabilities,
    )
    weighted_object_position_error = latency_weighted_context_mean(
        metrics.object_position_error,
        latency_probabilities,
    )
    weighted_object_velocity_error = latency_weighted_context_mean(
        metrics.object_velocity_error,
        latency_probabilities,
    )
    weighted_qpos_gain = latency_weighted_context_mean(
        metrics.qpos_conditioning_gain,
        latency_probabilities,
    )
    weighted_false_coupling = _masked_latency_weighted_context_mean(
        metrics.predicted_object_position_effect,
        metrics.object_effect_is_invariant,
        latency_probabilities,
    )
    for branch_index, branch_id in enumerate(branch_ids):
        valid = metrics.qpos_effect_cosine_valid[:, branch_index]
        result["branches"][branch_id] = {
            "qpos_error_rad": float(np.mean(metrics.qpos_error[:, branch_index])),
            "qpos_error_latency_weighted_rad": float(weighted_qpos_error[branch_index]),
            "object_position_error_m": float(
                np.mean(metrics.object_position_error[:, branch_index])
            ),
            "object_position_error_latency_weighted_m": float(
                weighted_object_position_error[branch_index]
            ),
            "object_velocity_error_m_s": float(
                np.mean(metrics.object_velocity_error[:, branch_index])
            ),
            "object_velocity_error_latency_weighted_m_s": float(
                weighted_object_velocity_error[branch_index]
            ),
            "qpos_conditioning_gain_rad": float(
                np.mean(metrics.qpos_conditioning_gain[:, branch_index])
            ),
            "qpos_conditioning_gain_latency_weighted_rad": float(
                weighted_qpos_gain[branch_index]
            ),
            "qpos_effect_cosine": (
                float(np.mean(metrics.qpos_effect_cosine[:, branch_index][valid]))
                if np.any(valid)
                else 0.0
            ),
            "token_delta_rms": float(np.mean(metrics.token_delta_rms[:, branch_index])),
            "invariant_object_false_coupling_m": (
                float(
                    np.mean(
                        metrics.predicted_object_position_effect[:, branch_index][
                            metrics.object_effect_is_invariant[:, branch_index]
                        ]
                    )
                )
                if np.any(metrics.object_effect_is_invariant[:, branch_index])
                else 0.0
            ),
            "invariant_object_false_coupling_latency_weighted_m": float(
                weighted_false_coupling[branch_index]
            ),
        }
    phases = tuple(sorted({row.phase for row in contexts}))
    for phase in phases:
        rows = np.asarray([row.phase == phase for row in contexts])
        phase_qpos_error = latency_weighted_context_mean(
            metrics.qpos_error[rows],
            latency_probabilities[rows],
        )
        phase_object_position_error = latency_weighted_context_mean(
            metrics.object_position_error[rows],
            latency_probabilities[rows],
        )
        phase_qpos_gain = latency_weighted_context_mean(
            metrics.qpos_conditioning_gain[rows],
            latency_probabilities[rows],
        )
        result["phases"][phase] = {
            branch_id: {
                "qpos_error_rad": float(np.mean(metrics.qpos_error[rows, branch_index])),
                "qpos_error_latency_weighted_rad": float(
                    phase_qpos_error[branch_index]
                ),
                "object_position_error_m": float(
                    np.mean(metrics.object_position_error[rows, branch_index])
                ),
                "object_position_error_latency_weighted_m": float(
                    phase_object_position_error[branch_index]
                ),
                "qpos_conditioning_gain_rad": float(
                    np.mean(metrics.qpos_conditioning_gain[rows, branch_index])
                ),
                "qpos_conditioning_gain_latency_weighted_rad": float(
                    phase_qpos_gain[branch_index]
                ),
            }
            for branch_index, branch_id in enumerate(branch_ids)
        }
    return result


def evaluate_buffer_counterfactual_level(
    *,
    project_root: Path,
    simulation_manifest: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    checkpoint_dir: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    split_config_path: Path,
    analysis_config_path: Path,
    output_dir: Path,
    level: int,
    device: str,
) -> Path:
    """Evaluate one frozen level checkpoint on exact simulator branches."""

    root = project_root.resolve()
    paths = {
        "simulation_manifest": simulation_manifest.resolve(),
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "checkpoint_dir": checkpoint_dir.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "nominal_law": nominal_law_path.resolve(),
        "family_config": family_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "analysis_config": analysis_config_path.resolve(),
    }
    for name, path in paths.items():
        if name == "checkpoint_dir":
            if not path.is_dir():
                raise FileNotFoundError(f"buffer-causality checkpoint directory is missing: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"buffer-causality input is missing: {path}")
    config = load_buffer_causality_config(paths["analysis_config"])
    sim_manifest, all_contexts, sim_arrays = _load_simulation(paths["simulation_manifest"])
    indices = [index for index, row in enumerate(all_contexts) if row.level == level]
    if not indices:
        raise ValueError("buffer-causality simulator artifact has no requested level")
    contexts = tuple(all_contexts[index] for index in indices)
    branch_ids = tuple(sim_manifest.get("branch_ids", ()))
    if branch_ids != config.branch_ids:
        raise ValueError("buffer-causality branch inventory does not match analysis config")
    branch_actions = sim_arrays["branch_actions"][indices]
    targets = sim_arrays["target_states"][indices]

    spec = load_vision_encoder_spec(paths["vision_config"])
    corpus = load_level_feature_belief_corpus(
        project_root=root,
        source_bulk_manifest=paths["source_bulk_manifest"],
        cache_run_manifest=paths["cache_run_manifest"],
        expected_spec=spec,
        temporal_config_path=paths["temporal_config"],
        latency_law_path=paths["nominal_law"],
        latency_law_family_path=paths["family_config"],
        split_plan_path=paths["split_config"],
        level=level,
    )
    reference_by_identity = {}
    for offset, (record_offset, index_offset) in enumerate(
        corpus.sample_references[ProbeSplit.VALIDATION]
    ):
        record = corpus.records[record_offset]
        source_tick = record.indices[index_offset].source_tick
        reference_by_identity[(record.episode_id, record.scene_seed, source_tick)] = offset
    checkpoint_manifest = _load_json(paths["checkpoint_dir"] / "manifest.json")
    checkpoint_eligible = verify_flow_run_eligibility(
        checkpoint_dir=paths["checkpoint_dir"],
        level=level,
    )
    flow_config = FlowBeliefConfig(**checkpoint_manifest["config"])
    loaded = load_verified_flow_quality_level(
        checkpoint_dir=paths["checkpoint_dir"],
        flow_config=flow_config,
        expected_level=level,
        device=device,
    )
    normalization = loaded.normalization
    predictions = []
    tokens = []
    latency_probabilities = []
    for context_index, context_identity in enumerate(contexts):
        identity = (
            context_identity.episode_id,
            context_identity.scene_seed,
            context_identity.source_tick,
        )
        if identity not in reference_by_identity:
            raise ValueError(
                f"buffer-causality context is absent from validation corpus: {identity}"
            )
        context = corpus.materialize(
            ProbeSplit.VALIDATION,
            reference_by_identity[identity],
        )
        if not np.array_equal(context.remaining_actions, branch_actions[context_index, 0]):
            raise ValueError("buffer-causality expert branch does not match Belief context")
        print(
            f"[buffer-causality-eval] L{level} context {context_index + 1}/{len(contexts)} "
            f"seed={context_identity.scene_seed} tick={context_identity.source_tick} "
            f"phase={context_identity.phase}",
            flush=True,
        )
        branch_count = len(branch_ids)
        vision = np.repeat(context.vision_history[None], branch_count, axis=0)
        proprio = np.repeat(
            (
                (context.robot_proprio_history - normalization.proprio_mean)
                / normalization.proprio_std
            )[None],
            branch_count,
            axis=0,
        ).astype(np.float32)
        actions = (
            (branch_actions[context_index] - normalization.action_mean)
            / normalization.action_std
        ).astype(np.float32)
        latency = np.repeat(
            context.latency_probabilities[None],
            branch_count,
            axis=0,
        ).astype(np.float32)
        rng = np.random.default_rng(
            config.sampling_seed
            + level * 1_000_000
            + context_identity.scene_seed * 1_000
            + context_identity.source_tick
        )
        base_noise = rng.standard_normal((1, 20, config.sample_count, 22), dtype=np.float32)
        noise = np.repeat(base_noise, branch_count, axis=0)
        delays = np.repeat(np.arange(1, 21, dtype=np.int64)[None], branch_count, axis=0)
        with torch.inference_mode():
            belief = loaded.model.encoder(
                vision_history=torch.from_numpy(np.array(vision, copy=True)).to(device),
                proprio_history=torch.from_numpy(proprio).to(device),
                remaining_actions=torch.from_numpy(actions).to(device),
                latency_probabilities=torch.from_numpy(latency).to(device),
            )
            normalized = sample_flow_belief(
                vector_field=loaded.model.vector_field,
                belief_tokens=belief,
                delay_ticks=torch.from_numpy(delays).to(device),
                noise=torch.from_numpy(noise).to(device),
                solver=config.solver,
                step_count=config.solver_step_count,
            )
        physical = (
            normalized.cpu().numpy() * normalization.target_std + normalization.target_mean
        )
        predictions.append(physical)
        tokens.append(belief.cpu().numpy())
        latency_probabilities.append(np.asarray(context.latency_probabilities, dtype=np.float64))
    prediction_array = np.stack(predictions).astype(np.float32)
    token_array = np.stack(tokens).astype(np.float32)
    latency_array = np.stack(latency_probabilities)
    metrics = compute_buffer_causality_metrics(
        predictions=prediction_array,
        targets=targets,
        belief_tokens=token_array,
        branch_ids=branch_ids,
    )
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"buffer-causality evaluation output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "contexts.json", [asdict(row) for row in contexts])
        _write_json(
            building / "summary.json",
            _summary(
                contexts=contexts,
                branch_ids=branch_ids,
                metrics=metrics,
                latency_probabilities=latency_array,
            ),
        )
        with (building / "evaluation.npz").open("xb") as handle:
            np.savez(
                handle,
                predictions=prediction_array,
                targets=targets,
                belief_tokens=token_array,
                latency_probabilities=latency_array,
                **{name: getattr(metrics, name) for name in metrics.__dataclass_fields__},
            )
            handle.flush()
            os.fsync(handle.fileno())
        artifacts = {
            name: sha256_file(building / name)
            for name in ("contexts.json", "summary.json", "evaluation.npz")
        }
        provenance = collect_implementation_provenance(root)
        eligible = (
            not provenance.dirty
            and sim_manifest.get("eligible") is True
            and checkpoint_eligible
        )
        blockers = []
        if provenance.dirty:
            blockers.append("implementation_dirty")
        if sim_manifest.get("eligible") is not True:
            blockers.append("simulation_artifact_ineligible")
        if not checkpoint_eligible:
            blockers.append("checkpoint_ineligible")
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_buffer_causality_evaluation_v1",
            "eligible": eligible,
            "blockers": blockers,
            "level": level,
            "context_count": len(contexts),
            "branch_ids": list(branch_ids),
            "sample_count": config.sample_count,
            "solver": config.solver,
            "solver_step_count": config.solver_step_count,
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "simulation_manifest_sha256": sha256_file(paths["simulation_manifest"]),
            "checkpoint_model_sha256": sha256_file(paths["checkpoint_dir"] / "model.safetensors"),
            "input_sha256": {
                name: sha256_file(path)
                for name, path in paths.items()
                if name not in {"checkpoint_dir"}
            },
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        import shutil

        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
