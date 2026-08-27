"""Verified deterministic Flow samples for selected quality contexts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.model import FlowBeliefModel
from latency_meta_mdp.belief.flow.quality_config import FlowBeliefQualitySampleConfig
from latency_meta_mdp.belief.flow.quality_selection import FormalFlowSummary
from latency_meta_mdp.belief.flow.quality_types import (
    QualitySampleBundle,
    QualitySelection,
)
from latency_meta_mdp.belief.flow.sampler import sample_flow_belief
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class LoadedFlowQualityLevel:
    model: FlowBeliefModel
    normalization: FlowBeliefNormalization
    level_manifest: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_flow_belief_normalization(path: Path) -> FlowBeliefNormalization:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "proprio_mean",
            "proprio_std",
            "action_mean",
            "action_std",
            "target_mean",
            "target_std",
        }
        if set(source.files) != required:
            raise ValueError("Flow quality normalization fields are invalid")
        normalization = FlowBeliefNormalization(
            proprio_mean=np.array(source["proprio_mean"], dtype=np.float32, copy=True),
            proprio_std=np.array(source["proprio_std"], dtype=np.float32, copy=True),
            action_mean=np.array(source["action_mean"], dtype=np.float32, copy=True),
            action_std=np.array(source["action_std"], dtype=np.float32, copy=True),
            target_mean=np.array(source["target_mean"], dtype=np.float32, copy=True),
            target_std=np.array(source["target_std"], dtype=np.float32, copy=True),
        )
    expected_shapes = {
        "proprio_mean": (16,),
        "proprio_std": (16,),
        "action_mean": (7,),
        "action_std": (7,),
        "target_mean": (22,),
        "target_std": (22,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(normalization, name))
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Flow quality normalization {name} is invalid")
    if (
        np.any(normalization.proprio_std <= 0.0)
        or np.any(normalization.action_std <= 0.0)
        or np.any(normalization.target_std <= 0.0)
    ):
        raise ValueError("Flow quality normalization scales must be positive")
    return normalization


def load_verified_flow_quality_level(
    *,
    checkpoint_dir: Path,
    flow_config: FlowBeliefConfig,
    expected_level: int,
    device: str,
) -> LoadedFlowQualityLevel:
    checkpoint = checkpoint_dir.resolve()
    manifest = _load_json(checkpoint / "manifest.json")
    if manifest.get("format_id") != "level_flow_belief_v1":
        raise ValueError("Flow quality checkpoint manifest format is invalid")
    if manifest.get("level") != expected_level or manifest.get("config") != dataclasses.asdict(
        flow_config
    ):
        raise ValueError("Flow quality checkpoint level or config is invalid")
    required = {
        "model.safetensors",
        "encoder.safetensors",
        "normalization.npz",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("Flow quality checkpoint artifact inventory is invalid")
    for name in sorted(required):
        path = checkpoint / name
        if not path.is_file() or sha256_file(path) != artifacts[name]:
            raise ValueError(f"Flow quality checkpoint hash mismatch: {name}")
    model = FlowBeliefModel(flow_config).to(device)
    model.load_state_dict(load_safetensors(checkpoint / "model.safetensors"), strict=True)
    model.requires_grad_(False)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Flow quality model did not enter frozen evaluation mode")
    return LoadedFlowQualityLevel(
        model=model,
        normalization=load_flow_belief_normalization(checkpoint / "normalization.npz"),
        level_manifest=manifest,
    )


def _normalized_inputs(
    contexts: list,
    normalization: FlowBeliefNormalization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vision = np.stack([context.vision_history for context in contexts])
    proprio = np.stack(
        [
            (context.robot_proprio_history - normalization.proprio_mean) / normalization.proprio_std
            for context in contexts
        ]
    ).astype(np.float32)
    actions = np.stack(
        [
            (context.remaining_actions - normalization.action_mean) / normalization.action_std
            for context in contexts
        ]
    ).astype(np.float32)
    latency = np.stack([context.latency_probabilities for context in contexts]).astype(np.float32)
    return vision, proprio, actions, latency


def _sample_selected_offsets(
    *,
    corpus: FeatureBeliefCorpus,
    loaded: LoadedFlowQualityLevel,
    flow_config: FlowBeliefConfig,
    config: FlowBeliefQualitySampleConfig,
    selected_offsets: tuple[int, ...],
    device: str,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    selected = set(selected_offsets)
    sampled_by_offset: dict[int, np.ndarray] = {}
    noise_sha256_by_offset: dict[int, str] = {}
    batch_starts = sorted(
        {offset // flow_config.batch_size * flow_config.batch_size for offset in selected}
    )
    delay_ticks = np.arange(1, 21, dtype=np.int64)
    for batch_start in batch_starts:
        batch_stop = min(batch_start + flow_config.batch_size, len(references))
        contexts = [
            corpus.materialize(ProbeSplit.VALIDATION, offset)
            for offset in range(batch_start, batch_stop)
        ]
        vision, proprio, actions, latency = _normalized_inputs(
            contexts,
            loaded.normalization,
        )
        noises = []
        for offset in range(batch_start, batch_stop):
            rng = np.random.default_rng(
                flow_config.evaluation_seed + corpus.level * 1_000_000 + offset
            )
            noises.append(
                rng.standard_normal(
                    (20, config.sample_count, 22),
                    dtype=np.float32,
                )
            )
        for offset, noise in zip(range(batch_start, batch_stop), noises, strict=True):
            if offset in selected:
                noise_sha256_by_offset[offset] = hashlib.sha256(
                    np.asarray(noise, dtype=np.float32).tobytes(order="C")
                ).hexdigest()
        with torch.inference_mode():
            belief = loaded.model.encoder(
                vision_history=torch.from_numpy(vision).to(device),
                proprio_history=torch.from_numpy(proprio).to(device),
                remaining_actions=torch.from_numpy(actions).to(device),
                latency_probabilities=torch.from_numpy(latency).to(device),
            )
            samples = (
                sample_flow_belief(
                    vector_field=loaded.model.vector_field,
                    belief_tokens=belief,
                    delay_ticks=torch.from_numpy(
                        np.broadcast_to(delay_ticks, (len(contexts), 20)).copy()
                    ).to(device),
                    noise=torch.from_numpy(np.stack(noises)).to(device),
                    solver=config.solver,
                    step_count=config.solver_step_count,
                )
                .cpu()
                .numpy()
            )
        for offset in range(batch_start, batch_stop):
            if offset in selected:
                sampled_by_offset[offset] = np.array(
                    samples[offset - batch_start],
                    dtype=np.float32,
                    copy=True,
                )
    if set(sampled_by_offset) != selected or set(noise_sha256_by_offset) != selected:
        raise RuntimeError("Flow quality sampling did not produce every selected context")
    return sampled_by_offset, noise_sha256_by_offset


def _selection_rows(selections: tuple[QualitySelection, ...]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(selection) for selection in selections]


def _selected_metrics(bundle: QualitySampleBundle) -> dict[str, Any]:
    sample_mean = bundle.physical_samples.mean(axis=2)
    error = sample_mean - bundle.physical_targets
    group_errors = {
        "robot_joint_position_rad": np.sqrt(np.mean(np.square(error[..., :7]), axis=-1)),
        "object_position_m": np.linalg.norm(error[..., 16:19], axis=-1),
        "object_linear_velocity_m_s": np.linalg.norm(error[..., 19:22], axis=-1),
    }
    quantiles = (0.5, 0.9, 0.95)
    result: dict[str, Any] = {
        "error_quantiles": {},
        "signed_object_position_bias_m": np.mean(error[..., 16:19], axis=(0, 1)).tolist(),
        "nonfinite_sample_count": int(
            np.size(bundle.physical_samples)
            - np.count_nonzero(np.isfinite(bundle.physical_samples))
        ),
    }
    for name, values in group_errors.items():
        result["error_quantiles"][name] = {
            f"p{int(quantile * 100)}": float(np.quantile(values, quantile))
            for quantile in quantiles
        }
    return result


def export_level_quality_samples(
    *,
    corpus: FeatureBeliefCorpus,
    config: FlowBeliefQualitySampleConfig,
    flow_config: FlowBeliefConfig,
    flow_checkpoint_dir: Path,
    evaluation_summary: FormalFlowSummary,
    selections: tuple[QualitySelection, ...],
    output_dir: Path,
    device: str,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow quality level output already exists: {target}")
    if not selections or any(selection.identity.level != corpus.level for selection in selections):
        raise ValueError("Flow quality selections do not match the corpus level")
    offsets = tuple(selection.identity.validation_offset for selection in selections)
    if offsets != tuple(sorted(set(offsets))):
        raise ValueError("Flow quality selection offsets must be sorted and unique")
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    if evaluation_summary.context_count != len(references):
        raise ValueError("Flow quality summary and validation corpus counts disagree")
    for selection in selections:
        context = corpus.materialize(
            ProbeSplit.VALIDATION,
            selection.identity.validation_offset,
        )
        if (
            context.episode_id != selection.identity.episode_id
            or context.scene_seed != selection.identity.scene_seed
            or context.source_tick != selection.identity.source_tick
        ):
            raise ValueError("Flow quality selection identity does not match the corpus")
    loaded = load_verified_flow_quality_level(
        checkpoint_dir=flow_checkpoint_dir,
        flow_config=flow_config,
        expected_level=corpus.level,
        device=device,
    )
    sampled_by_offset, noise_sha256_by_offset = _sample_selected_offsets(
        corpus=corpus,
        loaded=loaded,
        flow_config=flow_config,
        config=config,
        selected_offsets=offsets,
        device=device,
    )
    reproduced = np.stack([sampled_by_offset[offset] for offset in offsets])
    expected_mean = evaluation_summary.sample_mean_normalized[list(offsets)]
    expected_std = evaluation_summary.sample_std_normalized[list(offsets)]
    reproduced_mean = reproduced.mean(axis=2)
    reproduced_std = reproduced.std(axis=2)
    mean_allclose = bool(
        np.allclose(
            reproduced_mean,
            expected_mean,
            atol=config.summary_parity_atol,
            rtol=config.summary_parity_rtol,
        )
    )
    std_allclose = bool(
        np.allclose(
            reproduced_std,
            expected_std,
            atol=config.summary_parity_atol,
            rtol=config.summary_parity_rtol,
        )
    )
    if not mean_allclose or not std_allclose:
        raise ValueError("Flow quality reproduced samples do not match the formal summary")
    display_rows = np.asarray(config.display_delay_ticks, dtype=np.int64) - 1
    contexts = [corpus.materialize(ProbeSplit.VALIDATION, offset) for offset in offsets]
    normalized_targets = evaluation_summary.target_normalized[
        np.ix_(np.asarray(offsets), display_rows)
    ]
    normalized_samples = reproduced[:, display_rows]
    target_mean = np.asarray(loaded.normalization.target_mean, dtype=np.float32)
    target_std = np.asarray(loaded.normalization.target_std, dtype=np.float32)
    physical_samples = normalized_samples * target_std + target_mean
    physical_targets = normalized_targets * target_std + target_mean
    bundle = QualitySampleBundle(
        selections=selections,
        display_delay_ticks=np.asarray(config.display_delay_ticks),
        latency_probabilities=np.stack(
            [context.latency_probabilities[display_rows] for context in contexts]
        ),
        normalized_samples=normalized_samples,
        normalized_targets=normalized_targets,
        physical_samples=physical_samples,
        physical_targets=physical_targets,
        interaction_mode=evaluation_summary.interaction_mode[
            np.ix_(np.asarray(offsets), display_rows)
        ],
        absorbing=evaluation_summary.absorbing[np.ix_(np.asarray(offsets), display_rows)],
    )
    parity = {
        "mean_allclose": mean_allclose,
        "std_allclose": std_allclose,
        "mean_max_abs_error": float(np.max(np.abs(reproduced_mean - expected_mean))),
        "std_max_abs_error": float(np.max(np.abs(reproduced_std - expected_std))),
        "atol": config.summary_parity_atol,
        "rtol": config.summary_parity_rtol,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "selection.json", _selection_rows(selections))
        _write_json(building / "parity.json", parity)
        _write_json(building / "metrics.json", _selected_metrics(bundle))
        np.savez(
            building / "samples.npz",
            validation_offsets=np.asarray(offsets, dtype=np.int64),
            display_delay_ticks=bundle.display_delay_ticks,
            latency_probabilities=bundle.latency_probabilities,
            normalized_samples=bundle.normalized_samples,
            normalized_targets=bundle.normalized_targets,
            physical_samples=bundle.physical_samples,
            physical_targets=bundle.physical_targets,
            interaction_mode=bundle.interaction_mode,
            absorbing=bundle.absorbing,
        )
        artifacts = {
            name: sha256_file(building / name)
            for name in ("selection.json", "parity.json", "metrics.json", "samples.npz")
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_quality_samples_v1",
            "eligible": True,
            "level": corpus.level,
            "context_count": len(selections),
            "sample_count": config.sample_count,
            "role_counts": dict(
                sorted(Counter(role for row in selections for role in row.roles).items())
            ),
            "display_delay_ticks": list(config.display_delay_ticks),
            "solver": config.solver,
            "solver_step_count": config.solver_step_count,
            "generated_noise_sha256": {
                str(offset): noise_sha256_by_offset[offset] for offset in offsets
            },
            "checkpoint_model_sha256": sha256_file(flow_checkpoint_dir / "model.safetensors"),
            "checkpoint_encoder_sha256": sha256_file(flow_checkpoint_dir / "encoder.safetensors"),
            "checkpoint_normalization_sha256": sha256_file(
                flow_checkpoint_dir / "normalization.npz"
            ),
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        building.rename(target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
