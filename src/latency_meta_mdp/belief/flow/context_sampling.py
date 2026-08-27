"""Reusable deterministic Flow sampling for validation launch contexts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.model import FlowBeliefModel
from latency_meta_mdp.belief.flow.sampler import sample_flow_belief
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LoadedFlowQualityLevel:
    model: FlowBeliefModel
    normalization: FlowBeliefNormalization
    level_manifest: dict[str, Any]


@dataclass(frozen=True)
class FlowValidationContextSamples:
    validation_offsets: tuple[int, ...]
    delay_ticks: np.ndarray
    normalized_samples: np.ndarray
    normalized_targets: np.ndarray
    physical_samples: np.ndarray
    physical_targets: np.ndarray
    latency_probabilities: np.ndarray
    interaction_mode: np.ndarray
    absorbing: np.ndarray
    noise_sha256_by_offset: Mapping[int, str]

    def __post_init__(self) -> None:
        offsets = tuple(self.validation_offsets)
        if (
            not offsets
            or offsets != tuple(sorted(set(offsets)))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in offsets
            )
        ):
            raise ValueError("Flow validation sample offsets are invalid")
        context_count = len(offsets)
        delays = np.asarray(self.delay_ticks)
        if delays.ndim != 1 or len(delays) == 0:
            raise ValueError("Flow validation sample delays are invalid")
        query_count = len(delays)
        sample_array = np.asarray(self.normalized_samples)
        if sample_array.ndim != 4 or sample_array.shape[:2] != (context_count, query_count):
            raise ValueError("Flow validation samples have invalid context or delay axes")
        sample_count = sample_array.shape[2]
        expected_shapes = {
            "normalized_samples": (context_count, query_count, sample_count, 22),
            "normalized_targets": (context_count, query_count, 22),
            "physical_samples": (context_count, query_count, sample_count, 22),
            "physical_targets": (context_count, query_count, 22),
            "latency_probabilities": (context_count, query_count),
            "interaction_mode": (context_count, query_count),
            "absorbing": (context_count, query_count),
        }
        for name, shape in expected_shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"Flow validation sample {name} has invalid shape")
        numeric_names = (
            "normalized_samples",
            "normalized_targets",
            "physical_samples",
            "physical_targets",
            "latency_probabilities",
        )
        if any(not np.all(np.isfinite(getattr(self, name))) for name in numeric_names):
            raise ValueError("Flow validation sample arrays must be finite")
        hashes = dict(self.noise_sha256_by_offset)
        if set(hashes) != set(offsets) or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        ):
            raise ValueError("Flow validation sample noise hashes are invalid")
        object.__setattr__(self, "validation_offsets", offsets)
        for name, dtype in (
            ("delay_ticks", np.int64),
            ("normalized_samples", np.float32),
            ("normalized_targets", np.float32),
            ("physical_samples", np.float32),
            ("physical_targets", np.float32),
            ("latency_probabilities", np.float64),
            ("interaction_mode", np.int8),
            ("absorbing", np.bool_),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "noise_sha256_by_offset", MappingProxyType(hashes))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


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


def _validate_request(
    *,
    corpus: Any,
    flow_config: FlowBeliefConfig,
    validation_offsets: tuple[int, ...],
    delay_ticks: tuple[int, ...],
    sample_count: int,
    solver: str,
    solver_step_count: int,
) -> None:
    if not isinstance(flow_config, FlowBeliefConfig):
        raise TypeError("Flow validation sampling requires a typed Flow config")
    if (
        not validation_offsets
        or validation_offsets != tuple(sorted(set(validation_offsets)))
        or any(
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            for offset in validation_offsets
        )
    ):
        raise ValueError("Flow validation offsets must be sorted and unique")
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    if validation_offsets[-1] >= len(references):
        raise ValueError("Flow validation offsets exceed the validation corpus")
    if (
        not delay_ticks
        or delay_ticks != tuple(sorted(set(delay_ticks)))
        or any(
            isinstance(delay, bool) or not isinstance(delay, int) or not 1 <= delay <= 20
            for delay in delay_ticks
        )
    ):
        raise ValueError("Flow validation delay ticks must be sorted unique values in [1, 20]")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("Flow validation sample count must be positive")
    if solver not in ("euler", "heun"):
        raise ValueError("Flow validation solver is invalid")
    if (
        isinstance(solver_step_count, bool)
        or not isinstance(solver_step_count, int)
        or solver_step_count <= 0
    ):
        raise ValueError("Flow validation solver-step count must be positive")


def sample_flow_validation_contexts(
    *,
    corpus: FeatureBeliefCorpus,
    loaded: LoadedFlowQualityLevel,
    flow_config: FlowBeliefConfig,
    validation_offsets: tuple[int, ...],
    delay_ticks: tuple[int, ...],
    sample_count: int,
    solver: str,
    solver_step_count: int,
    device: str,
) -> FlowValidationContextSamples:
    _validate_request(
        corpus=corpus,
        flow_config=flow_config,
        validation_offsets=validation_offsets,
        delay_ticks=delay_ticks,
        sample_count=sample_count,
        solver=solver,
        solver_step_count=solver_step_count,
    )
    references = corpus.sample_references[ProbeSplit.VALIDATION]
    selected = set(validation_offsets)
    sampled_by_offset: dict[int, np.ndarray] = {}
    noise_sha256_by_offset: dict[int, str] = {}
    batch_starts = sorted(
        {offset // flow_config.batch_size * flow_config.batch_size for offset in selected}
    )
    full_delay_ticks = np.arange(1, 21, dtype=np.int64)
    for batch_start in batch_starts:
        batch_stop = min(batch_start + flow_config.batch_size, len(references))
        contexts = [
            corpus.materialize(ProbeSplit.VALIDATION, offset)
            for offset in range(batch_start, batch_stop)
        ]
        vision, proprio, actions, latency = _normalized_inputs(contexts, loaded.normalization)
        noises = []
        for offset in range(batch_start, batch_stop):
            rng = np.random.default_rng(
                flow_config.evaluation_seed + corpus.level * 1_000_000 + offset
            )
            noises.append(
                rng.standard_normal(
                    (20, sample_count, 22),
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
                        np.broadcast_to(full_delay_ticks, (len(contexts), 20)).copy()
                    ).to(device),
                    noise=torch.from_numpy(np.stack(noises)).to(device),
                    solver=solver,
                    step_count=solver_step_count,
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
        raise RuntimeError("Flow validation sampling did not produce every requested context")
    display_rows = np.asarray(delay_ticks, dtype=np.int64) - 1
    contexts = [corpus.materialize(ProbeSplit.VALIDATION, offset) for offset in validation_offsets]
    normalized_samples = np.stack(
        [sampled_by_offset[offset][display_rows] for offset in validation_offsets]
    )
    physical_targets = np.stack(
        [context.target_states[display_rows] for context in contexts]
    ).astype(np.float32)
    target_mean = np.asarray(loaded.normalization.target_mean, dtype=np.float32)
    target_std = np.asarray(loaded.normalization.target_std, dtype=np.float32)
    normalized_targets = (physical_targets - target_mean) / target_std
    physical_samples = normalized_samples * target_std + target_mean
    return FlowValidationContextSamples(
        validation_offsets=validation_offsets,
        delay_ticks=np.asarray(delay_ticks),
        normalized_samples=normalized_samples,
        normalized_targets=normalized_targets,
        physical_samples=physical_samples,
        physical_targets=physical_targets,
        latency_probabilities=np.stack(
            [context.latency_probabilities[display_rows] for context in contexts]
        ),
        interaction_mode=np.stack(
            [context.target_interaction_mode[display_rows] for context in contexts]
        ),
        absorbing=np.stack([context.target_absorbing[display_rows] for context in contexts]),
        noise_sha256_by_offset=noise_sha256_by_offset,
    )
