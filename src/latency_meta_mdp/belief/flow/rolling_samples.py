"""Per-level deterministic sample artifacts for rolling Flow inspection."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.belief.flow.config import FlowBeliefConfig
from latency_meta_mdp.belief.flow.context_sampling import (
    FlowValidationContextSamples,
    load_verified_flow_quality_level,
    sample_flow_validation_contexts,
)
from latency_meta_mdp.belief.flow.rolling_config import FlowBeliefRollingConfig
from latency_meta_mdp.belief.flow.rolling_selection import RollingSeedSelection


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RollingLevelSampleBundle:
    level: int
    episode_id: str
    scene_seed: int
    validation_offsets: tuple[int, ...]
    source_ticks: np.ndarray
    history_start_ticks: np.ndarray
    source_phase: np.ndarray
    critical_end_tick: int
    handoff_tick: int
    delay_ticks: np.ndarray
    target_ticks: np.ndarray
    target_phase: np.ndarray
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
            self.level not in (1, 2, 3)
            or not self.episode_id
            or isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
            or not offsets
            or offsets != tuple(sorted(set(offsets)))
        ):
            raise ValueError("rolling level sample identity is invalid")
        window_count = len(offsets)
        query_count = len(np.asarray(self.delay_ticks))
        sample_array = np.asarray(self.normalized_samples)
        if sample_array.ndim != 4 or sample_array.shape[:2] != (window_count, query_count):
            raise ValueError("rolling level sample distribution axes are invalid")
        sample_count = sample_array.shape[2]
        expected_shapes = {
            "source_ticks": (window_count,),
            "history_start_ticks": (window_count,),
            "source_phase": (window_count,),
            "delay_ticks": (query_count,),
            "target_ticks": (window_count, query_count),
            "target_phase": (window_count, query_count),
            "normalized_samples": (window_count, query_count, sample_count, 22),
            "normalized_targets": (window_count, query_count, 22),
            "physical_samples": (window_count, query_count, sample_count, 22),
            "physical_targets": (window_count, query_count, 22),
            "latency_probabilities": (window_count, query_count),
            "interaction_mode": (window_count, query_count),
            "absorbing": (window_count, query_count),
        }
        for name, shape in expected_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"rolling level sample {name} has invalid shape")
        if np.any(self.absorbing):
            raise ValueError("rolling level samples cannot contain absorbing targets")
        if any(phase not in ("pregrasp", "approach") for phase in self.source_phase):
            raise ValueError("rolling level sample source phase is invalid")
        if any(phase not in ("pregrasp", "approach") for phase in self.target_phase.reshape(-1)):
            raise ValueError("rolling level sample target phase is invalid")
        if np.any(self.target_ticks > self.critical_end_tick):
            raise ValueError("rolling level sample target crosses the approach cutoff")
        hashes = dict(self.noise_sha256_by_offset)
        if set(hashes) != set(offsets):
            raise ValueError("rolling level sample noise hashes are misaligned")
        object.__setattr__(self, "validation_offsets", offsets)
        for name, dtype in (
            ("source_ticks", np.int64),
            ("history_start_ticks", np.int64),
            ("source_phase", str),
            ("delay_ticks", np.int64),
            ("target_ticks", np.int64),
            ("target_phase", str),
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


def _boundary_target_phase(expert_phase: np.ndarray, target_ticks: np.ndarray) -> np.ndarray:
    phases = np.asarray(expert_phase)
    targets = np.asarray(target_ticks, dtype=np.int64)
    if (
        phases.ndim != 1
        or targets.ndim != 2
        or np.any(targets <= 0)
        or np.any(targets > len(phases))
    ):
        raise ValueError("rolling target ticks cannot be mapped to boundary phases")
    return np.asarray(phases[targets - 1], dtype=str)


def build_rolling_level_sample_bundle(
    *,
    selection: RollingSeedSelection,
    sampled: FlowValidationContextSamples,
    expert_phase: np.ndarray,
) -> RollingLevelSampleBundle:
    if not isinstance(selection, RollingSeedSelection) or not isinstance(
        sampled, FlowValidationContextSamples
    ):
        raise TypeError("rolling sample bundle requires typed selection and samples")
    expected_offsets = tuple(row.validation_offset for row in selection.windows)
    if sampled.validation_offsets != expected_offsets:
        raise ValueError("rolling sample validation offsets do not match the selection")
    source_ticks = np.asarray([row.source_tick for row in selection.windows], dtype=np.int64)
    delay_ticks = np.asarray(sampled.delay_ticks, dtype=np.int64)
    target_ticks = source_ticks[:, None] + delay_ticks[None, :]
    source_phase = np.asarray([row.source_phase for row in selection.windows], dtype=str)
    phase_values = np.asarray(expert_phase)
    if np.any(source_ticks >= len(phase_values)) or not np.array_equal(
        source_phase,
        phase_values[source_ticks],
    ):
        raise ValueError("rolling sample source phases do not match the episode")
    return RollingLevelSampleBundle(
        level=selection.level,
        episode_id=selection.episode_id,
        scene_seed=selection.scene_seed,
        validation_offsets=expected_offsets,
        source_ticks=source_ticks,
        history_start_ticks=np.asarray(
            [row.history_start_tick for row in selection.windows], dtype=np.int64
        ),
        source_phase=source_phase,
        critical_end_tick=selection.critical_end_tick,
        handoff_tick=selection.handoff_tick,
        delay_ticks=delay_ticks,
        target_ticks=target_ticks,
        target_phase=_boundary_target_phase(phase_values, target_ticks),
        normalized_samples=sampled.normalized_samples,
        normalized_targets=sampled.normalized_targets,
        physical_samples=sampled.physical_samples,
        physical_targets=sampled.physical_targets,
        latency_probabilities=sampled.latency_probabilities,
        interaction_mode=sampled.interaction_mode,
        absorbing=sampled.absorbing,
        noise_sha256_by_offset=sampled.noise_sha256_by_offset,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_level_rolling_samples(
    *,
    corpus: FeatureBeliefCorpus,
    selection: RollingSeedSelection,
    rolling_config: FlowBeliefRollingConfig,
    flow_config: FlowBeliefConfig,
    flow_checkpoint_dir: Path,
    output_dir: Path,
    device: str,
) -> Path:
    started = time.perf_counter()
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow rolling level output already exists: {target}")
    if selection.level != corpus.level or selection.scene_seed < 0:
        raise ValueError("rolling selection and corpus level disagree")
    matching_records = [
        record
        for record in corpus.records
        if record.episode_id == selection.episode_id and record.scene_seed == selection.scene_seed
    ]
    if len(matching_records) != 1:
        raise ValueError("rolling sample export requires one matching corpus record")
    loaded = load_verified_flow_quality_level(
        checkpoint_dir=flow_checkpoint_dir,
        flow_config=flow_config,
        expected_level=corpus.level,
        device=device,
    )
    sampled = sample_flow_validation_contexts(
        corpus=corpus,
        loaded=loaded,
        flow_config=flow_config,
        validation_offsets=tuple(row.validation_offset for row in selection.windows),
        delay_ticks=rolling_config.display_delay_ticks,
        sample_count=rolling_config.sample_count,
        solver=rolling_config.solver,
        solver_step_count=rolling_config.solver_step_count,
        device=device,
    )
    bundle = build_rolling_level_sample_bundle(
        selection=selection,
        sampled=sampled,
        expert_phase=matching_records[0].tail.expert_phase,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "selection.json", dataclasses.asdict(selection))
        _write_json(
            building / "metrics.json",
            {
                "window_count": len(selection.windows),
                "source_phase_counts": dict(sorted(Counter(bundle.source_phase).items())),
                "target_phase_counts": dict(
                    sorted(Counter(bundle.target_phase.reshape(-1)).items())
                ),
                "absorbing_target_count": int(np.count_nonzero(bundle.absorbing)),
                "nonfinite_sample_count": int(
                    bundle.physical_samples.size
                    - np.count_nonzero(np.isfinite(bundle.physical_samples))
                ),
            },
        )
        np.savez(
            building / "samples.npz",
            validation_offsets=np.asarray(bundle.validation_offsets, dtype=np.int64),
            source_ticks=bundle.source_ticks,
            history_start_ticks=bundle.history_start_ticks,
            source_phase=bundle.source_phase,
            critical_end_tick=np.asarray(bundle.critical_end_tick, dtype=np.int64),
            handoff_tick=np.asarray(bundle.handoff_tick, dtype=np.int64),
            display_delay_ticks=bundle.delay_ticks,
            target_ticks=bundle.target_ticks,
            target_phase=bundle.target_phase,
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
            for name in ("selection.json", "metrics.json", "samples.npz")
        }
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_rolling_samples_v1",
            "eligible": True,
            "level": bundle.level,
            "episode_id": bundle.episode_id,
            "scene_seed": bundle.scene_seed,
            "window_count": len(bundle.validation_offsets),
            "inspection_stride_ticks": rolling_config.inspection_stride_ticks,
            "critical_start_tick": int(bundle.source_ticks[0]),
            "critical_end_tick": bundle.critical_end_tick,
            "handoff_tick": bundle.handoff_tick,
            "selected_source_ticks": bundle.source_ticks.tolist(),
            "display_delay_ticks": bundle.delay_ticks.tolist(),
            "sample_count": rolling_config.sample_count,
            "solver": rolling_config.solver,
            "solver_step_count": rolling_config.solver_step_count,
            "generated_noise_sha256": {
                str(offset): bundle.noise_sha256_by_offset[offset]
                for offset in bundle.validation_offsets
            },
            "checkpoint_model_sha256": sha256_file(flow_checkpoint_dir / "model.safetensors"),
            "checkpoint_encoder_sha256": sha256_file(flow_checkpoint_dir / "encoder.safetensors"),
            "checkpoint_normalization_sha256": sha256_file(
                flow_checkpoint_dir / "normalization.npz"
            ),
            "wall_seconds": time.perf_counter() - started,
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        building.rename(target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
