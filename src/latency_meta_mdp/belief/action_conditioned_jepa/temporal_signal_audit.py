"""No-training physical-time signal audit for JEPA temporal candidates."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.belief.action_conditioned_jepa.config import JepaTemporalSampling
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaEpisodeRecord,
    VerifiedJepaInputs,
    load_verified_jepa_record,
)
from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
    SharedJepaSampleIndex,
    build_shared_temporal_indices,
)
from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.source_corpus.schema import SourceFieldRole

_SIGNAL_FIELDS = (
    "formal_tick",
    "eef_position_world",
    "object_pose",
    "commanded_motion_segment_index",
    "left_pad_contact",
    "right_pad_contact",
    "handoff_state",
)
_SIGNAL_ROLES = frozenset(
    {
        SourceFieldRole.IDENTITY,
        SourceFieldRole.DEPLOYMENT_INPUT,
        SourceFieldRole.SUPERVISION_CANDIDATE,
        SourceFieldRole.AUDIT_ONLY,
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TemporalSignalEpisode:
    record: JepaEpisodeRecord
    object_position: np.ndarray
    eef_position: np.ndarray
    left_pad_contact: np.ndarray
    right_pad_contact: np.ndarray
    handoff_state: tuple[str, ...]
    motion_segment_index: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.record, JepaEpisodeRecord):
            raise TypeError("record must be a JepaEpisodeRecord")
        count = self.record.terminal_tick + 1
        object_position = _readonly(self.object_position, dtype=np.dtype(np.float32))
        eef_position = _readonly(self.eef_position, dtype=np.dtype(np.float32))
        left = _readonly(self.left_pad_contact, dtype=np.dtype(np.bool_))
        right = _readonly(self.right_pad_contact, dtype=np.dtype(np.bool_))
        segments = _readonly(self.motion_segment_index, dtype=np.dtype(np.int64))
        if (
            object_position.shape != (count, 3)
            or eef_position.shape != (count, 3)
            or left.shape != (count,)
            or right.shape != (count,)
            or segments.shape != (count,)
            or not np.all(np.isfinite(object_position))
            or not np.all(np.isfinite(eef_position))
        ):
            raise ValueError("temporal signal arrays disagree with the episode timeline")
        if (
            type(self.handoff_state) is not tuple
            or len(self.handoff_state) != count
            or any(type(value) is not str or not value for value in self.handoff_state)
        ):
            raise ValueError("handoff_state must contain one string per boundary")
        object.__setattr__(self, "object_position", object_position)
        object.__setattr__(self, "eef_position", eef_position)
        object.__setattr__(self, "left_pad_contact", left)
        object.__setattr__(self, "right_pad_contact", right)
        object.__setattr__(self, "motion_segment_index", segments)


def load_temporal_signal_episodes(
    *,
    inputs: VerifiedJepaInputs,
    episode_ids: tuple[str, ...],
    level: int,
    split: str,
) -> tuple[TemporalSignalEpisode, ...]:
    if not isinstance(inputs, VerifiedJepaInputs):
        raise TypeError("inputs must be VerifiedJepaInputs")
    if (
        type(episode_ids) is not tuple
        or not episode_ids
        or any(type(value) is not str or not value for value in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError("episode_ids must be a non-empty unique tuple")
    if type(level) is not int or level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")

    result = []
    for episode_id in episode_ids:
        record = load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=level,
            split=split,
        )
        rows = inputs.source.read_fields(
            episode_id,
            fields=_SIGNAL_FIELDS,
            allowed_roles=_SIGNAL_ROLES,
        ).to_pylist()
        if [row["formal_tick"] for row in rows] != list(range(record.terminal_tick + 1)):
            raise ValueError("temporal signal rows are not boundary-aligned")
        result.append(
            TemporalSignalEpisode(
                record=record,
                object_position=np.asarray(
                    [row["object_pose"][:3] for row in rows],
                    dtype=np.float32,
                ),
                eef_position=np.asarray(
                    [row["eef_position_world"] for row in rows],
                    dtype=np.float32,
                ),
                left_pad_contact=np.asarray(
                    [row["left_pad_contact"] for row in rows],
                    dtype=np.bool_,
                ),
                right_pad_contact=np.asarray(
                    [row["right_pad_contact"] for row in rows],
                    dtype=np.bool_,
                ),
                handoff_state=tuple(str(row["handoff_state"]) for row in rows),
                motion_segment_index=np.asarray(
                    [row["commanded_motion_segment_index"] for row in rows],
                    dtype=np.int64,
                ),
            )
        )
    return tuple(result)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("temporal audit metric must contain finite observations")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def _crosses(values: np.ndarray | tuple[str, ...], start: int, stop: int) -> bool:
    window = values[start : stop + 1]
    first = window[0]
    return any(value != first for value in window[1:])


def _sample_indices(
    values: tuple[SharedJepaSampleIndex, ...],
    count: int,
) -> tuple[SharedJepaSampleIndex, ...]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return tuple(values[int(position)] for position in positions)


def summarize_temporal_signals(
    *,
    episodes: tuple[TemporalSignalEpisode, ...],
    samplings: tuple[JepaTemporalSampling, ...],
    latent_samples_per_episode: int = 4,
) -> dict[str, Any]:
    if (
        type(episodes) is not tuple
        or not episodes
        or any(not isinstance(value, TemporalSignalEpisode) for value in episodes)
    ):
        raise ValueError("episodes must be a non-empty tuple of temporal signal episodes")
    if type(latent_samples_per_episode) is not int or latent_samples_per_episode <= 0:
        raise ValueError("latent_samples_per_episode must be a positive integer")
    records = tuple(value.record for value in episodes)
    if len({record.episode_id for record in records}) != len(records):
        raise ValueError("temporal signal episodes must have unique identities")
    shared = build_shared_temporal_indices(records=records, samplings=samplings)
    by_episode: dict[str, list[SharedJepaSampleIndex]] = defaultdict(list)
    for index in shared:
        by_episode[index.episode_id].append(index)
    episode_by_id = {value.record.episode_id: value for value in episodes}
    reference_work = None
    candidates: dict[str, Any] = {}

    for sampling in samplings:
        latent_rms: list[float] = []
        object_displacement: list[float] = []
        eef_displacement: list[float] = []
        action_magnitude: list[float] = []
        contact_crossing = 0
        handoff_crossing = 0
        segment_crossing = 0
        phase_counts: Counter[str] = Counter()

        for episode_id, episode_indices in sorted(by_episode.items()):
            episode = episode_by_id[episode_id]
            terminal = episode.record.terminal_tick
            for index in episode_indices:
                start = index.source_tick
                stop = min(start + sampling.model_stride_ticks, terminal)
                object_displacement.append(
                    float(
                        np.linalg.norm(
                            episode.object_position[stop] - episode.object_position[start]
                        )
                    )
                )
                eef_displacement.append(
                    float(np.linalg.norm(episode.eef_position[stop] - episode.eef_position[start]))
                )
                controls = episode.record.controls[start:stop]
                if len(controls):
                    action_magnitude.append(float(np.linalg.norm(controls[:, :6], axis=1).mean()))
                else:
                    action_magnitude.append(0.0)
                contact = np.logical_or(episode.left_pad_contact, episode.right_pad_contact)
                contact_crossing += int(_crosses(contact, start, stop))
                handoff_crossing += int(_crosses(episode.handoff_state, start, stop))
                segment_crossing += int(_crosses(episode.motion_segment_index, start, stop))
                phase_counts[str(episode.record.phases[start])] += 1

            for index in _sample_indices(tuple(episode_indices), latent_samples_per_episode):
                start = index.source_tick
                stop = min(start + sampling.model_stride_ticks, terminal)
                current = np.asarray(episode.record.cache.features[start], dtype=np.float32)
                target = np.asarray(episode.record.cache.features[stop], dtype=np.float32)
                latent_rms.append(float(np.sqrt(np.mean(np.square(target - current)))))

        context_count = len(shared)
        tokens_per_time = 2 * 196 + 1
        attention_work = (
            sampling.native_rollout_steps
            * (sampling.history_observation_count * tokens_per_time) ** 2
        )
        if reference_work is None:
            reference_work = attention_work
        candidates[sampling.config_id] = {
            "model_stride_ticks": sampling.model_stride_ticks,
            "model_step_ms": sampling.model_stride_ticks * 20,
            "history_observation_count": sampling.history_observation_count,
            "history_span_ms": sampling.history_span_ticks * 20,
            "native_rollout_steps": sampling.native_rollout_steps,
            "native_anchor_ticks": list(sampling.native_future_offsets),
            "macro_action_dim": sampling.model_stride_ticks * 7,
            "shared_context_count": context_count,
            "absorbing_context_count": sum(
                index.boundary_disposition == "certified_absorbing_extension" for index in shared
            ),
            "native_rollout_transition_count": context_count * sampling.native_rollout_steps,
            "one_step_latent_sample_count": len(latent_rms),
            "one_step_latent_rms": _summary(latent_rms),
            "one_step_object_displacement_m": _summary(object_displacement),
            "one_step_eef_displacement_m": _summary(eef_displacement),
            "one_step_arm_action_norm": _summary(action_magnitude),
            "one_step_contact_crossing_fraction": contact_crossing / context_count,
            "one_step_handoff_crossing_fraction": handoff_crossing / context_count,
            "one_step_motion_segment_crossing_fraction": segment_crossing / context_count,
            "source_phase_counts": dict(sorted(phase_counts.items())),
            "attention_work_relative_to_dense_reference": attention_work / reference_work,
        }

    return {
        "schema_version": 1,
        "format_id": "action_conditioned_jepa_temporal_signal_audit",
        "episode_count": len(episodes),
        "shared_context_count": len(shared),
        "shared_context_start_tick": min(index.source_tick for index in shared),
        "latent_samples_per_episode": latent_samples_per_episode,
        "candidates": candidates,
    }


def write_temporal_signal_report(
    *,
    output_dir: Path,
    report: dict[str, Any],
    source_manifest_sha256: str,
    cache_manifest_sha256: str,
    split_manifest_sha256: str,
) -> Path:
    if (
        type(report) is not dict
        or report.get("schema_version") != 1
        or report.get("format_id") != "action_conditioned_jepa_temporal_signal_audit"
    ):
        raise ValueError("temporal signal report is invalid")
    provenance = {}
    for name, value in (
        ("source_manifest_sha256", source_manifest_sha256),
        ("cache_manifest_sha256", cache_manifest_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
    ):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        provenance[name] = value
    payload = dict(report)
    if "provenance" in payload:
        raise ValueError("report provenance must be supplied by the publisher")
    payload["provenance"] = provenance
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_file_fsynced(building / "report.json", encoded)
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "report.json"
