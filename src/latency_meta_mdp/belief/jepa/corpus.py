"""Leakage-safe lazy data view for nominal Action-Conditioned JEPA training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.config import (
    ActionConditionedJepaConfig,
)
from latency_meta_mdp.belief.jepa.contracts import (
    FutureLatentRollout,
    LaunchContextBatch,
)
from latency_meta_mdp.belief.jepa.identity import data_domain
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.data.source.loader import (
    VerifiedSourceCorpus,
    load_verified_source_corpus,
)
from latency_meta_mdp.data.source.schema import SourceFieldRole
from latency_meta_mdp.data.source.split_view import (
    SourceSplitManifest,
    load_verified_source_split,
)
from latency_meta_mdp.data.vision.cache import (
    EpisodeVisionFeatureCache,
    load_episode_vision_feature_cache,
)
from latency_meta_mdp.data.vision.extract import (
    load_verified_vision_feature_cache_run,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HISTORY_TICKS = 6
_EXECUTED_CONTROL_TICKS = 5
_MAXIMUM_DELAY_TICKS = 20
_PROPRIO_DIM = 16
_ACTION_DIM = 7
_SOURCE_FIELDS = (
    "formal_tick",
    "robot_qpos",
    "robot_qvel",
    "gripper_qpos",
    "gripper_qvel",
    "outcome_status",
    "expert_action",
    "action_mask",
    "phase_id",
)
_SOURCE_ROLES = frozenset(
    {
        SourceFieldRole.IDENTITY,
        SourceFieldRole.DEPLOYMENT_INPUT,
        SourceFieldRole.AUDIT_ONLY,
    }
)
_NORMALIZATION_FORMAT = "action_conditioned_jepa_proprio_normalization_v1"


def _require_sha256(value: str, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _readonly(array: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, order=True)
class JepaSampleIndex:
    level: int
    split: str
    episode_id: str
    source_tick: int

    def __post_init__(self) -> None:
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("JEPA sample level must be 1, 2, or 3")
        if self.split not in {"train", "validation"}:
            raise ValueError("JEPA sample split must be train or validation")
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("JEPA sample episode_id cannot be empty")
        if type(self.source_tick) is not int or self.source_tick < _HISTORY_TICKS - 1:
            raise ValueError("dense-reference JEPA sample lacks six-boundary history")


@dataclass(frozen=True)
class JepaEpisodeRecord:
    episode_id: str
    task_instance_id: str
    logical_master_task_index: int
    level: int | None
    split: str
    terminal_tick: int
    cache: EpisodeVisionFeatureCache
    proprio_physical: np.ndarray
    controls: np.ndarray
    phases: tuple[str | None, ...]
    statuses: tuple[str, ...]
    task_id: str | None = None
    action_contract_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("JEPA episode_id cannot be empty")
        if type(self.task_instance_id) is not str or not self.task_instance_id:
            raise ValueError("JEPA task_instance_id cannot be empty")
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("JEPA logical master-task index is invalid")
        data_domain(self)
        if self.split not in {"train", "validation"}:
            raise ValueError("JEPA episode split must be train or validation")
        if type(self.terminal_tick) is not int or self.terminal_tick < _HISTORY_TICKS:
            raise ValueError("JEPA episode is too short for dense-reference history")
        if not isinstance(self.cache, EpisodeVisionFeatureCache) or not isinstance(
            self.cache.features, np.memmap
        ):
            raise TypeError("JEPA episode features must remain a NumPy memmap")
        boundary_count = self.terminal_tick + 1
        if self.cache.features.shape != (boundary_count, 2, 196, 384):
            raise ValueError("JEPA episode cache shape disagrees with terminal_tick")
        if self.cache.features.dtype != np.float16:
            raise ValueError("JEPA episode cache must use float16")
        proprio = _readonly(self.proprio_physical, dtype=np.dtype(np.float32))
        controls = _readonly(self.controls, dtype=np.dtype(np.float32))
        if proprio.shape != (boundary_count, _PROPRIO_DIM) or not np.all(np.isfinite(proprio)):
            raise ValueError("JEPA episode proprio contract is invalid")
        if (
            controls.shape != (self.terminal_tick, _ACTION_DIM)
            or not np.all(np.isfinite(controls))
            or np.any(controls < -1.0)
            or np.any(controls > 1.0)
        ):
            raise ValueError("JEPA episode control contract is invalid")
        if (
            type(self.phases) is not tuple
            or len(self.phases) != boundary_count
            or any(value is not None and type(value) is not str for value in self.phases)
        ):
            raise ValueError("JEPA episode phases are invalid")
        if (
            type(self.statuses) is not tuple
            or len(self.statuses) != boundary_count
            or any(type(value) is not str or not value for value in self.statuses)
            or self.statuses[-1] != "success"
        ):
            raise ValueError("JEPA episode statuses are invalid")
        object.__setattr__(self, "proprio_physical", proprio)
        object.__setattr__(self, "controls", controls)

    @property
    def legal_source_ticks(self) -> tuple[int, ...]:
        return tuple(range(_HISTORY_TICKS - 1, self.terminal_tick))

    @property
    def last_real_gripper_command(self) -> float:
        return float(self.controls[-1, -1])


def _indices_for_records(records: tuple[JepaEpisodeRecord, ...]) -> tuple[JepaSampleIndex, ...]:
    return tuple(
        JepaSampleIndex(
            level=record.level,
            split=record.split,
            episode_id=record.episode_id,
            source_tick=source_tick,
        )
        for record in records
        for source_tick in record.legal_source_ticks
    )


def _sample_index_sha256(records: tuple[JepaEpisodeRecord, ...]) -> str:
    if records[0].task_id is not None:
        # Compact description of the complete Direct pair support, not millions of rows.
        payload = [
            {
                "episode_id": r.episode_id,
                "domain": data_domain(r),
                "split": r.split,
                "first_source_tick": 10,
                "terminal_tick": r.terminal_tick,
                "query_ticks": [1, 20],
            }
            for r in records
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload = [
        {
            "episode_id": index.episode_id,
            "level": index.level,
            "source_tick": index.source_tick,
            "split": index.split,
        }
        for index in _indices_for_records(records)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class JepaProprioNormalization:
    level: int | None
    mean: np.ndarray
    scale: np.ndarray
    constant_dimension_mask: np.ndarray
    episode_ids: tuple[str, ...]
    boundary_count: int
    source_manifest_sha256: str
    split_manifest_sha256: str
    sample_index_sha256: str
    task_id: str | None = None
    action_contract_id: str | None = None

    def __post_init__(self) -> None:
        data_domain(self)
        mean = _readonly(self.mean, dtype=np.dtype(np.float32))
        scale = _readonly(self.scale, dtype=np.dtype(np.float32))
        mask = _readonly(self.constant_dimension_mask, dtype=np.dtype(np.bool_))
        if (
            mean.shape != (_PROPRIO_DIM,)
            or scale.shape != (_PROPRIO_DIM,)
            or mask.shape != (_PROPRIO_DIM,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
            or np.any(scale[mask] != 1.0)
        ):
            raise ValueError("JEPA proprio normalization arrays are invalid")
        if (
            type(self.episode_ids) is not tuple
            or not self.episode_ids
            or self.episode_ids != tuple(sorted(set(self.episode_ids)))
        ):
            raise ValueError("normalization episode inventory is invalid")
        if type(self.boundary_count) is not int or self.boundary_count <= 0:
            raise ValueError("normalization boundary count must be positive")
        for name in (
            "source_manifest_sha256",
            "split_manifest_sha256",
            "sample_index_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "constant_dimension_mask", mask)

    def normalize(self, proprio: np.ndarray) -> np.ndarray:
        value = np.asarray(proprio, dtype=np.float32)
        if value.shape[-1:] != (_PROPRIO_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError("proprio input must be finite with final dimension 16")
        return np.asarray((value - self.mean) / self.scale, dtype=np.float32)

    def denormalize(self, proprio: np.ndarray) -> np.ndarray:
        value = np.asarray(proprio, dtype=np.float32)
        if value.shape[-1:] != (_PROPRIO_DIM,) or not np.all(np.isfinite(value)):
            raise ValueError("normalized proprio must be finite with final dimension 16")
        return np.asarray(value * self.scale + self.mean, dtype=np.float32)

    def to_mapping(self) -> dict[str, Any]:
        identity = (
            {"level": self.level}
            if self.task_id is None
            else {"task_id": self.task_id, "action_contract_id": self.action_contract_id}
        )
        return {
            "schema_version": 1 if self.task_id is None else 2,
            "format_id": _NORMALIZATION_FORMAT,
            **identity,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "constant_dimension_mask": self.constant_dimension_mask.tolist(),
            "episode_ids": list(self.episode_ids),
            "boundary_count": self.boundary_count,
            "source_manifest_sha256": self.source_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "sample_index_sha256": self.sample_index_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> JepaProprioNormalization:
        expected = {
            "schema_version",
            "format_id",
            "level",
            "mean",
            "scale",
            "constant_dimension_mask",
            "episode_ids",
            "boundary_count",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "sample_index_sha256",
        }
        if isinstance(value, dict) and value.get("schema_version") == 2:
            expected = (expected - {"level"}) | {"task_id", "action_contract_id"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("JEPA proprio normalization fields are invalid")
        if value["schema_version"] not in (1, 2) or value["format_id"] != _NORMALIZATION_FORMAT:
            raise ValueError("unsupported JEPA proprio normalization format")
        for name in ("mean", "scale"):
            values = value[name]
            if (
                type(values) is not list
                or len(values) != _PROPRIO_DIM
                or any(type(item) not in {int, float} for item in values)
            ):
                raise ValueError(f"normalization {name} is invalid")
        mask = value["constant_dimension_mask"]
        if (
            type(mask) is not list
            or len(mask) != _PROPRIO_DIM
            or any(type(item) is not bool for item in mask)
        ):
            raise ValueError("normalization constant_dimension_mask is invalid")
        episode_ids = value["episode_ids"]
        if type(episode_ids) is not list or any(type(item) is not str for item in episode_ids):
            raise ValueError("normalization episode_ids are invalid")
        return cls(
            level=value.get("level"),
            task_id=value.get("task_id"),
            action_contract_id=value.get("action_contract_id"),
            mean=np.asarray(value["mean"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
            constant_dimension_mask=np.asarray(mask, dtype=np.bool_),
            episode_ids=tuple(episode_ids),
            boundary_count=value["boundary_count"],
            source_manifest_sha256=value["source_manifest_sha256"],
            split_manifest_sha256=value["split_manifest_sha256"],
            sample_index_sha256=value["sample_index_sha256"],
        )


def compute_jepa_proprio_normalization(
    *,
    records: tuple[JepaEpisodeRecord, ...],
    source_manifest_sha256: str,
    split_manifest_sha256: str,
) -> JepaProprioNormalization:
    if (
        type(records) is not tuple
        or not records
        or any(not isinstance(record, JepaEpisodeRecord) for record in records)
    ):
        raise ValueError("normalization records must be a non-empty tuple")
    if any(record.split != "train" for record in records):
        raise ValueError("proprio normalization may use only train records")
    levels = {data_domain(record) for record in records}
    if len(levels) != 1:
        raise ValueError("proprio normalization must be level-specific")
    ordered = tuple(sorted(records, key=lambda record: record.episode_id))
    if len({record.episode_id for record in ordered}) != len(ordered):
        raise ValueError("proprio normalization records must have unique episodes")
    values = np.concatenate([record.proprio_physical for record in ordered], axis=0).astype(
        np.float64
    )
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    span = values.max(axis=0) - values.min(axis=0)
    constant = span <= np.finfo(np.float32).eps
    scale[constant] = 1.0
    return JepaProprioNormalization(
        level=ordered[0].level,
        task_id=ordered[0].task_id,
        action_contract_id=ordered[0].action_contract_id,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        constant_dimension_mask=constant,
        episode_ids=tuple(record.episode_id for record in ordered),
        boundary_count=sum(record.terminal_tick + 1 for record in ordered),
        source_manifest_sha256=_require_sha256(
            source_manifest_sha256, name="source_manifest_sha256"
        ),
        split_manifest_sha256=_require_sha256(split_manifest_sha256, name="split_manifest_sha256"),
        sample_index_sha256=_sample_index_sha256(ordered),
    )


def write_jepa_proprio_normalization(
    path: Path,
    normalization: JepaProprioNormalization,
) -> Path:
    if not isinstance(normalization, JepaProprioNormalization):
        raise TypeError("normalization must be a JepaProprioNormalization")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    payload = (
        json.dumps(
            normalization.to_mapping(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    try:
        _write_file_fsynced(staging, payload)
        _rename_noreplace(staging, path)
        _fsync_directory(path.parent)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return path


def load_jepa_proprio_normalization(path: Path) -> JepaProprioNormalization:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JEPA proprio normalization must contain valid JSON") from error
    return JepaProprioNormalization.from_mapping(value)


@dataclass(frozen=True)
class JepaTrainingSample:
    index: JepaSampleIndex
    vision_history: torch.Tensor
    proprio_history: torch.Tensor
    executed_controls: torch.Tensor
    executable_controls: torch.Tensor
    future_visual_latents: torch.Tensor
    future_proprio: torch.Tensor
    source_phase: str
    source_status: str
    boundary_ticks: np.ndarray
    executed_control_ticks: np.ndarray
    executable_control_ticks: np.ndarray
    target_ticks: np.ndarray
    target_phases: tuple[str | None, ...]
    target_statuses: tuple[str, ...]
    target_absorbing: np.ndarray

    @property
    def launch_context(self) -> LaunchContextBatch:
        return LaunchContextBatch(
            vision_history=self.vision_history.unsqueeze(0),
            proprio_history=self.proprio_history.unsqueeze(0),
            executed_controls=self.executed_controls.unsqueeze(0).unsqueeze(2),
            executable_controls=self.executable_controls.unsqueeze(0).unsqueeze(2),
        )

    @property
    def future_rollout(self) -> FutureLatentRollout:
        return FutureLatentRollout(
            native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            future_visual_latents=self.future_visual_latents.unsqueeze(0),
            future_proprio=self.future_proprio.unsqueeze(0),
        )


def formal_hold_suffix(gripper_command: float, count: int) -> np.ndarray:
    if not np.isfinite(gripper_command) or not -1.0 <= gripper_command <= 1.0:
        raise ValueError("hold gripper command must be finite and controller-native")
    if type(count) is not int or count < 0:
        raise ValueError("hold suffix count must be non-negative")
    result = np.zeros((count, _ACTION_DIM), dtype=np.float32)
    result[:, -1] = gripper_command
    return result


def _as_tensor(array: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.array(array, copy=True)).to(dtype=dtype)


def materialize_jepa_sample(
    record: JepaEpisodeRecord,
    *,
    source_tick: int,
    normalization: JepaProprioNormalization,
) -> JepaTrainingSample:
    if not isinstance(record, JepaEpisodeRecord):
        raise TypeError("record must be a JepaEpisodeRecord")
    if not isinstance(normalization, JepaProprioNormalization):
        raise TypeError("normalization must be a JepaProprioNormalization")
    if normalization.level != record.level:
        raise ValueError("proprio normalization level does not match the episode")
    if source_tick not in record.legal_source_ticks:
        raise ValueError(f"illegal JEPA source tick {source_tick} for {record.episode_id}")
    boundary_ticks = np.arange(source_tick - 5, source_tick + 1, dtype=np.int64)
    executed_ticks = np.arange(source_tick - 5, source_tick, dtype=np.int64)
    executable_ticks = np.arange(source_tick, source_tick + 20, dtype=np.int64)
    target_ticks = np.arange(source_tick + 1, source_tick + 21, dtype=np.int64)
    terminal = record.terminal_tick
    target_rows = np.minimum(target_ticks, terminal)
    target_absorbing = target_ticks > terminal

    executable = np.empty((_MAXIMUM_DELAY_TICKS, _ACTION_DIM), dtype=np.float32)
    real_control = executable_ticks < terminal
    executable[real_control] = record.controls[executable_ticks[real_control]]
    executable[~real_control] = formal_hold_suffix(
        record.last_real_gripper_command,
        int((~real_control).sum()),
    )
    future_proprio = np.array(record.proprio_physical[target_rows], copy=True)
    future_proprio[target_absorbing, 7:14] = 0.0
    future_proprio[target_absorbing, 15] = 0.0
    future_visual = np.array(record.cache.features[target_rows], copy=True)
    target_phases = tuple(
        record.phases[tick] if tick <= terminal else "absorbing" for tick in target_ticks
    )
    target_statuses = tuple(
        record.statuses[tick] if tick <= terminal else record.statuses[terminal]
        for tick in target_ticks
    )
    for value in (boundary_ticks, executed_ticks, executable_ticks, target_ticks, target_absorbing):
        value.setflags(write=False)
    return JepaTrainingSample(
        index=JepaSampleIndex(
            level=record.level,
            split=record.split,
            episode_id=record.episode_id,
            source_tick=source_tick,
        ),
        vision_history=_as_tensor(record.cache.features[boundary_ticks], dtype=torch.float16),
        proprio_history=_as_tensor(
            normalization.normalize(record.proprio_physical[boundary_ticks]),
            dtype=torch.float32,
        ),
        executed_controls=_as_tensor(record.controls[executed_ticks], dtype=torch.float32),
        executable_controls=_as_tensor(executable, dtype=torch.float32),
        future_visual_latents=_as_tensor(future_visual, dtype=torch.float16),
        future_proprio=_as_tensor(future_proprio, dtype=torch.float32),
        source_phase=record.phases[source_tick] or "terminal",
        source_status=record.statuses[source_tick],
        boundary_ticks=boundary_ticks,
        executed_control_ticks=executed_ticks,
        executable_control_ticks=executable_ticks,
        target_ticks=target_ticks,
        target_phases=target_phases,
        target_statuses=target_statuses,
        target_absorbing=target_absorbing,
    )


@dataclass(frozen=True)
class ActionConditionedJepaCorpus:
    records: tuple[JepaEpisodeRecord, ...]
    normalization: JepaProprioNormalization
    source_manifest_sha256: str
    cache_manifest_sha256: str
    split_manifest_sha256: str
    indices: tuple[JepaSampleIndex, ...] = field(init=False)
    _records_by_id: MappingProxyType = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.records) is not tuple or not self.records:
            raise ValueError("JEPA corpus requires at least one episode record")
        ordered = tuple(sorted(self.records, key=lambda record: record.episode_id))
        if len({record.episode_id for record in ordered}) != len(ordered):
            raise ValueError("JEPA corpus episode IDs must be unique")
        levels = {record.level for record in ordered}
        splits = {record.split for record in ordered}
        if len(levels) != 1 or len(splits) != 1:
            raise ValueError("one JEPA corpus must contain exactly one level and split")
        if self.normalization.level != ordered[0].level:
            raise ValueError("JEPA corpus normalization level is inconsistent")
        if (
            self.normalization.source_manifest_sha256 != self.source_manifest_sha256
            or self.normalization.split_manifest_sha256 != self.split_manifest_sha256
        ):
            raise ValueError("JEPA corpus normalization provenance is inconsistent")
        indices = _indices_for_records(ordered)
        if ordered[0].split == "train":
            if self.normalization.episode_ids != tuple(record.episode_id for record in ordered):
                raise ValueError("train normalization episode inventory is inconsistent")
            if self.normalization.sample_index_sha256 != _sample_index_sha256(ordered):
                raise ValueError("train normalization sample index is inconsistent")
        for name in (
            "source_manifest_sha256",
            "cache_manifest_sha256",
            "split_manifest_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        object.__setattr__(self, "records", ordered)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(
            self,
            "_records_by_id",
            MappingProxyType({record.episode_id: record for record in ordered}),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> JepaTrainingSample:
        sample = self.indices[index]
        return materialize_jepa_sample(
            self._records_by_id[sample.episode_id],
            source_tick=sample.source_tick,
            normalization=self.normalization,
        )

    def sample_indices_for_episode(self, episode_id: str) -> tuple[JepaSampleIndex, ...]:
        if episode_id not in self._records_by_id:
            raise KeyError(f"unknown JEPA corpus episode: {episode_id}")
        return tuple(index for index in self.indices if index.episode_id == episode_id)


@dataclass(frozen=True)
class VerifiedJepaInputs:
    source: VerifiedSourceCorpus
    split: SourceSplitManifest
    cache_root: Path
    cache_manifest: MappingProxyType
    cache_rows_by_episode: MappingProxyType
    config: ActionConditionedJepaConfig
    source_manifest_sha256: str
    cache_manifest_sha256: str
    split_manifest_sha256: str
    verify_payloads: bool = True


def load_verified_jepa_inputs(
    *,
    source_root: Path,
    cache_run_manifest: Path,
    split_manifest_path: Path,
    config: ActionConditionedJepaConfig,
    verify_payloads: bool = True,
    required_episode_ids: tuple[str, ...] | None = None,
) -> VerifiedJepaInputs:
    if not isinstance(config, ActionConditionedJepaConfig):
        raise TypeError("config must be an ActionConditionedJepaConfig")
    source = load_verified_source_corpus(Path(source_root), verify_payloads=verify_payloads)
    source_sha = _hash_file(source.root / "manifest.json")
    split_path = Path(split_manifest_path).resolve()
    split = load_verified_source_split(split_path, source)
    cache_path = Path(cache_run_manifest).resolve()
    if cache_path.name != "manifest.json":
        raise ValueError("cache_run_manifest must name manifest.json")
    cache_manifest = load_verified_vision_feature_cache_run(
        cache_path.parent,
        expected_source_manifest_sha256=source_sha,
        expected_spec=config.vision_encoder,
        verify_payloads=verify_payloads,
    )
    if cache_manifest["eligible"] is not True or cache_manifest["blockers"] != []:
        raise ValueError("JEPA requires an eligible complete vision cache")
    source_ids = {
        episode_id for level in (1, 2, 3) for episode_id in source.episode_ids(level=level)
    }
    required_ids = source_ids if required_episode_ids is None else set(required_episode_ids)
    if not required_ids or not required_ids <= source_ids:
        raise ValueError("Required cache episodes are not part of this source")
    cache_rows = {row["episode_id"]: row for row in cache_manifest["episodes"]}
    if (
        len(cache_rows) != len(cache_manifest["episodes"])
        or not required_ids <= set(cache_rows) <= source_ids
        or cache_manifest["source_corpus_id"] != source.manifest.corpus_id
    ):
        raise ValueError("JEPA source/cache episode join is incomplete")
    return VerifiedJepaInputs(
        source=source,
        split=split,
        cache_root=cache_path.parent,
        cache_manifest=MappingProxyType(cache_manifest),
        cache_rows_by_episode=MappingProxyType(cache_rows),
        config=config,
        source_manifest_sha256=source_sha,
        cache_manifest_sha256=_hash_file(cache_path),
        split_manifest_sha256=_hash_file(split_path),
        verify_payloads=verify_payloads,
    )


def _physical_proprio(rows: list[dict[str, Any]]) -> np.ndarray:
    qpos = np.asarray([row["robot_qpos"] for row in rows], dtype=np.float32)
    qvel = np.asarray([row["robot_qvel"] for row in rows], dtype=np.float32)
    gripper_qpos = np.asarray([row["gripper_qpos"] for row in rows], dtype=np.float32)
    gripper_qvel = np.asarray([row["gripper_qvel"] for row in rows], dtype=np.float32)
    width = (gripper_qpos[:, :1] - gripper_qpos[:, 1:2]).astype(np.float32)
    velocity = (gripper_qvel[:, :1] - gripper_qvel[:, 1:2]).astype(np.float32)
    return np.concatenate((qpos, qvel, width, velocity), axis=1)


def load_verified_jepa_record(
    inputs: VerifiedJepaInputs,
    *,
    episode_id: str,
    level: int,
    split: str,
) -> JepaEpisodeRecord:
    if not isinstance(inputs, VerifiedJepaInputs):
        raise TypeError("inputs must be VerifiedJepaInputs")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    admitted = (
        inputs.split.train_episode_ids if split == "train" else inputs.split.validation_episode_ids
    )
    if episode_id not in admitted or episode_id not in inputs.cache_rows_by_episode:
        raise ValueError("episode is not admitted by the requested JEPA split")
    cache_row = inputs.cache_rows_by_episode[episode_id]
    if cache_row["level"] != level:
        raise ValueError("episode level disagrees with the requested JEPA level")
    table = inputs.source.read_fields(
        episode_id,
        fields=_SOURCE_FIELDS,
        allowed_roles=_SOURCE_ROLES,
    )
    rows = table.to_pylist()
    ticks = [row["formal_tick"] for row in rows]
    if ticks != list(range(len(rows))) or len(rows) != cache_row["boundary_count"]:
        raise ValueError("JEPA source boundaries are not contiguous or cache-aligned")
    terminal_tick = len(rows) - 1
    if (
        rows[-1]["outcome_status"] != "success"
        or rows[-1]["expert_action"] is not None
        or rows[-1]["action_mask"] is not None
        or any(row["expert_action"] is None for row in rows[:-1])
        or any(row["action_mask"] != [True] * _ACTION_DIM for row in rows[:-1])
    ):
        raise ValueError("JEPA source terminal/action semantics are invalid")
    controls = np.asarray([row["expert_action"] for row in rows[:-1]], dtype=np.float32)
    cache = load_episode_vision_feature_cache(
        inputs.cache_root / Path(cache_row["cache_manifest"]).parent,
        expected_spec=inputs.config.vision_encoder,
        verify_payloads=inputs.verify_payloads,
    )
    return JepaEpisodeRecord(
        episode_id=episode_id,
        task_instance_id=cache_row["task_instance_id"],
        logical_master_task_index=cache_row["logical_master_task_index"],
        level=level,
        split=split,
        terminal_tick=terminal_tick,
        cache=cache,
        proprio_physical=_physical_proprio(rows),
        controls=controls,
        phases=tuple(row["phase_id"] for row in rows),
        statuses=tuple(row["outcome_status"] for row in rows),
    )


def load_action_conditioned_jepa_corpus(
    *,
    source_root: Path,
    cache_run_manifest: Path,
    split_manifest_path: Path,
    level: int,
    split: str,
    config: ActionConditionedJepaConfig,
    normalization: JepaProprioNormalization | None = None,
    verify_payloads: bool = True,
) -> ActionConditionedJepaCorpus:
    if type(level) is not int or level not in (1, 2, 3) or config.level != level:
        raise ValueError("requested level must match the JEPA level config")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    inputs = load_verified_jepa_inputs(
        source_root=source_root,
        cache_run_manifest=cache_run_manifest,
        split_manifest_path=split_manifest_path,
        config=config,
        verify_payloads=verify_payloads,
    )
    admitted = (
        inputs.split.train_episode_ids if split == "train" else inputs.split.validation_episode_ids
    )
    level_ids = set(inputs.source.episode_ids(level=level))
    episode_ids = tuple(sorted(level_ids.intersection(admitted)))
    records = tuple(
        load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=level,
            split=split,
        )
        for episode_id in episode_ids
    )
    if split == "train":
        if normalization is None:
            normalization = compute_jepa_proprio_normalization(
                records=records,
                source_manifest_sha256=inputs.source_manifest_sha256,
                split_manifest_sha256=inputs.split_manifest_sha256,
            )
    elif normalization is None:
        raise ValueError("validation corpus requires the matching train normalization")
    return ActionConditionedJepaCorpus(
        records=records,
        normalization=normalization,
        source_manifest_sha256=inputs.source_manifest_sha256,
        cache_manifest_sha256=inputs.cache_manifest_sha256,
        split_manifest_sha256=inputs.split_manifest_sha256,
    )
