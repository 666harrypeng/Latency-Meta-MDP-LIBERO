"""Index-only physical-time views over the canonical 50 Hz JEPA source cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.config import JepaTemporalSampling
from latency_meta_mdp.belief.jepa.contracts import (
    FutureLatentRollout,
    LaunchContextBatch,
)
from latency_meta_mdp.belief.jepa.corpus import (
    JepaEpisodeRecord,
    JepaProprioNormalization,
    formal_hold_suffix,
)

BoundaryDisposition = Literal["recorded_complete", "certified_absorbing_extension"]
_DISPOSITIONS = frozenset(("recorded_complete", "certified_absorbing_extension"))


def _readonly_int64(values: tuple[int, ...] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


def _readonly_bool(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.bool_)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, order=True)
class SharedJepaSampleIndex:
    """One launch identity shared by every temporal candidate during selection."""

    level: int
    split: str
    episode_id: str
    source_tick: int
    boundary_disposition: BoundaryDisposition

    def __post_init__(self) -> None:
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("shared JEPA sample level must be 1, 2, or 3")
        if self.split not in {"train", "validation"}:
            raise ValueError("shared JEPA sample split must be train or validation")
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("shared JEPA sample episode_id cannot be empty")
        if type(self.source_tick) is not int or self.source_tick < 0:
            raise ValueError("shared JEPA sample source_tick must be nonnegative")
        if self.boundary_disposition not in _DISPOSITIONS:
            raise ValueError("shared JEPA sample boundary disposition is invalid")


@dataclass(frozen=True)
class TemporalJepaTrainingSample:
    index: SharedJepaSampleIndex
    temporal_config_id: str
    vision_history: torch.Tensor
    proprio_history: torch.Tensor
    proprio_history_physical: torch.Tensor
    past_macro_controls: torch.Tensor
    future_macro_controls: torch.Tensor
    future_visual_latents: torch.Tensor
    future_proprio_normalized: torch.Tensor
    future_proprio: torch.Tensor
    history_ticks: np.ndarray
    past_macro_control_ticks: np.ndarray
    future_macro_control_ticks: np.ndarray
    native_target_ticks: np.ndarray
    native_target_rows: np.ndarray
    target_absorbing: np.ndarray

    @property
    def launch_context(self) -> LaunchContextBatch:
        return LaunchContextBatch(
            vision_history=self.vision_history.unsqueeze(0),
            proprio_history=self.proprio_history.unsqueeze(0),
            executed_controls=self.past_macro_controls.unsqueeze(0),
            executable_controls=self.future_macro_controls.unsqueeze(0),
        )

    @property
    def future_rollout(self) -> FutureLatentRollout:
        return FutureLatentRollout(
            native_delay_ticks=torch.from_numpy(np.array(self.native_target_ticks, copy=True))
            - self.index.source_tick,
            future_visual_latents=self.future_visual_latents.unsqueeze(0),
            future_proprio=self.future_proprio.unsqueeze(0),
        )


@dataclass(frozen=True)
class TemporalJepaBatch:
    indices: tuple[SharedJepaSampleIndex, ...]
    temporal_config_id: str
    context: LaunchContextBatch
    target_visual_latents: torch.Tensor
    target_proprio_normalized: torch.Tensor
    target_proprio_physical: torch.Tensor
    target_absorbing: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.context.batch_size

    def to(self, device: torch.device, *, non_blocking: bool = False) -> TemporalJepaBatch:
        context = _context_to(self.context, device=device, non_blocking=non_blocking)
        return TemporalJepaBatch(
            indices=self.indices,
            temporal_config_id=self.temporal_config_id,
            context=context,
            target_visual_latents=self.target_visual_latents.to(device, non_blocking=non_blocking),
            target_proprio_normalized=self.target_proprio_normalized.to(
                device, non_blocking=non_blocking
            ),
            target_proprio_physical=self.target_proprio_physical.to(
                device, non_blocking=non_blocking
            ),
            target_absorbing=self.target_absorbing.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class TemporalJepaEvaluationBatch:
    indices: tuple[SharedJepaSampleIndex, ...]
    temporal_config_id: str
    context: LaunchContextBatch
    current_proprio_physical: torch.Tensor
    target_visual_latents: torch.Tensor
    target_proprio_physical: torch.Tensor
    target_absorbing: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.context.batch_size

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> TemporalJepaEvaluationBatch:
        context = _context_to(self.context, device=device, non_blocking=non_blocking)
        return TemporalJepaEvaluationBatch(
            indices=self.indices,
            temporal_config_id=self.temporal_config_id,
            context=context,
            current_proprio_physical=self.current_proprio_physical.to(
                device, non_blocking=non_blocking
            ),
            target_visual_latents=self.target_visual_latents.to(device, non_blocking=non_blocking),
            target_proprio_physical=self.target_proprio_physical.to(
                device, non_blocking=non_blocking
            ),
            target_absorbing=self.target_absorbing.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class TemporalJepaDeployedEvaluationSample:
    """One native model input paired with the complete 20 ms D20 target grid."""

    native_sample: TemporalJepaTrainingSample
    dense_delay_ticks: np.ndarray
    dense_target_visual_latents: torch.Tensor
    dense_target_proprio_physical: torch.Tensor
    dense_target_absorbing: np.ndarray


@dataclass(frozen=True)
class TemporalJepaDeployedEvaluationBatch:
    indices: tuple[SharedJepaSampleIndex, ...]
    temporal_config_id: str
    context: LaunchContextBatch
    native_delay_ticks: torch.Tensor
    dense_delay_ticks: torch.Tensor
    current_proprio_physical: torch.Tensor
    dense_target_visual_latents: torch.Tensor
    dense_target_proprio_physical: torch.Tensor
    dense_target_absorbing: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.context.batch_size

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> TemporalJepaDeployedEvaluationBatch:
        return TemporalJepaDeployedEvaluationBatch(
            indices=self.indices,
            temporal_config_id=self.temporal_config_id,
            context=_context_to(self.context, device=device, non_blocking=non_blocking),
            native_delay_ticks=self.native_delay_ticks.to(device, non_blocking=non_blocking),
            dense_delay_ticks=self.dense_delay_ticks.to(device, non_blocking=non_blocking),
            current_proprio_physical=self.current_proprio_physical.to(
                device, non_blocking=non_blocking
            ),
            dense_target_visual_latents=self.dense_target_visual_latents.to(
                device, non_blocking=non_blocking
            ),
            dense_target_proprio_physical=self.dense_target_proprio_physical.to(
                device, non_blocking=non_blocking
            ),
            dense_target_absorbing=self.dense_target_absorbing.to(
                device, non_blocking=non_blocking
            ),
        )


def _context_to(
    context: LaunchContextBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> LaunchContextBatch:
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device")
    return LaunchContextBatch(
        vision_history=context.vision_history.to(device, non_blocking=non_blocking),
        proprio_history=context.proprio_history.to(device, non_blocking=non_blocking),
        executed_controls=context.executed_controls.to(device, non_blocking=non_blocking),
        executable_controls=context.executable_controls.to(device, non_blocking=non_blocking),
    )


class TemporalJepaCorpus:
    """Lazy fold view over verified records and one shared launch-index inventory."""

    def __init__(
        self,
        *,
        records: tuple[JepaEpisodeRecord, ...],
        indices: tuple[SharedJepaSampleIndex, ...],
        episode_ids: tuple[str, ...],
        partition: Literal["fit", "development"],
        sampling: JepaTemporalSampling,
        normalization: JepaProprioNormalization,
    ) -> None:
        if (
            type(records) is not tuple
            or not records
            or any(not isinstance(value, JepaEpisodeRecord) for value in records)
        ):
            raise ValueError("records must be a non-empty tuple")
        records_by_id = {record.episode_id: record for record in records}
        if len(records_by_id) != len(records):
            raise ValueError("records must have unique episode identities")
        if (
            type(episode_ids) is not tuple
            or not episode_ids
            or episode_ids != tuple(sorted(set(episode_ids)))
            or not set(episode_ids).issubset(records_by_id)
        ):
            raise ValueError("episode_ids must select a non-empty record subset")
        if not isinstance(sampling, JepaTemporalSampling):
            raise TypeError("sampling must be JepaTemporalSampling")
        if not isinstance(normalization, JepaProprioNormalization):
            raise TypeError("normalization must be JepaProprioNormalization")
        if partition not in {"fit", "development"}:
            raise ValueError("partition must be fit or development")
        normalized_episodes = set(normalization.episode_ids)
        selected_episodes = set(episode_ids)
        if partition == "fit" and normalized_episodes != selected_episodes:
            raise ValueError("fit normalization must match selected fit episodes exactly")
        if partition == "development" and normalized_episodes.intersection(selected_episodes):
            raise ValueError("development episodes cannot enter fit normalization")
        selected = tuple(index for index in indices if index.episode_id in set(episode_ids))
        if not selected or selected != tuple(sorted(set(selected))):
            raise ValueError("indices must provide a sorted unique selected inventory")
        self.records = tuple(records_by_id[value] for value in episode_ids)
        self.indices = selected
        self.partition = partition
        self.sampling = sampling
        self.normalization = normalization
        self._records_by_id = records_by_id

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> TemporalJepaTrainingSample:
        index = self.indices[item]
        return materialize_temporal_jepa_sample(
            record=self._records_by_id[index.episode_id],
            index=index,
            sampling=self.sampling,
            normalization=self.normalization,
        )


class TemporalJepaDeployedEvaluationCorpus:
    """Evaluation-only D20 view; training remains limited to native D1/K2 targets."""

    def __init__(self, native_corpus: TemporalJepaCorpus) -> None:
        if not isinstance(native_corpus, TemporalJepaCorpus):
            raise TypeError("native_corpus must be TemporalJepaCorpus")
        if native_corpus.partition != "development":
            raise ValueError("deployed evaluation requires a development corpus")
        self.native_corpus = native_corpus
        self.indices = native_corpus.indices

    def __len__(self) -> int:
        return len(self.native_corpus)

    def __getitem__(self, item: int) -> TemporalJepaDeployedEvaluationSample:
        native_sample = self.native_corpus[item]
        record = self.native_corpus._records_by_id[native_sample.index.episode_id]
        dense_delay_ticks = _readonly_int64(tuple(range(1, 21)))
        target_ticks = native_sample.index.source_tick + dense_delay_ticks
        target_rows = np.minimum(target_ticks, record.terminal_tick)
        target_absorbing = _readonly_bool(target_ticks > record.terminal_tick)
        future_proprio = np.array(record.proprio_physical[target_rows], copy=True)
        future_proprio[target_absorbing, 7:14] = 0.0
        future_proprio[target_absorbing, 15] = 0.0
        return TemporalJepaDeployedEvaluationSample(
            native_sample=native_sample,
            dense_delay_ticks=dense_delay_ticks,
            dense_target_visual_latents=torch.from_numpy(
                np.array(record.cache.features[target_rows], copy=True)
            ).to(torch.float16),
            dense_target_proprio_physical=torch.from_numpy(future_proprio).to(torch.float32),
            dense_target_absorbing=target_absorbing,
        )


def collate_temporal_jepa_samples(
    samples: Sequence[TemporalJepaTrainingSample],
) -> TemporalJepaBatch:
    values = tuple(samples)
    if not values or any(not isinstance(value, TemporalJepaTrainingSample) for value in values):
        raise ValueError("samples must contain TemporalJepaTrainingSample values")
    config_ids = {sample.temporal_config_id for sample in values}
    if len(config_ids) != 1:
        raise ValueError("one temporal batch cannot mix model configurations")
    context = LaunchContextBatch(
        vision_history=torch.stack(tuple(value.vision_history for value in values)),
        proprio_history=torch.stack(tuple(value.proprio_history for value in values)),
        executed_controls=torch.stack(tuple(value.past_macro_controls for value in values)),
        executable_controls=torch.stack(tuple(value.future_macro_controls for value in values)),
    )
    return TemporalJepaBatch(
        indices=tuple(value.index for value in values),
        temporal_config_id=next(iter(config_ids)),
        context=context,
        target_visual_latents=torch.stack(
            tuple(value.future_visual_latents[:2] for value in values)
        ),
        target_proprio_normalized=torch.stack(
            tuple(value.future_proprio_normalized[:2] for value in values)
        ),
        target_proprio_physical=torch.stack(tuple(value.future_proprio[:2] for value in values)),
        target_absorbing=torch.from_numpy(
            np.stack(tuple(value.target_absorbing[:2] for value in values))
        ).to(torch.bool),
    )


def collate_temporal_jepa_evaluation_samples(
    samples: Sequence[TemporalJepaTrainingSample],
) -> TemporalJepaEvaluationBatch:
    values = tuple(samples)
    if not values or any(not isinstance(value, TemporalJepaTrainingSample) for value in values):
        raise ValueError("samples must contain TemporalJepaTrainingSample values")
    config_ids = {sample.temporal_config_id for sample in values}
    if len(config_ids) != 1:
        raise ValueError("one evaluation batch cannot mix model configurations")
    context = LaunchContextBatch(
        vision_history=torch.stack(tuple(value.vision_history for value in values)),
        proprio_history=torch.stack(tuple(value.proprio_history for value in values)),
        executed_controls=torch.stack(tuple(value.past_macro_controls for value in values)),
        executable_controls=torch.stack(tuple(value.future_macro_controls for value in values)),
    )
    return TemporalJepaEvaluationBatch(
        indices=tuple(value.index for value in values),
        temporal_config_id=next(iter(config_ids)),
        context=context,
        current_proprio_physical=torch.stack(
            tuple(value.proprio_history_physical[-1] for value in values)
        ),
        target_visual_latents=torch.stack(tuple(value.future_visual_latents for value in values)),
        target_proprio_physical=torch.stack(tuple(value.future_proprio for value in values)),
        target_absorbing=torch.from_numpy(
            np.stack(tuple(value.target_absorbing for value in values))
        ).to(torch.bool),
    )


def collate_temporal_jepa_deployed_evaluation_samples(
    samples: Sequence[TemporalJepaDeployedEvaluationSample],
) -> TemporalJepaDeployedEvaluationBatch:
    values = tuple(samples)
    if not values or any(
        not isinstance(value, TemporalJepaDeployedEvaluationSample) for value in values
    ):
        raise ValueError("samples must contain TemporalJepaDeployedEvaluationSample values")
    native_values = tuple(value.native_sample for value in values)
    native_batch = collate_temporal_jepa_evaluation_samples(native_values)
    dense_ticks = tuple(tuple(value.dense_delay_ticks.tolist()) for value in values)
    if len(set(dense_ticks)) != 1:
        raise ValueError("deployed evaluation samples must share the D20 grid")
    native_delay_ticks = native_values[0].future_rollout.native_delay_ticks.detach().clone()
    if any(
        not torch.equal(value.future_rollout.native_delay_ticks, native_delay_ticks)
        for value in native_values[1:]
    ):
        raise ValueError("deployed evaluation samples must share native delay anchors")
    return TemporalJepaDeployedEvaluationBatch(
        indices=native_batch.indices,
        temporal_config_id=native_batch.temporal_config_id,
        context=native_batch.context,
        native_delay_ticks=native_delay_ticks,
        dense_delay_ticks=torch.from_numpy(np.array(values[0].dense_delay_ticks, copy=True)).to(
            torch.int64
        ),
        current_proprio_physical=native_batch.current_proprio_physical,
        dense_target_visual_latents=torch.stack(
            tuple(value.dense_target_visual_latents for value in values)
        ),
        dense_target_proprio_physical=torch.stack(
            tuple(value.dense_target_proprio_physical for value in values)
        ),
        dense_target_absorbing=torch.from_numpy(
            np.stack(tuple(value.dense_target_absorbing for value in values))
        ).to(torch.bool),
    )


def build_shared_temporal_indices(
    *,
    records: tuple[JepaEpisodeRecord, ...],
    samplings: tuple[JepaTemporalSampling, ...],
) -> tuple[SharedJepaSampleIndex, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(not isinstance(record, JepaEpisodeRecord) for record in records)
    ):
        raise ValueError("records must be a non-empty tuple of JEPA episode records")
    if (
        type(samplings) is not tuple
        or not samplings
        or any(not isinstance(value, JepaTemporalSampling) for value in samplings)
    ):
        raise ValueError("samplings must be a non-empty tuple of JEPA temporal configs")
    if len({value.config_id for value in samplings}) != len(samplings):
        raise ValueError("temporal config identities must be unique")
    maximum_history = max(value.history_span_ticks for value in samplings)
    maximum_delay = {value.maximum_delay_ticks for value in samplings}
    if maximum_delay != {20}:
        raise ValueError("all temporal configs must share the D20 physical horizon")

    result = []
    for record in sorted(records, key=lambda value: value.episode_id):
        for source_tick in range(maximum_history, record.terminal_tick):
            disposition: BoundaryDisposition = (
                "recorded_complete"
                if source_tick + 20 <= record.terminal_tick
                else "certified_absorbing_extension"
            )
            result.append(
                SharedJepaSampleIndex(
                    level=record.level,
                    split=record.split,
                    episode_id=record.episode_id,
                    source_tick=source_tick,
                    boundary_disposition=disposition,
                )
            )
    return tuple(result)


def materialize_temporal_jepa_sample(
    *,
    record: JepaEpisodeRecord,
    index: SharedJepaSampleIndex,
    sampling: JepaTemporalSampling,
    normalization: JepaProprioNormalization,
) -> TemporalJepaTrainingSample:
    if not isinstance(record, JepaEpisodeRecord):
        raise TypeError("record must be a JepaEpisodeRecord")
    if not isinstance(index, SharedJepaSampleIndex):
        raise TypeError("index must be a SharedJepaSampleIndex")
    if not isinstance(sampling, JepaTemporalSampling):
        raise TypeError("sampling must be a JepaTemporalSampling")
    if not isinstance(normalization, JepaProprioNormalization):
        raise TypeError("normalization must be JepaProprioNormalization")
    if (index.episode_id, index.level, index.split) != (
        record.episode_id,
        record.level,
        record.split,
    ):
        raise ValueError("sample index does not identify the provided episode")
    if normalization.level != record.level:
        raise ValueError("proprio normalization level does not match the episode")
    if index.source_tick < sampling.history_span_ticks or index.source_tick >= record.terminal_tick:
        raise ValueError("source tick lacks required history or precedes no future transition")

    expected_disposition: BoundaryDisposition = (
        "recorded_complete"
        if index.source_tick + sampling.maximum_delay_ticks <= record.terminal_tick
        else "certified_absorbing_extension"
    )
    if index.boundary_disposition != expected_disposition:
        raise ValueError("sample boundary disposition disagrees with the episode timeline")

    source_tick = index.source_tick
    history_ticks = _readonly_int64(
        tuple(source_tick + offset for offset in sampling.history_source_offsets)
    )
    past_control_ticks = _readonly_int64(
        np.asarray(sampling.past_macro_control_offsets, dtype=np.int64) + source_tick
    )
    future_control_ticks = _readonly_int64(
        np.asarray(sampling.future_macro_control_offsets, dtype=np.int64) + source_tick
    )
    target_ticks = _readonly_int64(
        tuple(source_tick + offset for offset in sampling.native_future_offsets)
    )
    target_rows = _readonly_int64(np.minimum(target_ticks, record.terminal_tick))
    target_absorbing = _readonly_bool(target_ticks > record.terminal_tick)

    future_control_rows = future_control_ticks.reshape(-1)
    future_controls = np.empty((sampling.maximum_delay_ticks, 7), dtype=np.float32)
    real = future_control_rows < record.terminal_tick
    future_controls[real] = record.controls[future_control_rows[real]]
    future_controls[~real] = formal_hold_suffix(
        record.last_real_gripper_command,
        int((~real).sum()),
    )
    future_controls = future_controls.reshape(
        sampling.native_rollout_steps,
        sampling.model_stride_ticks,
        7,
    )

    future_proprio = np.array(record.proprio_physical[target_rows], copy=True)
    future_proprio[target_absorbing, 7:14] = 0.0
    future_proprio[target_absorbing, 15] = 0.0

    return TemporalJepaTrainingSample(
        index=index,
        temporal_config_id=sampling.config_id,
        vision_history=torch.from_numpy(
            np.array(record.cache.features[history_ticks], copy=True)
        ).to(torch.float16),
        proprio_history=torch.from_numpy(
            normalization.normalize(record.proprio_physical[history_ticks])
        ).to(torch.float32),
        proprio_history_physical=torch.from_numpy(
            np.array(record.proprio_physical[history_ticks], copy=True)
        ).to(torch.float32),
        past_macro_controls=torch.from_numpy(
            np.array(record.controls[past_control_ticks], copy=True)
        ).to(torch.float32),
        future_macro_controls=torch.from_numpy(np.array(future_controls, copy=True)).to(
            torch.float32
        ),
        future_visual_latents=torch.from_numpy(
            np.array(record.cache.features[target_rows], copy=True)
        ).to(torch.float16),
        future_proprio_normalized=torch.from_numpy(normalization.normalize(future_proprio)).to(
            torch.float32
        ),
        future_proprio=torch.from_numpy(future_proprio).to(torch.float32),
        history_ticks=history_ticks,
        past_macro_control_ticks=past_control_ticks,
        future_macro_control_ticks=future_control_ticks,
        native_target_ticks=target_ticks,
        native_target_rows=target_rows,
        target_absorbing=target_absorbing,
    )
