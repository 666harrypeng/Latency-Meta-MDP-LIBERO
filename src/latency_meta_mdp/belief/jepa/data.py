"""Native 50 Hz (source, horizon) pairs; cache data stays shared and memory mapped."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from latency_meta_mdp.belief.jepa.contracts import ForecastQuery
from latency_meta_mdp.belief.jepa.corpus import (
    JepaEpisodeRecord,
    JepaProprioNormalization,
)
from latency_meta_mdp.belief.jepa.identity import data_domain

FIRST_SOURCE_TICK = 10  # Matches the admitted stride4 training source start.


@dataclass(frozen=True)
class DirectPredictionSample:
    query: ForecastQuery
    target_visual: torch.Tensor
    target_proprio: torch.Tensor


def materialize_direct_sample(
    record: JepaEpisodeRecord,
    *,
    source_tick: int,
    query_ticks: int,
    normalization: JepaProprioNormalization,
) -> DirectPredictionSample:
    if (
        type(source_tick) is not int
        or source_tick < FIRST_SOURCE_TICK
        or type(query_ticks) is not int
        or not 1 <= query_ticks <= 20
        or source_tick + query_ticks > record.terminal_tick
    ):
        raise ValueError(
            "direct training pair must have real history, controls and future endpoint"
        )
    query = materialize_direct_query(
        record, source_tick=source_tick, query_ticks=query_ticks, normalization=normalization
    )
    return DirectPredictionSample(
        query=query,
        target_visual=torch.from_numpy(
            np.array(record.cache.features[source_tick + query_ticks], copy=True)
        )
        .half()
        .unsqueeze(0),
        target_proprio=torch.from_numpy(
            np.array(record.proprio_physical[source_tick + query_ticks], copy=True)
        )
        .float()
        .unsqueeze(0),
    )


def materialize_direct_query(
    record: JepaEpisodeRecord,
    *,
    source_tick: int,
    query_ticks: int,
    normalization: JepaProprioNormalization,
    executable_controls: np.ndarray | None = None,
) -> ForecastQuery:
    """Build inputs only; explicit deployment buffers need no recorded future endpoint.

    Without an explicit buffer, only real recorded controls may be used. No GT future
    image or proprio is read, and later-than-q controls are removed before model input.
    """
    if (
        type(source_tick) is not int
        or not FIRST_SOURCE_TICK <= source_tick <= record.terminal_tick
        or type(query_ticks) is not int
        or not 0 <= query_ticks <= 20
    ):
        raise ValueError("direct query requires real history and a supported query horizon")
    if data_domain(normalization) != data_domain(record):
        raise ValueError("direct sample normalization domain identity mismatch")
    if executable_controls is None:
        if source_tick + query_ticks > record.terminal_tick:
            raise ValueError("query beyond the recording requires an explicit executable buffer")
        prefix = record.controls[source_tick : source_tick + query_ticks]
    else:
        buffer = np.asarray(executable_controls)
        if buffer.shape != (20, 7):
            raise ValueError("executable buffer must have shape20x7")
        prefix = buffer[:query_ticks]
    future_controls = np.zeros((20, 7), dtype=np.float32)
    future_controls[:query_ticks] = prefix

    def tensor(value, dtype=torch.float32):
        return torch.from_numpy(np.array(value, copy=True)).to(dtype).unsqueeze(0)

    history = np.array([source_tick - 8, source_tick - 4, source_tick])
    query = ForecastQuery(
        vision_history=tensor(record.cache.features[history], torch.float16),
        proprio_history=tensor(normalization.normalize(record.proprio_physical[history])),
        executed_controls=tensor(record.controls[source_tick - 8 : source_tick].reshape(2, 4, 7)),
        executable_controls=tensor(future_controls),
        control_mask=torch.arange(20)[None] < query_ticks,
        query_ticks=torch.tensor([query_ticks]),
        source_ticks=torch.tensor([source_tick]),
    )
    query.validate_finite()
    return query


class DirectPredictionDataset(Dataset):
    """Compact horizon-major index; every legal pair exists, without synthetic tails."""

    def __init__(
        self, *, records: tuple[JepaEpisodeRecord, ...], normalization: JepaProprioNormalization
    ):
        if not records or len({(data_domain(r), r.split) for r in records}) != 1:
            raise ValueError("direct dataset requires one nonempty level/split")
        if len({r.episode_id for r in records}) != len(records):
            raise ValueError("direct dataset episode IDs must be unique")
        self.records = tuple(sorted(records, key=lambda r: r.episode_id))
        self.normalization = normalization
        if data_domain(normalization) != data_domain(records[0]):
            raise ValueError("direct dataset normalization domain identity mismatch")
        if records[0].split == "train" and normalization.episode_ids != tuple(
            r.episode_id for r in self.records
        ):
            raise ValueError(
                "direct training normalization must match the complete train inventory"
            )
        if records[0].split == "validation" and set(normalization.episode_ids).intersection(
            r.episode_id for r in records
        ):
            raise ValueError("validation episodes cannot enter training normalization")
        self._episode_ends = tuple(
            np.cumsum(
                [max(0, r.terminal_tick - FIRST_SOURCE_TICK - q + 1) for r in self.records]
            ).tolist()
            for q in range(1, 21)
        )
        self.horizon_counts = tuple(ends[-1] for ends in self._episode_ends)
        self.horizon_ends = np.cumsum(self.horizon_counts).tolist()
        if any(count == 0 for count in self.horizon_counts):
            raise ValueError("direct dataset must contain real pairs for all twenty horizons")

    def __len__(self):
        return self.horizon_ends[-1]

    def pair_at(self, index: int) -> tuple[int, int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        q_index = bisect_right(self.horizon_ends, index)
        local = index - (self.horizon_ends[q_index - 1] if q_index else 0)
        ends = self._episode_ends[q_index]
        episode = bisect_right(ends, local)
        source = FIRST_SOURCE_TICK + local - (ends[episode - 1] if episode else 0)
        return episode, source, q_index + 1

    def __getitem__(self, index):
        episode, source, q = self.pair_at(index)
        return materialize_direct_sample(
            self.records[episode],
            source_tick=source,
            query_ticks=q,
            normalization=self.normalization,
        )


class BalancedQuerySampler(Sampler):
    """Exactly equal q exposure per epoch, uniform valid pairs within q with replacement.

    An epoch is a declared exposure budget, not a traversal of every possible pair.
    Longer horizons are not downweighted just because fewer late sources admit them.
    """

    def __init__(self, dataset: DirectPredictionDataset, *, sample_count: int, seed: int):
        if type(sample_count) is not int or sample_count <= 0 or sample_count % 20:
            raise ValueError("balanced sample count must be a positive multiple of twenty")
        self.dataset, self.sample_count, self.seed, self.epoch = dataset, sample_count, seed, 0

    def __len__(self):
        return self.sample_count

    def set_epoch(self, epoch: int):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        indices = []
        offset = 0
        for count in self.dataset.horizon_counts:
            indices.extend((offset + rng.integers(count, size=self.sample_count // 20)).tolist())
            offset += count
        rng.shuffle(indices)
        return iter(indices)


def collate_direct_samples(samples: list[DirectPredictionSample]) -> DirectPredictionSample:
    return DirectPredictionSample(
        query=ForecastQuery(
            **{
                name: torch.cat([getattr(s.query, name) for s in samples])
                for name in vars(samples[0].query)
            }
        ),
        target_visual=torch.cat([s.target_visual for s in samples]),
        target_proprio=torch.cat([s.target_proprio for s in samples]),
    )
