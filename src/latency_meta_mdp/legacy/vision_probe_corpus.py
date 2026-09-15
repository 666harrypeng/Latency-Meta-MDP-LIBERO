"""Verified per-level corpus over synchronized episodes and DINO feature caches."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.data.episode_split import load_episode_split_plan
from latency_meta_mdp.data.vision.cache import (
    EpisodeVisionFeatureCache,
    load_episode_vision_feature_cache,
)
from latency_meta_mdp.data.vision.contracts import VisionEncoderSpec
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief_data import BeliefEpisodeView, load_belief_episode
from latency_meta_mdp.legacy.vision_probe_data import (
    ProbeSplit,
    VisionProbeSample,
    VisionProbeSampleIndex,
    build_probe_sample_indices,
    materialize_probe_sample,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class VisionProbeEpisodeRecord:
    episode: BeliefEpisodeView
    cache: EpisodeVisionFeatureCache
    indices: tuple[VisionProbeSampleIndex, ...]


@dataclass(frozen=True)
class VisionProbeCorpus:
    level: int
    history_sample_count: int
    records: tuple[VisionProbeEpisodeRecord, ...]
    sample_references: dict[ProbeSplit, tuple[tuple[int, int], ...]]

    @property
    def episode_counts(self) -> dict[ProbeSplit, int]:
        counts = Counter(record.indices[0].split for record in self.records if record.indices)
        return {split: counts[split] for split in ProbeSplit}

    @property
    def sample_counts(self) -> dict[ProbeSplit, int]:
        return {split: len(self.sample_references[split]) for split in ProbeSplit}

    @property
    def seeds(self) -> dict[ProbeSplit, frozenset[int]]:
        values: dict[ProbeSplit, set[int]] = defaultdict(set)
        for record in self.records:
            if record.indices:
                values[record.indices[0].split].add(record.episode.scene_seed)
        return {split: frozenset(values[split]) for split in ProbeSplit}

    def materialize(self, split: ProbeSplit, offset: int) -> VisionProbeSample:
        record_index, sample_index = self.sample_references[split][offset]
        record = self.records[record_index]
        return materialize_probe_sample(
            episode=record.episode,
            features=record.cache.features,
            index=record.indices[sample_index],
            history_sample_count=self.history_sample_count,
        )


def load_level_probe_corpus(
    *,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    expected_spec: VisionEncoderSpec,
    level: int,
    history_sample_count: int,
    split_plan_path: Path | None = None,
) -> VisionProbeCorpus:
    if level not in (1, 2, 3):
        raise ValueError("probe corpus level must be 1, 2, or 3")
    source_path = source_bulk_manifest.resolve()
    cache_path = cache_run_manifest.resolve()
    source = _load_json(source_path)
    cache_run = _load_json(cache_path)
    source_format = source.get("format_id")
    if (
        source_format
        not in {
            "panda_ball_bulk_first_tranche_v1",
            "panda_ball_formal_corpus_v1",
        }
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("probe corpus requires an eligible clean bulk source")
    if (
        cache_run.get("format_id") != "vision_feature_cache_run_v1"
        or cache_run.get("eligible") is not True
        or cache_run.get("implementation_dirty") is not False
        or cache_run.get("encoder_fingerprint") != expected_spec.fingerprint
        or cache_run.get("source_bulk_manifest_sha256") != sha256_file(source_path)
    ):
        raise ValueError("probe corpus requires a matching eligible feature cache")
    source_root = source_path.parent
    cache_root = cache_path.parent
    admitted = set(source["admitted_episode_manifests"])
    source_artifacts = source["artifacts"]
    cache_artifacts = cache_run["artifacts"]
    records: list[VisionProbeEpisodeRecord] = []
    references: dict[ProbeSplit, list[tuple[int, int]]] = defaultdict(list)
    selected = [item for item in cache_run["episodes"] if item["level"] == level]
    split_plan = load_episode_split_plan(split_plan_path) if split_plan_path is not None else None
    if source_format == "panda_ball_formal_corpus_v1" and split_plan is None:
        raise ValueError("formal probe corpus requires an explicit split plan")
    if split_plan is not None and (
        source.get("seed_start") != split_plan.source_seed_start
        or source.get("seed_count_per_level") != split_plan.source_seed_count
    ):
        raise ValueError("probe split plan does not cover the source corpus")
    expected_episode_count = split_plan.source_seed_count if split_plan is not None else 25
    if len(selected) != expected_episode_count:
        raise ValueError("probe corpus cached episode count disagrees with source mode")
    for item in selected:
        source_relative = item["source_episode_manifest"]
        cache_relative = item["cache_manifest"]
        if source_relative not in admitted:
            raise ValueError("cached probe episode is not admitted by the bulk source")
        source_episode_manifest = source_root / source_relative
        if sha256_file(source_episode_manifest) != source_artifacts.get(source_relative):
            raise ValueError("probe source episode manifest hash mismatch")
        cache_manifest = cache_root / cache_relative
        if sha256_file(cache_manifest) != cache_artifacts.get(cache_relative):
            raise ValueError("probe cache episode manifest hash mismatch")
        episode = load_belief_episode(source_episode_manifest.parent)
        episode_cache = load_episode_vision_feature_cache(
            cache_manifest.parent,
            expected_spec=expected_spec,
        )
        if (
            episode.level != level
            or episode.scene_seed != item["scene_seed"]
            or episode.episode_id != item["episode_id"]
            or episode_cache.manifest["episode_id"] != episode.episode_id
            or episode_cache.features.shape[0] != episode.boundary_count
        ):
            raise ValueError("probe raw episode and feature cache identity disagree")
        indices = build_probe_sample_indices(
            episode=episode,
            history_sample_count=history_sample_count,
            split=(
                ProbeSplit(split_plan.split_for_seed(episode.scene_seed))
                if split_plan is not None
                else None
            ),
        )
        record_index = len(records)
        records.append(
            VisionProbeEpisodeRecord(
                episode=episode,
                cache=episode_cache,
                indices=indices,
            )
        )
        for sample_index, index in enumerate(indices):
            references[index.split].append((record_index, sample_index))
    return VisionProbeCorpus(
        level=level,
        history_sample_count=history_sample_count,
        records=tuple(records),
        sample_references={split: tuple(references[split]) for split in ProbeSplit},
    )
