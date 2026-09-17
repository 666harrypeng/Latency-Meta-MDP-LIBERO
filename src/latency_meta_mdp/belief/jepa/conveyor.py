"""Conveyor source adapter for the shared Direct-query dataset and predictor."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.jepa.config import load_action_conditioned_jepa_config
from latency_meta_mdp.belief.jepa.corpus import (
    JepaEpisodeRecord,
    compute_jepa_proprio_normalization,
    load_jepa_proprio_normalization,
    write_jepa_proprio_normalization,
)
from latency_meta_mdp.belief.jepa.data import DirectPredictionDataset
from latency_meta_mdp.data.conveyor.corpus import split_sources
from latency_meta_mdp.data.conveyor.vision import verify_feature_cache
from latency_meta_mdp.data.vision.cache import EpisodeVisionFeatureCache
from latency_meta_mdp.io.artifacts import sha256_file


@dataclass(frozen=True)
class ConveyorDirectCorpus:
    records: tuple[JepaEpisodeRecord, ...]
    source_manifest_sha256: str
    cache_manifest_sha256: str
    split_manifest_sha256: str


def load_conveyor_records(job, *, split):
    source_manifest = job.source_root / "manifest.json"
    cache = verify_feature_cache(job.vision_cache_manifest, corpus_manifest=source_manifest)
    if cache["purpose"] != "training_source":
        raise ValueError("conveyor Belief job requires training_source")
    config = load_action_conditioned_jepa_config(
        model_path=job.project_root / "configs/models/jepa/model.yaml",
        temporal_sampling_path=job.project_root
        / "configs/models/jepa/stride4_80ms_history_160ms.yaml",
        task_id=job.task_id,
        control_path=job.control_config,
    )
    if (
        cache["action_contract"] != config.action_contract.contract_id
        or cache["identity"]["encoder_fingerprint"] != config.vision_encoder.fingerprint
    ):
        raise ValueError("Belief source controller/encoder differs from model config")
    entries = {r["episode_id"]: r for r in cache["episodes"]}
    records = []
    for row, source in split_sources(source_manifest, split=split):
        entry = entries[row["episode_id"]]
        n = len(source.actions)
        records.append(
            JepaEpisodeRecord(
                episode_id=row["episode_id"],
                task_instance_id=row["group_id"],
                logical_master_task_index=row["seed"],
                level=None,
                split=split,
                terminal_tick=n,
                task_id=job.task_id,
                action_contract_id=config.action_contract.contract_id,
                cache=EpisodeVisionFeatureCache(
                    manifest=entry,
                    features=np.load(
                        job.vision_cache_manifest.parent / entry["features"],
                        mmap_mode="r",
                        allow_pickle=False,
                    ),
                ),
                proprio_physical=source.states,
                controls=source.actions,
                phases=(None,) * (n + 1),
                statuses=("running",) * n + ("success",),
            )
        )
    if not records:
        raise ValueError(f"no conveyor {split} records")
    source_hash = sha256_file(source_manifest)
    return config, ConveyorDirectCorpus(
        tuple(records), source_hash, sha256_file(job.vision_cache_manifest), source_hash
    )


def prepare_normalization(job):
    _, corpus = load_conveyor_records(job, split="train")
    norm = compute_jepa_proprio_normalization(
        records=corpus.records,
        source_manifest_sha256=corpus.source_manifest_sha256,
        split_manifest_sha256=corpus.split_manifest_sha256,
    )
    if job.normalization.exists():
        if load_jepa_proprio_normalization(job.normalization).to_mapping() != norm.to_mapping():
            raise ValueError("existing normalization differs from train inventory")
    else:
        write_jepa_proprio_normalization(job.normalization, norm)
    return job.normalization


def load_conveyor_direct_data(job, *, split):
    config, corpus = load_conveyor_records(job, split=split)
    norm = load_jepa_proprio_normalization(job.normalization)
    if (
        norm.source_manifest_sha256 != corpus.source_manifest_sha256
        or norm.split_manifest_sha256 != corpus.split_manifest_sha256
    ):
        raise ValueError("normalization source/split identity mismatch")
    dataset = DirectPredictionDataset(records=corpus.records, normalization=norm)
    return config, norm, corpus, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from latency_meta_mdp.belief.jepa.job import load_direct_job

    job = load_direct_job(args.config, project_root=Path.cwd())
    if job.task_id != "conveyor_sort":
        raise ValueError("this preparation entrypoint requires a conveyor job")
    path = prepare_normalization(job)
    norm = load_jepa_proprio_normalization(path)
    print(
        json.dumps(
            {
                "normalization": str(path),
                "train_episodes": len(norm.episode_ids),
                "train_boundaries": norm.boundary_count,
                "training_started": False,
            }
        ),
        flush=True,
    )
