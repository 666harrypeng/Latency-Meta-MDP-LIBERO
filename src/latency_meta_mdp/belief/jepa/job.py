"""Level-aware Direct predictor training with explicit data and resume identities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from latency_meta_mdp.belief.jepa.optimization import (
    DirectTrainingConfig,
)


@dataclass(frozen=True)
class DirectJob:
    project_root: Path
    config_path: Path
    level: int
    training_config: Path
    source_root: Path
    vision_cache_manifest: Path
    split_manifest: Path
    normalization: Path
    microbatch_size: int
    num_workers: int
    device: str
    training: DirectTrainingConfig


def load_direct_job(path: Path, *, project_root: Path) -> DirectJob:
    root = project_root.resolve()
    value = yaml.safe_load(path.read_text())
    if value.pop("schema_version") != 1 or value["level"] not in (1, 2, 3):
        raise ValueError("Invalid Direct job schema/level")
    for key in (
        "training_config",
        "source_root",
        "vision_cache_manifest",
        "split_manifest",
        "normalization",
    ):
        value[key] = (root / value[key]).resolve()
    training = DirectTrainingConfig(**yaml.safe_load(value["training_config"].read_text()))
    if value["microbatch_size"] <= 0 or training.global_batch_size % value["microbatch_size"]:
        raise ValueError("microbatch must divide global batch")
    if value["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    return DirectJob(project_root=root, config_path=path.resolve(), training=training, **value)


def load_direct_data(job: DirectJob, *, split: str):
    from latency_meta_mdp.belief.jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.jepa.corpus import (
        load_action_conditioned_jepa_corpus,
        load_jepa_proprio_normalization,
    )
    from latency_meta_mdp.belief.jepa.data import (
        DirectPredictionDataset,
    )

    root = job.project_root
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        level_path=root / f"configs/models/jepa/l{job.level}.yaml",
        temporal_sampling_path=root / "configs/models/jepa/stride4_80ms_history_160ms.yaml",
    )
    norm = load_jepa_proprio_normalization(job.normalization)
    if norm.level != job.level:
        raise ValueError("Direct job normalization level mismatch")
    # Formal recordings already have immutable manifests. Check metadata, sizes and
    # array/episode contracts without rereading all large payloads for hashing.
    corpus = load_action_conditioned_jepa_corpus(
        source_root=job.source_root,
        cache_run_manifest=job.vision_cache_manifest,
        split_manifest_path=job.split_manifest,
        level=job.level,
        split=split,
        config=config,
        normalization=norm,
        verify_payloads=False,
    )
    dataset = DirectPredictionDataset(records=corpus.records, normalization=norm)
    return config, norm, corpus, dataset


def atomic_json(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def reconcile_epoch_journal(path: Path, *, completed_epochs: int, last_report: dict):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["completed_epochs"] <= completed_epochs:
                rows.append(row)
    if rows and rows[-1]["completed_epochs"] == completed_epochs:
        rows[-1] = last_report
    elif completed_epochs:
        rows.append(last_report)
    if [r["completed_epochs"] for r in rows] != list(range(1, completed_epochs + 1)):
        raise ValueError("Epoch journal is missing earlier completed epochs")
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temp.replace(path)
