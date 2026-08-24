"""Episode-split index artifact for lazy return-belief training contexts."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.artifacts import (
    collect_implementation_provenance,
    sha256_file,
)
from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.belief_data_artifact import load_belief_data_view_config
from latency_meta_mdp.belief_training_data import (
    ROBOT_PROPRIO_DIM,
    build_belief_training_indices,
)
from latency_meta_mdp.bulk_plan import load_bulk_collection_plan
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.return_belief_geometry import RETURN_STATE_DIM
from latency_meta_mdp.terminal_absorbing_tail import build_terminal_absorbing_tail


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "supervised_holdout"


@dataclass(frozen=True)
class BeliefTrainingSplitPlan:
    schema_version: int
    split_id: str
    formal_seed_start: int
    formal_seed_count: int
    formal_train_count: int
    formal_validation_count: int
    formal_holdout_count: int
    first_tranche_count: int
    first_tranche_train_count: int
    first_tranche_validation_count: int
    first_tranche_holdout_count: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.split_id != "belief_training_split_v1":
            raise ValueError("unsupported belief training split schema")
        for name, value in self.__dict__.items():
            if name in {"split_id", "schema_version"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if sum(self.formal_counts) != self.formal_seed_count:
            raise ValueError("formal split counts must cover the expert seed bank")
        if sum(self.first_tranche_counts) != self.first_tranche_count:
            raise ValueError("first-tranche split counts must cover the tranche")
        if self.first_tranche_count > self.formal_seed_count:
            raise ValueError("first tranche cannot exceed the formal seed bank")

    @property
    def formal_counts(self) -> tuple[int, int, int]:
        return (
            self.formal_train_count,
            self.formal_validation_count,
            self.formal_holdout_count,
        )

    @property
    def first_tranche_counts(self) -> tuple[int, int, int]:
        return (
            self.first_tranche_train_count,
            self.first_tranche_validation_count,
            self.first_tranche_holdout_count,
        )

    def split_for_seed(self, seed: int, *, first_tranche: bool) -> SplitName:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        offset = seed - self.formal_seed_start
        counts = self.first_tranche_counts if first_tranche else self.formal_counts
        limit = self.first_tranche_count if first_tranche else self.formal_seed_count
        if not 0 <= offset < limit:
            raise ValueError("seed is outside the selected supervised split bank")
        train_count, validation_count, _holdout_count = counts
        if offset < train_count:
            return SplitName.TRAIN
        if offset < train_count + validation_count:
            return SplitName.VALIDATION
        return SplitName.HOLDOUT


def load_belief_training_split_plan(path: Path) -> BeliefTrainingSplitPlan:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(
        BeliefTrainingSplitPlan.__dataclass_fields__
    ):
        raise ValueError("belief training split config fields are invalid")
    return BeliefTrainingSplitPlan(**raw)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_source(path: Path) -> tuple[dict[str, Any], Path]:
    source = _load_json(path)
    if (
        source.get("format_id") != "panda_ball_bulk_first_tranche_v1"
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("belief index requires an eligible clean bulk source")
    admitted = source.get("admitted_episode_manifests")
    artifacts = source.get("artifacts")
    if not isinstance(admitted, list) or not admitted or not isinstance(artifacts, dict):
        raise ValueError("bulk source inventory is invalid")
    root = path.parent.resolve()
    for relative, digest in artifacts.items():
        artifact = (root / relative).resolve()
        if (
            not isinstance(relative, str)
            or not artifact.is_relative_to(root)
            or not artifact.is_file()
            or sha256_file(artifact) != digest
        ):
            raise ValueError("bulk source artifact verification failed")
    return source, root


def write_belief_training_index_artifact(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    bulk_plan_path: Path,
    view_config_path: Path,
    latency_law_path: Path,
    output_dir: Path,
) -> Path:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"belief training index output already exists: {target}")
    project = project_root.resolve()
    source_path = source_bulk_manifest.resolve()
    split_path = split_config_path.resolve()
    bulk_path = bulk_plan_path.resolve()
    view_path = view_config_path.resolve()
    law_path = latency_law_path.resolve()
    source, source_root = _validate_source(source_path)
    split_plan = load_belief_training_split_plan(split_path)
    bulk_plan = load_bulk_collection_plan(bulk_path)
    if (
        split_plan.formal_seed_start != bulk_plan.train.start
        or split_plan.formal_seed_count != bulk_plan.train.count
        or split_plan.first_tranche_count != bulk_plan.first_tranche_count
    ):
        raise ValueError("belief split and bulk collection plans disagree")
    view_config = load_belief_data_view_config(view_path)
    law = load_latency_law(law_path)
    action_path = project / "configs/control/panda_osc_pose_delta_v1.yaml"
    action_contract = load_action_contract(action_path)

    episodes = []
    episode_manifests = []
    seeds = []
    for relative in source["admitted_episode_manifests"]:
        if not isinstance(relative, str):
            raise ValueError("admitted episode paths must be strings")
        episode_manifest = (source_root / relative).resolve()
        episode = load_belief_episode(episode_manifest.parent)
        episodes.append(episode)
        episode_manifests.append(relative)
        seeds.append(episode.scene_seed)
    first_tranche = all(
        split_plan.formal_seed_start
        <= seed
        < split_plan.formal_seed_start + split_plan.first_tranche_count
        for seed in seeds
    )
    rows = []
    split_episode_ids: dict[str, set[str]] = defaultdict(set)
    split_master_seeds: dict[str, set[int]] = defaultdict(set)
    split_context_counts: Counter[str] = Counter()
    for episode, relative in zip(episodes, episode_manifests, strict=True):
        split = split_plan.split_for_seed(
            episode.scene_seed,
            first_tranche=first_tranche,
        )
        tail = build_terminal_absorbing_tail(
            episode=episode,
            temporal_contract=view_config.temporal_contract,
            action_contract=action_contract,
        )
        indices = build_belief_training_indices(
            tail_view=tail,
            temporal_contract=view_config.temporal_contract,
        )
        split_episode_ids[split.value].add(episode.episode_id)
        split_master_seeds[split.value].add(episode.scene_seed)
        split_context_counts[split.value] += len(indices)
        rows.extend(
            {
                "episode_manifest": relative,
                "episode_id": index.episode_id,
                "level": index.level,
                "scene_seed": episode.scene_seed,
                "history_start_tick": index.history_start_tick,
                "source_tick": index.source_tick,
                "source_phase": index.source_phase,
                "split": split.value,
            }
            for index in indices
        )
    arrays = {
        name: np.asarray([row[name] for row in rows], dtype=dtype)
        for name, dtype in (
            ("episode_manifest", np.str_),
            ("episode_id", np.str_),
            ("level", np.int8),
            ("scene_seed", np.int64),
            ("history_start_tick", np.int64),
            ("source_tick", np.int64),
            ("source_phase", np.str_),
            ("split", np.str_),
        )
    }
    provenance = collect_implementation_provenance(project)
    split_rows = {
        split.value: {
            "episode_count": len(split_episode_ids[split.value]),
            "master_seed_count": len(split_master_seeds[split.value]),
            "context_count": split_context_counts[split.value],
        }
        for split in SplitName
    }
    summary = {
        "schema_version": 1,
        "format_id": "belief_training_index_summary_v1",
        "implementation_revision": provenance.revision,
        "implementation_source_sha256": provenance.source_sha256,
        "implementation_dirty": provenance.dirty,
        "source_bulk_manifest": source_path.as_posix(),
        "source_bulk_manifest_sha256": sha256_file(source_path),
        "split_config_sha256": sha256_file(split_path),
        "bulk_plan_sha256": sha256_file(bulk_path),
        "view_config_sha256": sha256_file(view_path),
        "latency_law_sha256": sha256_file(law_path),
        "action_contract_sha256": sha256_file(action_path),
        "split_mode": "first_tranche" if first_tranche else "formal",
        "episode_count": len(episodes),
        "master_seed_count": len(set(seeds)),
        "context_count": len(rows),
        "splits": split_rows,
        "context_contract": {
            "image_history_ticks": view_config.temporal_contract.history_sample_count,
            "camera_count": 2,
            "robot_proprio_dim": ROBOT_PROPRIO_DIM,
            "remaining_action_shape": [
                view_config.temporal_contract.remaining_buffer_coverage,
                7,
            ],
            "latency_probability_dim": law.bin_count,
            "sampled_realized_delay_visible_to_encoder": False,
        },
        "target_contract": {
            "delay_count": law.bin_count,
            "state_dim": RETURN_STATE_DIM,
            "interaction_mode_count": 4,
            "absorbing_tail_ticks": view_config.temporal_contract.target_tail_ticks,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"belief training staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_json(staging / "summary.json", summary)
        with (staging / "index.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        artifact_hashes = {
            name: sha256_file(staging / name)
            for name in ("index.npz", "summary.json")
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "belief_training_index_artifact_v1",
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "eligible": not provenance.dirty,
                "artifacts": artifact_hashes,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
