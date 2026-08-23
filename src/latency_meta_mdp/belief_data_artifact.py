"""Certification artifact for model-neutral BELIEF raw/index readiness."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief_data import (
    BeliefDeploymentStream,
    BeliefSupervisionStream,
    build_belief_sample_indices,
    load_belief_episode,
)
from latency_meta_mdp.latency_law import load_latency_law


@dataclass(frozen=True)
class BeliefDataViewConfig:
    schema_version: int
    view_id: str
    history_ticks: int
    buffer_protocol_status: str
    target_representation_status: str
    action_chunk_alignment_status: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.view_id != "belief_data_view_v1":
            raise ValueError("unsupported belief data view schema or identifier")
        if (
            isinstance(self.history_ticks, bool)
            or not isinstance(self.history_ticks, int)
            or self.history_ticks <= 0
        ):
            raise ValueError("history_ticks must be a positive integer")
        if self.buffer_protocol_status != "unbound":
            raise ValueError("belief data v1 must leave the buffer protocol unbound")
        if self.target_representation_status != "model_neutral_raw":
            raise ValueError("belief data v1 must retain model-neutral raw targets")
        if self.action_chunk_alignment_status != "unbound":
            raise ValueError("belief data v1 must leave action chunk alignment unbound")


def load_belief_data_view_config(path: Path) -> BeliefDataViewConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(BeliefDataViewConfig.__dataclass_fields__):
        raise ValueError("belief data view config fields are invalid")
    return BeliefDataViewConfig(**raw)


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


def certify_belief_data_contract(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    view_config_path: Path,
    latency_law_path: Path,
    output_dir: Path,
) -> Path:
    project = project_root.resolve()
    source_path = source_bulk_manifest.resolve()
    source_root = source_path.parent
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"belief data certification already exists: {target}")
    source = _load_json(source_path)
    if (
        source.get("format_id") != "panda_ball_bulk_first_tranche_v1"
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("belief certification requires an eligible clean bulk source")
    admitted = source.get("admitted_episode_manifests")
    artifacts = source.get("artifacts")
    if not isinstance(admitted, list) or not admitted or not isinstance(artifacts, dict):
        raise ValueError("bulk source does not contain admitted episodes and artifacts")
    for relative, digest in artifacts.items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"bulk source artifact hash mismatch: {relative}")

    config = load_belief_data_view_config(view_config_path)
    law = load_latency_law(latency_law_path)
    provenance = collect_implementation_provenance(project)
    episode_counts: Counter[int] = Counter()
    boundary_counts: Counter[int] = Counter()
    transition_counts: Counter[int] = Counter()
    branch_counts: Counter[int] = Counter()
    per_delay: dict[int, Counter[int]] = defaultdict(Counter)
    episode_ids: dict[int, list[str]] = defaultdict(list)
    for relative in admitted:
        if not isinstance(relative, str):
            raise ValueError("admitted episode manifest paths must be strings")
        view = load_belief_episode((source_root / relative).parent)
        indices = build_belief_sample_indices(
            episode=view,
            history_ticks=config.history_ticks,
            delay_ticks=law.delay_ticks,
        )
        episode_counts[view.level] += 1
        boundary_counts[view.level] += view.boundary_count
        transition_counts[view.level] += view.transition_count
        branch_counts[view.level] += len(indices)
        episode_ids[view.level].append(view.episode_id)
        for index in indices:
            per_delay[view.level][index.branch_delay_tick] += 1

    levels = {}
    raw_index_ready = True
    for level in sorted(episode_counts):
        delay_counts = {str(delay): per_delay[level][delay] for delay in law.delay_ticks}
        raw_index_ready = bool(
            raw_index_ready
            and episode_counts[level] > 0
            and branch_counts[level] > 0
            and all(count > 0 for count in delay_counts.values())
        )
        levels[str(level)] = {
            "episode_count": episode_counts[level],
            "episode_ids": sorted(episode_ids[level]),
            "boundary_count": boundary_counts[level],
            "transition_count": transition_counts[level],
            "branch_index_count": branch_counts[level],
            "per_delay_index_count": delay_counts,
        }

    blockers = [] if raw_index_ready else ["raw_or_delay_index_contract_failed"]
    if provenance.dirty:
        blockers.append("implementation_dirty")
    open_items = [
        "action_buffer_protocol",
        "belief_target_representation",
        "action_chunk_alignment",
    ]
    report = {
        "schema_version": 1,
        "format_id": "belief_raw_index_certification_v1",
        "implementation_revision": provenance.revision,
        "implementation_source_sha256": provenance.source_sha256,
        "implementation_dirty": provenance.dirty,
        "source_bulk_manifest": source_path.as_posix(),
        "source_bulk_manifest_sha256": sha256_file(source_path),
        "view_config_sha256": sha256_file(view_config_path),
        "latency_law_sha256": sha256_file(latency_law_path),
        "view_id": config.view_id,
        "history_sample_count": config.history_ticks,
        "history_span_ms": (config.history_ticks - 1) * 20,
        "delay_ticks": list(law.delay_ticks),
        "latency_condition_dim": len(law.condition_vector),
        "latency_condition_probabilities": law.condition_vector.tolist(),
        "branch_delay_visible_in_launch_history": False,
        "deployment_fields": list(BeliefDeploymentStream.__dataclass_fields__),
        "privileged_supervision_fields": list(
            BeliefSupervisionStream.__dataclass_fields__
        ),
        "raw_index_ready": raw_index_ready,
        "belief_training_ready": False,
        "open_design_items": open_items,
        "levels": levels,
        "blockers": blockers,
        "eligible": not blockers,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    try:
        staging.mkdir()
        _write_json(staging / "manifest.json", report)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
