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

from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.legacy.belief_data import (
    BeliefDeploymentStream,
    BeliefSupervisionStream,
    SharpTeacherBufferAdapter,
    build_belief_sample_indices,
    load_belief_episode,
)
from latency_meta_mdp.runtime.latency_law import load_latency_law
from latency_meta_mdp.runtime.temporal_contract import TemporalContract, load_temporal_contract


@dataclass(frozen=True)
class BeliefDataViewConfig:
    schema_version: int
    view_id: str
    temporal_contract: TemporalContract
    buffer_protocol_status: str
    target_representation_status: str
    action_chunk_alignment_status: str

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.view_id != "belief_data_view_h50_v2":
            raise ValueError("unsupported belief data view schema or identifier")
        if not isinstance(self.temporal_contract, TemporalContract):
            raise TypeError("temporal_contract must be a TemporalContract")
        if self.buffer_protocol_status != "sharp_teacher_buffer_bound":
            raise ValueError("belief data v2 requires the sharp teacher buffer")
        if self.target_representation_status != "model_neutral_raw":
            raise ValueError("belief data v2 must retain model-neutral raw targets")
        if self.action_chunk_alignment_status != "return_time":
            raise ValueError("belief data v2 requires return-time chunk alignment")

    @property
    def history_ticks(self) -> int:
        return self.temporal_contract.history_sample_count


def load_belief_data_view_config(path: Path) -> BeliefDataViewConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(BeliefDataViewConfig.__dataclass_fields__):
        raise ValueError("belief data view config fields are invalid")
    temporal_path = raw["temporal_contract"]
    if not isinstance(temporal_path, str) or not temporal_path:
        raise ValueError("temporal_contract must be a non-empty relative path")
    raw["temporal_contract"] = load_temporal_contract((path.parent / temporal_path).resolve())
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
    if (
        law.bin_count != config.temporal_contract.maximum_delay_ticks
        or round(law.control_tick_seconds * 1_000_000) != config.temporal_contract.formal_tick_us
    ):
        raise ValueError("latency law and temporal contract are inconsistent")
    buffer_adapter = SharpTeacherBufferAdapter(config.temporal_contract)
    provenance = collect_implementation_provenance(project)
    episode_counts: Counter[int] = Counter()
    boundary_counts: Counter[int] = Counter()
    transition_counts: Counter[int] = Counter()
    branch_counts: Counter[int] = Counter()
    per_delay: dict[int, Counter[int]] = defaultdict(Counter)
    episode_ids: dict[int, list[str]] = defaultdict(list)
    teacher_buffer_counts: Counter[int] = Counter()
    for relative in admitted:
        if not isinstance(relative, str):
            raise ValueError("admitted episode manifest paths must be strings")
        view = load_belief_episode((source_root / relative).parent)
        indices = build_belief_sample_indices(
            episode=view,
            temporal_contract=config.temporal_contract,
            delay_ticks=law.delay_ticks,
        )
        source_interval = config.temporal_contract.belief_source_interval(
            episode_action_count=view.transition_count
        )
        for source_tick in range(source_interval.minimum, source_interval.maximum + 1):
            buffer_adapter.build(episode=view, source_tick=source_tick)
            teacher_buffer_counts[view.level] += 1
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
            "teacher_buffer_source_count": teacher_buffer_counts[level],
            "per_delay_index_count": delay_counts,
        }

    blockers = [] if raw_index_ready else ["raw_or_delay_index_contract_failed"]
    if provenance.dirty:
        blockers.append("implementation_dirty")
    open_items = ["belief_target_representation", "belief_action_policy_interface"]
    report = {
        "schema_version": 2,
        "format_id": "belief_raw_index_certification_v2",
        "implementation_revision": provenance.revision,
        "implementation_source_sha256": provenance.source_sha256,
        "implementation_dirty": provenance.dirty,
        "source_bulk_manifest": source_path.as_posix(),
        "source_bulk_manifest_sha256": sha256_file(source_path),
        "view_config_sha256": sha256_file(view_config_path),
        "temporal_contract_sha256": sha256_file(
            repository_root() / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"
        ),
        "latency_law_sha256": sha256_file(latency_law_path),
        "view_id": config.view_id,
        "temporal_contract_id": config.temporal_contract.contract_id,
        "prediction_horizon": config.temporal_contract.prediction_horizon,
        "launch_trigger_horizon": (config.temporal_contract.launch_trigger_horizon),
        "history_sample_count": config.history_ticks,
        "history_span_ms": config.temporal_contract.history_span_us // 1_000,
        "delay_ticks": list(law.delay_ticks),
        "latency_condition_dim": len(law.condition_vector),
        "latency_condition_probabilities": law.condition_vector.tolist(),
        "branch_delay_visible_in_launch_history": False,
        "deployment_fields": list(BeliefDeploymentStream.__dataclass_fields__),
        "privileged_supervision_fields": list(BeliefSupervisionStream.__dataclass_fields__),
        "raw_index_ready": raw_index_ready,
        "teacher_buffer_ready": raw_index_ready
        and all(teacher_buffer_counts[level] > 0 for level in episode_counts),
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
