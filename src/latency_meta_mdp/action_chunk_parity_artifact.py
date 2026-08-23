"""Atomic certification artifacts for the warm-start sharp chunk client."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.action_chunk_client import load_action_chunk_client_config
from latency_meta_mdp.action_chunk_parity import ActionChunkParityResult
from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.latency_parity import ParityLaneTrace


def preflight_action_chunk_parity_output(output_dir: Path) -> None:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"action-chunk parity output already exists: {target}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_trace(path: Path, trace: ParityLaneTrace) -> None:
    arrays = {
        name: getattr(trace, name)
        for name in ParityLaneTrace.array_field_names()
    }
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _physical_events(trace: ParityLaneTrace) -> list[dict[str, Any]]:
    return [
        {"kind": kind, "time_us": time_us, "terminal_reason": terminal_reason}
        for kind, time_us, terminal_reason in trace.physical_events
    ]


def _harness_events(result: ActionChunkParityResult) -> list[dict[str, Any]]:
    return [
        {
            "kind": event.kind.value,
            "formal_tick": event.formal_tick,
            "time_us": event.time_us,
            "request_id": event.request_id,
            "launch_formal_tick": event.launch_formal_tick,
            "arrival_formal_tick": event.arrival_formal_tick,
            "realized_delay_ticks": event.realized_delay_ticks,
            "wall_start_ns": event.wall_start_ns,
            "wall_end_ns": event.wall_end_ns,
            "wall_duration_ns": event.wall_duration_ns,
            "action": None if event.action is None else event.action.tolist(),
        }
        for event in result.harness_events
    ]


def _chunk_events(result: ActionChunkParityResult) -> list[dict[str, Any]]:
    return [
        {
            "kind": event.kind.value,
            "formal_tick": event.formal_tick,
            "time_us": event.time_us,
            "chunk_id": event.chunk_id,
            "source_request_id": event.source_request_id,
            "old_chunk_id": event.old_chunk_id,
            "discarded_action_count": event.discarded_action_count,
            "installed_chunk_index": event.installed_chunk_index,
            "executed_chunk_index": event.executed_chunk_index,
            "wall_start_ns": event.wall_start_ns,
            "wall_end_ns": event.wall_end_ns,
            "wall_duration_ns": event.wall_duration_ns,
            "simulation_time_before_us": event.simulation_time_before_us,
            "simulation_time_after_us": event.simulation_time_after_us,
            "action": None if event.action is None else event.action.tolist(),
        }
        for event in result.chunk_events
    ]


def write_action_chunk_parity_artifact(
    *,
    result: ActionChunkParityResult,
    project_root: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    """Write one exact chunk-client calibration without overwriting evidence."""

    preflight_action_chunk_parity_output(output_dir)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("action-chunk parity seed must be a non-negative integer")
    result.validate_exact()
    root = project_root.resolve()
    provenance = collect_implementation_provenance(root)
    config_paths = {
        "runtime": root / "configs/runtime/robosuite_v1.yaml",
        "task": root / "configs/task/dynamic_grasp_lift_l0.yaml",
        "motion": root / "configs/motion/dynamic_grasp_lift_l1.yaml",
        "control": root / "configs/control/panda_osc_pose_delta_v1.yaml",
        "expert": root / "configs/expert/panda_ball_feedback_v1.yaml",
        "client": root / "configs/client/sharp_return_time_h50_e25_v1.yaml",
        "temporal": root / "configs/temporal/h50_e25_d20_k6_v1.yaml",
    }
    if any(not path.is_file() for path in config_paths.values()):
        raise FileNotFoundError("action-chunk parity configuration is incomplete")
    config_sha256 = {name: sha256_file(path) for name, path in config_paths.items()}
    client_config = load_action_chunk_client_config(config_paths["client"])
    target = output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"action-chunk parity staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_trace(staging / "direct_trace.npz", result.direct)
        _write_trace(staging / "chunk_trace.npz", result.chunk)
        _write_json(
            staging / "events.json",
            {
                "schema_version": 1,
                "direct_physical_events": _physical_events(result.direct),
                "chunk_physical_events": _physical_events(result.chunk),
                "harness_events": _harness_events(result),
                "chunk_events": _chunk_events(result),
            },
        )
        _write_json(
            staging / "report.json",
            {
                "schema_version": 1,
                "seed": seed,
                "exact": True,
                "mismatches": [],
                "generator_kind": result.generator_kind,
                "prediction_horizon": client_config.prediction_horizon,
                "launch_trigger_horizon": client_config.launch_trigger_horizon,
                "maximum_delay_ticks": (
                    client_config.temporal_contract.maximum_delay_ticks
                ),
                "bootstrap_simulation_time_before_us": (
                    result.bootstrap.simulation_time_before_us
                ),
                "bootstrap_simulation_time_after_us": (
                    result.bootstrap.simulation_time_after_us
                ),
                "bootstrap_wall_duration_ns": result.bootstrap.wall_duration_ns,
                "boundary_count": int(result.direct.boundary_time_us.shape[0]),
                "transition_count": int(result.direct.executed_action.shape[0]),
                "chunk_launch_ticks": list(result.chunk_launch_ticks),
                "starvation_ticks": list(result.starvation_ticks),
                "terminal_status": result.direct.terminal_status.value,
                "terminal_reason": result.direct.terminal_reason.value,
                "terminal_time_us": result.direct.terminal_time_us,
            },
        )
        artifacts = {
            name: sha256_file(staging / name)
            for name in (
                "direct_trace.npz",
                "chunk_trace.npz",
                "events.json",
                "report.json",
            )
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 2,
                "format_id": "sharp_return_time_chunk_parity_v2",
                "eligible": not provenance.dirty,
                "blockers": [] if not provenance.dirty else ["implementation_dirty"],
                "seed": seed,
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "config_sha256": config_sha256,
                "artifacts": artifacts,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
