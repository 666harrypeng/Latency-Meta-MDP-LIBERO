"""Atomic evidence publication for direct versus zero-delay parity."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.io.artifacts import (
    collect_implementation_provenance,
    sha256_file,
)
from latency_meta_mdp.runtime.latency_parity import LatencyParityResult, ParityLaneTrace


def preflight_latency_parity_output(output_dir: Path) -> None:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"latency parity output already exists: {target}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_trace(path: Path, trace: ParityLaneTrace) -> None:
    arrays = {name: getattr(trace, name) for name in ParityLaneTrace.array_field_names()}
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _physical_events(trace: ParityLaneTrace) -> list[dict[str, Any]]:
    return [
        {"kind": kind, "time_us": time_us, "terminal_reason": terminal_reason}
        for kind, time_us, terminal_reason in trace.physical_events
    ]


def _harness_events(result: LatencyParityResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in result.harness_events:
        rows.append(
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
        )
    return rows


def write_latency_parity_artifact(
    *,
    result: LatencyParityResult,
    project_root: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    """Write one exact parity report atomically without overwriting prior evidence."""

    preflight_latency_parity_output(output_dir)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("parity artifact seed must be a non-negative integer")
    result.validate_exact()
    root = project_root.resolve()
    provenance = collect_implementation_provenance(root)
    config_paths = {
        "runtime": root / "configs/runtime/robosuite_v1.yaml",
        "task": root / "configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml",
        "motion": root / "configs/tasks/moving_ball/motion/dynamic_grasp_lift_l1.yaml",
        "control": root / "configs/runtime/control/panda_osc_pose_delta_v1.yaml",
        "expert": root / "configs/data/expert/panda_ball_feedback_v1.yaml",
    }
    if any(not path.is_file() for path in config_paths.values()):
        raise FileNotFoundError("latency parity configuration is incomplete")
    config_sha256 = {name: sha256_file(path) for name, path in config_paths.items()}
    target = output_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"latency parity staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_trace(staging / "direct_trace.npz", result.direct)
        _write_trace(staging / "zero_delay_trace.npz", result.harness)
        _write_json(
            staging / "events.json",
            {
                "schema_version": 1,
                "direct_physical_events": _physical_events(result.direct),
                "zero_delay_physical_events": _physical_events(result.harness),
                "harness_events": _harness_events(result),
            },
        )
        _write_json(
            staging / "report.json",
            {
                "schema_version": 1,
                "seed": seed,
                "formal_tick_us": 20_000,
                "exact": True,
                "mismatches": [],
                "boundary_count": int(result.direct.boundary_time_us.shape[0]),
                "transition_count": int(result.direct.executed_action.shape[0]),
                "terminal_status": result.direct.terminal_status.value,
                "terminal_reason": result.direct.terminal_reason.value,
                "terminal_time_us": result.direct.terminal_time_us,
                "harness_event_count": len(result.harness_events),
            },
        )
        artifacts = {
            name: sha256_file(staging / name)
            for name in (
                "direct_trace.npz",
                "zero_delay_trace.npz",
                "events.json",
                "report.json",
            )
        }
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "logical_latency_parity_v1",
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
