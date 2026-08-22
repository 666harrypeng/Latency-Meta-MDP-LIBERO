"""Deterministic coverage audit for formal Panda-ball seed banks."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.bulk_plan import BulkCollectionPlan, SeedBank, load_bulk_collection_plan
from latency_meta_mdp.motion import MotionConfig, build_motion_profile, load_motion_config
from latency_meta_mdp.task import load_task_spec


def _quantiles(values: list[float]) -> dict[str, float]:
    result = np.quantile(values, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        name: float(value)
        for name, value in zip(("min", "p10", "median", "p90", "max"), result, strict=True)
    }


def _level_coverage(
    *,
    config: MotionConfig,
    seeds: tuple[int, ...],
    workspace_z: float,
) -> dict[str, Any]:
    speeds: list[float] = []
    quadrants: Counter[int] = Counter()
    curve_signs: Counter[str] = Counter()
    segment_counts: Counter[int] = Counter()
    jumps: list[float] = []
    upper_scale = {1: 1.0, 2: 0.65, 3: 0.60}[config.level]
    speed_upper = config.max_speed_mps * upper_scale

    for seed in seeds:
        profile = build_motion_profile(config=config, seed=seed, workspace_z=workspace_z)
        start = profile.sample(0).position[:2]
        midpoint = profile.sample(config.anchor_time_us // 2).position[:2]
        end = profile.sample(config.anchor_time_us).position[:2]
        chord = end - start
        angle = float(np.arctan2(chord[1], chord[0])) % (2.0 * np.pi)
        quadrants[int(angle // (np.pi / 2.0))] += 1
        deviation = midpoint - 0.5 * (start + end)
        signed_curve = float(chord[0] * deviation[1] - chord[1] * deviation[0])
        if abs(signed_curve) > 1e-12:
            curve_signs["positive" if signed_curve > 0.0 else "negative"] += 1

        if config.level == 1:
            speeds.append(float(np.linalg.norm(profile.velocity_xy)))
        elif config.level == 2:
            duration_s = profile.segment.duration_us / 1_000_000
            speeds.append(
                float(
                    np.linalg.norm(profile.segment.end_xy - profile.segment.start_xy)
                    / duration_s
                )
            )
        else:
            segment_counts[profile.segment_count] += 1
            for segment in profile.segments:
                duration_s = segment.duration_us / 1_000_000
                speeds.append(
                    float(np.linalg.norm(segment.end_xy - segment.start_xy) / duration_s)
                )
            jumps.extend(
                float(np.linalg.norm(right.start_velocity_xy - left.end_velocity_xy))
                for left, right in zip(profile.segments, profile.segments[1:])
            )

    speed_boundary_hits = sum(
        abs(speed - config.min_speed_mps) <= 1e-12
        or abs(speed - speed_upper) <= 1e-12
        for speed in speeds
    )
    quadrants_complete = set(quadrants) == {0, 1, 2, 3}
    signs_complete = config.level == 1 or set(curve_signs) == {"negative", "positive"}
    segments_complete = config.level != 3 or set(segment_counts) == {2, 3}
    jumps_valid = config.level != 3 or all(
        config.velocity_jump_range_mps[0]
        <= jump
        <= config.velocity_jump_range_mps[1]
        for jump in jumps
    )
    passed = bool(
        speed_boundary_hits == 0
        and quadrants_complete
        and signs_complete
        and segments_complete
        and jumps_valid
    )
    return {
        "level": config.level,
        "profile_id": config.profile_id,
        "passed": passed,
        "quadrant_counts": {str(index): quadrants[index] for index in range(4)},
        "curve_sign_counts": {
            "negative": curve_signs["negative"],
            "positive": curve_signs["positive"],
        },
        "segment_count_counts": {
            "2": segment_counts[2],
            "3": segment_counts[3],
        },
        "speed_sample_count": len(speeds),
        "speed_quantiles_mps": _quantiles(speeds),
        "speed_boundary_hit_count": speed_boundary_hits,
        "velocity_jump_quantiles_mps": _quantiles(jumps) if jumps else None,
    }


def build_bulk_motion_coverage(
    *,
    project_root: Path,
    plan: BulkCollectionPlan,
) -> dict[str, Any]:
    project = project_root.resolve()
    config_rows: list[dict[str, Any]] = []
    configs: dict[int, MotionConfig] = {}
    workspace_z: dict[int, float] = {}
    task_path = project / "configs/task/dynamic_grasp_lift_l0.yaml"
    task_spec = load_task_spec(task_path)
    for level in plan.levels:
        motion_path = project / f"configs/motion/dynamic_grasp_lift_l{level}.yaml"
        configs[level] = load_motion_config(motion_path)
        workspace_z[level] = task_spec.ball_initial_position[2]
        config_rows.append(
            {
                "level": level,
                "profile_id": configs[level].profile_id,
                "motion_config_sha256": sha256_file(motion_path),
                "task_config_sha256": sha256_file(task_path),
            }
        )

    split_banks: tuple[tuple[str, SeedBank], ...] = (
        ("train", plan.train),
        ("development", plan.development),
        ("test", plan.test),
    )
    splits = []
    for name, bank in split_banks:
        levels = [
            _level_coverage(
                config=configs[level],
                seeds=bank.seeds,
                workspace_z=workspace_z[level],
            )
            for level in plan.levels
        ]
        splits.append(
            {
                "name": name,
                "seed_start": bank.start,
                "seed_count": bank.count,
                "collect_expert": bank.collect_expert,
                "levels": levels,
                "passed": all(level["passed"] for level in levels),
            }
        )
    return {
        "schema_version": 1,
        "format_id": "panda_ball_bulk_motion_coverage_v1",
        "collection_id": plan.collection_id,
        "motion_configs": config_rows,
        "splits": splits,
        "coverage_passed": all(split["passed"] for split in splits),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_bulk_motion_coverage(
    *,
    project_root: Path,
    plan_path: Path,
    output_dir: Path,
) -> Path:
    project = project_root.resolve()
    plan_file = plan_path.resolve()
    report = build_bulk_motion_coverage(
        project_root=project,
        plan=load_bulk_collection_plan(plan_file),
    )
    provenance = collect_implementation_provenance(project)
    report.update(
        {
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "eligible": report["coverage_passed"] and not provenance.dirty,
            "bulk_plan_sha256": sha256_file(plan_file),
        }
    )

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"bulk coverage output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"bulk coverage staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_json(staging / "manifest.json", report)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
