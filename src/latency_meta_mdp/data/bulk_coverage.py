"""Deterministic coverage audit for formal Panda-ball seed banks."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.data.bulk_plan import BulkCollectionPlan, SeedBank, load_bulk_collection_plan
from latency_meta_mdp.envs.motion import (
    CubicPolynomialProfile,
    MotionConfig,
    PiecewisePolynomialProfile,
    build_motion_profile,
    load_motion_config,
    sample_shared_geometry,
)
from latency_meta_mdp.envs.task import load_task_spec
from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file


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
    quadrants: Counter[int] = Counter()
    curve_signs: Counter[str] = Counter()
    segment_counts: Counter[int] = Counter()
    segment_kinds: Counter[str] = Counter()
    cubic_degrees: Counter[int] = Counter()
    chord_lengths: list[float] = []
    path_lengths: list[float] = []
    deviations: list[float] = []
    peak_speeds: list[float] = []
    peak_accelerations: list[float] = []
    jumps: list[float] = []
    turn_angles: list[float] = []
    change_times: list[float] = []
    curvature_change_times: list[float] = []

    for seed in seeds:
        profile = build_motion_profile(config=config, seed=seed, workspace_z=workspace_z)
        samples = [
            profile.sample(time_us) for time_us in range(0, config.anchor_time_us + 1, 20_000)
        ]
        positions = np.stack([sample.position[:2] for sample in samples])
        speeds = np.array([np.linalg.norm(sample.velocity[:2]) for sample in samples])
        accelerations = np.array([np.linalg.norm(sample.acceleration[:2]) for sample in samples])
        start = positions[0]
        midpoint = profile.sample(config.anchor_time_us // 2).position[:2]
        end = positions[-1]
        chord = end - start
        chord_length = float(np.linalg.norm(chord))
        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        relative = positions - start
        deviation = float(
            np.max(np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0])) / chord_length
        )
        chord_lengths.append(chord_length)
        path_lengths.append(path_length)
        deviations.append(deviation)
        peak_speeds.append(float(speeds.max()))
        peak_accelerations.append(float(accelerations.max()))
        angle = float(np.arctan2(chord[1], chord[0])) % (2.0 * np.pi)
        quadrants[int(angle // (np.pi / 2.0))] += 1
        if isinstance(profile, CubicPolynomialProfile):
            first_waypoint = profile.waypoint_positions_xy[1]
            first_fraction = profile.waypoint_times_us[1] / config.anchor_time_us
            first_deviation = first_waypoint - (start + first_fraction * chord)
            signed_curve = float(chord[0] * first_deviation[1] - chord[1] * first_deviation[0])
            curve_signs["positive" if signed_curve > 0.0 else "negative"] += 1
            degree = max(
                index
                for index, coefficient in enumerate(profile.coefficients_xy)
                if np.linalg.norm(coefficient) > 1e-12
            )
            cubic_degrees[degree] += 1
            curvature_cross = np.array(
                [
                    sample.velocity[0] * sample.acceleration[1]
                    - sample.velocity[1] * sample.acceleration[0]
                    for sample in samples
                ]
            )
            meaningful = np.flatnonzero(np.abs(curvature_cross) > 1e-10)
            signs = np.sign(curvature_cross[meaningful])
            changes = np.flatnonzero(signs[1:] != signs[:-1])
            if len(changes) != 1:
                raise RuntimeError("validated L2 profile lost its curvature sign change")
            curvature_change_times.append(float(meaningful[changes[0] + 1] * 20_000 / 1_000_000))
        elif isinstance(profile, PiecewisePolynomialProfile):
            midpoint_deviation = midpoint - 0.5 * (start + end)
            signed_curve = float(
                chord[0] * midpoint_deviation[1] - chord[1] * midpoint_deviation[0]
            )
            if abs(signed_curve) > 1e-12:
                curve_signs["positive" if signed_curve > 0.0 else "negative"] += 1
            segment_counts[profile.segment_count] += 1
            segment_kinds.update(segment.kind for segment in profile.segments)
            change_times.extend(time_us / 1_000_000 for time_us in profile.change_times_us)
            for left, right in zip(profile.segments, profile.segments[1:]):
                jumps.append(float(np.linalg.norm(right.start_velocity_xy - left.end_velocity_xy)))
                cosine = np.clip(
                    np.dot(left.end_velocity_xy, right.start_velocity_xy)
                    / (
                        np.linalg.norm(left.end_velocity_xy)
                        * np.linalg.norm(right.start_velocity_xy)
                    ),
                    -1.0,
                    1.0,
                )
                turn_angles.append(float(np.degrees(np.arccos(cosine))))

    if config.min_chord_length_m is None or config.max_chord_length_m is None:
        raise ValueError("dynamic chord bounds are missing")
    chord_boundary_hits = sum(
        abs(length - config.min_chord_length_m) <= 1e-12
        or abs(length - config.max_chord_length_m) <= 1e-12
        for length in chord_lengths
    )
    quadrants_complete = set(quadrants) == {0, 1, 2, 3}
    signs_complete = config.level == 1 or set(curve_signs) == {"negative", "positive"}
    segments_complete = config.level != 3 or set(segment_counts) == {2, 3}
    kinds_complete = config.level != 3 or set(segment_kinds) == {"line", "cubic"}
    degrees_complete = config.level != 2 or cubic_degrees == {3: len(seeds)}
    jumps_valid = config.level != 3 or all(
        config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]
        for jump in jumps
    )
    turns_valid = config.level != 3 or all(
        config.turn_angle_degrees_range[0] <= angle <= config.turn_angle_degrees_range[1]
        for angle in turn_angles
    )
    passed = bool(
        chord_boundary_hits == 0
        and quadrants_complete
        and signs_complete
        and segments_complete
        and kinds_complete
        and degrees_complete
        and jumps_valid
        and turns_valid
    )
    return {
        "level": config.level,
        "profile_id": config.profile_id,
        "passed": passed,
        "chord_length_quantiles_m": _quantiles(chord_lengths),
        "path_length_quantiles_m": _quantiles(path_lengths),
        "maximum_deviation_quantiles_m": _quantiles(deviations),
        "peak_speed_quantiles_mps": _quantiles(peak_speeds),
        "peak_acceleration_quantiles_mps2": _quantiles(peak_accelerations),
        "quadrant_counts": {str(index): quadrants[index] for index in range(4)},
        "curve_sign_counts": {
            "negative": curve_signs["negative"],
            "positive": curve_signs["positive"],
        },
        "segment_count_counts": {
            "2": segment_counts[2],
            "3": segment_counts[3],
        },
        "segment_kind_counts": {
            "line": segment_kinds["line"],
            "cubic": segment_kinds["cubic"],
        },
        "cubic_degree_counts": {
            str(degree): count for degree, count in sorted(cubic_degrees.items()) if count
        },
        "chord_boundary_hit_count": chord_boundary_hits,
        "velocity_jump_quantiles_mps": _quantiles(jumps) if jumps else None,
        "turn_angle_quantiles_degrees": _quantiles(turn_angles) if turn_angles else None,
        "change_time_quantiles_seconds": _quantiles(change_times) if change_times else None,
        "curvature_change_time_quantiles_seconds": (
            _quantiles(curvature_change_times) if curvature_change_times else None
        ),
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
    task_path = project / "configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml"
    task_spec = load_task_spec(task_path)
    for level in plan.levels:
        motion_path = project / f"configs/tasks/moving_ball/motion/dynamic_grasp_lift_l{level}.yaml"
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
        shared_geometry_matches = all(
            all(
                np.array_equal(geometry.start_xy, geometries[0].start_xy)
                and np.array_equal(geometry.end_xy, geometries[0].end_xy)
                for geometry in geometries[1:]
            )
            for geometries in (
                tuple(
                    sample_shared_geometry(config=configs[level], seed=seed)
                    for level in plan.levels
                )
                for seed in bank.seeds
            )
        )
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
                "shared_geometry_matches_across_levels": shared_geometry_matches,
                "passed": shared_geometry_matches and all(level["passed"] for level in levels),
            }
        )
    return {
        "schema_version": 1,
        "format_id": "panda_ball_bulk_motion_coverage_v2",
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
