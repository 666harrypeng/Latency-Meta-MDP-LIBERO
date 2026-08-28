"""Physical-trajectory and scaling evidence for Flow belief data sufficiency."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import FeatureBeliefCorpus
from latency_meta_mdp.episode_split import load_episode_split_plan
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class FlowDataSufficiencyConfig:
    schema_version: int
    audit_id: str
    subset_sizes: tuple[int, ...]
    subset_seed: int
    model_seeds: tuple[int, ...]
    workspace_bin_count_per_axis: int
    minimum_occupied_bin_count: int
    scaling_improvement_trigger: float
    model_seed_disagreement_trigger: float
    additional_tranche_size: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.audit_id != "dinov3_flow_belief_data_scaling_v1":
            raise ValueError("unsupported Flow data-sufficiency config")
        if (
            not self.subset_sizes
            or self.subset_sizes != tuple(sorted(set(self.subset_sizes)))
            or any(value <= 0 for value in self.subset_sizes)
            or len(self.model_seeds) != 2
            or self.model_seeds != tuple(sorted(set(self.model_seeds)))
            or any(value < 0 for value in self.model_seeds)
        ):
            raise ValueError("Flow data-sufficiency subset or model seeds are invalid")
        for name in (
            "subset_seed",
            "workspace_bin_count_per_axis",
            "minimum_occupied_bin_count",
            "additional_tranche_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Flow data-sufficiency {name} must be positive")
        for name in (
            "scaling_improvement_trigger",
            "model_seed_disagreement_trigger",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"Flow data-sufficiency {name} must lie in (0, 1)")


def load_flow_data_sufficiency_config(path: Path) -> FlowDataSufficiencyConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = set(FlowDataSufficiencyConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Flow data-sufficiency config fields are invalid")
    value["subset_sizes"] = tuple(value["subset_sizes"])
    value["model_seeds"] = tuple(value["model_seeds"])
    return FlowDataSufficiencyConfig(**value)


def build_nested_episode_subsets(
    *,
    training_episode_ids: tuple[str, ...],
    validation_episode_ids: tuple[str, ...],
    sizes: tuple[int, ...],
    seed: int,
) -> dict[int, tuple[str, ...]]:
    train = tuple(training_episode_ids)
    validation = tuple(validation_episode_ids)
    if (
        not train
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or any(not value for value in train + validation)
    ):
        raise ValueError("episode subset identities are invalid")
    if set(train).intersection(validation):
        raise ValueError("training and validation episode identities overlap")
    if (
        not sizes
        or sizes != tuple(sorted(set(sizes)))
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes)
        or sizes[-1] != len(train)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("nested episode subset sizes or seed are invalid")
    order = np.random.default_rng(seed).permutation(len(train))
    shuffled = tuple(train[index] for index in order)
    return {size: shuffled[:size] for size in sizes}


def subset_feature_belief_corpus(
    *,
    corpus: Any,
    training_episode_ids: tuple[str, ...],
) -> FeatureBeliefCorpus:
    selected = tuple(training_episode_ids)
    if not selected or len(set(selected)) != len(selected) or any(not value for value in selected):
        raise ValueError("feature-corpus training episode selection is invalid")
    train_by_id = {
        record.episode_id: record for record in corpus.records if record.split is ProbeSplit.TRAIN
    }
    if not set(selected) <= set(train_by_id):
        raise ValueError("feature-corpus training episode selection is unavailable")
    selected_set = set(selected)
    records = tuple(
        record
        for record in corpus.records
        if record.split is not ProbeSplit.TRAIN or record.episode_id in selected_set
    )
    references: dict[ProbeSplit, list[tuple[int, int]]] = defaultdict(list)
    for record_offset, record in enumerate(records):
        for index_offset in range(len(record.indices)):
            references[record.split].append((record_offset, index_offset))
    return FeatureBeliefCorpus(
        level=corpus.level,
        temporal_contract=corpus.temporal_contract,
        latency_law=corpus.latency_law,
        records=records,
        sample_references={split: tuple(references[split]) for split in ProbeSplit},
    )


def decide_data_scaling(
    *,
    metrics_by_size_seed: dict[int, dict[int, dict[str, float]]],
    improvement_trigger: float,
    seed_disagreement_trigger: float,
) -> dict[str, Any]:
    sizes = tuple(sorted(metrics_by_size_seed))
    if (
        len(sizes) < 2
        or sizes[-2:] != (120, 180)
        or not 0.0 < improvement_trigger < 1.0
        or not 0.0 < seed_disagreement_trigger < 1.0
    ):
        raise ValueError("data-scaling decision inputs are invalid")
    seed_sets = [set(metrics_by_size_seed[size]) for size in sizes]
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("data-scaling model seeds are inconsistent")
    metric_names = set(next(iter(metrics_by_size_seed[sizes[0]].values())))
    if not metric_names:
        raise ValueError("data-scaling metrics are empty")
    medians = {}
    for size in sizes:
        rows = metrics_by_size_seed[size]
        if any(set(row) != metric_names for row in rows.values()):
            raise ValueError("data-scaling metric fields are inconsistent")
        medians[size] = {}
        for name in sorted(metric_names):
            values = np.asarray([rows[seed][name] for seed in sorted(rows)], dtype=np.float64)
            if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("data-scaling metrics must be finite and positive")
            medians[size][name] = float(np.median(values))
    improvements = {
        name: (medians[120][name] - medians[180][name]) / medians[120][name]
        for name in sorted(metric_names)
    }
    disagreements = {}
    for name in sorted(metric_names):
        values = np.asarray(
            [metrics_by_size_seed[180][seed][name] for seed in sorted(seed_sets[0])],
            dtype=np.float64,
        )
        disagreements[name] = float((np.max(values) - np.min(values)) / np.median(values))
    reasons = [
        f"unsaturated:{name}" for name, value in improvements.items() if value > improvement_trigger
    ]
    reasons.extend(
        f"seed_disagreement:{name}"
        for name, value in disagreements.items()
        if value > seed_disagreement_trigger
    )
    return {
        "decision": "collect_more" if reasons else "reuse",
        "improvement_trigger": improvement_trigger,
        "seed_disagreement_trigger": seed_disagreement_trigger,
        "median_metrics_by_size": {str(size): medians[size] for size in sizes},
        "improvements_120_to_180": improvements,
        "seed_disagreement_at_180": disagreements,
        "reasons": reasons,
    }


@dataclass(frozen=True)
class EpisodeMotionSummary:
    episode_id: str
    level: int
    scene_seed: int
    profile_type: str
    segment_count: int
    segment_kinds: tuple[str, ...]
    segment_transition_count: int
    planned_start_xy: tuple[float, float]
    planned_end_xy: tuple[float, float]
    handoff_xy: tuple[float, float]
    pre_handoff_path_length_m: float
    pre_handoff_chord_length_m: float
    pre_handoff_path_chord_ratio: float
    mean_speed_m_s: float
    maximum_speed_m_s: float
    speed_percentiles_m_s: tuple[float, float, float]
    signed_turning_angle_rad: float
    absolute_turning_angle_rad: float
    positive_curvature_fraction: float
    negative_curvature_fraction: float
    curvature_sign_change_count: int
    maximum_velocity_jump_m_s: float
    first_contact_time_seconds: float
    handoff_time_seconds: float
    success_time_seconds: float


def _event_time_seconds(events: dict[str, Any], kind: str) -> float:
    rows = events.get("events")
    if not isinstance(rows, list):
        raise ValueError("episode event stream is invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"episode requires exactly one {kind} event")
    time_us = matches[0].get("time_us")
    if isinstance(time_us, bool) or not isinstance(time_us, int) or time_us < 0:
        raise ValueError(f"episode {kind} time is invalid")
    return time_us / 1_000_000.0


def _turning_angles(displacements: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(displacements, axis=1)
    valid = displacements[norms > 1e-9]
    if len(valid) < 2:
        return np.empty(0, dtype=np.float64)
    left = valid[:-1]
    right = valid[1:]
    cross = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    dot = np.sum(left * right, axis=1)
    return np.arctan2(cross, dot)


def _xy_pair(value: Any, *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"motion profile {name} must be a finite XY pair")
    return float(array[0]), float(array[1])


def _planned_profile_geometry(
    profile: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], tuple[str, ...]]:
    profile_type = profile["type"]
    if profile_type == "constant_velocity":
        return (
            _xy_pair(profile.get("start_xy"), name="start_xy"),
            _xy_pair(profile.get("end_xy"), name="end_xy"),
            ("line",),
        )
    if profile_type == "cubic_polynomial":
        waypoints = profile.get("waypoint_positions_xy")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("cubic motion profile waypoints are invalid")
        return (
            _xy_pair(waypoints[0], name="first waypoint"),
            _xy_pair(waypoints[-1], name="last waypoint"),
            ("cubic",),
        )
    if profile_type == "piecewise_polynomial":
        segments = profile.get("segments")
        if (
            not isinstance(segments, list)
            or len(segments) < 2
            or any(
                not isinstance(segment, dict) or segment.get("kind") not in {"line", "cubic"}
                for segment in segments
            )
        ):
            raise ValueError("piecewise motion profile segments are invalid")
        return (
            _xy_pair(segments[0].get("start_xy"), name="first segment start"),
            _xy_pair(segments[-1].get("end_xy"), name="last segment end"),
            tuple(str(segment["kind"]) for segment in segments),
        )
    raise ValueError("motion profile type is unsupported")


def summarize_episode_motion(
    *,
    metadata: dict[str, Any],
    events: dict[str, Any],
    arrays: Any,
) -> EpisodeMotionSummary:
    episode_id = metadata.get("episode_id")
    level = metadata.get("level")
    scene_seed = metadata.get("scene_seed")
    profile = metadata.get("motion_profile")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or level not in (1, 2, 3)
        or isinstance(scene_seed, bool)
        or not isinstance(scene_seed, int)
        or not isinstance(profile, dict)
        or not isinstance(profile.get("type"), str)
    ):
        raise ValueError("episode motion metadata is invalid")
    handoff_seconds = _event_time_seconds(events, "handoff")
    boundary_time_us = np.asarray(arrays["boundary_time_us"], dtype=np.int64)
    position = np.asarray(arrays["commanded_motion_position"], dtype=np.float64)
    velocity = np.asarray(arrays["commanded_motion_velocity"], dtype=np.float64)
    acceleration = np.asarray(arrays["commanded_motion_acceleration"], dtype=np.float64)
    segment_index = np.asarray(arrays["commanded_motion_segment_index"], dtype=np.int64)
    count = len(boundary_time_us)
    if (
        count < 2
        or position.shape != (count, 3)
        or velocity.shape != (count, 3)
        or acceleration.shape != (count, 3)
        or segment_index.shape != (count,)
        or any(not np.all(np.isfinite(value)) for value in (position, velocity, acceleration))
    ):
        raise ValueError("episode commanded-motion arrays are invalid")
    active = boundary_time_us <= round(handoff_seconds * 1_000_000)
    if np.count_nonzero(active) < 2:
        raise ValueError("episode motion has insufficient pre-handoff boundaries")
    xy = position[active, :2]
    velocity_xy = velocity[active, :2]
    acceleration_xy = acceleration[active, :2]
    segments = segment_index[active]
    displacement = np.diff(xy, axis=0)
    path_length = float(np.linalg.norm(displacement, axis=1).sum())
    chord_length = float(np.linalg.norm(xy[-1] - xy[0]))
    ratio = path_length / chord_length if chord_length > 1e-9 else float("inf")
    speed = np.linalg.norm(velocity_xy, axis=1)
    curvature_signal = (
        velocity_xy[:, 0] * acceleration_xy[:, 1] - velocity_xy[:, 1] * acceleration_xy[:, 0]
    )
    moving = speed > 1e-6
    signs = np.sign(curvature_signal[moving])
    nonzero_signs = signs[signs != 0]
    denominator = max(int(np.count_nonzero(moving)), 1)
    sign_changes = (
        int(np.count_nonzero(np.diff(nonzero_signs) != 0)) if len(nonzero_signs) > 1 else 0
    )
    transitions = np.flatnonzero(np.diff(segments) != 0) + 1
    velocity_jumps = (
        np.linalg.norm(velocity_xy[transitions] - velocity_xy[transitions - 1], axis=1)
        if len(transitions)
        else np.zeros(1, dtype=np.float64)
    )
    angles = _turning_angles(displacement)
    profile_segments = profile.get("segments")
    planned_start, planned_end, segment_kinds = _planned_profile_geometry(profile)
    segment_count = len(profile_segments) if isinstance(profile_segments, list) else 1
    return EpisodeMotionSummary(
        episode_id=episode_id,
        level=level,
        scene_seed=scene_seed,
        profile_type=profile["type"],
        segment_count=segment_count,
        segment_kinds=segment_kinds,
        segment_transition_count=len(transitions),
        planned_start_xy=planned_start,
        planned_end_xy=planned_end,
        handoff_xy=(float(xy[-1, 0]), float(xy[-1, 1])),
        pre_handoff_path_length_m=path_length,
        pre_handoff_chord_length_m=chord_length,
        pre_handoff_path_chord_ratio=ratio,
        mean_speed_m_s=float(np.mean(speed)),
        maximum_speed_m_s=float(np.max(speed)),
        speed_percentiles_m_s=tuple(float(value) for value in np.quantile(speed, (0.1, 0.5, 0.9))),
        signed_turning_angle_rad=float(np.sum(angles)),
        absolute_turning_angle_rad=float(np.sum(np.abs(angles))),
        positive_curvature_fraction=float(np.count_nonzero(curvature_signal > 1e-9) / denominator),
        negative_curvature_fraction=float(np.count_nonzero(curvature_signal < -1e-9) / denominator),
        curvature_sign_change_count=sign_changes,
        maximum_velocity_jump_m_s=float(np.max(velocity_jumps)),
        first_contact_time_seconds=_event_time_seconds(events, "first_contact"),
        handoff_time_seconds=handoff_seconds,
        success_time_seconds=_event_time_seconds(events, "success"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("physical-data summary values must be finite and non-empty")
    quantiles = np.quantile(array, (0.1, 0.5, 0.9))
    return {
        "minimum": float(np.min(array)),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _workspace_histogram(
    *,
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    bin_count: int,
    minimum_count: int,
) -> dict[str, Any]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("workspace points must have shape [N, 2]")
    x_min, x_max, y_min, y_max = bounds
    histogram, x_edges, y_edges = np.histogram2d(
        array[:, 0],
        array[:, 1],
        bins=bin_count,
        range=((x_min, x_max), (y_min, y_max)),
    )
    counts = histogram.astype(np.int64)
    occupied = counts[counts > 0]
    return {
        "bounds_xy": list(bounds),
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "counts": counts.tolist(),
        "occupied_bin_count": int(np.count_nonzero(counts)),
        "empty_bin_count": int(np.count_nonzero(counts == 0)),
        "underpopulated_occupied_bin_count": int(
            np.count_nonzero((counts > 0) & (counts < minimum_count))
        ),
        "minimum_occupied_count": int(np.min(occupied)) if len(occupied) else 0,
    }


def _curvature_category(summary: EpisodeMotionSummary) -> str:
    if summary.absolute_turning_angle_rad < 0.05:
        return "approximately_straight"
    if summary.positive_curvature_fraction > 0.0 and summary.negative_curvature_fraction > 0.0:
        return "mixed_sign"
    if summary.positive_curvature_fraction > 0.0:
        return "positive"
    if summary.negative_curvature_fraction > 0.0:
        return "negative"
    return "near_zero_curvature_signal"


def _level_summary(
    *,
    level: int,
    rows: list[EpisodeMotionSummary],
    split_by_episode: dict[str, str],
    workspace_bounds: tuple[float, float, float, float],
    config: FlowDataSufficiencyConfig,
) -> dict[str, Any]:
    training_count = sum(split_by_episode[row.episode_id] == "train" for row in rows)
    validation_count = sum(split_by_episode[row.episode_id] == "validation" for row in rows)
    return {
        "level": level,
        "episode_count": len(rows),
        "training_episode_count": training_count,
        "validation_episode_count": validation_count,
        "profile_type_counts": dict(sorted(Counter(row.profile_type for row in rows).items())),
        "segment_count_counts": {
            str(key): value
            for key, value in sorted(Counter(row.segment_count for row in rows).items())
        },
        "segment_kind_counts": dict(
            sorted(Counter(kind for row in rows for kind in row.segment_kinds).items())
        ),
        "encountered_segment_transition_count_counts": {
            str(key): value
            for key, value in sorted(Counter(row.segment_transition_count for row in rows).items())
        },
        "curvature_category_counts": dict(
            sorted(Counter(_curvature_category(row) for row in rows).items())
        ),
        "pre_handoff_path_length_m": _numeric_summary(
            [row.pre_handoff_path_length_m for row in rows]
        ),
        "pre_handoff_path_chord_ratio": _numeric_summary(
            [row.pre_handoff_path_chord_ratio for row in rows]
        ),
        "mean_speed_m_s": _numeric_summary([row.mean_speed_m_s for row in rows]),
        "maximum_speed_m_s": _numeric_summary([row.maximum_speed_m_s for row in rows]),
        "absolute_turning_angle_rad": _numeric_summary(
            [row.absolute_turning_angle_rad for row in rows]
        ),
        "maximum_velocity_jump_m_s": _numeric_summary(
            [row.maximum_velocity_jump_m_s for row in rows]
        ),
        "first_contact_time_seconds": _numeric_summary(
            [row.first_contact_time_seconds for row in rows]
        ),
        "handoff_time_seconds": _numeric_summary([row.handoff_time_seconds for row in rows]),
        "success_time_seconds": _numeric_summary([row.success_time_seconds for row in rows]),
        "planned_start_workspace": _workspace_histogram(
            points=[row.planned_start_xy for row in rows],
            bounds=workspace_bounds,
            bin_count=config.workspace_bin_count_per_axis,
            minimum_count=config.minimum_occupied_bin_count,
        ),
        "planned_end_workspace": _workspace_histogram(
            points=[row.planned_end_xy for row in rows],
            bounds=workspace_bounds,
            bin_count=config.workspace_bin_count_per_axis,
            minimum_count=config.minimum_occupied_bin_count,
        ),
        "handoff_workspace": _workspace_histogram(
            points=[row.handoff_xy for row in rows],
            bounds=workspace_bounds,
            bin_count=config.workspace_bin_count_per_axis,
            minimum_count=config.minimum_occupied_bin_count,
        ),
    }


def write_flow_belief_physical_data_audit(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    audit_config_path: Path,
    output_dir: Path,
) -> Path:
    source_path = source_bulk_manifest.resolve()
    split_path = split_config_path.resolve()
    config_path = audit_config_path.resolve()
    for path in (source_path, split_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"physical-data audit input does not exist: {path}")
    source = _load_json(source_path)
    if (
        source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or source.get("levels") != [1, 2, 3]
        or source.get("episode_count") != 600
    ):
        raise ValueError("physical-data audit requires the eligible formal corpus")
    admitted = source.get("admitted_episode_manifests")
    if not isinstance(admitted, list) or len(admitted) != 600:
        raise ValueError("physical-data audit episode inventory is invalid")
    split_plan = load_episode_split_plan(split_path)
    config = load_flow_data_sufficiency_config(config_path)
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"physical-data audit output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    source_root = source_path.parent
    provenance = collect_implementation_provenance(project_root)
    level_manifests = {}
    episode_counts = {}
    try:
        for level in (1, 2, 3):
            relative_manifests = sorted(
                relative for relative in admitted if relative.startswith(f"episodes/L{level}/")
            )
            if len(relative_manifests) != 200:
                raise ValueError(f"physical-data audit L{level} episode count is invalid")
            motion_config = yaml.safe_load(
                (
                    project_root.resolve() / f"configs/motion/dynamic_grasp_lift_l{level}.yaml"
                ).read_text(encoding="utf-8")
            )
            endpoint_bounds = tuple(float(value) for value in motion_config["endpoint_bounds_xy"])
            rows = []
            split_by_episode = {}
            for relative in relative_manifests:
                episode_dir = (source_root / relative).parent
                episode_manifest = _load_json(episode_dir / "manifest.json")
                if (
                    episode_manifest.get("format_id") != "synchronized_episode_npz_v3"
                    or episode_manifest.get("terminal_status") != "success"
                    or episode_manifest.get("record_profile") != "belief"
                ):
                    raise ValueError("physical-data audit found an ineligible episode")
                metadata = _load_json(episode_dir / "metadata.json")
                events = _load_json(episode_dir / "events.json")
                with np.load(episode_dir / "arrays.npz", allow_pickle=False) as arrays:
                    summary = summarize_episode_motion(
                        metadata=metadata,
                        events=events,
                        arrays=arrays,
                    )
                rows.append(summary)
                split_by_episode[summary.episode_id] = split_plan.split_for_seed(summary.scene_seed)
            training_ids = tuple(
                row.episode_id for row in rows if split_by_episode[row.episode_id] == "train"
            )
            validation_ids = tuple(
                row.episode_id for row in rows if split_by_episode[row.episode_id] == "validation"
            )
            if config.subset_sizes[-1] != len(training_ids):
                raise ValueError("physical-data audit largest subset must equal training count")
            subsets = build_nested_episode_subsets(
                training_episode_ids=training_ids,
                validation_episode_ids=validation_ids,
                sizes=config.subset_sizes,
                seed=config.subset_seed + level * 1_000_000,
            )
            level_dir = building / f"L{level}"
            level_dir.mkdir()
            _write_json(level_dir / "episodes.json", [asdict(row) for row in rows])
            _write_json(
                level_dir / "nested_subsets.json",
                {str(size): list(episode_ids) for size, episode_ids in subsets.items()},
            )
            _write_json(
                level_dir / "summary.json",
                _level_summary(
                    level=level,
                    rows=rows,
                    split_by_episode=split_by_episode,
                    workspace_bounds=endpoint_bounds,
                    config=config,
                ),
            )
            artifacts = {
                name: sha256_file(level_dir / name)
                for name in ("episodes.json", "nested_subsets.json", "summary.json")
            }
            _write_json(
                level_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "format_id": "level_flow_belief_physical_data_audit_v1",
                    "level": level,
                    "episode_count": len(rows),
                    "artifacts": artifacts,
                },
            )
            level_manifests[f"L{level}"] = f"L{level}/manifest.json"
            episode_counts[f"L{level}"] = len(rows)
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "flow_belief_physical_data_audit_v1",
                "eligible": not provenance.dirty,
                "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "levels": [1, 2, 3],
                "episode_counts": episode_counts,
                "level_manifests": level_manifests,
                "input_sha256": {
                    "source_bulk_manifest": sha256_file(source_path),
                    "split_config": sha256_file(split_path),
                    "audit_config": sha256_file(config_path),
                },
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
