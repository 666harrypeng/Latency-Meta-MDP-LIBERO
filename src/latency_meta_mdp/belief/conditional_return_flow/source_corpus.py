"""Verified lazy access to immutable Conditional Return Flow source episodes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchCorpusConfig,
    SourceContextIdentity,
    SourceSelectionExclusion,
    normalize_scene_seed_ranges,
)
from latency_meta_mdp.episode_split import load_episode_split_plan
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec
from latency_meta_mdp.recording import RecordProfile
from latency_meta_mdp.temporal_contract import TemporalContract

_CONFIG_RELATIVE_PATHS = {
    "runtime": "configs/runtime/robosuite_v1.yaml",
    "task": "configs/task/dynamic_grasp_lift_l0.yaml",
    "control": "configs/control/panda_osc_pose_delta_v1.yaml",
    "expert": "configs/expert/panda_ball_feedback_v1.yaml",
}
_REQUIRED_ARRAYS = {
    "boundary_formal_tick",
    "boundary_time_us",
    "boundary_outcome_status",
    "agentview_rgb",
    "robot0_eye_in_hand_rgb",
    "robot_qpos",
    "robot_qvel",
    "gripper_qpos",
    "gripper_qvel",
    "eef_position_world",
    "eef_orientation_matrix_world",
    "object_pose",
    "object_velocity",
    "expert_action",
    "expert_phase",
}
_OBSERVATION_ARRAYS = (
    "boundary_formal_tick",
    "boundary_time_us",
    "agentview_rgb",
    "robot0_eye_in_hand_rgb",
    "robot_qpos",
    "robot_qvel",
    "gripper_qpos",
    "gripper_qvel",
)
_EPISODE_MANIFEST_RE = re.compile(
    r"^episodes/L(?P<level>[123])/seed_(?P<scene_seed>[0-9]{6})/manifest\.json$"
)


@dataclass(frozen=True)
class SourceContextSelection:
    contexts: tuple[SelectedSourceContext, ...]
    exclusions: tuple[SourceSelectionExclusion, ...]
    expected_slot_count: int

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        exclusions = tuple(self.exclusions)
        if (
            isinstance(self.expected_slot_count, bool)
            or not isinstance(self.expected_slot_count, int)
            or self.expected_slot_count <= 0
            or len(contexts) + len(exclusions) != self.expected_slot_count
        ):
            raise ValueError("source selection does not account for every episode-phase slot")
        context_keys = {(row.identity.episode_id, row.identity.source_phase) for row in contexts}
        exclusion_keys = {(row.episode_id, row.source_phase) for row in exclusions}
        if (
            len(context_keys) != len(contexts)
            or len(exclusion_keys) != len(exclusions)
            or context_keys & exclusion_keys
        ):
            raise ValueError("source selection episode-phase slots must be unique")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "exclusions", exclusions)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class VerifiedSourceEpisode:
    episode_dir: Path
    episode_manifest_sha256: str
    arrays_sha256: str
    metadata_sha256: str
    episode_id: str
    level: int
    scene_seed: int
    split: str
    motion_seed: int
    expert_seed: int
    record_profile: str
    camera_width: int
    camera_height: int
    runtime_config_path: Path
    task_config_path: Path
    motion_config_path: Path
    control_config_path: Path
    expert_config_path: Path
    motion_profile_mapping: Mapping[str, object]
    motion_profile_sha256: str
    arrays_path: Path

    def to_expert_spec(self) -> ExpertEpisodeSpec:
        return ExpertEpisodeSpec(
            episode_id=self.episode_id,
            level=self.level,
            scene_seed=self.scene_seed,
            motion_seed=self.motion_seed,
            expert_seed=self.expert_seed,
            record_profile=RecordProfile(self.record_profile),
            camera_width=self.camera_width,
            camera_height=self.camera_height,
        )

    def load_arrays(self, names: tuple[str, ...]) -> Mapping[str, np.ndarray]:
        return load_verified_episode_arrays(self, names=names)


@dataclass(frozen=True)
class SelectedSourceContext:
    identity: SourceContextIdentity
    episode: VerifiedSourceEpisode


def load_verified_episode_arrays(
    episode: VerifiedSourceEpisode,
    *,
    names: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    if not names or len(set(names)) != len(names):
        raise ValueError("requested episode arrays must be non-empty and unique")
    if sha256_file(episode.arrays_path) != episode.arrays_sha256:
        raise ValueError("source arrays hash mismatch after episode admission")
    with np.load(episode.arrays_path, allow_pickle=False) as source:
        missing = set(names) - set(source.files)
        if missing:
            raise ValueError(f"source episode arrays are missing fields: {sorted(missing)}")
        result = {name: _readonly(source[name]) for name in names}
    return MappingProxyType(result)


def _config_paths(project_root: Path, *, level: int) -> dict[str, Path]:
    result = {
        name: project_root / relative for name, relative in _CONFIG_RELATIVE_PATHS.items()
    }
    result["motion"] = project_root / f"configs/motion/dynamic_grasp_lift_l{level}.yaml"
    return result


def _verify_nested_episode(
    *,
    project_root: Path,
    source_root: Path,
    top_manifest: dict[str, Any],
    relative_manifest: str,
    split_for_seed: Any,
) -> VerifiedSourceEpisode:
    episode_manifest_path = (source_root / relative_manifest).resolve()
    try:
        episode_manifest_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("source episode manifest escapes formal corpus") from error
    artifacts = top_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("formal corpus artifact inventory is invalid")
    if (
        not episode_manifest_path.is_file()
        or sha256_file(episode_manifest_path) != artifacts.get(relative_manifest)
    ):
        raise ValueError("source episode manifest hash mismatch")
    nested = _load_json(episode_manifest_path)
    nested_artifacts = nested.get("artifacts")
    if (
        nested.get("schema_version") != 3
        or nested.get("format_id") != "synchronized_episode_npz_v3"
        or nested.get("record_profile") != "belief"
        or not isinstance(nested_artifacts, dict)
    ):
        raise ValueError("source episode manifest contract is invalid")
    episode_dir = episode_manifest_path.parent
    child_paths = {name: episode_dir / name for name in nested_artifacts}
    for name, path in child_paths.items():
        top_relative = path.relative_to(source_root).as_posix()
        if (
            not path.is_file()
            or sha256_file(path) != nested_artifacts[name]
            or sha256_file(path) != artifacts.get(top_relative)
        ):
            raise ValueError(f"source episode artifact hash mismatch: {name}")
    arrays_path = episode_dir / "arrays.npz"
    metadata_path = episode_dir / "metadata.json"
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise ValueError("source episode lacks arrays or metadata")
    metadata = _load_json(metadata_path)
    level = metadata.get("level")
    seed = metadata.get("scene_seed")
    if level not in (1, 2, 3) or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("source episode level or scene seed is invalid")
    if (
        metadata.get("episode_id") != nested.get("episode_id")
        or metadata.get("record_profile") != "belief"
        or metadata.get("formal_tick_us") != 20_000
        or metadata.get("physics_dt_us") != 2_000
    ):
        raise ValueError("source episode metadata identity is invalid")
    paths = _config_paths(project_root, level=level)
    recorded_hashes = metadata.get("config_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(paths):
        raise ValueError("source config hash inventory is invalid")
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != recorded_hashes[name]:
            raise ValueError(f"source config hash mismatch: {name}")
    with np.load(arrays_path, allow_pickle=False) as source:
        if not _REQUIRED_ARRAYS <= set(source.files):
            raise ValueError("source episode lacks required branch arrays")
        boundary_ticks = np.asarray(source["boundary_formal_tick"])
        boundary_times = np.asarray(source["boundary_time_us"])
        agent_rgb = np.asarray(source["agentview_rgb"])
        wrist_rgb = np.asarray(source["robot0_eye_in_hand_rgb"])
        expert_actions = np.asarray(source["expert_action"])
        if (
            boundary_ticks.ndim != 1
            or not np.array_equal(boundary_ticks, np.arange(len(boundary_ticks)))
            or not np.array_equal(boundary_times, boundary_ticks * 20_000)
            or agent_rgb.ndim != 4
            or wrist_rgb.shape != agent_rgb.shape
            or agent_rgb.shape[-1] != 3
            or len(expert_actions) + 1 != len(boundary_ticks)
        ):
            raise ValueError("source episode synchronized array contract is invalid")
        camera_height, camera_width = agent_rgb.shape[1:3]
    motion_profile = metadata.get("motion_profile")
    if not isinstance(motion_profile, dict):
        raise ValueError("source episode motion profile is invalid")
    frozen_profile = json.loads(json.dumps(motion_profile, allow_nan=False))
    return VerifiedSourceEpisode(
        episode_dir=episode_dir,
        episode_manifest_sha256=sha256_file(episode_manifest_path),
        arrays_sha256=sha256_file(arrays_path),
        metadata_sha256=sha256_file(metadata_path),
        episode_id=str(metadata["episode_id"]),
        level=level,
        scene_seed=seed,
        split=split_for_seed(seed),
        motion_seed=int(metadata["motion_seed"]),
        expert_seed=int(metadata["expert_seed"]),
        record_profile=str(metadata["record_profile"]),
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        runtime_config_path=paths["runtime"],
        task_config_path=paths["task"],
        motion_config_path=paths["motion"],
        control_config_path=paths["control"],
        expert_config_path=paths["expert"],
        motion_profile_mapping=MappingProxyType(frozen_profile),
        motion_profile_sha256=_canonical_json_sha256(frozen_profile),
        arrays_path=arrays_path,
    )


def load_verified_source_episodes(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    levels: tuple[int, ...],
    allowed_scene_seed_ranges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[VerifiedSourceEpisode, ...]:
    if levels != tuple(sorted(set(levels))) or any(level not in (1, 2, 3) for level in levels):
        raise ValueError("source levels must be sorted unique values from 1, 2, 3")
    root = project_root.resolve()
    seed_ranges = normalize_scene_seed_ranges(allowed_scene_seed_ranges)
    source_path = source_bulk_manifest.resolve()
    source_root = source_path.parent
    top = _load_json(source_path)
    if (
        top.get("schema_version") != 1
        or top.get("format_id") != "panda_ball_formal_corpus_v1"
        or top.get("eligible") is not True
        or top.get("implementation_dirty") is not False
    ):
        raise ValueError("formal source corpus is not eligible")
    admitted = top.get("admitted_episode_manifests")
    if not isinstance(admitted, list) or not admitted or len(set(admitted)) != len(admitted):
        raise ValueError("formal source episode inventory is invalid")
    split = load_episode_split_plan(split_config_path)
    episodes = []
    for relative in admitted:
        if not isinstance(relative, str):
            raise ValueError("formal source episode path must be a string")
        identity = _EPISODE_MANIFEST_RE.fullmatch(relative)
        if identity is None:
            raise ValueError("formal source episode path is malformed")
        if int(identity.group("level")) not in levels:
            continue
        scene_seed = int(identity.group("scene_seed"))
        if seed_ranges is not None and not any(
            start <= scene_seed < stop for start, stop in seed_ranges
        ):
            continue
        episode = _verify_nested_episode(
            project_root=root,
            source_root=source_root,
            top_manifest=top,
            relative_manifest=relative,
            split_for_seed=split.split_for_seed,
        )
        episodes.append(episode)
    if not episodes or set(levels) - {episode.level for episode in episodes}:
        raise ValueError("formal source corpus does not cover requested levels")
    return tuple(sorted(episodes, key=lambda row: (row.level, row.scene_seed, row.episode_id)))


def load_one_verified_source(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    level: int,
    scene_seed: int,
) -> VerifiedSourceEpisode:
    if level not in (1, 2, 3):
        raise ValueError("source level must be 1, 2, or 3")
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int) or scene_seed < 0:
        raise ValueError("source scene seed must be a non-negative integer")
    root = project_root.resolve()
    source_path = source_bulk_manifest.resolve()
    source_root = source_path.parent
    top = _load_json(source_path)
    if (
        top.get("schema_version") != 1
        or top.get("format_id") != "panda_ball_formal_corpus_v1"
        or top.get("eligible") is not True
        or top.get("implementation_dirty") is not False
    ):
        raise ValueError("formal source corpus is not eligible")
    relative = f"episodes/L{level}/seed_{scene_seed:06d}/manifest.json"
    admitted = top.get("admitted_episode_manifests")
    if not isinstance(admitted, list) or relative not in admitted:
        raise ValueError("formal source corpus does not admit the requested episode")
    split = load_episode_split_plan(split_config_path)
    return _verify_nested_episode(
        project_root=root,
        source_root=source_root,
        top_manifest=top,
        relative_manifest=relative,
        split_for_seed=split.split_for_seed,
    )


def source_observation_sha256(
    *,
    arrays: Mapping[str, np.ndarray],
    source_tick: int,
    k: int,
) -> str:
    if isinstance(source_tick, bool) or not isinstance(source_tick, int):
        raise TypeError("source tick must be an integer")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0 or source_tick < k - 1:
        raise ValueError("source observation window is invalid")
    start = source_tick - k + 1
    digest = hashlib.sha256()
    for name in _OBSERVATION_ARRAYS:
        if name not in arrays:
            raise ValueError(f"source observation is missing {name}")
        value = np.ascontiguousarray(np.asarray(arrays[name])[start : source_tick + 1])
        encoded_name = name.encode()
        encoded_dtype = value.dtype.str.encode()
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "big"))
        digest.update(encoded_dtype)
        digest.update(np.asarray(value.shape, dtype=">i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def select_source_contexts(
    *,
    episodes: tuple[VerifiedSourceEpisode, ...],
    config: BranchCorpusConfig,
    temporal: TemporalContract,
) -> SourceContextSelection:
    selected = []
    exclusions = []
    for episode in episodes:
        arrays = episode.load_arrays(
            names=(
                "boundary_formal_tick",
                "boundary_time_us",
                "boundary_outcome_status",
                "agentview_rgb",
                "robot0_eye_in_hand_rgb",
                "robot_qpos",
                "robot_qvel",
                "gripper_qpos",
                "gripper_qvel",
                "expert_action",
                "expert_phase",
            )
        )
        actions = arrays["expert_action"]
        phases = arrays["expert_phase"].astype(str)
        outcomes = arrays["boundary_outcome_status"].astype(str)
        interval = temporal.belief_source_interval(episode_action_count=len(actions))
        for phase in config.phases:
            phase_candidates = np.flatnonzero(phases == phase)
            interval_candidates = phase_candidates[
                (phase_candidates >= interval.minimum)
                & (phase_candidates <= interval.maximum)
            ]
            candidates = interval_candidates[outcomes[interval_candidates] == "running"]
            if len(candidates) == 0:
                reason = (
                    "phase_absent"
                    if len(phase_candidates) == 0
                    else "no_phase_tick_in_source_interval"
                    if len(interval_candidates) == 0
                    else "no_running_phase_tick_in_source_interval"
                )
                exclusions.append(
                    SourceSelectionExclusion(
                        episode_id=episode.episode_id,
                        level=episode.level,
                        scene_seed=episode.scene_seed,
                        split=episode.split,
                        source_phase=phase,
                        reason=reason,
                        source_interval_minimum=interval.minimum,
                        source_interval_maximum=interval.maximum,
                        phase_tick_count=len(phase_candidates),
                        interval_phase_tick_count=len(interval_candidates),
                        running_interval_phase_tick_count=len(candidates),
                    )
                )
                continue
            source_tick = int(candidates[(len(candidates) - 1) // 2])
            observation_hash = source_observation_sha256(
                arrays=arrays,
                source_tick=source_tick,
                k=temporal.history_sample_count,
            )
            identity = SourceContextIdentity(
                source_context_id=(
                    f"L{episode.level}-seed{episode.scene_seed:06d}-tick{source_tick:06d}"
                ),
                episode_id=episode.episode_id,
                level=episode.level,
                scene_seed=episode.scene_seed,
                split=episode.split,
                source_tick=source_tick,
                source_phase=phase,
                history_start_tick=source_tick - temporal.history_sample_count + 1,
                source_episode_manifest_sha256=episode.episode_manifest_sha256,
                source_arrays_sha256=episode.arrays_sha256,
                source_metadata_sha256=episode.metadata_sha256,
                source_observation_sha256=observation_hash,
                motion_profile_sha256=episode.motion_profile_sha256,
            )
            selected.append(SelectedSourceContext(identity=identity, episode=episode))
    contexts = tuple(
        sorted(
            selected,
            key=lambda row: (
                row.identity.level,
                row.identity.scene_seed,
                config.phases.index(row.identity.source_phase),
                row.identity.source_tick,
            ),
        )
    )
    ordered_exclusions = tuple(
        sorted(
            exclusions,
            key=lambda row: (
                row.level,
                row.scene_seed,
                config.phases.index(row.source_phase),
            ),
        )
    )
    return SourceContextSelection(
        contexts=contexts,
        exclusions=ordered_exclusions,
        expected_slot_count=len(episodes) * len(config.phases),
    )
