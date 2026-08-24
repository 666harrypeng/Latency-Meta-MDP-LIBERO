"""Leakage-safe temporal sample views for frozen-vision state probes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

ROBOT_PROPRIO_DIM = 16
PROBE_TARGET_DIM = 9


class ProbeSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


def first_tranche_probe_split(seed: int) -> ProbeSplit:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1000 <= seed <= 1024:
        raise ValueError("probe seed is outside the first-tranche bank")
    if seed <= 1019:
        return ProbeSplit.TRAIN
    if seed <= 1021:
        return ProbeSplit.VALIDATION
    return ProbeSplit.HOLDOUT


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class VisionProbeSampleIndex:
    episode_id: str
    level: int
    scene_seed: int
    split: ProbeSplit
    history_start_tick: int
    source_tick: int
    pre_handoff: bool


@dataclass(frozen=True)
class VisionProbeSample:
    index: VisionProbeSampleIndex
    vision_history: np.ndarray
    robot_proprio_history: np.ndarray
    target_state: np.ndarray
    pre_handoff: bool

    def __post_init__(self) -> None:
        if self.vision_history.ndim != 4 or self.vision_history.shape[1:] != (
            2,
            196,
            384,
        ):
            raise ValueError("probe vision history does not satisfy the patch-token contract")
        history_count = self.vision_history.shape[0]
        if self.robot_proprio_history.shape != (history_count, ROBOT_PROPRIO_DIM):
            raise ValueError("probe proprioception history has invalid shape")
        if self.target_state.shape != (PROBE_TARGET_DIM,):
            raise ValueError("probe target state has invalid shape")
        if (
            self.vision_history.dtype != np.float16
            or not np.all(np.isfinite(self.vision_history))
            or not np.all(np.isfinite(self.robot_proprio_history))
            or not np.all(np.isfinite(self.target_state))
        ):
            raise ValueError("probe sample contains invalid numeric values")
        object.__setattr__(self, "vision_history", _readonly(self.vision_history))
        object.__setattr__(
            self,
            "robot_proprio_history",
            _readonly(self.robot_proprio_history, dtype=np.float32),
        )
        object.__setattr__(self, "target_state", _readonly(self.target_state, dtype=np.float32))


def build_probe_sample_indices(
    *,
    episode: Any,
    history_sample_count: int,
) -> tuple[VisionProbeSampleIndex, ...]:
    if (
        isinstance(history_sample_count, bool)
        or not isinstance(history_sample_count, int)
        or history_sample_count <= 0
    ):
        raise ValueError("probe history sample count must be a positive integer")
    if (
        not isinstance(episode.episode_id, str)
        or not episode.episode_id
        or episode.level not in (1, 2, 3)
        or episode.boundary_count < history_sample_count
    ):
        raise ValueError("episode cannot provide the requested probe histories")
    split = first_tranche_probe_split(episode.scene_seed)
    return tuple(
        VisionProbeSampleIndex(
            episode_id=episode.episode_id,
            level=episode.level,
            scene_seed=episode.scene_seed,
            split=split,
            history_start_tick=source_tick - history_sample_count + 1,
            source_tick=source_tick,
            pre_handoff=str(episode.supervision.handoff_state[source_tick]) != "physical",
        )
        for source_tick in range(history_sample_count - 1, episode.boundary_count)
    )


def _robot_proprio_history(episode: Any, history: slice) -> np.ndarray:
    deployment = episode.deployment
    width = deployment.gripper_qpos[:, 0] - deployment.gripper_qpos[:, 1]
    width_velocity = deployment.gripper_qvel[:, 0] - deployment.gripper_qvel[:, 1]
    return np.concatenate(
        (
            deployment.robot_qpos[history],
            deployment.robot_qvel[history],
            width[history, None],
            width_velocity[history, None],
        ),
        axis=1,
    )


def materialize_probe_sample(
    *,
    episode: Any,
    features: np.ndarray,
    index: VisionProbeSampleIndex,
    history_sample_count: int,
) -> VisionProbeSample:
    if (
        index.episode_id != episode.episode_id
        or index.level != episode.level
        or index.scene_seed != episode.scene_seed
        or index.source_tick - index.history_start_tick + 1 != history_sample_count
    ):
        raise ValueError("probe index and episode disagree")
    feature_array = np.asarray(features)
    if feature_array.shape != (episode.boundary_count, 2, 196, 384):
        raise ValueError("probe feature cache and episode boundaries disagree")
    history = slice(index.history_start_tick, index.source_tick + 1)
    target_tick = index.source_tick
    target = np.concatenate(
        (
            episode.supervision.object_pose[target_tick, :3],
            episode.supervision.object_velocity[target_tick, :3],
            episode.supervision.relative_geometry[target_tick, :3],
        )
    )
    return VisionProbeSample(
        index=index,
        vision_history=feature_array[history],
        robot_proprio_history=_robot_proprio_history(episode, history),
        target_state=target,
        pre_handoff=index.pre_handoff,
    )
