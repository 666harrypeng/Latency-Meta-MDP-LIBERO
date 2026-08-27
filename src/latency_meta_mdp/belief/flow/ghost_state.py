"""Kinematic return-state reconstruction and Flow sample medoids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _readonly(value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("reconstructed return-state array is invalid")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ReconstructedReturnState:
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    object_qpos: np.ndarray
    object_qvel: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("robot_qpos", (7,)),
            ("robot_qvel", (7,)),
            ("gripper_qpos", (2,)),
            ("gripper_qvel", (2,)),
            ("object_qpos", (7,)),
            ("object_qvel", (6,)),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), shape=shape))


def _symmetric_pair(value: float, target_pair: np.ndarray) -> np.ndarray:
    target = np.asarray(target_pair, dtype=np.float64)
    if target.shape != (2,) or not np.all(np.isfinite(target)) or not np.isfinite(value):
        raise ValueError("gripper reconstruction inputs are invalid")
    midpoint = float(np.mean(target))
    return np.asarray([midpoint + value / 2.0, midpoint - value / 2.0])


def reconstruct_return_state(
    *,
    predicted_state: np.ndarray,
    target_gripper_qpos: np.ndarray,
    target_gripper_qvel: np.ndarray,
    target_object_pose: np.ndarray,
    target_object_velocity: np.ndarray,
) -> ReconstructedReturnState:
    predicted = np.asarray(predicted_state, dtype=np.float64)
    object_pose = np.asarray(target_object_pose, dtype=np.float64)
    object_velocity = np.asarray(target_object_velocity, dtype=np.float64)
    if (
        predicted.shape != (22,)
        or object_pose.shape != (7,)
        or object_velocity.shape != (6,)
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(object_pose))
        or not np.all(np.isfinite(object_velocity))
    ):
        raise ValueError("return-state reconstruction inputs are invalid")
    return ReconstructedReturnState(
        robot_qpos=predicted[:7],
        robot_qvel=predicted[7:14],
        gripper_qpos=_symmetric_pair(float(predicted[14]), target_gripper_qpos),
        gripper_qvel=_symmetric_pair(float(predicted[15]), target_gripper_qvel),
        object_qpos=np.concatenate((predicted[16:19], object_pose[3:])),
        object_qvel=np.concatenate((predicted[19:22], object_velocity[3:])),
    )


def valid_sample_mask(
    *,
    physical_samples: np.ndarray,
    joint_ranges: np.ndarray,
    gripper_width_range: tuple[float, float],
    object_position_bounds: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(physical_samples, dtype=np.float64)
    joints = np.asarray(joint_ranges, dtype=np.float64)
    bounds = np.asarray(object_position_bounds, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 22:
        raise ValueError("physical samples must have shape [S, 22]")
    if joints.shape != (7, 2) or bounds.shape != (3, 2):
        raise ValueError("sample validity ranges have invalid shapes")
    width_min, width_max = gripper_width_range
    if (
        not np.all(np.isfinite(joints))
        or not np.all(np.isfinite(bounds))
        or not np.isfinite(width_min)
        or not np.isfinite(width_max)
        or np.any(joints[:, 0] > joints[:, 1])
        or np.any(bounds[:, 0] > bounds[:, 1])
        or width_min > width_max
    ):
        raise ValueError("sample validity ranges are invalid")
    finite = np.all(np.isfinite(samples), axis=1)
    joint_valid = np.all(
        (samples[:, :7] >= joints[:, 0]) & (samples[:, :7] <= joints[:, 1]),
        axis=1,
    )
    width_valid = (samples[:, 14] >= width_min) & (samples[:, 14] <= width_max)
    position_valid = np.all(
        (samples[:, 16:19] >= bounds[:, 0]) & (samples[:, 16:19] <= bounds[:, 1]),
        axis=1,
    )
    return finite & joint_valid & width_valid & position_valid


def select_sample_medoid(
    normalized_samples: np.ndarray,
    valid_mask: np.ndarray,
) -> int:
    samples = np.asarray(normalized_samples, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if samples.ndim != 2 or valid.shape != (len(samples),):
        raise ValueError("medoid samples and validity mask have invalid shapes")
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise ValueError("medoid selection requires at least one valid sample")
    selected = samples[indices]
    if not np.all(np.isfinite(selected)):
        raise ValueError("valid medoid samples must be finite")
    distances = np.linalg.norm(selected[:, None] - selected[None, :], axis=-1)
    local_index = int(np.argmin(distances.mean(axis=1)))
    return int(indices[local_index])
