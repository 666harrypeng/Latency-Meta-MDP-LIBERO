"""Pure branch-minus-nominal metrics for the JEPA J4 gate."""

from __future__ import annotations

import numpy as np


def _effect_statistics(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    predicted_norm = np.linalg.norm(predicted, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    selected = valid & (target_norm > 1e-8)
    if not np.any(selected):
        return {
            "valid_count": 0,
            "effect_cosine_mean": None,
            "effect_magnitude_ratio_median": None,
            "effect_error_mean": None,
        }
    cosine = np.sum(predicted * target, axis=-1) / np.maximum(
        predicted_norm * target_norm,
        1e-12,
    )
    ratio = predicted_norm / np.maximum(target_norm, 1e-12)
    error = np.linalg.norm(predicted - target, axis=-1)
    return {
        "valid_count": int(selected.sum()),
        "effect_cosine_mean": float(cosine[selected].mean()),
        "effect_magnitude_ratio_median": float(np.median(ratio[selected])),
        "effect_error_mean": float(error[selected].mean()),
    }


def _effect_cosine_per_anchor(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> list[float | None]:
    predicted_norm = np.linalg.norm(predicted, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    cosine = np.sum(predicted * target, axis=-1) / np.maximum(
        predicted_norm * target_norm,
        1e-12,
    )
    result = []
    for anchor in range(predicted.shape[-2]):
        selected = valid[..., anchor] & (target_norm[..., anchor] > 1e-8)
        result.append(None if not np.any(selected) else float(cosine[..., anchor][selected].mean()))
    return result


def summarize_j4_effects(
    *,
    predicted_proprio: np.ndarray,
    predicted_object: np.ndarray,
    gt_proprio: np.ndarray,
    gt_object: np.ndarray,
    handoff_physical: np.ndarray,
    context_categories: tuple[str, ...],
    branch_names: tuple[str, ...],
    native_delay_ticks: tuple[int, ...],
) -> dict[str, object]:
    predicted_robot = np.asarray(predicted_proprio, dtype=np.float64)
    target_robot = np.asarray(gt_proprio, dtype=np.float64)
    predicted_world = np.asarray(predicted_object, dtype=np.float64)
    target_world = np.asarray(gt_object, dtype=np.float64)
    handoff = np.asarray(handoff_physical, dtype=np.bool_)
    if (
        predicted_robot.shape != target_robot.shape
        or predicted_world.shape != target_world.shape
        or predicted_robot.ndim != 4
        or predicted_robot.shape[-1] != 16
        or predicted_world.shape != (*predicted_robot.shape[:-1], 6)
        or handoff.shape != predicted_robot.shape[:-1]
        or predicted_robot.shape[0] != len(context_categories)
        or predicted_robot.shape[1] != len(branch_names)
        or predicted_robot.shape[2] != len(native_delay_ticks)
        or branch_names != ("nominal", "hold", "scale_0.5", "prefix4_then_hold")
        or not all(
            np.all(np.isfinite(value))
            for value in (predicted_robot, target_robot, predicted_world, target_world)
        )
    ):
        raise ValueError("J4 effect arrays or identities are incompatible")
    predicted_robot_effect = predicted_robot[:, 1:, :, :7] - predicted_robot[:, :1, :, :7]
    target_robot_effect = target_robot[:, 1:, :, :7] - target_robot[:, :1, :, :7]
    predicted_object_effect = predicted_world[:, 1:, :, :3] - predicted_world[:, :1, :, :3]
    target_object_effect = target_world[:, 1:, :, :3] - target_world[:, :1, :, :3]
    valid = np.ones(predicted_robot_effect.shape[:-1], dtype=np.bool_)
    branches = {}
    for branch_index, branch_name in enumerate(branch_names[1:]):
        robot = _effect_statistics(
            predicted=predicted_robot_effect[:, branch_index],
            target=target_robot_effect[:, branch_index],
            valid=valid[:, branch_index],
        )
        branches[branch_name] = {
            "robot_qpos_effect_valid_count": robot["valid_count"],
            "robot_qpos_effect_cosine_mean": robot["effect_cosine_mean"],
            "robot_qpos_effect_magnitude_ratio_median": robot["effect_magnitude_ratio_median"],
            "robot_qpos_effect_error_mean_rad": robot["effect_error_mean"],
            "robot_qpos_effect_cosine_per_anchor": _effect_cosine_per_anchor(
                predicted=predicted_robot_effect[:, branch_index],
                target=target_robot_effect[:, branch_index],
                valid=valid[:, branch_index],
            ),
        }

    pre_handoff = (~handoff[:, :1]) & (~handoff[:, 1:])
    predicted_spurious = np.linalg.norm(predicted_object_effect, axis=-1)
    post_handoff = handoff[:, :1] & handoff[:, 1:]
    post = _effect_statistics(
        predicted=predicted_object_effect,
        target=target_object_effect,
        valid=post_handoff,
    )
    prefix_index = branch_names.index("prefix4_then_hold") - 1
    target_object_norm = np.linalg.norm(target_object_effect, axis=-1)
    pre_count = int(pre_handoff.sum())
    pre_spurious_per_anchor = []
    for anchor in range(predicted_spurious.shape[-1]):
        selected = pre_handoff[..., anchor]
        pre_spurious_per_anchor.append(
            None
            if not np.any(selected)
            else float(predicted_spurious[..., anchor][selected].mean())
        )
    return {
        "context_count": predicted_robot.shape[0],
        "native_delay_ticks": list(native_delay_ticks),
        "branches": branches,
        "prefix4_first_anchor_gt_robot_effect_max": float(
            np.linalg.norm(target_robot_effect[:, prefix_index, 0], axis=-1).max()
        ),
        "prefix4_first_anchor_predicted_robot_effect_max": float(
            np.linalg.norm(predicted_robot_effect[:, prefix_index, 0], axis=-1).max()
        ),
        "pre_handoff_value_count": pre_count,
        "pre_handoff_gt_object_effect_max_m": (
            None if pre_count == 0 else float(target_object_norm[pre_handoff].max())
        ),
        "pre_handoff_predicted_object_spurious_effect_mean_m": (
            None if pre_count == 0 else float(predicted_spurious[pre_handoff].mean())
        ),
        "pre_handoff_predicted_object_spurious_effect_max_m": (
            None if pre_count == 0 else float(predicted_spurious[pre_handoff].max())
        ),
        "pre_handoff_predicted_object_spurious_effect_mean_per_anchor_m": (pre_spurious_per_anchor),
        "post_handoff_object_effect_valid_count": post["valid_count"],
        "post_handoff_object_effect_cosine_mean": post["effect_cosine_mean"],
        "post_handoff_object_effect_magnitude_ratio_median": post["effect_magnitude_ratio_median"],
        "post_handoff_object_effect_error_mean_m": post["effect_error_mean"],
        "post_handoff_object_effect_cosine_per_anchor": _effect_cosine_per_anchor(
            predicted=predicted_object_effect,
            target=target_object_effect,
            valid=post_handoff,
        ),
    }
