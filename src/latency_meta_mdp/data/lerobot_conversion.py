"""LeRobot v2.1 writer for the synchronized Panda policy contract."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.data.policy import (
    ACTION_CONTRACT_ID,
    ACTION_DIM,
    ACTION_HORIZON,
    FORMAL_TICK_US,
    LAUNCH_TRIGGER_HORIZON,
    POLICY_STATE_DIM,
    TEMPORAL_CONTRACT_ID,
    PolicyEpisode,
)
from latency_meta_mdp.io.artifacts import sha256_file

FPS = 50
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
STRUCTURED_STATE_CONTRACT = "joint_qpos_qvel_gripper_width_velocity"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_episode(episode: PolicyEpisode) -> tuple[int, int, int]:
    frame_count = len(episode.source_formal_tick)
    structured = episode.state_contract == STRUCTURED_STATE_CONTRACT
    if not structured and episode.state_contract != "eef_xyz_rotvec_gripper_qpos_v1":
        raise ValueError("unsupported policy state contract")
    if structured and (
        type(episode.logical_master_task_index) is not int or episode.logical_master_task_index < 0
    ):
        raise ValueError("structured policy episode requires a master task identity")
    state_dim = 16 if structured else POLICY_STATE_DIM
    expected_valid_count = frame_count if structured else frame_count - ACTION_HORIZON + 1
    if episode.action_horizon != ACTION_HORIZON or expected_valid_count <= 0:
        raise ValueError("policy episode cannot provide the canonical H50 target")
    if not (
        episode.source_formal_tick.shape == (frame_count,)
        and episode.source_time_us.shape == (frame_count,)
        and np.array_equal(episode.source_formal_tick, np.arange(frame_count))
        and np.array_equal(episode.source_time_us, np.arange(frame_count) * FORMAL_TICK_US)
        and episode.state.shape == (frame_count, state_dim)
        and episode.actions.shape == (frame_count, ACTION_DIM)
        and episode.agentview_rgb.ndim == 4
        and episode.wrist_rgb.ndim == 4
        and episode.agentview_rgb.shape[0] == frame_count
        and episode.wrist_rgb.shape[0] == frame_count
        and episode.agentview_rgb.shape[-1] == 3
        and episode.wrist_rgb.shape[-1] == 3
        and episode.agentview_rgb.dtype == np.uint8
        and episode.wrist_rgb.dtype == np.uint8
        and np.all(np.isfinite(episode.state))
        and np.all(np.isfinite(episode.actions))
        and np.array_equal(
            episode.valid_action_chunk_sources,
            np.arange(expected_valid_count),
        )
    ):
        raise ValueError("policy episode does not satisfy its synchronized state/action contract")
    return frame_count, episode.agentview_rgb.shape[1], episode.agentview_rgb.shape[2]


def _resolve_dataset_factory(dataset_factory: Callable[..., Any] | None) -> Callable[..., Any]:
    if dataset_factory is not None:
        return dataset_factory
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required only for dataset writing; run this command in the pinned "
            "Python 3.11 OpenPI environment"
        ) from exc
    return LeRobotDataset.create


def _features(*, height: int, width: int, state_dim: int) -> dict[str, dict[str, Any]]:
    return {
        "image": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
    }


def _artifact_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "metamdp_dataset.json"
    }


def write_lerobot_policy_dataset(
    *,
    episodes: Iterable[PolicyEpisode],
    output_dir: Path,
    repo_id: str,
    dataset_factory: Callable[..., Any] | None = None,
) -> Path:
    """Write a level-specific, no-overwrite LeRobot dataset atomically."""

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"LeRobot output already exists: {target}")
    iterator = iter(episodes)
    try:
        episode = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one policy episode is required") from exc
    _, height, width = _validate_episode(episode)
    expected_level = episode.level
    expected_task_id = episode.task_id
    expected_instruction = episode.instruction
    state_contract = episode.state_contract
    structured = state_contract == STRUCTURED_STATE_CONTRACT
    state_dim = episode.state.shape[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"LeRobot staging output already exists: {staging}")

    create_dataset = _resolve_dataset_factory(dataset_factory)
    try:
        dataset = create_dataset(
            repo_id=repo_id.strip(),
            root=staging,
            robot_type="panda",
            fps=FPS,
            features=_features(height=height, width=width, state_dim=state_dim),
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        episode_rows = []
        episode_ids: set[str] = set()
        while True:
            frame_count, episode_height, episode_width = _validate_episode(episode)
            if (
                episode.level != expected_level
                or episode.task_id != expected_task_id
                or episode.instruction != expected_instruction
                or episode.state_contract != state_contract
            ):
                raise ValueError("one LeRobot dataset must contain exactly one task level")
            if (episode_height, episode_width) != (height, width) or episode.wrist_rgb.shape[
                1:3
            ] != (height, width):
                raise ValueError("all policy cameras must share one fixed image shape")
            if episode.episode_id in episode_ids:
                raise ValueError("policy episode ids must be unique")
            episode_ids.add(episode.episode_id)
            for index in range(len(episode.source_formal_tick)):
                dataset.add_frame(
                    {
                        "image": episode.agentview_rgb[index],
                        "wrist_image": episode.wrist_rgb[index],
                        "state": episode.state[index],
                        "actions": episode.actions[index],
                        "task": episode.instruction,
                    }
                )
            dataset.save_episode()
            episode_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "level": episode.level,
                    "frame_count": frame_count,
                    "valid_action_chunk_source_count": len(episode.valid_action_chunk_sources),
                    "first_source_time_us": int(episode.source_time_us[0]),
                    "last_source_time_us": int(episode.source_time_us[-1]),
                    **(
                        {"logical_master_task_index": episode.logical_master_task_index}
                        if structured
                        else {}
                    ),
                }
            )
            try:
                episode = next(iterator)
            except StopIteration:
                break
        _write_json(
            staging / "metamdp_dataset.json",
            {
                "schema_version": 1,
                "format_id": "metamdp_lerobot_v21",
                "repo_id": repo_id.strip(),
                "source_format_id": (
                    "structured_expert_source_corpus_v3"
                    if structured
                    else "synchronized_episode_npz_v3"
                ),
                "level": expected_level,
                "task_id": expected_task_id,
                "instruction": expected_instruction,
                "fps": FPS,
                "formal_tick_us": FORMAL_TICK_US,
                "temporal_contract_id": TEMPORAL_CONTRACT_ID,
                "action_horizon": ACTION_HORIZON,
                "launch_trigger_horizon": LAUNCH_TRIGGER_HORIZON,
                "drop_n_last_frames": 0 if structured else ACTION_HORIZON - 1,
                "action_target_contract": (
                    "masked_h50_real_actions_v1" if structured else "complete_h50_v1"
                ),
                "state_contract": state_contract,
                "state_dim": state_dim,
                "action_contract": ACTION_CONTRACT_ID,
                "action_dim": ACTION_DIM,
                "image_storage": "embedded_png_in_parquet",
                "video_keys": [],
                "openpi_revision": OPENPI_REVISION,
                "lerobot_revision": LEROBOT_REVISION,
                "episode_count": len(episode_rows),
                "frame_count": sum(row["frame_count"] for row in episode_rows),
                "valid_action_chunk_source_count": sum(
                    row["valid_action_chunk_source_count"] for row in episode_rows
                ),
                "episodes": episode_rows,
                "artifacts": _artifact_inventory(staging),
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "metamdp_dataset.json"
