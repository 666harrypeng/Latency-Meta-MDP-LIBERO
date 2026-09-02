"""Lossless PNG and episode-aligned Parquet serialization for formal source data."""

from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
)
from latency_meta_mdp.expert_realization.source_corpus.config import SourceCorpusConfig
from latency_meta_mdp.expert_realization.source_corpus.contracts import (
    FormalSourceSynchronizedEpisode,
)
from latency_meta_mdp.expert_realization.source_corpus.schema import SOURCE_FRAME_SCHEMA

_SHARD_NAME = re.compile(r"^shard-[0-9]{5}[.]parquet$")


@dataclass(frozen=True)
class EpisodeLocation:
    episode_id: str
    row_group_index: int
    row_offset: int
    row_count: int


@dataclass(frozen=True)
class PublishedShard:
    relative_name: str
    byte_count: int
    row_count: int
    row_group_count: int
    sha256: str


def encode_png(rgb: np.ndarray, *, compress_level: int) -> bytes:
    source = np.asarray(rgb)
    if source.dtype != np.uint8 or source.shape != (256, 256, 3):
        raise ValueError("source RGB must be uint8[256,256,3]")
    if type(compress_level) is not int or not 0 <= compress_level <= 9:
        raise ValueError("compress_level must be an integer in [0, 9]")
    output = io.BytesIO()
    Image.fromarray(source, mode="RGB").save(
        output,
        format="PNG",
        compress_level=compress_level,
        optimize=False,
    )
    return output.getvalue()


def decode_png(payload: bytes) -> np.ndarray:
    if type(payload) is not bytes:
        raise TypeError("PNG payload must be bytes")
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
            raise ValueError("source PNG must decode to RGB 256x256")
        result = np.asarray(image, dtype=np.uint8).copy()
    if result.shape != (256, 256, 3):
        raise ValueError("source PNG has invalid decoded shape")
    result.setflags(write=False)
    return result


def _list(value: np.ndarray) -> list[Any]:
    return np.asarray(value).reshape(-1).tolist()


def _optional_list(value: np.ndarray | None) -> list[Any] | None:
    return None if value is None else _list(value)


def _image_row(
    *, episode_id: str, camera: str, tick: int, rgb: np.ndarray, config: SourceCorpusConfig
) -> dict[str, Any]:
    return {
        "bytes": encode_png(rgb, compress_level=config.png_compress_level),
        "path": f"{episode_id}/{camera}/frame_{tick:06d}.png",
    }


def episode_to_frame_table(
    episode: FormalSourceSynchronizedEpisode,
    *,
    config: SourceCorpusConfig,
) -> pa.Table:
    if not isinstance(episode, FormalSourceSynchronizedEpisode):
        raise TypeError("episode must be FormalSourceSynchronizedEpisode")
    if not isinstance(config, SourceCorpusConfig):
        raise TypeError("config must be SourceCorpusConfig")
    if (episode.metadata.camera_height, episode.metadata.camera_width) != (256, 256):
        raise ValueError("formal source cameras must be 256x256")
    rows = []
    transition_count = len(episode.transitions)
    for index, boundary in enumerate(episode.boundaries):
        deployment = boundary.deployment
        qualification = boundary.qualification
        transition = episode.transitions[index] if index < transition_count else None
        audit = None if transition is None else transition.expert_audit
        rows.append(
            {
                "episode_id": episode.metadata.episode_id,
                "formal_tick": boundary.formal_tick_index,
                "physics_step": boundary.physics_step_index,
                "time_us": boundary.time_us,
                "agentview_rgb": _image_row(
                    episode_id=episode.metadata.episode_id,
                    camera="agentview",
                    tick=index,
                    rgb=deployment.agentview_rgb,
                    config=config,
                ),
                "wrist_rgb": _image_row(
                    episode_id=episode.metadata.episode_id,
                    camera="wrist",
                    tick=index,
                    rgb=deployment.robot0_eye_in_hand_rgb,
                    config=config,
                ),
                "robot_qpos": _list(deployment.robot_qpos),
                "robot_qvel": _list(deployment.robot_qvel),
                "gripper_qpos": _list(deployment.gripper_qpos),
                "gripper_qvel": _list(deployment.gripper_qvel),
                "eef_position_world": _list(deployment.eef_position_world),
                "eef_orientation_matrix_world": _list(
                    deployment.eef_orientation_matrix_world
                ),
                "outcome_status": boundary.outcome_status,
                "object_pose": _list(qualification.object_pose),
                "object_velocity": _list(qualification.object_velocity),
                "commanded_motion_position": _list(
                    qualification.commanded_motion_position
                ),
                "commanded_motion_velocity": _list(
                    qualification.commanded_motion_velocity
                ),
                "commanded_motion_acceleration": _list(
                    qualification.commanded_motion_acceleration
                ),
                "commanded_motion_segment_index": (
                    qualification.commanded_motion_segment_index
                ),
                "left_pad_contact": qualification.left_pad_contact,
                "right_pad_contact": qualification.right_pad_contact,
                "handoff_state": qualification.handoff_state,
                "relative_geometry": _list(qualification.relative_geometry),
                "actuator_ctrl": _list(qualification.actuator_ctrl),
                "applied_reference": _optional_list(qualification.applied_reference),
                "applied_reference_source_tick": (
                    qualification.applied_reference_source_tick
                ),
                "nullspace_joint_position_error": _optional_list(
                    qualification.nullspace_joint_position_error
                ),
                "eef_position_error": _optional_list(qualification.eef_position_error),
                "eef_orientation_error_rotvec": _optional_list(
                    qualification.eef_orientation_error_rotvec
                ),
                "expert_action": None if transition is None else _list(transition.expert_action),
                "action_mask": None if transition is None else _list(transition.action_mask),
                "phase_id": None if audit is None else audit.phase_id,
                "reference_kind": None if audit is None else audit.reference_kind,
                "selected_reference_index": (
                    None if audit is None else audit.selected_reference_index
                ),
                "target_eef_position_world": (
                    None if audit is None else _list(audit.target_eef_position_world)
                ),
                "target_eef_orientation_matrix_world": (
                    None if audit is None else _list(audit.target_eef_orientation_matrix_world)
                ),
                "estimated_object_velocity_world": (
                    None if audit is None else _list(audit.estimated_object_velocity_world)
                ),
            }
        )
    return pa.Table.from_pylist(rows, schema=SOURCE_FRAME_SCHEMA)


class SourceParquetShardWriter:
    """Write multiple complete episodes as separate row groups in one immutable shard."""

    def __init__(self, *, target: Path, level: int, config: SourceCorpusConfig) -> None:
        target = Path(target)
        if type(level) is not int or level not in (1, 2, 3):
            raise ValueError("level must be one of 1, 2, or 3")
        if not isinstance(config, SourceCorpusConfig):
            raise TypeError("config must be SourceCorpusConfig")
        if _SHARD_NAME.fullmatch(target.name) is None:
            raise ValueError("target must use shard-NNNNN.parquet naming")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        self.target = target
        self.level = level
        self.config = config
        self._building = target.with_name(f".{target.name}.building-{uuid.uuid4().hex}")
        self._writer = pq.ParquetWriter(
            self._building,
            SOURCE_FRAME_SCHEMA,
            compression=config.parquet_compression,
            compression_level=config.parquet_compression_level,
            use_dictionary=True,
        )
        self._locations: list[EpisodeLocation] = []
        self._row_count = 0
        self._closed = False

    @property
    def current_byte_count(self) -> int:
        if self._closed:
            return self.target.stat().st_size
        return self._building.stat().st_size

    def add_episode(self, episode: FormalSourceSynchronizedEpisode) -> EpisodeLocation:
        if self._closed:
            raise RuntimeError("source shard writer is closed")
        if episode.metadata.task_instance_id.level != self.level:
            raise ValueError("episode level does not match source shard")
        table = episode_to_frame_table(episode, config=self.config)
        location = EpisodeLocation(
            episode_id=episode.metadata.episode_id,
            row_group_index=len(self._locations),
            row_offset=self._row_count,
            row_count=table.num_rows,
        )
        self._writer.write_table(table, row_group_size=table.num_rows)
        self._locations.append(location)
        self._row_count += table.num_rows
        return location

    def close(self) -> PublishedShard:
        if self._closed:
            raise RuntimeError("source shard writer is closed")
        if not self._locations:
            raise RuntimeError("source shard writer cannot publish an empty shard")
        self._writer.close()
        self._closed = True
        with self._building.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(self._building.parent)
        _rename_noreplace(self._building, self.target)
        _fsync_directory(self.target.parent)
        return PublishedShard(
            relative_name=self.target.name,
            byte_count=self.target.stat().st_size,
            row_count=self._row_count,
            row_group_count=len(self._locations),
            sha256=_hash_file(self.target),
        )
