"""Simulator-free review videos derived from a verified source corpus."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.data.collection.artifacts import (
    _cleanup_owned_staging,
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.data.collection.review_collection import _review_frames
from latency_meta_mdp.data.source.loader import (
    VerifiedSourceCorpus,
    load_verified_source_corpus,
)
from latency_meta_mdp.data.source.parquet import decode_png

_SPEEDS = frozenset({1, 2, 3, 5, 10})
_FPS = 50


def _validate_selection(
    corpus: VerifiedSourceCorpus,
    *,
    level: int | None,
    episode_range: tuple[int, int] | None,
) -> tuple[int, ...]:
    if not isinstance(corpus, VerifiedSourceCorpus):
        raise TypeError("corpus must be a VerifiedSourceCorpus")
    available = tuple(sorted(int(value) for value in corpus.manifest.episodes_by_level))
    if level is not None and (type(level) is not int or level not in available):
        raise ValueError("level is outside the source corpus")
    if episode_range is not None:
        if level is None:
            raise ValueError("episode range requires one level")
        if (
            type(episode_range) is not tuple
            or len(episode_range) != 2
            or any(type(value) is not int for value in episode_range)
            or episode_range[0] < 0
            or episode_range[0] >= episode_range[1]
        ):
            raise ValueError("episode range must be an ordered non-negative pair")
    return available if level is None else (level,)


def select_review_episodes(
    corpus: VerifiedSourceCorpus,
    *,
    level: int | None,
    episode_range: tuple[int, int] | None,
) -> dict[int, tuple[str, ...]]:
    levels = _validate_selection(corpus, level=level, episode_range=episode_range)
    result = {}
    for selected_level in levels:
        rows = sorted(
            (row for row in corpus._episodes.values() if row["level"] == selected_level),
            key=lambda row: (
                row["logical_master_task_index"],
                row.get("accepted_slot", row.get("realization_index")),
            ),
        )
        start, stop = (0, len(rows)) if episode_range is None else episode_range
        if stop > len(rows):
            raise ValueError("episode range is outside the selected level")
        result[selected_level] = tuple(row["episode_id"] for row in rows[start:stop])
    return result


def review_frame_indices(*, frame_count: int, speed: int) -> tuple[int, ...]:
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if type(speed) is not int or speed not in _SPEEDS:
        raise ValueError("speed must be one of 1, 2, 3, 5, or 10")
    values = list(range(0, frame_count, speed))
    if values[-1] != frame_count - 1:
        values.append(frame_count - 1)
    return tuple(values)


def review_plan_summary(
    corpus: VerifiedSourceCorpus,
    *,
    level: int | None,
    episode_range: tuple[int, int] | None,
    speed: int,
) -> dict[str, Any]:
    review_frame_indices(frame_count=1, speed=speed)
    selected = select_review_episodes(corpus, level=level, episode_range=episode_range)
    levels = {}
    for selected_level, episode_ids in selected.items():
        total = int(corpus.manifest.episodes_by_level[str(selected_level)])
        start, stop = (0, total) if episode_range is None else episode_range
        levels[str(selected_level)] = {
            "start": start,
            "stop": stop,
            "episode_count": len(episode_ids),
            "episode_ids": list(episode_ids),
        }
    return {
        "source_root": str(corpus.root),
        "source_manifest_sha256": hashlib.sha256(
            (corpus.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_fps": _FPS,
        "output_fps": _FPS,
        "speed": speed,
        "levels": levels,
    }


def _encode_level(
    *,
    corpus: VerifiedSourceCorpus,
    episode_ids: tuple[str, ...],
    canonical_start: int,
    speed: int,
    target: Path,
) -> tuple[int, int, list[dict[str, Any]]]:
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "512x288",
            "-r",
            str(_FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not create a video input pipe")
    source_frames = 0
    output_frames = 0
    rows = []
    try:
        for offset, episode_id in enumerate(episode_ids):
            episode = corpus.read_episode(episode_id)
            frames = episode.frames
            agent = np.stack(
                [decode_png(value["bytes"]) for value in frames["agentview_rgb"].to_pylist()]
            )
            wrist = np.stack(
                [decode_png(value["bytes"]) for value in frames["wrist_rgb"].to_pylist()]
            )
            phases = tuple(
                "terminal" if value is None else value for value in frames["phase_id"].to_pylist()
            )
            metadata = episode.metadata
            accepted_slot = metadata.get("accepted_slot", metadata.get("realization_index"))
            rendered = _review_frames(
                agentview_rgb=agent,
                wrist_rgb=wrist,
                phase_by_tick=phases,
                label=(
                    f"L{metadata['level']} episode_index={canonical_start + offset:04d} "
                    f"task={metadata['logical_master_task_index']:06d} "
                    f"realization={accepted_slot:04d} "
                    f"family={metadata['strategy_family']}"
                ),
            )
            indices = review_frame_indices(frame_count=len(rendered), speed=speed)
            process.stdin.write(np.ascontiguousarray(rendered[list(indices)]).tobytes())
            source_frames += len(rendered)
            output_frames += len(indices)
            rows.append(
                {
                    "episode_index": canonical_start + offset,
                    "episode_id": episode_id,
                    "source_frame_count": len(rendered),
                    "output_frame_count": len(indices),
                }
            )
    finally:
        process.stdin.close()
    stderr = b"" if process.stderr is None else process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg review encoding failed: {stderr.decode()[-2000:]}")
    with target.open("rb") as handle:
        os.fsync(handle.fileno())
    return source_frames, output_frames, rows


def render_source_review_videos(
    *,
    source_root: Path,
    output_root: Path,
    level: int | None,
    episode_range: tuple[int, int] | None,
    speed: int,
) -> Path:
    corpus = load_verified_source_corpus(Path(source_root))
    summary = review_plan_summary(
        corpus,
        level=level,
        episode_range=episode_range,
        speed=speed,
    )
    selected = select_review_episodes(corpus, level=level, episode_range=episode_range)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(output_root.parent)
    staging = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    staging_stat = os.lstat(staging)
    videos = []
    try:
        for selected_level, episode_ids in selected.items():
            total = int(corpus.manifest.episodes_by_level[str(selected_level)])
            start, stop = (0, total) if episode_range is None else episode_range
            relative = (
                f"level-{selected_level}_episodes-{start:04d}-{stop - 1:04d}_speed-{speed}x.mp4"
            )
            source_count, output_count, episode_rows = _encode_level(
                corpus=corpus,
                episode_ids=episode_ids,
                canonical_start=start,
                speed=speed,
                target=staging / relative,
            )
            videos.append(
                {
                    "level": selected_level,
                    "start": start,
                    "stop": stop,
                    "speed": speed,
                    "source_frame_count": source_count,
                    "output_frame_count": output_count,
                    "duration_seconds": output_count / _FPS,
                    "path": relative,
                    "sha256": _hash_file(staging / relative),
                    "episodes": episode_rows,
                }
            )
        manifest = {
            "schema_version": 1,
            "format_id": "structured_source_review_v1",
            **{key: value for key, value in summary.items() if key != "levels"},
            "videos": videos,
        }
        _write_file_fsynced(
            staging / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _fsync_directory(staging)
        _rename_noreplace(staging, output_root)
        _fsync_directory(output_root.parent)
    except BaseException:
        _cleanup_owned_staging(
            staging,
            expected_device=staging_stat.st_dev,
            expected_inode=staging_stat.st_ino,
            parent=output_root.parent,
            expected_parent_device=parent_stat.st_dev,
            expected_parent_inode=parent_stat.st_ino,
        )
        raise
    return output_root / "manifest.json"
