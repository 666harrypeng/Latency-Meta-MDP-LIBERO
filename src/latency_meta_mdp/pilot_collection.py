"""Small synchronized expert pilot campaigns for recorder qualification."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.recording import PhysicalEventKind, RecordProfile, SynchronizedEpisode
from latency_meta_mdp.review_video import write_review_video

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PilotRunSpec:
    levels: tuple[int, ...]
    seeds: tuple[int, ...]
    record_profile: RecordProfile
    camera_width: int
    camera_height: int
    review_video_fps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("pilot levels must be non-empty and unique")
        if any(level not in (1, 2, 3) for level in self.levels):
            raise ValueError("pilot levels must be drawn from L1, L2, and L3")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("pilot seeds must be non-empty and unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("pilot seeds must be non-negative integers")
        if self.record_profile is not RecordProfile.PILOT_DEBUG:
            raise ValueError("pilot runs must retain the pilot_debug record profile")
        for name in ("camera_width", "camera_height", "review_video_fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.camera_width % 2 or self.camera_height % 2:
            raise ValueError("pilot camera dimensions must be even for review video encoding")


def load_pilot_run_spec(path: Path) -> PilotRunSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "collection_id",
        "levels",
        "seeds",
        "record_profile",
        "camera_width",
        "camera_height",
        "review_video_fps",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("pilot collection config fields are invalid")
    if raw["schema_version"] != 1 or raw["collection_id"] != "panda_ball_pilot_v1":
        raise ValueError("unsupported pilot collection schema or identifier")
    if not isinstance(raw["levels"], list) or not isinstance(raw["seeds"], list):
        raise TypeError("pilot levels and seeds must be lists")
    return PilotRunSpec(
        levels=tuple(raw["levels"]),
        seeds=tuple(raw["seeds"]),
        record_profile=RecordProfile(raw["record_profile"]),
        camera_width=raw["camera_width"],
        camera_height=raw["camera_height"],
        review_video_fps=raw["review_video_fps"],
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _event_time(episode: SynchronizedEpisode, kind: PhysicalEventKind) -> int:
    matches = [event.time_us for event in episode.physical_events if event.kind is kind]
    if len(matches) != 1:
        raise ValueError(f"episode does not contain exactly one {kind.value} event")
    return matches[0]


def collect_expert_pilot_run(
    *,
    project_root: Path,
    output_root: Path,
    run_id: str,
    spec: PilotRunSpec,
) -> Path:
    """Collect one atomic multi-level pilot run with lossless raw data and review videos."""

    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe path component")
    provenance = collect_implementation_provenance(project_root)
    root = output_root.resolve()
    target = root / run_id
    if target.exists():
        raise FileExistsError(f"pilot run already exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{run_id}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"pilot staging output already exists: {staging}")
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    try:
        staging.mkdir()
        for level in spec.levels:
            for seed in spec.seeds:
                episode_id = f"l{level}-seed-{seed:06d}-attempt-000"
                episode = collect_expert_episode(
                    project_root=project_root,
                    spec=ExpertEpisodeSpec(
                        episode_id=episode_id,
                        level=level,
                        scene_seed=seed,
                        motion_seed=seed,
                        expert_seed=seed,
                        record_profile=spec.record_profile,
                        camera_width=spec.camera_width,
                        camera_height=spec.camera_height,
                    ),
                )
                episode_root = staging / "episodes" / f"L{level}" / f"seed_{seed:06d}"
                episode_manifest = write_synchronized_episode_artifact(
                    episode=episode,
                    output_dir=episode_root,
                )
                review_path = staging / "review" / f"L{level}_seed_{seed:06d}.mp4"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                write_review_video(
                    episode=episode,
                    output_path=review_path,
                    fps=spec.review_video_fps,
                )
                relative_episode_manifest = episode_manifest.relative_to(staging).as_posix()
                relative_review = review_path.relative_to(staging).as_posix()
                artifacts[relative_episode_manifest] = sha256_file(episode_manifest)
                artifacts[relative_review] = sha256_file(review_path)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "level": level,
                        "seed": seed,
                        "boundary_count": len(episode.boundaries),
                        "transition_count": len(episode.transitions),
                        "handoff_time_us": _event_time(
                            episode, PhysicalEventKind.HANDOFF
                        ),
                        "terminal_time_us": episode.boundaries[-1].time_us,
                        "episode_manifest": relative_episode_manifest,
                        "review_video": relative_review,
                    }
                )
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "expert_pilot_run_v1",
                "run_id": run_id,
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "record_profile": spec.record_profile.value,
                "camera_width": spec.camera_width,
                "camera_height": spec.camera_height,
                "review_video_fps": spec.review_video_fps,
                "episode_count": len(rows),
                "episodes": rows,
                "artifacts": dict(sorted(artifacts.items())),
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
