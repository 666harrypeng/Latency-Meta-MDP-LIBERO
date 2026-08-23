"""Atomic first-tranche expert collection for the Panda-ball corpus."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.bulk_plan import BulkCollectionPlan, load_bulk_collection_plan
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, run_expert_episode_attempt
from latency_meta_mdp.outcomes import OutcomeStatus, TerminalReason
from latency_meta_mdp.recording import PhysicalEventKind
from latency_meta_mdp.review_video import write_review_video

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BulkAttemptSpec:
    level: int
    seed: int

    def __post_init__(self) -> None:
        if self.level not in (1, 2, 3):
            raise ValueError("bulk attempt level must be L1, L2, or L3")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("bulk attempt seed must be a non-negative integer")


@dataclass(frozen=True)
class BulkAttemptResult:
    level: int
    seed: int
    terminal_status: OutcomeStatus
    terminal_reason: TerminalReason
    handoff_time_us: int | None
    terminal_time_us: int
    episode_manifest: str

    def __post_init__(self) -> None:
        BulkAttemptSpec(level=self.level, seed=self.seed)
        if not isinstance(self.terminal_status, OutcomeStatus):
            raise TypeError("terminal_status must be an OutcomeStatus")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise TypeError("terminal_reason must be a TerminalReason")
        if (
            isinstance(self.terminal_time_us, bool)
            or not isinstance(self.terminal_time_us, int)
            or self.terminal_time_us <= 0
            or self.terminal_time_us % 20_000
        ):
            raise ValueError("terminal_time_us must be a positive formal-grid time")
        if self.handoff_time_us is not None and (
            isinstance(self.handoff_time_us, bool)
            or not isinstance(self.handoff_time_us, int)
            or self.handoff_time_us < 0
            or self.handoff_time_us % 2_000
        ):
            raise ValueError("handoff_time_us must lie on the physics grid")
        if self.terminal_status is OutcomeStatus.SUCCESS:
            if self.terminal_reason is not TerminalReason.LIFT_SUCCEEDED:
                raise ValueError("successful bulk attempt must use lift_succeeded")
            if self.handoff_time_us is None:
                raise ValueError("successful bulk attempt must contain a handoff")
        elif self.terminal_reason is TerminalReason.LIFT_SUCCEEDED:
            raise ValueError("failed bulk attempt cannot use lift_succeeded")
        path = PurePosixPath(self.episode_manifest)
        if path.is_absolute() or ".." in path.parts or path.name != "manifest.json":
            raise ValueError("episode_manifest must be a safe relative manifest path")

    @property
    def succeeded(self) -> bool:
        return self.terminal_status is OutcomeStatus.SUCCESS

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_manifest": self.episode_manifest,
            "handoff_time_us": self.handoff_time_us,
            "level": self.level,
            "seed": self.seed,
            "succeeded": self.succeeded,
            "terminal_reason": self.terminal_reason.value,
            "terminal_status": self.terminal_status.value,
            "terminal_time_us": self.terminal_time_us,
        }


def select_first_tranche(plan: BulkCollectionPlan) -> tuple[BulkAttemptSpec, ...]:
    seeds = plan.train.seeds[: plan.first_tranche_count]
    return tuple(BulkAttemptSpec(level=level, seed=seed) for level in plan.levels for seed in seeds)


def evaluate_first_tranche_gate(
    *,
    results: tuple[BulkAttemptResult, ...],
    minimum_success_rate: float,
) -> dict[str, Any]:
    if not 0 < minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must lie in (0, 1]")
    identities = [(result.level, result.seed) for result in results]
    if not results or len(set(identities)) != len(identities):
        raise ValueError("bulk results must be non-empty with unique level/seed identities")
    levels: dict[str, dict[str, Any]] = {}
    for level in (1, 2, 3):
        level_results = [result for result in results if result.level == level]
        if not level_results:
            raise ValueError("bulk results must contain every dynamic level")
        attempt_count = len(level_results)
        success_count = sum(result.succeeded for result in level_results)
        required = math.ceil(minimum_success_rate * attempt_count)
        levels[str(level)] = {
            "attempt_count": attempt_count,
            "failure_count": attempt_count - success_count,
            "passed": success_count >= required,
            "required_success_count": required,
            "success_count": success_count,
            "success_rate": success_count / attempt_count,
        }
    return {
        "levels": levels,
        "minimum_success_rate": minimum_success_rate,
        "passed": all(row["passed"] for row in levels.values()),
    }


def select_review_attempts(
    *,
    results: tuple[BulkAttemptResult, ...],
    count_per_level: int,
) -> tuple[BulkAttemptResult, ...]:
    if (
        isinstance(count_per_level, bool)
        or not isinstance(count_per_level, int)
        or count_per_level <= 0
    ):
        raise ValueError("count_per_level must be a positive integer")
    selected: list[BulkAttemptResult] = []
    for level in (1, 2, 3):
        successes = sorted(
            (result for result in results if result.level == level and result.succeeded),
            key=lambda result: result.seed,
        )
        if len(successes) < count_per_level:
            raise ValueError(f"L{level} does not contain enough successes for review videos")
        selected.extend(successes[:count_per_level])
    return tuple(selected)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _handoff_time_us(episode) -> int | None:
    matches = [
        event.time_us
        for event in episode.physical_events
        if event.kind is PhysicalEventKind.HANDOFF
    ]
    if episode.terminal_status is OutcomeStatus.SUCCESS:
        if len(matches) != 1:
            raise ValueError("successful bulk attempt must contain exactly one handoff")
        return matches[0]
    if len(matches) > 1:
        raise ValueError("failed bulk attempt contains duplicate handoff events")
    return matches[0] if matches else None


def collect_first_tranche(
    *,
    project_root: Path,
    plan_path: Path,
    output_root: Path,
    run_id: str,
    on_attempt: Callable[[int, int, BulkAttemptResult], None] | None = None,
) -> Path:
    """Collect and publish one no-retry first tranche atomically."""

    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe path component")
    project = project_root.resolve()
    plan_file = plan_path.resolve()
    plan = load_bulk_collection_plan(plan_file)
    target = output_root.resolve() / run_id
    if target.exists():
        raise FileExistsError(f"bulk first-tranche run already exists: {target}")
    provenance = collect_implementation_provenance(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"bulk first-tranche staging already exists: {staging}")

    results: list[BulkAttemptResult] = []
    artifacts: dict[str, str] = {}
    review_videos: dict[tuple[int, int], str] = {}
    review_success_count = {level: 0 for level in plan.levels}
    try:
        staging.mkdir()
        attempts = select_first_tranche(plan)
        for index, attempt in enumerate(attempts, start=1):
            episode_id = f"l{attempt.level}-seed-{attempt.seed:06d}-attempt-000"
            episode = run_expert_episode_attempt(
                project_root=project,
                spec=ExpertEpisodeSpec(
                    episode_id=episode_id,
                    level=attempt.level,
                    scene_seed=attempt.seed,
                    motion_seed=attempt.seed,
                    expert_seed=attempt.seed,
                    record_profile=plan.record_profile,
                    camera_width=plan.camera_width,
                    camera_height=plan.camera_height,
                ),
            )
            episode_root = (
                staging / "attempts" / f"L{attempt.level}" / f"seed_{attempt.seed:06d}"
            )
            episode_manifest = write_synchronized_episode_artifact(
                episode=episode,
                output_dir=episode_root,
            )
            relative_manifest = episode_manifest.relative_to(staging).as_posix()
            artifacts[relative_manifest] = sha256_file(episode_manifest)
            result = BulkAttemptResult(
                level=attempt.level,
                seed=attempt.seed,
                terminal_status=episode.terminal_status,
                terminal_reason=episode.terminal_reason,
                handoff_time_us=_handoff_time_us(episode),
                terminal_time_us=episode.boundaries[-1].time_us,
                episode_manifest=relative_manifest,
            )
            results.append(result)
            if (
                result.succeeded
                and review_success_count[attempt.level] < plan.review_video_count_per_level
            ):
                review_path = (
                    staging
                    / "review"
                    / f"L{attempt.level}_seed_{attempt.seed:06d}.mp4"
                )
                review_path.parent.mkdir(parents=True, exist_ok=True)
                write_review_video(
                    episode=episode,
                    output_path=review_path,
                    fps=plan.review_video_fps,
                )
                relative_review = review_path.relative_to(staging).as_posix()
                artifacts[relative_review] = sha256_file(review_path)
                review_videos[(attempt.level, attempt.seed)] = relative_review
                review_success_count[attempt.level] += 1
            if on_attempt is not None:
                on_attempt(index, len(attempts), result)

        result_tuple = tuple(results)
        gate = evaluate_first_tranche_gate(
            results=result_tuple,
            minimum_success_rate=plan.minimum_first_attempt_success_rate,
        )
        selected_reviews = select_review_attempts(
            results=result_tuple,
            count_per_level=plan.review_video_count_per_level,
        )
        expected_review_ids = {(result.level, result.seed) for result in selected_reviews}
        if set(review_videos) != expected_review_ids:
            raise RuntimeError("written review videos do not match deterministic selection")
        blockers = [
            f"L{level}_first_attempt_success_below_threshold"
            for level, row in gate["levels"].items()
            if not row["passed"]
        ]
        if provenance.dirty:
            blockers.append("implementation_dirty")
        admitted = [result.episode_manifest for result in result_tuple if result.succeeded]
        manifest = {
            "schema_version": 1,
            "format_id": "panda_ball_bulk_first_tranche_v1",
            "run_id": run_id,
            "collection_id": plan.collection_id,
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "bulk_plan_sha256": sha256_file(plan_file),
            "levels": list(plan.levels),
            "seed_start": plan.train.start,
            "seed_count_per_level": plan.first_tranche_count,
            "record_profile": plan.record_profile.value,
            "camera_width": plan.camera_width,
            "camera_height": plan.camera_height,
            "review_video_fps": plan.review_video_fps,
            "attempt_count": len(result_tuple),
            "success_count": len(admitted),
            "failure_count": len(result_tuple) - len(admitted),
            "attempts": [result.to_mapping() for result in result_tuple],
            "admitted_episode_manifests": admitted,
            "review_videos": [
                review_videos[(result.level, result.seed)] for result in selected_reviews
            ],
            "gate": gate,
            "blockers": blockers,
            "eligible": not blockers,
            "artifacts": dict(sorted(artifacts.items())),
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
