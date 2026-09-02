"""Bounded non-training collection and video rendering for expert-behavior review."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from latency_meta_mdp.expert_realization.config import (
    CANONICAL_FAMILIES,
    FormalCorpusConfig,
)


@dataclass(frozen=True)
class BehaviorReviewRequest:
    schema_version: int
    review_id: str
    corpus_id: str
    logical_task_index_start: int
    task_instance_count: int
    levels: tuple[int, ...]
    realizations_per_task: int
    families: tuple[str, ...]
    family_allocation: str
    planner_candidate_count: int
    maximum_formal_ticks: int
    review_video_fps: int
    bounded_review_only: bool
    training_authorized: bool

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "logical_task_index_start",
            "task_instance_count",
            "realizations_per_task",
            "planner_candidate_count",
            "maximum_formal_ticks",
            "review_video_fps",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.schema_version != 1:
            raise ValueError("review schema_version must equal 1")
        if (
            self.review_id != "panda-ball-smooth-random-review-3x3x3-v1"
            or self.corpus_id != "panda-ball-smooth-random-review-v1"
        ):
            raise ValueError("unsupported behavior-review identity")
        if self.logical_task_index_start < 0:
            raise ValueError("logical_task_index_start must be non-negative")
        if self.task_instance_count != 3 or self.realizations_per_task != 3:
            raise ValueError(
                "this bounded review must be exactly 3 task instances x 3 realizations"
            )
        if self.levels != (1, 2, 3) or self.families != CANONICAL_FAMILIES:
            raise ValueError("review levels or family support are invalid")
        if self.family_allocation != "iid_uniform_seeded":
            raise ValueError("review family allocation must be iid_uniform_seeded")
        if (
            self.planner_candidate_count != 8
            or self.maximum_formal_ticks != 220
            or self.review_video_fps != 25
        ):
            raise ValueError("review planning, clock, or video settings are invalid")
        if self.bounded_review_only is not True or self.training_authorized is not False:
            raise ValueError("behavior review cannot authorize training")

    @property
    def requested_trajectory_count(self) -> int:
        return self.task_instance_count * len(self.levels) * self.realizations_per_task

    def to_formal_config(self) -> FormalCorpusConfig:
        return FormalCorpusConfig(
            schema_version=1,
            corpus_id=self.corpus_id,
            logical_task_index_start=self.logical_task_index_start,
            task_instance_count=self.task_instance_count,
            levels=self.levels,
            realizations_per_task=self.realizations_per_task,
            families=self.families,
            family_allocation=self.family_allocation,
            reserve_task_instance_count=0,
            require_complete_realization_block=True,
            split_unit="master_task_index",
        )


def load_review_request(path: Path) -> BehaviorReviewRequest:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise ValueError("behavior-review config must be a YAML mapping")
    expected = set(BehaviorReviewRequest.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("behavior-review config fields are invalid")
    if type(raw["levels"]) is not list or type(raw["families"]) is not list:
        raise TypeError("behavior-review levels/families must be YAML lists")
    raw = dict(raw)
    raw["levels"] = tuple(raw["levels"])
    raw["families"] = tuple(raw["families"])
    return BehaviorReviewRequest(**raw)


def phase_timeline(decisions: tuple[Any, ...], *, boundary_count: int) -> tuple[str, ...]:
    if type(decisions) is not tuple or type(boundary_count) is not int or boundary_count <= 0:
        raise ValueError("phase timeline inputs are invalid")
    phases = ["shared_prefix"] * boundary_count
    for decision in decisions:
        tick = decision.source_formal_tick
        if type(tick) is not int or not 0 <= tick < boundary_count - 1:
            raise ValueError("decision source tick is outside the boundary timeline")
        phases[tick] = decision.phase.value
    phases[-1] = "terminal"
    return tuple(phases)


def _review_frames(
    *,
    agentview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    phase_by_tick: tuple[str, ...],
    label: str,
) -> np.ndarray:
    agent = np.asarray(agentview_rgb)
    wrist = np.asarray(wrist_rgb)
    if (
        agent.dtype != np.uint8
        or wrist.dtype != np.uint8
        or agent.ndim != 4
        or wrist.shape != agent.shape
        or agent.shape[-1] != 3
        or len(agent) != len(phase_by_tick)
        or not len(agent)
    ):
        raise ValueError("review videos require aligned nonempty uint8 RGB views and phases")
    height, width = agent.shape[1:3]
    output = np.zeros((len(agent), height + 32, width * 2, 3), dtype=np.uint8)
    output[:, 32:, :width] = agent
    output[:, 32:, width:] = wrist
    for tick, phase in enumerate(phase_by_tick):
        image = Image.fromarray(output[tick])
        draw = ImageDraw.Draw(image)
        draw.text((4, 2), f"{label}  tick={tick:03d}  phase={phase}", fill=(255, 255, 255))
        draw.text((4, 17), "agentview", fill=(100, 220, 255))
        draw.text((width + 4, 17), "wrist", fill=(255, 190, 100))
        output[tick] = np.asarray(image)
    return output


def encode_review_video(
    *,
    agentview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    phase_by_tick: tuple[str, ...],
    level: int,
    logical_task_index: int,
    realization_slot: int,
    family: str,
    fps: int,
    target: Path,
) -> None:
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer")
    for name, value in (
        ("level", level),
        ("logical_task_index", logical_task_index),
        ("realization_slot", realization_slot),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if family not in CANONICAL_FAMILIES:
        raise ValueError("unknown strategy family")
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = _review_frames(
        agentview_rgb=agentview_rgb,
        wrist_rgb=wrist_rgb,
        phase_by_tick=phase_by_tick,
        label=(
            f"L{level} task={logical_task_index:03d} "
            f"realization={realization_slot:02d} family={family}"
        ),
    )
    height, width = frames.shape[1:3]
    temporary = target.with_name(f".{target.stem}.building.mp4")
    if temporary.exists():
        raise FileExistsError(temporary)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg review encoding failed: {completed.stderr.decode()[-2000:]}")
    temporary.rename(target)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.building")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(_canonical_json(value), encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.rename(path)


def _intent_summary(intent: Any) -> dict[str, Any]:
    return {
        "family": intent.family.value,
        "strategy": intent.strategy.to_mapping(),
        "capture_position_world": intent.capture_position_world.tolist(),
        "capture_velocity_world": intent.capture_velocity_world.tolist(),
        "approach": {
            "soft_guide_regions_world": intent.approach.soft_guide_regions_world.tolist(),
            "funnel_entry_target_tick": intent.approach.funnel_entry_target_tick,
            "funnel_entry_deadline_tick": intent.approach.funnel_entry_deadline_tick,
            "funnel_entry_position_world": intent.approach.funnel_entry_position_world.tolist(),
            "funnel_entry_tangent_world": intent.approach.funnel_entry_tangent_world.tolist(),
            "time_scaling_profile": intent.approach.time_scaling_profile,
        },
        "grasp_funnel": {
            "close_target_tick": intent.grasp_funnel.close_target_tick,
            "handoff_deadline_tick": intent.grasp_funnel.handoff_deadline_tick,
            "bilateral_contact_acquisition_ticks": (
                intent.grasp_funnel.bilateral_contact_acquisition_ticks
            ),
        },
    }


def collect_behavior_review(
    *,
    project_root: Path,
    config_path: Path,
    target: Path,
    planner_worker_python: Path,
    maximum_task_level_groups: int | None = None,
    on_progress: Any | None = None,
) -> Path:
    """Collect or resume the bounded 3x3x3 review; never publish training data."""
    from latency_meta_mdp.expert_realization.contracts import (
        build_formal_realization_requests,
        build_formal_request_universe,
    )
    from latency_meta_mdp.expert_realization.planner import (
        generate_planner_candidates,
        load_planner_candidates,
        write_planner_candidates,
    )
    from latency_meta_mdp.expert_realization.robot_bridge import build_panda_planning_bridge
    from latency_meta_mdp.expert_realization.rollout import execute_structured_realization
    from latency_meta_mdp.expert_realization.selector import select_task_instance_plan_set
    from latency_meta_mdp.expert_realization.strategy import (
        StructuredStrategyConfig,
        sample_requested_strategy,
    )
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    target = Path(target).resolve()
    request = load_review_request(config_path)
    if maximum_task_level_groups is not None and (
        type(maximum_task_level_groups) is not int or maximum_task_level_groups <= 0
    ):
        raise ValueError("maximum_task_level_groups must be a positive integer or None")
    structured = StructuredStrategyConfig.from_path(
        root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    request_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    universe = build_formal_request_universe(
        request.to_formal_config(),
        corpus_config_sha256=request_sha,
        structured_expert_config_sha256=structured.source_sha256,
    )
    run_record = {
        "review_request": {
            **request.__dict__,
            "levels": list(request.levels),
            "families": list(request.families),
        },
        "review_config_sha256": request_sha,
        "structured_expert_config_sha256": structured.source_sha256,
        "request_universe": universe.to_mapping(),
        "bounded_review_only": True,
        "training_authorized": False,
    }
    target.mkdir(parents=True, exist_ok=True)
    run_path = target / "run_request.json"
    if run_path.exists():
        if json.loads(run_path.read_text(encoding="utf-8")) != run_record:
            raise ValueError("existing review target belongs to a different request")
    else:
        _write_json(run_path, run_record)

    bridge = build_panda_planning_bridge(root)
    rows: list[dict[str, Any]] = []
    group_failures: list[dict[str, Any]] = []
    processed_groups = 0
    stop = False
    for master in universe.primary_tasks:
        for level in request.levels:
            if maximum_task_level_groups is not None and (
                processed_groups >= maximum_task_level_groups
            ):
                stop = True
                break
            task = materialize_task_instance(
                project_root=root,
                level=level,
                task_instance_seed=master.master_task_seed,
            )
            realization_requests = build_formal_realization_requests(
                universe,
                task.task_instance_id,
            )
            group_root = (
                target
                / f"level-{level}"
                / f"task-{master.logical_task_index:03d}-seed-{master.master_task_seed}"
            )
            group_root.mkdir(parents=True, exist_ok=True)
            candidates_by_key = {}
            intents_by_slot = {}
            for realization_request in realization_requests:
                key = realization_request.to_expert_realization_key()
                strategy = sample_requested_strategy(
                    task,
                    realization_request,
                    structured,
                    universe=universe,
                )
                intent = build_trajectory_intent(task, task.expected_anchor, strategy)
                intents_by_slot[realization_request.realization_slot] = intent
                candidate_root = (
                    group_root
                    / f"realization-{realization_request.realization_slot:02d}"
                    / "candidates"
                )
                candidate_root.parent.mkdir(parents=True, exist_ok=True)
                if candidate_root.exists():
                    candidates = load_planner_candidates(
                        candidate_root,
                        structured_expert_config_sha256=structured.source_sha256,
                    )
                else:
                    candidates = generate_planner_candidates(
                        expert_realization_key=key,
                        structured_expert_config_sha256=structured.source_sha256,
                        bridge=bridge,
                        intent=intent,
                        start_qpos=task.expected_anchor.anchor_robot_qpos,
                        worker_python=planner_worker_python,
                    )
                    building = candidate_root.with_name(f".{candidate_root.name}.building")
                    write_planner_candidates(building, candidates)
                    building.rename(candidate_root)
                candidates_by_key[key] = candidates
                if on_progress is not None:
                    on_progress(
                        f"planned L{level} task={master.logical_task_index} "
                        f"realization={realization_request.realization_slot}"
                    )
            try:
                plan_set = select_task_instance_plan_set(
                    task_instance=task,
                    candidates_by_key=candidates_by_key,
                )
            except (ValueError, RuntimeError) as error:
                failure = {
                    "level": level,
                    "logical_task_index": master.logical_task_index,
                    "master_task_seed": master.master_task_seed,
                    "reason": str(error),
                }
                _write_json(group_root / "group_failure.json", failure)
                group_failures.append(failure)
                processed_groups += 1
                continue

            for realization_request in realization_requests:
                slot = realization_request.realization_slot
                realization_root = group_root / f"realization-{slot:02d}"
                summary_path = realization_root / "summary.json"
                video_path = realization_root / "dual_view_review_25fps.mp4"
                trajectory_path = realization_root / "trajectory.npz"
                if summary_path.exists() and video_path.exists() and trajectory_path.exists():
                    row = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    rollout = execute_structured_realization(
                        task_instance=task,
                        intent=intents_by_slot[slot],
                        reference=plan_set.references[slot],
                        maximum_formal_ticks=request.maximum_formal_ticks,
                    )
                    phases = phase_timeline(
                        rollout.decisions,
                        boundary_count=len(rollout.agentview_rgb),
                    )
                    encode_review_video(
                        agentview_rgb=rollout.agentview_rgb,
                        wrist_rgb=rollout.wrist_rgb,
                        phase_by_tick=phases,
                        level=level,
                        logical_task_index=master.logical_task_index,
                        realization_slot=slot,
                        family=realization_request.assigned_family.value,
                        fps=request.review_video_fps,
                        target=video_path,
                    )
                    np.savez_compressed(
                        trajectory_path,
                        actions=rollout.actions,
                        eef_positions_world=rollout.eef_positions_world,
                        object_positions_world=rollout.object_positions_world,
                        phase_by_tick=np.asarray(phases),
                    )
                    row = {
                        "level": level,
                        "logical_task_index": master.logical_task_index,
                        "master_task_seed": master.master_task_seed,
                        "realization_slot": slot,
                        "family": realization_request.assigned_family.value,
                        "terminal_status": rollout.terminal_status,
                        "terminal_reason": rollout.terminal_reason,
                        "terminal_tick": rollout.terminal_tick,
                        "physical_handoff_tick": rollout.physical_handoff_tick,
                        "video": str(video_path.relative_to(target)),
                        "trajectory": str(trajectory_path.relative_to(target)),
                        "intent": _intent_summary(intents_by_slot[slot]),
                    }
                    _write_json(summary_path, row)
                rows.append(row)
                if on_progress is not None:
                    on_progress(
                        f"executed L{level} task={master.logical_task_index} "
                        f"realization={slot} status={row['terminal_status']}"
                    )
            processed_groups += 1
        if stop:
            break

    complete = processed_groups == request.task_instance_count * len(request.levels)
    manifest = {
        "schema_version": 1,
        "format_id": "smooth_expert_behavior_review_v1",
        "review_id": request.review_id,
        "bounded_review_only": True,
        "training_authorized": False,
        "complete": complete,
        "requested_trajectory_count": request.requested_trajectory_count,
        "materialized_video_count": len(rows),
        "successful_trajectory_count": sum(row["terminal_status"] == "success" for row in rows),
        "group_failures": group_failures,
        "trajectories": rows,
    }
    manifest_path = target / ("manifest.json" if complete else "partial_manifest.json")
    if manifest_path.exists():
        manifest_path.unlink()
    _write_json(manifest_path, manifest)
    return manifest_path
