"""Transparent reserve-realization extensions for incomplete behavior-review task groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.contracts import (
    FormalRealizationRequest,
    FormalRequestUniverse,
    build_formal_realization_requests,
    derive_realization_seed,
    sample_strategy_parameters,
    sample_uniform_family_for_slot,
)
from latency_meta_mdp.expert_realization.planner import (
    PlannerCandidateStatus,
    generate_planner_candidates,
    load_planner_candidates,
    write_planner_candidates,
)
from latency_meta_mdp.expert_realization.review_collection import (
    encode_review_video,
    intent_summary,
    load_review_request,
    phase_timeline,
    write_review_json,
)
from latency_meta_mdp.expert_realization.robot_bridge import build_panda_planning_bridge
from latency_meta_mdp.expert_realization.rollout import execute_structured_realization
from latency_meta_mdp.expert_realization.selector import (
    discrete_frechet,
    freeze_selected_reference,
)
from latency_meta_mdp.expert_realization.strategy import StructuredStrategyConfig
from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent


def _summary_rows(group_root: Path) -> dict[int, dict[str, Any]]:
    return {
        int(path.parent.name.split("-")[-1]): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(group_root.glob("realization-*/summary.json"))
    }


def _successful_eef_paths(
    target: Path,
    rows: dict[int, dict[str, Any]],
) -> dict[int, np.ndarray]:
    result = {}
    for slot, row in rows.items():
        if row["terminal_status"] != "success":
            continue
        with np.load(target / row["trajectory"], allow_pickle=False) as arrays:
            result[slot] = np.array(arrays["eef_positions_world"], copy=True)
    return result


def fill_review_task_realizations(
    *,
    project_root: Path,
    config_path: Path,
    target: Path,
    level: int,
    logical_task_index: int,
    planner_worker_python: Path,
    maximum_realization_slot: int = 8,
    on_progress: Any | None = None,
) -> Path:
    """Append explicit slots until one task has three qualified trajectories."""
    root = Path(project_root).resolve()
    target = Path(target).resolve()
    request = load_review_request(config_path)
    run = json.loads((target / "run_request.json").read_text(encoding="utf-8"))
    universe = FormalRequestUniverse.from_mapping(run["request_universe"])
    structured = StructuredStrategyConfig.from_path(
        root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    if (
        run["structured_expert_config_sha256"] != structured.source_sha256
        or universe.structured_expert_config_sha256 != structured.source_sha256
    ):
        raise ValueError("review extension config does not match the parent request")
    masters = [
        row
        for row in universe.primary_tasks + universe.reserve_tasks
        if row.logical_task_index == logical_task_index
    ]
    if len(masters) != 1 or level not in request.levels:
        raise ValueError("review extension task identity is outside the parent request")
    master = masters[0]
    task = materialize_task_instance(
        project_root=root,
        level=level,
        task_instance_seed=master.master_task_seed,
    )
    group_matches = list(
        (target / f"level-{level}").glob(
            f"task-{logical_task_index:03d}-seed-{master.master_task_seed}"
        )
    )
    if len(group_matches) != 1:
        raise ValueError("review extension requires one existing task-group directory")
    group_root = group_matches[0]
    rows = _summary_rows(group_root)
    successful_paths = _successful_eef_paths(target, rows)
    if len(successful_paths) >= request.realizations_per_task:
        selected = sorted(successful_paths)[: request.realizations_per_task]
        write_review_json(
            group_root / "group_result.json",
            {"admitted": True, "selected_realization_slots": selected, "trajectory_statuses": []},
        )
        return group_root / "group_result.json"

    base_requests = list(build_formal_realization_requests(universe, task.task_instance_id))
    namespace = base_requests[0].realization_namespace_sha256
    bridge = build_panda_planning_bridge(root)
    for slot in range(request.realizations_per_task, maximum_realization_slot + 1):
        if len(successful_paths) >= request.realizations_per_task:
            break
        family = sample_uniform_family_for_slot(
            families=universe.config.families,
            master_task_seed=master.master_task_seed,
            structured_expert_config_sha256=structured.source_sha256,
            realization_slot=slot,
        )
        realization_request = FormalRealizationRequest(
            task_instance_id=task.task_instance_id,
            realization_slot=slot,
            assigned_family=family,
            realization_namespace_sha256=namespace,
            realization_seed=derive_realization_seed(
                task.task_instance_id,
                slot,
                namespace,
            ),
        )
        key = realization_request.to_expert_realization_key()
        strategy = sample_strategy_parameters(
            structured.expert,
            family=family,
            realization_seed=key.realization_seed,
        )
        intent = build_trajectory_intent(task, task.expected_anchor, strategy)
        realization_root = group_root / f"realization-{slot:02d}"
        realization_root.mkdir(parents=True, exist_ok=True)
        extension_path = realization_root / "extension_request.json"
        extension_record = {
            "parent_review_id": request.review_id,
            "parent_request_sha256": universe.request_sha256,
            "logical_task_index": logical_task_index,
            "level": level,
            "realization_request": realization_request.to_mapping(),
        }
        if extension_path.exists():
            if json.loads(extension_path.read_text(encoding="utf-8")) != extension_record:
                raise ValueError("existing realization extension has a different identity")
        else:
            write_review_json(extension_path, extension_record)
        candidate_root = realization_root / "candidates"
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
        successful_candidates = sorted(
            (row for row in candidates if row.status is PlannerCandidateStatus.SUCCESS),
            key=lambda row: (row.planner_cost, row.eef_path_length_m, row.fingerprint),
        )
        if not successful_candidates:
            if on_progress is not None:
                on_progress(
                    f"extension L{level} task={logical_task_index} slot={slot} planner_failed"
                )
            continue
        reference = freeze_selected_reference(
            successful_candidates[0],
            fixed_orientation_world=task.expected_anchor.anchor_eef_orientation_matrix_world,
        )
        rollout = execute_structured_realization(
            task_instance=task,
            intent=intent,
            reference=reference,
            maximum_formal_ticks=request.maximum_formal_ticks,
        )
        phases = phase_timeline(rollout.decisions, boundary_count=len(rollout.agentview_rgb))
        video_path = realization_root / "dual_view_review_25fps.mp4"
        trajectory_path = realization_root / "trajectory.npz"
        if not video_path.exists():
            encode_review_video(
                agentview_rgb=rollout.agentview_rgb,
                wrist_rgb=rollout.wrist_rgb,
                phase_by_tick=phases,
                level=level,
                logical_task_index=logical_task_index,
                realization_slot=slot,
                family=family.value,
                fps=request.review_video_fps,
                target=video_path,
            )
        if not trajectory_path.exists():
            np.savez_compressed(
                trajectory_path,
                actions=rollout.actions,
                eef_positions_world=rollout.eef_positions_world,
                object_positions_world=rollout.object_positions_world,
                phase_by_tick=np.asarray(phases),
            )
        terminal_status = rollout.terminal_status
        terminal_reason = rollout.terminal_reason
        if terminal_status == "success" and any(
            discrete_frechet(rollout.eef_positions_world, path) < 0.005
            for path in successful_paths.values()
        ):
            terminal_status = "diversity_rejection"
            terminal_reason = "EEF trajectory is within 5 mm discrete Frechet distance"
        summary = {
            "level": level,
            "logical_task_index": logical_task_index,
            "master_task_seed": master.master_task_seed,
            "realization_slot": slot,
            "family": family.value,
            "extension": True,
            "terminal_status": terminal_status,
            "terminal_reason": terminal_reason,
            "terminal_tick": rollout.terminal_tick,
            "physical_handoff_tick": rollout.physical_handoff_tick,
            "video": str(video_path.relative_to(target)),
            "trajectory": str(trajectory_path.relative_to(target)),
            "intent": intent_summary(intent),
        }
        write_review_json(realization_root / "summary.json", summary)
        rows[slot] = summary
        if terminal_status == "success":
            successful_paths[slot] = rollout.eef_positions_world
        if on_progress is not None:
            on_progress(
                f"extension L{level} task={logical_task_index} slot={slot} "
                f"family={family.value} status={terminal_status}"
            )

    if len(successful_paths) < request.realizations_per_task:
        raise RuntimeError("review realization extension exhausted before completing the task")
    selected_slots = sorted(successful_paths)[: request.realizations_per_task]
    result = {
        "admitted": True,
        "selected_realization_slots": selected_slots,
        "trajectory_statuses": [
            {
                "realization_slot": slot,
                "terminal_status": rows[slot]["terminal_status"],
                "terminal_reason": rows[slot]["terminal_reason"],
            }
            for slot in sorted(rows)
        ],
    }
    write_review_json(group_root / "group_result.json", result)
    return group_root / "group_result.json"
