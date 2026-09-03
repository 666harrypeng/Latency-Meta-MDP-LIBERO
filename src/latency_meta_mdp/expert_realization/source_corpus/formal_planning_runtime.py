"""Side-effecting planning for one success-quota realization draw."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse
from latency_meta_mdp.expert_realization.source_corpus.config import SourceExecutionConfig


def _sha_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class PlannedSourceRealization:
    logical_master_task_index: int
    accepted_slot: int
    task: Any
    realization_request: Any
    strategy: Any
    intent: Any
    first_plan: Any
    task_lineage_sha256: str
    plan_lineage_sha256: str
    realization_universe_sha256: str

    @property
    def level(self) -> int:
        return self.task.task_instance_id.level

    @property
    def realization_draw_index(self) -> int:
        return self.realization_request.realization_draw_index

    @property
    def realization_index(self) -> int:
        return self.accepted_slot


def _plan_payload_path(
    workspace: Any, *, logical: int, level: int, draw_index: int
) -> Path:
    return (
        workspace.payload_root
        / "plans"
        / f"L{level}"
        / f"task-{logical:06d}"
        / f"draw-{draw_index:04d}"
    )


def _run_candidate_request(
    *,
    realization_key: Any,
    structured_config_sha256: str,
    bridge: Any,
    intent: Any,
    start_qpos: Any,
    planner_python: Path,
) -> Any:
    from latency_meta_mdp.expert_realization.planner import (
        load_single_candidate_result,
        run_curobo_candidate_process,
        write_planner_request,
    )
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        PlannerInfrastructureError,
    )

    with tempfile.TemporaryDirectory(prefix="formal-source-draw-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        result_path = root / "result"
        write_planner_request(
            expert_realization_key=realization_key,
            structured_expert_config_sha256=structured_config_sha256,
            candidate_index=0,
            bridge=bridge,
            intent=intent,
            start_qpos=start_qpos,
            timeout_seconds=5.0,
            path=request_path,
        )
        try:
            run_curobo_candidate_process(
                request_path,
                result_path,
                worker_python=planner_python,
            )
        except RuntimeError as error:
            raise PlannerInfrastructureError(str(error)) from error
        return load_single_candidate_result(
            result_path,
            structured_expert_config_sha256=structured_config_sha256,
        )


def plan_formal_realization_draw(
    *,
    project_root: Path,
    logical_master_task_index: int,
    level: int,
    realization_draw_index: int,
    accepted_slot: int,
    admitted_reference_paths: tuple[Any, ...],
    formal_request: FormalRequestUniverse,
    execution_config: SourceExecutionConfig,
    workspace: Any,
    structured: Any,
    bridge: Any,
    planner_python: Path,
    canary_levels: set[int],
    on_progress: Callable[[str], None],
) -> PlannedSourceRealization:
    from latency_meta_mdp.expert_realization.contracts import (
        build_formal_realization_draw_request,
    )
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidateStatus,
        load_single_candidate_result,
        write_single_candidate_result,
    )
    from latency_meta_mdp.expert_realization.selector import freeze_selected_reference
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        DrawRejected,
        FirstQualifiedPlan,
        PlannerInfrastructureError,
        PlanningAttemptsExhausted,
        generate_single_draw_plan,
        qualify_draw_diversity,
        verify_level_planner_canary,
    )
    from latency_meta_mdp.expert_realization.strategy import sample_requested_strategy
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    masters = {
        item.logical_task_index: item
        for item in formal_request.primary_tasks + formal_request.reserve_tasks
    }
    master = masters.get(logical_master_task_index)
    if master is None:
        raise ValueError("master task is outside the formal request")
    task = materialize_task_instance(
        project_root=project_root,
        level=level,
        task_instance_seed=master.master_task_seed,
    )
    request = build_formal_realization_draw_request(
        formal_request,
        task.task_instance_id,
        realization_draw_index,
    )
    draw = workspace.begin_next_draw(
        logical_master_task_index=logical_master_task_index,
        request=request,
    )
    if draw.status in {
        "planner_failure",
        "task_failure",
        "safety_failure",
        "diversity_rejection",
    }:
        raise DrawRejected(draw.status, draw.terminal_reason)
    if draw.status == "accepted":
        raise ValueError("an accepted draw cannot be planned again")

    strategy = sample_requested_strategy(task, request, structured, universe=formal_request)
    intent = build_trajectory_intent(task, task.expected_anchor, strategy)
    key = request.to_expert_realization_key()
    payload_path = _plan_payload_path(
        workspace,
        logical=logical_master_task_index,
        level=level,
        draw_index=realization_draw_index,
    )
    candidate = None
    if payload_path.exists():
        candidate = load_single_candidate_result(
            payload_path,
            structured_expert_config_sha256=structured.source_sha256,
        )
        if (
            candidate.status is not PlannerCandidateStatus.SUCCESS
            or candidate.expert_realization_key != key
            or candidate.candidate_index != 0
        ):
            raise RuntimeError("stored draw plan does not match workspace identity")
        if draw.status == "planning":
            relative_plan = payload_path.relative_to(workspace.root).as_posix()
            draw = workspace.record_draw_plan_qualified(
                draw,
                selected_candidate_fingerprint=candidate.fingerprint,
                plan_path=relative_plan,
            )
    if candidate is None:

        def run_candidate(_candidate_index: int):
            try:
                return _run_candidate_request(
                    realization_key=key,
                    structured_config_sha256=structured.source_sha256,
                    bridge=bridge,
                    intent=intent,
                    start_qpos=task.expected_anchor.anchor_robot_qpos,
                    planner_python=planner_python,
                )
            except PlannerInfrastructureError as error:
                workspace.record_draw_infrastructure_retry(draw, reason=str(error))
                raise

        try:
            first_plan = generate_single_draw_plan(
                realization_key=key,
                intent=intent,
                execution=execution_config,
                run_candidate=run_candidate,
                on_progress=lambda message: on_progress(
                    f"[draw] master={logical_master_task_index} L{level} "
                    f"draw={realization_draw_index} {message}"
                ),
            )
        except PlanningAttemptsExhausted as error:
            workspace.record_draw_failure(
                draw,
                failure_class="planner_failure",
                reason=str(error),
            )
            raise DrawRejected("planner_failure", str(error)) from error
        candidate = first_plan.candidate
        try:
            qualify_draw_diversity(
                candidate.eef_positions_world,
                admitted_reference_paths,
                minimum_frechet_m=0.005,
            )
        except Exception as error:
            from latency_meta_mdp.expert_realization.selector import DiversitySelectionError

            if not isinstance(error, DiversitySelectionError):
                raise
            workspace.record_draw_failure(
                draw,
                failure_class="diversity_rejection",
                reason=str(error),
            )
            raise DrawRejected("diversity_rejection", str(error)) from error
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        write_single_candidate_result(payload_path, candidate)
        relative_plan = payload_path.relative_to(workspace.root).as_posix()
        workspace.record_draw_plan_qualified(
            draw,
            selected_candidate_fingerprint=candidate.fingerprint,
            plan_path=relative_plan,
        )
    first_plan = FirstQualifiedPlan(
        candidate=candidate,
        reference=freeze_selected_reference(
            candidate,
            fixed_orientation_world=task.expected_anchor.anchor_eef_orientation_matrix_world,
        ),
        attempted_candidate_indices=(0,),
        semantic_failures=(),
    )
    if level not in canary_levels:
        on_progress(
            f"[canary] master={logical_master_task_index} L{level} "
            f"draw={realization_draw_index} start"
        )
        verify_level_planner_canary(
            first_plan,
            replay_candidate=lambda: _run_candidate_request(
                realization_key=key,
                structured_config_sha256=structured.source_sha256,
                bridge=bridge,
                intent=intent,
                start_qpos=task.expected_anchor.anchor_robot_qpos,
                planner_python=planner_python,
            ),
        )
        canary_levels.add(level)
        on_progress(f"[canary] L{level} draw={realization_draw_index} passed")

    task_lineage = _sha_mapping(
        {
            "task_instance_id": task.task_instance_id.to_mapping(),
            "task_config_sha256": task.task_config_sha256,
            "motion_config_sha256": task.motion_config_sha256,
            "runtime_config_sha256": task.runtime_config_sha256,
            "controller_config_sha256": task.controller_config_sha256,
        }
    )
    draw_universe = _sha_mapping(
        {
            "formal_request_sha256": formal_request.request_sha256,
            "task_instance_id": task.task_instance_id.to_mapping(),
            "realization_draw_index": realization_draw_index,
        }
    )
    plan_lineage = _sha_mapping(
        {
            "task_lineage_sha256": task_lineage,
            "draw_request": request.to_mapping(),
            "candidate_fingerprint": candidate.fingerprint,
            "reference_fingerprint": first_plan.reference.fingerprint,
            "accepted_slot": accepted_slot,
        }
    )
    return PlannedSourceRealization(
        logical_master_task_index=logical_master_task_index,
        accepted_slot=accepted_slot,
        task=task,
        realization_request=request,
        strategy=strategy,
        intent=intent,
        first_plan=first_plan,
        task_lineage_sha256=task_lineage,
        plan_lineage_sha256=plan_lineage,
        realization_universe_sha256=draw_universe,
    )
