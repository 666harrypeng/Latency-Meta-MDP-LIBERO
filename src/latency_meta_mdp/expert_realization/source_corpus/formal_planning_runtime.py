"""Side-effecting sequential planning for formal source collection."""

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
    task: Any
    realization_request: Any
    strategy: Any
    intent: Any
    first_plan: Any
    plan_set: Any
    task_lineage_sha256: str
    plan_lineage_sha256: str
    realization_universe_sha256: str

    @property
    def level(self) -> int:
        return self.task.task_instance_id.level

    @property
    def realization_index(self) -> int:
        return self.realization_request.realization_slot


def _plan_payload_path(
    workspace: Any, *, logical: int, level: int, realization: int
) -> Path:
    return (
        workspace.payload_root
        / "plans"
        / f"L{level}"
        / f"task-{logical:06d}"
        / f"realization-{realization:04d}"
    )


def _run_candidate_request(
    *,
    realization_key: Any,
    candidate_index: int,
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

    with tempfile.TemporaryDirectory(prefix="formal-source-plan-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        result_path = root / "result"
        write_planner_request(
            expert_realization_key=realization_key,
            structured_expert_config_sha256=structured_config_sha256,
            candidate_index=candidate_index,
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


def plan_formal_master_block(
    *,
    project_root: Path,
    logical_master_task_index: int,
    formal_request: FormalRequestUniverse,
    execution_config: SourceExecutionConfig,
    workspace: Any,
    structured: Any,
    bridge: Any,
    planner_python: Path,
    canary_levels: set[int],
    on_progress: Callable[[str], None],
) -> Any:
    from latency_meta_mdp.expert_realization.contracts import (
        build_formal_realization_requests,
        build_realization_universe_identity,
    )
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidateStatus,
        load_single_candidate_result,
        write_single_candidate_result,
    )
    from latency_meta_mdp.expert_realization.selector import (
        DiversitySelectionError,
        freeze_selected_reference,
        select_first_qualified_plan_set,
    )
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        BlockPlanningFailure,
        FirstQualifiedPlan,
        PlannedMasterBlock,
        PlannerInfrastructureError,
        PlanningAttemptsExhausted,
        generate_first_qualified_plan,
        verify_level_planner_canary,
    )
    from latency_meta_mdp.expert_realization.strategy import sample_requested_strategy
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    masters = {
        item.logical_task_index: item
        for item in formal_request.primary_tasks + formal_request.reserve_tasks
    }
    if logical_master_task_index not in masters:
        raise ValueError("master task is outside the formal request")
    block_status = workspace.formal_block_status(logical_master_task_index)
    if block_status.status == "rejected":
        raise BlockPlanningFailure(block_status.terminal_reason)
    if block_status.status == "pending":
        workspace.begin_formal_block(logical_master_task_index)
        block_status = workspace.formal_block_status(logical_master_task_index)
    if block_status.status not in {
        "planning",
        "plan_ready",
        "executing",
        "admitted",
    }:
        raise RuntimeError("formal block cannot be reconstructed from its current state")

    planned_rows = []
    for level in formal_request.config.levels:
        task = materialize_task_instance(
            project_root=project_root,
            level=level,
            task_instance_seed=masters[logical_master_task_index].master_task_seed,
        )
        requests = build_formal_realization_requests(formal_request, task.task_instance_id)
        realization_universe = build_realization_universe_identity(
            request_sha256=formal_request.request_sha256,
            task_instance_id=task.task_instance_id,
            realization_slots=tuple(item.realization_slot for item in requests),
        )
        plans = {}
        strategies = {}
        intents = {}
        for request in requests:
            slot = request.realization_slot
            key = request.to_expert_realization_key()
            strategy = sample_requested_strategy(
                task,
                request,
                structured,
                universe=formal_request,
            )
            intent = build_trajectory_intent(task, task.expected_anchor, strategy)
            strategies[slot] = strategy
            intents[slot] = intent
            state = workspace.formal_plan_status(
                logical_master_task_index, level, slot
            )
            payload_path = _plan_payload_path(
                workspace,
                logical=logical_master_task_index,
                level=level,
                realization=slot,
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
                    or candidate.candidate_index != state.current_candidate_index
                ):
                    raise RuntimeError("stored selected candidate does not match workspace state")
                if state.status != "qualified":
                    workspace.record_formal_plan_success(
                        logical_master_task_index=logical_master_task_index,
                        level=level,
                        realization_index=slot,
                        candidate_index=candidate.candidate_index,
                        selected_candidate_fingerprint=candidate.fingerprint,
                    )
                    state = workspace.formal_plan_status(
                        logical_master_task_index, level, slot
                    )
            if state.status == "qualified":
                if candidate is None:
                    raise RuntimeError("qualified workspace plan has no selected candidate payload")
                if state.selected_candidate_fingerprint != candidate.fingerprint:
                    raise RuntimeError("workspace selected candidate fingerprint changed")
                first_plan = FirstQualifiedPlan(
                    candidate=candidate,
                    reference=freeze_selected_reference(
                        candidate,
                        fixed_orientation_world=(
                            task.expected_anchor.anchor_eef_orientation_matrix_world
                        ),
                    ),
                    attempted_candidate_indices=tuple(
                        range(candidate.candidate_index + 1)
                    ),
                    semantic_failures=state.semantic_failures,
                )
            else:
                def run_candidate(candidate_index: int):
                    try:
                        value = _run_candidate_request(
                            realization_key=key,
                            candidate_index=candidate_index,
                            structured_config_sha256=structured.source_sha256,
                            bridge=bridge,
                            intent=intent,
                            start_qpos=task.expected_anchor.anchor_robot_qpos,
                            planner_python=planner_python,
                        )
                    except PlannerInfrastructureError as error:
                        workspace.record_formal_infrastructure_retry(
                            logical_master_task_index=logical_master_task_index,
                            level=level,
                            realization_index=slot,
                            candidate_index=candidate_index,
                            reason=str(error),
                        )
                        raise
                    if value.status is not PlannerCandidateStatus.SUCCESS:
                        workspace.record_formal_candidate_failure(
                            logical_master_task_index=logical_master_task_index,
                            level=level,
                            realization_index=slot,
                            candidate_index=candidate_index,
                            reason=value.failure_reason,
                        )
                    return value

                try:
                    first_plan = generate_first_qualified_plan(
                        realization_key=key,
                        intent=intent,
                        execution=execution_config,
                        run_candidate=run_candidate,
                        on_progress=lambda message, level=level, slot=slot: on_progress(
                            f"[plan] master={logical_master_task_index} "
                            f"L{level} r{slot} {message}"
                        ),
                        start_candidate_index=state.current_candidate_index,
                        prior_semantic_failures=state.semantic_failures,
                    )
                except PlanningAttemptsExhausted as error:
                    raise BlockPlanningFailure(str(error)) from error
                except PlannerInfrastructureError as error:
                    workspace.reject_formal_block(
                        logical_master_task_index,
                        reason=str(error),
                    )
                    raise BlockPlanningFailure(str(error)) from error
                payload_path.parent.mkdir(parents=True, exist_ok=True)
                write_single_candidate_result(payload_path, first_plan.candidate)
                workspace.record_formal_plan_success(
                    logical_master_task_index=logical_master_task_index,
                    level=level,
                    realization_index=slot,
                    candidate_index=first_plan.candidate.candidate_index,
                    selected_candidate_fingerprint=first_plan.candidate.fingerprint,
                )
            plans[key] = first_plan

        try:
            plan_set = select_first_qualified_plan_set(
                task_instance=task,
                plans_by_key=plans,
            )
        except DiversitySelectionError as error:
            workspace.reject_formal_block(
                logical_master_task_index,
                reason=f"L{level} four-plan diversity failed: {error}",
            )
            raise BlockPlanningFailure(str(error)) from error
        if level not in canary_levels:
            selected = plans[requests[0].to_expert_realization_key()]
            on_progress(
                f"[canary] master={logical_master_task_index} L{level} start "
                f"candidate={selected.candidate.candidate_index}"
            )
            verify_level_planner_canary(
                selected,
                replay_candidate=lambda selected=selected, intent=intents[0]: (
                    _run_candidate_request(
                        realization_key=selected.candidate.expert_realization_key,
                        candidate_index=selected.candidate.candidate_index,
                        structured_config_sha256=structured.source_sha256,
                        bridge=bridge,
                        intent=intent,
                        start_qpos=task.expected_anchor.anchor_robot_qpos,
                        planner_python=planner_python,
                    )
                ),
            )
            canary_levels.add(level)
            on_progress(f"[canary] master={logical_master_task_index} L{level} passed")

        task_lineage = _sha_mapping(
            {
                "task_instance_id": task.task_instance_id.to_mapping(),
                "task_config_sha256": task.task_config_sha256,
                "motion_config_sha256": task.motion_config_sha256,
                "runtime_config_sha256": task.runtime_config_sha256,
                "controller_config_sha256": task.controller_config_sha256,
            }
        )
        plan_lineage = _sha_mapping(
            {
                "task_lineage_sha256": task_lineage,
                "candidate_set_sha256": plan_set.candidate_set_sha256,
                "selected_candidate_fingerprints": dict(
                    plan_set.selected_candidate_fingerprints
                ),
                "selected_reference_fingerprints": {
                    str(index): reference.fingerprint
                    for index, reference in plan_set.references.items()
                },
                "realization_universe_sha256": realization_universe.universe_sha256,
            }
        )
        for request in requests:
            slot = request.realization_slot
            planned_rows.append(
                PlannedSourceRealization(
                    logical_master_task_index=logical_master_task_index,
                    task=task,
                    realization_request=request,
                    strategy=strategies[slot],
                    intent=intents[slot],
                    first_plan=plans[request.to_expert_realization_key()],
                    plan_set=plan_set,
                    task_lineage_sha256=task_lineage,
                    plan_lineage_sha256=plan_lineage,
                    realization_universe_sha256=realization_universe.universe_sha256,
                )
            )
        on_progress(
            f"[group] master={logical_master_task_index} L{level} "
            "plan diversity passed 4/4"
        )
    if workspace.formal_block_status(logical_master_task_index).status == "planning":
        workspace.mark_formal_block_plan_ready(logical_master_task_index)
    return PlannedMasterBlock(
        logical_master_task_index=logical_master_task_index,
        plan_identities=tuple(planned_rows),
    )
