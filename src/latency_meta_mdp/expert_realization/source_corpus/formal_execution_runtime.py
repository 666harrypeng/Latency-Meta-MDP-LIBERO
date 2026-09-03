"""Success-quota source rollout, persistence, and publication."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse
from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity
from latency_meta_mdp.expert_realization.source_corpus.config import (
    SourceCorpusConfig,
    SourceExecutionConfig,
)
from latency_meta_mdp.expert_realization.source_corpus.formal_planning_runtime import (
    PlannedSourceRealization,
    plan_formal_realization_draw,
)


def _sha_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _success_payload_path(
    workspace: Any, *, logical: int, level: int, draw_index: int
) -> tuple[Path, str]:
    relative = (
        f"payloads/successful/L{level}/task-{logical:06d}/draw-{draw_index:04d}"
    )
    return workspace.root / relative, relative


def _episode_metadata(
    planned: PlannedSourceRealization,
    *,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    structured: Any,
    curobo_config_sha256: str,
    implementation: ImplementationIdentity,
) -> Any:
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationId
    from latency_meta_mdp.expert_realization.source_corpus.contracts import (
        FormalSourceEpisodeMetadata,
    )

    task = planned.task
    request = planned.realization_request
    candidate = planned.first_plan.candidate
    key = request.to_expert_realization_key()
    return FormalSourceEpisodeMetadata(
        schema_version=3,
        record_profile="formal_source",
        episode_id=(
            f"source-L{planned.level}-task{planned.logical_master_task_index:06d}-"
            f"s{planned.accepted_slot:02d}-d{planned.realization_draw_index:04d}"
        ),
        corpus_id=formal_request.config.corpus_id,
        logical_master_task_index=planned.logical_master_task_index,
        task_instance_id=task.task_instance_id,
        expert_realization_id=ExpertRealizationId(key, planned.plan_lineage_sha256),
        task_id=task.task_id,
        instruction=task.instruction,
        physics_dt_us=2_000,
        formal_tick_us=20_000,
        camera_height=task.camera_height,
        camera_width=task.camera_width,
        action_contract_id="panda_osc_pose_delta_v1",
        action_dim=7,
        actuator_dim=9,
        expert_id=structured.expert.expert_id,
        strategy_family=request.assigned_family,
        formal_corpus_config_sha256=formal_request.corpus_config_sha256,
        source_corpus_config_sha256=source_config.sha256,
        task_config_sha256=task.task_config_sha256,
        motion_config_sha256=task.motion_config_sha256,
        runtime_config_sha256=task.runtime_config_sha256,
        controller_config_sha256=task.controller_config_sha256,
        structured_expert_config_sha256=structured.source_sha256,
        curobo_planner_config_sha256=curobo_config_sha256,
        task_instance_manifest_sha256=planned.task_lineage_sha256,
        frozen_plan_set_manifest_sha256=planned.plan_lineage_sha256,
        realization_universe_sha256=planned.realization_universe_sha256,
        strategy_sha256=_sha_mapping(planned.strategy.to_mapping()),
        planner_candidates_sha256=candidate.fingerprint,
        selected_reference_sha256=planned.first_plan.reference.fingerprint,
        implementation=implementation,
        accepted_slot=planned.accepted_slot,
        realization_draw_index=planned.realization_draw_index,
    )


def _episode_eef_path(episode: Any) -> np.ndarray:
    return np.stack(
        [boundary.deployment.eef_position_world for boundary in episode.boundaries],
        axis=0,
    ).astype(np.float64, copy=False)


def _reference_from_status(status: Any, *, workspace: Any, structured: Any) -> Any:
    from latency_meta_mdp.expert_realization.planner import load_single_candidate_result
    from latency_meta_mdp.expert_realization.selector import freeze_selected_reference

    if status.plan_path is None:
        raise ValueError("accepted draw has no persisted plan path")
    candidate = load_single_candidate_result(
        workspace.root / status.plan_path,
        structured_expert_config_sha256=structured.source_sha256,
    )
    return freeze_selected_reference(
        candidate,
        fixed_orientation_world=np.eye(3, dtype=np.float64),
    )


def execute_formal_realization_draw(
    planned: PlannedSourceRealization,
    *,
    existing_successes: tuple[Any, ...],
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    execution_config: SourceExecutionConfig,
    workspace: Any,
    structured: Any,
    bridge: Any,
    gate: Any,
    curobo_config_sha256: str,
    implementation: ImplementationIdentity,
    on_progress: Callable[[str], None],
) -> Any:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        DrawRejected,
        SourceSuccessPayloadRef,
        qualify_draw_diversity,
    )
    from latency_meta_mdp.expert_realization.source_corpus.recording import (
        SourceQualificationFailure,
        SourceRecordingFailure,
        execute_structured_source_recording,
    )
    from latency_meta_mdp.expert_realization.source_corpus.success_payload import (
        load_source_success_payload,
        write_source_success_payload,
    )

    logical = planned.logical_master_task_index
    level = planned.level
    draw_index = planned.realization_draw_index
    payload_path, relative_payload = _success_payload_path(
        workspace,
        logical=logical,
        level=level,
        draw_index=draw_index,
    )
    status = workspace.draw_status(logical, level, draw_index)
    if status.status == "accepted":
        loaded = load_source_success_payload(payload_path, source_config=source_config)
        del loaded
        return SourceSuccessPayloadRef(
            logical_master_task_index=logical,
            level=level,
            realization_draw_index=draw_index,
            accepted_slot=status.accepted_slot,
            payload_path=payload_path.absolute(),
        )
    resumed_running = status.status == "running"
    if status.status == "plan_qualified":
        status = workspace.record_draw_running(status)
    elif status.status != "running":
        raise ValueError("draw must be plan-qualified before execution")
    if resumed_running and not payload_path.exists():
        status = workspace.record_draw_infrastructure_retry(
            status,
            reason="previous process ended during source rollout",
        )
        if status.attempt_index > execution_config.infrastructure_retry_limit:
            raise RuntimeError("source rollout infrastructure retry limit exhausted")

    while True:
        if payload_path.exists():
            loaded = load_source_success_payload(payload_path, source_config=source_config)
            slot = workspace.record_draw_accepted(
                status,
                payload_path=relative_payload,
                terminal_reason=loaded.episode.terminal_reason,
            )
            del loaded
            if slot != planned.accepted_slot:
                raise RuntimeError("resumed draw accepted slot changed")
            break
        metadata = _episode_metadata(
            planned,
            formal_request=formal_request,
            source_config=source_config,
            structured=structured,
            curobo_config_sha256=curobo_config_sha256,
            implementation=implementation,
        )
        on_progress(
            f"[draw] master={logical} L{level} draw={draw_index} "
            f"rollout_start attempt={status.attempt_index}"
        )
        try:
            recording = execute_structured_source_recording(
                task_instance=planned.task,
                intent=planned.intent,
                reference=planned.first_plan.reference,
                metadata=metadata,
                maximum_formal_ticks=execution_config.maximum_formal_ticks,
                planning_bridge=bridge,
                gate=gate,
            )
        except SourceQualificationFailure as error:
            workspace.record_draw_failure(
                status,
                failure_class="safety_failure",
                reason=str(error),
            )
            raise DrawRejected("safety_failure", str(error)) from error
        except SourceRecordingFailure as error:
            workspace.record_draw_failure(
                status,
                failure_class="task_failure",
                reason=error.terminal_reason,
            )
            raise DrawRejected("task_failure", error.terminal_reason) from error
        except Exception as error:
            status = workspace.record_draw_infrastructure_retry(
                status,
                reason=f"{type(error).__name__}: {error}",
            )
            if status.attempt_index > execution_config.infrastructure_retry_limit:
                raise RuntimeError("source rollout infrastructure retry limit exhausted") from error
            continue
        admitted_paths = []
        for reference in existing_successes:
            previous = load_source_success_payload(
                reference.payload_path,
                source_config=source_config,
            )
            admitted_paths.append(_episode_eef_path(previous.episode))
            del previous
        try:
            qualify_draw_diversity(
                _episode_eef_path(recording.episode),
                tuple(admitted_paths),
                minimum_frechet_m=0.005,
            )
        except Exception as error:
            from latency_meta_mdp.expert_realization.selector import DiversitySelectionError

            if not isinstance(error, DiversitySelectionError):
                raise
            workspace.record_draw_failure(
                status,
                failure_class="diversity_rejection",
                reason=str(error),
            )
            raise DrawRejected("diversity_rejection", str(error)) from error
        write_source_success_payload(
            target=payload_path,
            recording=recording,
            source_config=source_config,
            strategy_parameters=planned.strategy.to_mapping(),
            selected_planner_fingerprint=planned.first_plan.candidate.fingerprint,
        )
        terminal_tick = len(recording.episode.transitions)
        terminal_reason = recording.episode.terminal_reason
        del recording
        slot = workspace.record_draw_accepted(
            status,
            payload_path=relative_payload,
            terminal_reason=terminal_reason,
        )
        if slot != planned.accepted_slot:
            raise RuntimeError("draw accepted slot changed")
        on_progress(
            f"[draw] master={logical} L{level} draw={draw_index} "
            f"accepted slot={slot} tick={terminal_tick}"
        )
        break
    return SourceSuccessPayloadRef(
        logical_master_task_index=logical,
        level=level,
        realization_draw_index=draw_index,
        accepted_slot=planned.accepted_slot,
        payload_path=payload_path.absolute(),
    )


def collect_formal_master_block(
    *,
    project_root: Path,
    logical_master_task_index: int,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    execution_config: SourceExecutionConfig,
    workspace: Any,
    structured: Any,
    bridge: Any,
    gate: Any,
    planner_python: Path,
    canary_levels: set[int],
    curobo_config_sha256: str,
    implementation: ImplementationIdentity,
    on_progress: Callable[[str], None],
) -> Any:
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
        DrawRejected,
        LevelQuotaExhausted,
        SourceSuccessPayloadRef,
    )

    block = workspace.formal_block_status(logical_master_task_index)
    if block.status == "pending":
        workspace.begin_formal_block(logical_master_task_index)
        block = workspace.formal_block_status(logical_master_task_index)
    if block.status == "rejected":
        raise LevelQuotaExhausted(block.terminal_reason)
    successes: list[SourceSuccessPayloadRef] = []
    for level in formal_request.config.levels:
        existing = workspace.accepted_draws(logical_master_task_index, level)
        level_successes = [
            SourceSuccessPayloadRef(
                logical_master_task_index=logical_master_task_index,
                level=level,
                realization_draw_index=row.realization_draw_index,
                accepted_slot=row.accepted_slot,
                payload_path=(workspace.root / row.payload_path).absolute(),
            )
            for row in existing
        ]
        while len(level_successes) < formal_request.config.realizations_per_task:
            quota = workspace.level_quota_status(logical_master_task_index, level)
            draw_index = quota.next_draw_index
            if draw_index >= execution_config.maximum_realization_draws_per_level:
                reason = (
                    f"master {logical_master_task_index} L{level} could not collect "
                    f"{formal_request.config.realizations_per_task} successes within "
                    f"{execution_config.maximum_realization_draws_per_level} draws"
                )
                workspace.reject_formal_block(logical_master_task_index, reason=reason)
                raise LevelQuotaExhausted(reason)
            admitted_reference_paths = tuple(
                _reference_from_status(row, workspace=workspace, structured=structured)
                .eef_positions_world
                for row in workspace.accepted_draws(logical_master_task_index, level)
            )
            try:
                planned = plan_formal_realization_draw(
                    project_root=project_root,
                    logical_master_task_index=logical_master_task_index,
                    level=level,
                    realization_draw_index=draw_index,
                    accepted_slot=len(level_successes),
                    admitted_reference_paths=admitted_reference_paths,
                    formal_request=formal_request,
                    execution_config=execution_config,
                    workspace=workspace,
                    structured=structured,
                    bridge=bridge,
                    planner_python=planner_python,
                    canary_levels=canary_levels,
                    on_progress=on_progress,
                )
                success = execute_formal_realization_draw(
                    planned,
                    existing_successes=tuple(level_successes),
                    formal_request=formal_request,
                    source_config=source_config,
                    execution_config=execution_config,
                    workspace=workspace,
                    structured=structured,
                    bridge=bridge,
                    gate=gate,
                    curobo_config_sha256=curobo_config_sha256,
                    implementation=implementation,
                    on_progress=on_progress,
                )
            except DrawRejected as error:
                on_progress(
                    f"[draw] master={logical_master_task_index} L{level} "
                    f"draw={draw_index} rejected class={error.failure_class} "
                    f"reason={error.reason}"
                )
                continue
            level_successes.append(success)
        on_progress(
            f"[quota] master={logical_master_task_index} L{level} "
            f"accepted={len(level_successes)}/{formal_request.config.realizations_per_task}"
        )
        successes.extend(level_successes)
    workspace.admit_formal_block(logical_master_task_index)
    on_progress(f"[block] master={logical_master_task_index} admitted 12/12")
    return CompletedMasterBlock(
        logical_master_task_index=logical_master_task_index,
        successes=tuple(successes),
    )


def publish_completed_blocks(
    completed_blocks: tuple[Any, ...],
    *,
    project_root: Path,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    workspace: Any,
    output_root: Path,
) -> Path:
    from latency_meta_mdp.expert_realization.source_corpus.collection import (
        AdmittedSourceEpisode,
        CollectionSummary,
        publish_source_corpus,
    )
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
    )
    from latency_meta_mdp.expert_realization.source_corpus.metadata import (
        SourceTaskMetadataEntry,
    )
    from latency_meta_mdp.expert_realization.source_corpus.success_payload import (
        load_source_success_payload,
    )
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance

    if (
        type(completed_blocks) is not tuple
        or len(completed_blocks) != formal_request.config.task_instance_count
        or any(not isinstance(item, CompletedMasterBlock) for item in completed_blocks)
    ):
        raise ValueError("publication requires the exact completed master-block target")
    masters = {
        item.logical_task_index: item
        for item in formal_request.primary_tasks + formal_request.reserve_tasks
    }
    task_entries = []
    success_refs = []
    for block in completed_blocks:
        logical = block.logical_master_task_index
        success_refs.extend(block.successes)
        for level in formal_request.config.levels:
            task = materialize_task_instance(
                project_root=project_root,
                level=level,
                task_instance_seed=masters[logical].master_task_seed,
            )
            task_entries.append(
                SourceTaskMetadataEntry(
                    task_instance_id=task.task_instance_id,
                    corpus_id=formal_request.config.corpus_id,
                    logical_master_task_index=logical,
                    instruction=task.instruction,
                    motion_profile_json=task.motion_profile_bytes.decode("utf-8"),
                    initial_state_npz=task.initial_state_bytes,
                    admitted_realization_count=formal_request.config.realizations_per_task,
                )
            )
    draws = workspace.draw_statuses()
    failures = Counter(
        row.status
        for row in draws
        if row.status
        in {"planner_failure", "task_failure", "safety_failure", "diversity_rejection"}
    )
    executed = sum(
        row.status in {"task_failure", "safety_failure", "accepted"}
        or (row.status == "diversity_rejection" and row.plan_path is not None)
        for row in draws
    )
    task_successes = sum(
        row.status == "accepted"
        or (row.status == "diversity_rejection" and row.plan_path is not None)
        for row in draws
    )
    summary = CollectionSummary(
        requested_realizations=len(draws),
        planned_realizations=sum(row.status != "planner_failure" for row in draws),
        executed_attempts=executed,
        successful_realizations=task_successes,
        admitted_realizations=len(success_refs),
        failures_by_class=dict(failures),
        attempted_family_counts=dict(Counter(row.family for row in draws)),
        admitted_family_counts=dict(
            Counter(row.family for row in draws if row.status == "accepted")
        ),
    )

    def admitted_stream():
        for reference in sorted(
            success_refs,
            key=lambda value: (
                value.level,
                value.logical_master_task_index,
                value.accepted_slot,
            ),
        ):
            success = load_source_success_payload(
                reference.payload_path,
                source_config=source_config,
            )
            metadata = success.episode.metadata
            if (
                metadata.logical_master_task_index != reference.logical_master_task_index
                or metadata.task_instance_id.level != reference.level
                or metadata.realization_draw_index != reference.realization_draw_index
                or metadata.accepted_slot != reference.accepted_slot
            ):
                raise ValueError("success payload reference identity changed")
            yield AdmittedSourceEpisode(
                episode=success.episode,
                strategy_parameters=success.strategy_parameters,
                selected_planner_fingerprint=success.selected_planner_fingerprint,
                qualification=success.qualification,
            )
            del success

    return publish_source_corpus(
        target=output_root,
        request=formal_request,
        source_config=source_config,
        task_entries=tuple(task_entries),
        admitted_episodes=admitted_stream(),
        collection_summary=summary,
    )
