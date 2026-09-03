"""Source rollout, success persistence, and publication for formal collection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse
from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity
from latency_meta_mdp.expert_realization.source_corpus.config import (
    SourceCorpusConfig,
    SourceExecutionConfig,
)
from latency_meta_mdp.expert_realization.source_corpus.formal_planning_runtime import (
    PlannedSourceRealization,
)


def _sha_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _success_payload_path(
    workspace: Any, *, logical: int, level: int, realization: int
) -> tuple[Path, str]:
    relative = (
        f"payloads/successful/L{level}/task-{logical:06d}/"
        f"realization-{realization:04d}"
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
        schema_version=2,
        record_profile="formal_source",
        episode_id=(
            f"source-L{planned.level}-task{planned.logical_master_task_index:06d}-"
            f"r{planned.realization_index:04d}"
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
    )


def execute_formal_master_block(
    planned_block: Any,
    *,
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
        BlockExecutionFailure,
        CompletedMasterBlock,
        PlannedMasterBlock,
        SourceSuccessPayloadRef,
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
    from latency_meta_mdp.expert_realization.source_corpus.workspace import (
        RealizationRunStatus,
    )

    if not isinstance(planned_block, PlannedMasterBlock):
        raise TypeError("planned_block must be PlannedMasterBlock")
    logical = planned_block.logical_master_task_index
    block_status = workspace.formal_block_status(logical)
    if block_status.status == "plan_ready":
        workspace.begin_formal_block_execution(logical)
        block_status = workspace.formal_block_status(logical)
    if block_status.status not in {"executing", "admitted"}:
        raise BlockExecutionFailure(
            "formal block is not executable",
            partial_successes=(),
        )
    successes = []

    def success_ref(item: PlannedSourceRealization, payload_path: Path):
        return SourceSuccessPayloadRef(
            logical_master_task_index=logical,
            level=item.level,
            realization_index=item.realization_index,
            payload_path=payload_path.absolute(),
        )

    for item in planned_block.plan_identities:
        if not isinstance(item, PlannedSourceRealization):
            raise TypeError("planned block contains an invalid realization")
        identity = {
            "logical_master_task_index": logical,
            "level": item.level,
            "realization_index": item.realization_index,
        }
        payload_path, relative_payload = _success_payload_path(
            workspace,
            logical=logical,
            level=item.level,
            realization=item.realization_index,
        )
        while True:
            status = workspace.status_for(**identity)
            if status.status == "success":
                loaded = load_source_success_payload(
                    payload_path,
                    source_config=source_config,
                )
                del loaded
                successes.append(success_ref(item, payload_path))
                break
            if status.status in {
                "task_failure",
                "safety_failure",
                "planner_failure",
                "diversity_rejection",
            }:
                raise BlockExecutionFailure(
                    status.terminal_reason,
                    partial_successes=tuple(successes),
                )
            if status.status == "running":
                workspace.record_formal_execution_status(
                    RealizationRunStatus(
                        **identity,
                        status="infrastructure_failure",
                        attempt_index=status.attempt_index,
                        terminal_reason="previous process ended during source rollout",
                    )
                )
                status = workspace.status_for(**identity)
            if status.status == "infrastructure_failure" and (
                status.attempt_index >= execution_config.infrastructure_retry_limit
            ):
                reason = "source rollout infrastructure retry limit exhausted"
                workspace.reject_formal_block(logical, reason=reason)
                raise BlockExecutionFailure(
                    reason,
                    partial_successes=tuple(successes),
                )
            if status.status == "planned":
                attempt_index = status.attempt_index
            elif status.status == "infrastructure_failure":
                attempt_index = status.attempt_index + 1
            else:
                raise RuntimeError("formal realization has an invalid execution state")
            workspace.record_formal_execution_status(
                RealizationRunStatus(
                    **identity,
                    status="running",
                    attempt_index=attempt_index,
                )
            )
            if payload_path.exists():
                loaded = load_source_success_payload(
                    payload_path,
                    source_config=source_config,
                )
                workspace.record_formal_execution_status(
                    RealizationRunStatus(
                        **identity,
                        status="success",
                        attempt_index=attempt_index,
                        payload_path=relative_payload,
                        terminal_reason=loaded.episode.terminal_reason,
                    )
                )
                del loaded
                successes.append(success_ref(item, payload_path))
                break
            metadata = _episode_metadata(
                item,
                formal_request=formal_request,
                source_config=source_config,
                structured=structured,
                curobo_config_sha256=curobo_config_sha256,
                implementation=implementation,
            )
            on_progress(
                f"[execute] master={logical} L{item.level} "
                f"r{item.realization_index} start attempt={attempt_index}"
            )
            try:
                recording = execute_structured_source_recording(
                    task_instance=item.task,
                    intent=item.intent,
                    reference=item.first_plan.reference,
                    metadata=metadata,
                    maximum_formal_ticks=execution_config.maximum_formal_ticks,
                    planning_bridge=bridge,
                    gate=gate,
                )
            except SourceQualificationFailure as error:
                workspace.record_formal_execution_status(
                    RealizationRunStatus(
                        **identity,
                        status="safety_failure",
                        attempt_index=attempt_index,
                        terminal_reason=str(error),
                    )
                )
                raise BlockExecutionFailure(
                    str(error), partial_successes=tuple(successes)
                ) from error
            except SourceRecordingFailure as error:
                workspace.record_formal_execution_status(
                    RealizationRunStatus(
                        **identity,
                        status="task_failure",
                        attempt_index=attempt_index,
                        terminal_reason=error.terminal_reason,
                    )
                )
                raise BlockExecutionFailure(
                    error.terminal_reason,
                    partial_successes=tuple(successes),
                ) from error
            except Exception as error:
                workspace.record_formal_execution_status(
                    RealizationRunStatus(
                        **identity,
                        status="infrastructure_failure",
                        attempt_index=attempt_index,
                        terminal_reason=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            write_source_success_payload(
                target=payload_path,
                recording=recording,
                source_config=source_config,
                strategy_parameters=item.strategy.to_mapping(),
                selected_planner_fingerprint=item.first_plan.candidate.fingerprint,
            )
            workspace.record_formal_execution_status(
                RealizationRunStatus(
                    **identity,
                    status="success",
                    attempt_index=attempt_index,
                    payload_path=relative_payload,
                    terminal_reason=recording.episode.terminal_reason,
                )
            )
            terminal_tick = len(recording.episode.transitions)
            del recording
            loaded = load_source_success_payload(
                payload_path,
                source_config=source_config,
            )
            del loaded
            successes.append(success_ref(item, payload_path))
            on_progress(
                f"[execute] master={logical} L{item.level} "
                f"r{item.realization_index} success "
                f"tick={terminal_tick}"
            )
            break
    if workspace.formal_block_status(logical).status == "executing":
        workspace.admit_formal_block(logical)
    on_progress(f"[block] master={logical} admitted 12/12")
    return CompletedMasterBlock(logical_master_task_index=logical, successes=tuple(successes))


def publish_completed_blocks(
    completed_blocks: tuple[Any, ...],
    *,
    project_root: Path,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    workspace: Any,
    output_root: Path,
) -> Path:
    from latency_meta_mdp.expert_realization.contracts import FailureClass
    from latency_meta_mdp.expert_realization.source_corpus.collection import (
        AdmittedSourceEpisode,
        CollectionSummary,
        publish_source_corpus,
    )
    from latency_meta_mdp.expert_realization.source_corpus.formal_collection import (
        CompletedMasterBlock,
        SourceSuccessPayloadRef,
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
    success_refs: list[SourceSuccessPayloadRef] = []
    for block in completed_blocks:
        logical = block.logical_master_task_index
        if logical not in masters:
            raise ValueError("completed block is outside the formal request")
        success_refs.extend(block.successes)
        for level in formal_request.config.levels:
            level_refs = [value for value in block.successes if value.level == level]
            if len(level_refs) != formal_request.config.realizations_per_task:
                raise ValueError("completed block level does not contain four successes")
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
                    admitted_realization_count=(
                        formal_request.config.realizations_per_task
                    ),
                )
            )
    statuses = workspace.status_rows()
    failure_values = {value.value for value in FailureClass}
    failures = Counter(row.status for row in statuses if row.status in failure_values)

    def execution_attempts(row: Any) -> int:
        if row.status in {
            "success",
            "task_failure",
            "safety_failure",
            "infrastructure_failure",
        }:
            return row.attempt_index + 1
        return 0

    summary = CollectionSummary(
        requested_realizations=len(statuses),
        planned_realizations=sum(row.status != "requested" for row in statuses),
        executed_attempts=sum(execution_attempts(row) for row in statuses),
        successful_realizations=sum(row.status == "success" for row in statuses),
        admitted_realizations=len(success_refs),
        failures_by_class=dict(failures),
    )

    def admitted_stream():
        for reference in sorted(
            success_refs,
            key=lambda value: (
                value.level,
                value.logical_master_task_index,
                value.realization_index,
            ),
        ):
            success = load_source_success_payload(
                reference.payload_path,
                source_config=source_config,
            )
            metadata = success.episode.metadata
            if (
                metadata.logical_master_task_index
                != reference.logical_master_task_index
                or metadata.task_instance_id.level != reference.level
                or metadata.expert_realization_id.expert_realization_key.realization_index
                != reference.realization_index
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
