"""Top-level orchestration for the formal paired source collector."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from latency_meta_mdp.data.collection.contracts import FormalRequestUniverse
from latency_meta_mdp.data.collection.recording_contracts import ImplementationIdentity
from latency_meta_mdp.data.source.config import (
    SourceCorpusConfig,
    SourceExecutionConfig,
)
from latency_meta_mdp.data.source.formal_execution_runtime import (
    collect_formal_master_block,
    publish_completed_blocks,
)

_IMPLEMENTATION_PATHS = (
    "src/latency_meta_mdp/envs/backend.py",
    "src/latency_meta_mdp/data/collection/actual_physics.py",
    "src/latency_meta_mdp/data/collection/config.py",
    "src/latency_meta_mdp/data/collection/contracts.py",
    "src/latency_meta_mdp/data/collection/selector.py",
    "src/latency_meta_mdp/data/collection/strategy.py",
    "src/latency_meta_mdp/data/source/collection.py",
    "src/latency_meta_mdp/data/source/config.py",
    "src/latency_meta_mdp/data/source/contracts.py",
    "src/latency_meta_mdp/data/source/formal_collection.py",
    "src/latency_meta_mdp/data/source/formal_execution_runtime.py",
    "src/latency_meta_mdp/data/source/formal_planning_runtime.py",
    "src/latency_meta_mdp/data/source/formal_runtime.py",
    "src/latency_meta_mdp/data/source/metadata.py",
    "src/latency_meta_mdp/data/source/recording.py",
    "src/latency_meta_mdp/data/source/schema.py",
    "src/latency_meta_mdp/data/source/success_payload.py",
    "src/latency_meta_mdp/data/source/workspace.py",
    "src/latency_meta_mdp/data/collection/task_instance.py",
)


def build_formal_collection_identity(
    *,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    execution_config: SourceExecutionConfig,
    qualification_gate_sha256: str,
    planner_environment_sha256: str,
    implementation: ImplementationIdentity,
) -> dict[str, Any]:
    if not isinstance(formal_request, FormalRequestUniverse):
        raise TypeError("formal_request must be FormalRequestUniverse")
    if not isinstance(source_config, SourceCorpusConfig):
        raise TypeError("source_config must be SourceCorpusConfig")
    if not isinstance(execution_config, SourceExecutionConfig):
        raise TypeError("execution_config must be SourceExecutionConfig")
    for name, value in (
        ("qualification_gate_sha256", qualification_gate_sha256),
        ("planner_environment_sha256", planner_environment_sha256),
    ):
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if not isinstance(implementation, ImplementationIdentity):
        raise TypeError("implementation must be ImplementationIdentity")
    return {
        "schema_version": 2,
        "format_id": "formal_source_success_quota_identity_v1",
        "formal_request_sha256": formal_request.request_sha256,
        "source_config_sha256": source_config.sha256,
        "execution_config_sha256": execution_config.sha256,
        "qualification_gate_sha256": qualification_gate_sha256,
        "planner_environment_sha256": planner_environment_sha256,
        "implementation": implementation.to_mapping(),
    }


def lexical_planner_launcher(project_root: Path, launcher: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(project_root)))
    value = Path(launcher)
    result = value.absolute() if value.is_absolute() else (root / value).absolute()
    if not result.is_file():
        raise FileNotFoundError(f"planner Python launcher is missing: {result}")
    return result


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def collect_formal_source(
    *,
    project_root: Path,
    formal_request: FormalRequestUniverse,
    source_config: SourceCorpusConfig,
    execution_config: SourceExecutionConfig,
    work_root: Path,
    output_root: Path,
    planner_python: Path,
    resume: bool,
    on_progress: Callable[[str], None],
) -> Path:
    """Run or resume one immutable formal source collection request."""
    from latency_meta_mdp.data.collection.calibration import (
        collect_scoped_implementation_identity,
    )
    from latency_meta_mdp.data.collection.config import load_pilot_gate_config
    from latency_meta_mdp.data.collection.robot_bridge import build_panda_planning_bridge
    from latency_meta_mdp.data.collection.strategy import StructuredStrategyConfig
    from latency_meta_mdp.data.source.formal_collection import (
        schedule_success_quota_master_blocks,
    )
    from latency_meta_mdp.data.source.workspace import (
        CollectionWorkspace,
    )

    root = Path(project_root).resolve()
    if not isinstance(formal_request, FormalRequestUniverse):
        raise TypeError("formal_request must be FormalRequestUniverse")
    if not isinstance(source_config, SourceCorpusConfig):
        raise TypeError("source_config must be SourceCorpusConfig")
    if not isinstance(execution_config, SourceExecutionConfig):
        raise TypeError("execution_config must be SourceExecutionConfig")
    if type(resume) is not bool or not callable(on_progress):
        raise TypeError("resume must be bool and on_progress must be callable")
    planner_launcher = lexical_planner_launcher(root, planner_python)
    work_root = Path(work_root).absolute()
    output_root = Path(output_root).absolute()
    if (
        work_root == output_root
        or work_root in output_root.parents
        or (output_root in work_root.parents)
    ):
        raise ValueError("formal work and output roots must be separate non-nested paths")
    if output_root.exists():
        raise FileExistsError(output_root)
    structured = StructuredStrategyConfig.from_path(
        root / "configs/data/expert_realization/panda_ball_structured.yaml"
    )
    if structured.source_sha256 != formal_request.structured_expert_config_sha256:
        raise ValueError("formal request structured-expert config changed")
    implementation = collect_scoped_implementation_identity(
        root,
        source_paths=_IMPLEMENTATION_PATHS,
    )
    if implementation.dirty:
        raise ValueError("formal source collection requires a clean scoped implementation")
    qualification_gate_path = root / "configs/analysis/panda_ball_structured_pilot_gate.yaml"
    curobo_config_path = root / "configs/data/expert_realization/curobo_panda.yaml"
    planner_environment_sha = _sha_mapping(
        {
            "planner_config_sha256": _sha_file(curobo_config_path),
            "runtime_config_sha256": _sha_file(
                root / "configs/data/expert_realization/curobo_runtime.yaml"
            ),
            "lock_sha256": _sha_file(root / "requirements/expert-realization-curobo.lock"),
        }
    )
    collection_identity = build_formal_collection_identity(
        formal_request=formal_request,
        source_config=source_config,
        execution_config=execution_config,
        qualification_gate_sha256=_sha_file(qualification_gate_path),
        planner_environment_sha256=planner_environment_sha,
        implementation=implementation,
    )
    if resume:
        workspace = CollectionWorkspace.resume_formal(
            work_root,
            formal_request,
            collection_identity=collection_identity,
        )
    else:
        workspace = CollectionWorkspace.create_formal(
            work_root,
            formal_request,
            collection_identity=collection_identity,
        )
    bridge = build_panda_planning_bridge(root)
    gate = load_pilot_gate_config(qualification_gate_path)
    canary_levels: set[int] = set()

    def collect_block(logical: int):
        return collect_formal_master_block(
            project_root=root,
            logical_master_task_index=logical,
            formal_request=formal_request,
            source_config=source_config,
            execution_config=execution_config,
            workspace=workspace,
            structured=structured,
            bridge=bridge,
            gate=gate,
            planner_python=planner_launcher,
            canary_levels=canary_levels,
            curobo_config_sha256=_sha_file(curobo_config_path),
            implementation=implementation,
            on_progress=on_progress,
        )

    return schedule_success_quota_master_blocks(
        master_task_indices=tuple(
            item.logical_task_index
            for item in formal_request.primary_tasks + formal_request.reserve_tasks
        ),
        target_block_count=formal_request.config.task_instance_count,
        collect_block=collect_block,
        publish_blocks=lambda blocks: publish_completed_blocks(
            blocks,
            project_root=root,
            formal_request=formal_request,
            source_config=source_config,
            workspace=workspace,
            output_root=output_root,
        ),
    )
