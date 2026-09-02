"""Collect or dry-run one reviewed structured-expert source corpus request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_meta_mdp.expert_realization.config import load_formal_corpus_config
from latency_meta_mdp.expert_realization.contracts import (
    FormalRequestUniverse,
    build_formal_request_universe,
)
from latency_meta_mdp.expert_realization.source_corpus.config import (
    MasterTaskSplitPlan,
    SourceCorpusConfig,
    SourceExecutionConfig,
    load_master_task_split_plan,
    load_source_corpus_config,
    load_source_execution_config,
)
from latency_meta_mdp.expert_realization.strategy import StructuredStrategyConfig


@dataclass(frozen=True)
class CollectionInputs:
    project_root: Path
    formal_request: FormalRequestUniverse
    source_config: SourceCorpusConfig
    split_plan: MasterTaskSplitPlan
    execution_config: SourceExecutionConfig
    work_root: Path
    output_root: Path
    planner_python: Path
    resume: bool
    dry_run: bool
    formal_config_sha256: str
    split_config_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--planner-python", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _absolute(root: Path, path: Path) -> Path:
    value = path if path.is_absolute() else root / path
    return Path(os.path.abspath(os.fspath(value)))


def _require_separate_roots(work_root: Path, output_root: Path) -> None:
    if (
        work_root == output_root
        or work_root in output_root.parents
        or output_root in work_root.parents
    ):
        raise ValueError("formal work and output roots must be separate non-nested paths")


def _lexical_launcher(project_root: Path, launcher: Path) -> Path:
    result = _absolute(project_root, launcher)
    if not result.is_file():
        raise FileNotFoundError(f"planner Python launcher is missing: {result}")
    return result


def load_collection_inputs(argv: list[str] | None = None) -> CollectionInputs:
    args = _parser().parse_args(argv)
    project_root = _absolute(Path.cwd(), args.project_root)
    formal_path = _absolute(project_root, args.formal_config)
    execution_path = _absolute(project_root, args.execution_config)
    source_path = _absolute(project_root, args.source_config)
    split_path = _absolute(project_root, args.split_config)
    structured = StructuredStrategyConfig.from_path(
        project_root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    formal_config_sha256 = hashlib.sha256(formal_path.read_bytes()).hexdigest()
    formal_request = build_formal_request_universe(
        load_formal_corpus_config(formal_path),
        corpus_config_sha256=formal_config_sha256,
        structured_expert_config_sha256=structured.source_sha256,
    )
    source_config = load_source_corpus_config(source_path)
    split_plan = load_master_task_split_plan(split_path)
    execution_config = load_source_execution_config(execution_path)
    requested_indices = tuple(
        row.logical_task_index
        for row in formal_request.primary_tasks + formal_request.reserve_tasks
    )
    split_plan.require_exact_indices(requested_indices)
    if split_plan.corpus_id != formal_request.config.corpus_id:
        raise ValueError("split plan does not match formal request corpus")
    work_root = _absolute(project_root, args.work_root)
    output_root = _absolute(project_root, args.output_root)
    _require_separate_roots(work_root, output_root)
    planner_python = _lexical_launcher(project_root, args.planner_python)
    return CollectionInputs(
        project_root=project_root,
        formal_request=formal_request,
        source_config=source_config,
        split_plan=split_plan,
        execution_config=execution_config,
        work_root=work_root,
        output_root=output_root,
        planner_python=planner_python,
        resume=args.resume,
        dry_run=args.dry_run,
        formal_config_sha256=formal_config_sha256,
        split_config_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
    )


def _dry_run_summary(inputs: CollectionInputs) -> dict[str, Any]:
    request = inputs.formal_request
    config = request.config
    assignments = {
        str(task_index): [row.to_mapping() for row in rows]
        for task_index, rows in sorted(request.family_assignments.items())
    }
    target_success_count = config.primary_trajectory_count
    expected_canaries = len(config.levels) * inputs.execution_config.determinism_canaries_per_level
    return {
        "mode": "dry_run",
        "corpus_id": config.corpus_id,
        "formal_config_sha256": inputs.formal_config_sha256,
        "formal_request_sha256": request.request_sha256,
        "source_config_sha256": inputs.source_config.sha256,
        "split_config_sha256": inputs.split_config_sha256,
        "split_plan_sha256": inputs.split_plan.sha256,
        "execution_config_sha256": inputs.execution_config.sha256,
        "primary_master_task_indices": [row.logical_task_index for row in request.primary_tasks],
        "reserve_master_task_indices": [row.logical_task_index for row in request.reserve_tasks],
        "levels": list(config.levels),
        "realizations_per_task": config.realizations_per_task,
        "family_assignments": assignments,
        "target_success_count": target_success_count,
        "predeclared_realization_count": (
            len(request.primary_tasks + request.reserve_tasks)
            * len(config.levels)
            * config.realizations_per_task
        ),
        "expected_planner_calls_if_candidate_zero_qualifies": (
            target_success_count + expected_canaries
        ),
        "work_root": str(inputs.work_root),
        "output_root": str(inputs.output_root),
        "planner_python": str(inputs.planner_python),
        "resume": inputs.resume,
    }


def main(
    argv: list[str] | None = None,
    *,
    collect_fn: Callable[..., Path] | None = None,
) -> int:
    inputs = load_collection_inputs(argv)
    if inputs.dry_run:
        print(json.dumps(_dry_run_summary(inputs), sort_keys=True), file=sys.stderr)
        return 0
    if collect_fn is None:
        from latency_meta_mdp.expert_realization.source_corpus.formal_runtime import (
            collect_formal_source,
        )

        collect_fn = collect_formal_source
    manifest = collect_fn(
        project_root=inputs.project_root,
        formal_request=inputs.formal_request,
        source_config=inputs.source_config,
        split_plan=inputs.split_plan,
        execution_config=inputs.execution_config,
        work_root=inputs.work_root,
        output_root=inputs.output_root,
        planner_python=inputs.planner_python,
        resume=inputs.resume,
        on_progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(Path(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
