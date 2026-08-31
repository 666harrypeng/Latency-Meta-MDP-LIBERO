"""Orchestrate exact same-source executable-control branch collection."""

from __future__ import annotations

import resource
from pathlib import Path
from time import perf_counter
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.conditional_return_flow.branch_artifacts import (
    ControlBranchCorpus,
    write_control_branch_corpus,
)
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    load_branch_corpus_config,
)
from latency_meta_mdp.belief.conditional_return_flow.branch_runtime import (
    build_branch_runtime,
    execute_control_branch,
    recorded_return_states,
    replay_to_source,
)
from latency_meta_mdp.belief.conditional_return_flow.control_continuations import (
    build_control_continuations,
)
from latency_meta_mdp.belief.conditional_return_flow.executable_prefix import (
    materialize_teacher_executable_prefix,
)
from latency_meta_mdp.belief.conditional_return_flow.source_corpus import (
    SelectedSourceContext,
    load_verified_source_episodes,
    select_source_contexts,
)
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.temporal_contract import load_temporal_contract


def _limit_contexts_per_level(
    contexts: tuple[SelectedSourceContext, ...],
    *,
    levels: tuple[int, ...],
    limit: int | None,
) -> tuple[SelectedSourceContext, ...]:
    if limit is None:
        return contexts
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("maximum contexts per level must be a positive integer")
    selected = []
    for level in levels:
        rows = [row for row in contexts if row.identity.level == level]
        if len(rows) < limit:
            raise ValueError(f"L{level} has fewer source contexts than the requested pilot limit")
        selected.extend(rows[:limit])
    return tuple(selected)


def _resolve_inputs(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    temporal_config_path: Path,
    branch_config_path: Path,
) -> dict[str, Path]:
    paths = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "split_config": split_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "branch_config": branch_config_path.resolve(),
        "control_config": (
            project_root / "configs/control/panda_osc_pose_delta_v1.yaml"
        ).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"conditional return branch input is missing: {name}={path}")
    return paths


def collect_control_branch_corpus(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    temporal_config_path: Path,
    branch_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    maximum_contexts_per_level: int | None = None,
) -> Path:
    started_at = perf_counter()
    if not levels or levels != tuple(sorted(set(levels))) or any(
        level not in (1, 2, 3) for level in levels
    ):
        raise ValueError("branch collection levels must be sorted unique values from 1, 2, 3")
    root = project_root.resolve()
    inputs = _resolve_inputs(
        project_root=root,
        source_bulk_manifest=source_bulk_manifest,
        split_config_path=split_config_path,
        temporal_config_path=temporal_config_path,
        branch_config_path=branch_config_path,
    )
    config = load_branch_corpus_config(inputs["branch_config"])
    temporal = load_temporal_contract(inputs["temporal_config"])
    action_contract = load_action_contract(inputs["control_config"])
    if temporal.maximum_delay_ticks != 20 or temporal.history_sample_count != 6:
        raise ValueError("branch collection requires the certified D20/K6 temporal contract")
    if (
        action_contract.formal_tick_us != temporal.formal_tick_us
        or action_contract.action_dim != 7
    ):
        raise ValueError("branch collection action and temporal contracts disagree")
    episodes = load_verified_source_episodes(
        project_root=root,
        source_bulk_manifest=inputs["source_bulk_manifest"],
        split_config_path=inputs["split_config"],
        levels=levels,
    )
    contexts = select_source_contexts(episodes=episodes, config=config, temporal=temporal)
    contexts = _limit_contexts_per_level(
        contexts,
        levels=levels,
        limit=maximum_contexts_per_level,
    )
    if not contexts:
        raise ValueError("branch collection selected no source contexts")

    rollouts = []
    canonical_fingerprints = []
    required_branch_kinds: tuple[str, ...] | None = None
    for context_index, context in enumerate(contexts, start=1):
        print(
            f"[conditional-return-branches] context {context_index}/{len(contexts)} "
            f"L{context.identity.level} seed={context.identity.scene_seed} "
            f"tick={context.identity.source_tick} phase={context.identity.source_phase}",
            flush=True,
        )
        expert_actions = context.episode.load_arrays(names=("expert_action",))["expert_action"]
        nominal_prefix = materialize_teacher_executable_prefix(
            expert_actions=expert_actions,
            source_tick=context.identity.source_tick,
            temporal=temporal,
            action_contract=action_contract,
        )
        continuations = build_control_continuations(
            nominal_prefix=nominal_prefix,
            config=config,
            action_contract=action_contract,
            source_context_id=context.identity.source_context_id,
        )
        branch_kinds = tuple(row.spec.kind for row in continuations)
        if required_branch_kinds is None:
            required_branch_kinds = branch_kinds
        elif branch_kinds != required_branch_kinds:
            raise RuntimeError("control branch inventory changed across source contexts")
        canonical_fingerprint: str | None = None
        nominal_targets = recorded_return_states(
            episode=context.episode,
            source_tick=context.identity.source_tick,
            maximum_delay_ticks=temporal.maximum_delay_ticks,
        )
        for continuation in continuations:
            runtime = build_branch_runtime(project_root=root, episode=context.episode)
            try:
                replay = replay_to_source(
                    runtime=runtime,
                    episode=context.episode,
                    context=context,
                )
                if canonical_fingerprint is None:
                    canonical_fingerprint = replay.fingerprint_sha256
                rollout = execute_control_branch(
                    runtime=runtime,
                    source=context,
                    replay=replay,
                    canonical_source_fingerprint=canonical_fingerprint,
                    continuation=continuation,
                    recorded_nominal_targets=(
                        nominal_targets if continuation.spec.kind == "nominal" else None
                    ),
                )
            finally:
                runtime.close()
            rollouts.append(rollout)
        if canonical_fingerprint is None:
            raise RuntimeError("control branch source produced no canonical replay fingerprint")
        canonical_fingerprints.append(canonical_fingerprint)
    if required_branch_kinds is None:
        raise RuntimeError("control branch inventory is empty")

    implementation = collect_implementation_provenance(root)
    manifest_fields: dict[str, Any] = {
        "implementation_revision": implementation.revision,
        "implementation_source_sha256": implementation.source_sha256,
        "implementation_dirty": implementation.dirty,
        "input_paths": {name: str(path) for name, path in inputs.items()},
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "collection_wall_time_seconds": perf_counter() - started_at,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    corpus = ControlBranchCorpus(
        contexts=tuple(row.identity for row in contexts),
        canonical_replay_fingerprints=tuple(canonical_fingerprints),
        rollouts=tuple(rollouts),
        selection_exclusions=(),
        required_branch_kinds=required_branch_kinds,
        requested_levels=levels,
    )
    return write_control_branch_corpus(
        corpus=corpus,
        output_dir=output_dir,
        manifest_fields=manifest_fields,
        bounded=maximum_contexts_per_level is not None,
    )
