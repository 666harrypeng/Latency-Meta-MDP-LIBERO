"""Exact RoboSuite targets for Flow Belief action-buffer counterfactuals."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.flow.buffer_causality import (
    BufferCausalityCandidate,
    build_buffer_branches,
    load_buffer_causality_config,
    select_buffer_causality_contexts,
)
from latency_meta_mdp.episode_split import load_episode_split_plan
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, build_expert_episode_runtime
from latency_meta_mdp.outcomes import OutcomeStatus
from latency_meta_mdp.recording import RecordProfile
from latency_meta_mdp.temporal_contract import load_temporal_contract

_PHASES = ("pregrasp", "approach", "close", "lift")


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def snapshot_return_state(snapshot: Any) -> np.ndarray:
    """Pack one deployment return state in the trained 22D coordinate order."""

    return np.concatenate(
        (
            np.asarray(snapshot.robot_qpos),
            np.asarray(snapshot.robot_qvel),
            np.asarray(
                [
                    snapshot.robot_gripper_qpos[0] - snapshot.robot_gripper_qpos[1],
                    snapshot.robot_gripper_qvel[0] - snapshot.robot_gripper_qvel[1],
                ]
            ),
            np.asarray(snapshot.object_qpos)[:3],
            np.asarray(snapshot.object_qvel)[:3],
        )
    ).astype(np.float32)


def recorded_return_state(arrays: dict[str, np.ndarray], tick: int) -> np.ndarray:
    """Pack a recorded boundary using the same 22D return-state contract."""

    return np.concatenate(
        (
            arrays["robot_qpos"][tick],
            arrays["robot_qvel"][tick],
            np.asarray(
                [
                    arrays["gripper_qpos"][tick, 0] - arrays["gripper_qpos"][tick, 1],
                    arrays["gripper_qvel"][tick, 0] - arrays["gripper_qvel"][tick, 1],
                ]
            ),
            arrays["object_pose"][tick, :3],
            arrays["object_velocity"][tick, :3],
        )
    ).astype(np.float32)


def replay_source_max_abs(
    *, snapshot: Any, arrays: dict[str, np.ndarray], source_tick: int
) -> float:
    """Compare every state field needed to certify an exact replay branch point."""

    pairs = (
        (snapshot.robot_qpos, arrays["robot_qpos"][source_tick]),
        (snapshot.robot_qvel, arrays["robot_qvel"][source_tick]),
        (snapshot.robot_gripper_qpos, arrays["gripper_qpos"][source_tick]),
        (snapshot.robot_gripper_qvel, arrays["gripper_qvel"][source_tick]),
        (snapshot.object_qpos, arrays["object_pose"][source_tick]),
        (snapshot.object_qvel, arrays["object_velocity"][source_tick]),
        (snapshot.eef_pos, arrays["eef_position_world"][source_tick]),
        (snapshot.eef_xmat, arrays["eef_orientation_matrix_world"][source_tick]),
    )
    return max(float(np.max(np.abs(np.asarray(left) - np.asarray(right)))) for left, right in pairs)


def phase_candidates_from_episode_arrays(
    *,
    episode_id: str,
    level: int,
    scene_seed: int,
    expert_phase: np.ndarray,
    minimum_source_tick: int,
    required_real_action_count: int,
) -> tuple[BufferCausalityCandidate, ...]:
    """Choose one central, entirely real source window per semantic phase."""

    phases = np.asarray(expert_phase).astype(str)
    if phases.ndim != 1 or minimum_source_tick < 0 or required_real_action_count <= 0:
        raise ValueError("episode phase candidate inputs are invalid")
    rows = []
    for phase in _PHASES:
        ticks = np.flatnonzero(phases == phase)
        ticks = ticks[
            (ticks >= minimum_source_tick)
            & (ticks + required_real_action_count <= len(phases))
        ]
        if len(ticks) == 0:
            continue
        source_tick = int(ticks[(len(ticks) - 1) // 2])
        rows.append(
            BufferCausalityCandidate(
                episode_id=episode_id,
                level=level,
                scene_seed=scene_seed,
                source_tick=source_tick,
                phase=phase,
                real_transition_count=len(phases),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class BufferCounterfactualSimulation:
    contexts: tuple[BufferCausalityCandidate, ...]
    branch_ids: tuple[str, ...]
    branch_actions: np.ndarray
    target_states: np.ndarray
    handoff_state: np.ndarray
    replay_max_abs: np.ndarray
    expert_reference_max_abs: np.ndarray

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        branches = tuple(self.branch_ids)
        if not contexts or not branches:
            raise ValueError("buffer counterfactual simulation cannot be empty")
        context_count = len(contexts)
        branch_count = len(branches)
        expected = {
            "branch_actions": (context_count, branch_count, 25, 7),
            "target_states": (context_count, branch_count, 20, 22),
            "handoff_state": (context_count, branch_count, 20),
            "replay_max_abs": (context_count,),
            "expert_reference_max_abs": (context_count,),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"buffer counterfactual {name} has invalid shape")
        numeric = (
            self.branch_actions,
            self.target_states,
            self.replay_max_abs,
            self.expert_reference_max_abs,
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("buffer counterfactual numeric arrays must be finite")
        if np.any(np.asarray(self.replay_max_abs) != 0.0):
            raise ValueError("buffer counterfactual source replay is not exact")
        if np.any(np.asarray(self.expert_reference_max_abs) > 1e-6):
            raise ValueError("float32 expert branch exceeds its reference tolerance")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "branch_ids", branches)
        object.__setattr__(self, "branch_actions", _readonly(self.branch_actions, dtype=np.float32))
        object.__setattr__(self, "target_states", _readonly(self.target_states, dtype=np.float32))
        object.__setattr__(self, "handoff_state", _readonly(self.handoff_state, dtype=str))
        object.__setattr__(self, "replay_max_abs", _readonly(self.replay_max_abs, dtype=np.float64))
        object.__setattr__(
            self,
            "expert_reference_max_abs",
            _readonly(self.expert_reference_max_abs, dtype=np.float64),
        )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_buffer_counterfactual_simulation(
    *,
    simulation: BufferCounterfactualSimulation,
    output_dir: Path,
    manifest_fields: dict[str, Any],
) -> Path:
    """Write one no-overwrite simulator counterfactual artifact atomically."""

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"buffer counterfactual output already exists: {target}")
    reserved = {
        "schema_version",
        "format_id",
        "eligible",
        "blockers",
        "context_count",
        "branch_ids",
        "maximum_replay_max_abs",
        "artifacts",
    }
    if reserved & set(manifest_fields):
        raise ValueError("buffer counterfactual manifest fields overwrite reserved fields")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "contexts.json", [asdict(row) for row in simulation.contexts])
        with (building / "counterfactuals.npz").open("xb") as handle:
            np.savez(
                handle,
                branch_actions=simulation.branch_actions,
                target_states=simulation.target_states,
                handoff_state=simulation.handoff_state,
                replay_max_abs=simulation.replay_max_abs,
                expert_reference_max_abs=simulation.expert_reference_max_abs,
            )
            handle.flush()
            os.fsync(handle.fileno())
        artifacts = {
            name: sha256_file(building / name)
            for name in ("contexts.json", "counterfactuals.npz")
        }
        manifest = {
            **manifest_fields,
            "schema_version": 1,
            "format_id": "flow_belief_buffer_counterfactual_sim_v1",
            "eligible": not bool(manifest_fields.get("implementation_dirty", False)),
            "blockers": (
                ["implementation_dirty"]
                if bool(manifest_fields.get("implementation_dirty", False))
                else []
            ),
            "context_count": len(simulation.contexts),
            "branch_ids": list(simulation.branch_ids),
            "maximum_replay_max_abs": float(np.max(simulation.replay_max_abs)),
            "maximum_expert_reference_max_abs": float(
                np.max(simulation.expert_reference_max_abs)
            ),
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        import shutil

        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _verified_episode_dir(*, source_path: Path, level: int, seed: int) -> Path:
    source = _load_json(source_path)
    if (
        source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or source.get("implementation_dirty") is not False
    ):
        raise ValueError("buffer counterfactuals require an eligible formal corpus")
    relative = f"episodes/L{level}/seed_{seed:06d}/manifest.json"
    if relative not in source.get("admitted_episode_manifests", []):
        raise ValueError("formal corpus does not admit the requested episode")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("formal corpus artifact inventory is invalid")
    episode_manifest = source_path.parent / relative
    if not episode_manifest.is_file() or sha256_file(episode_manifest) != artifacts.get(relative):
        raise ValueError("formal corpus episode manifest hash mismatch")
    nested = _load_json(episode_manifest)
    nested_artifacts = nested.get("artifacts")
    if nested.get("format_id") != "synchronized_episode_npz_v3" or not isinstance(
        nested_artifacts, dict
    ):
        raise ValueError("formal corpus episode format is invalid")
    for name in ("arrays.npz", "metadata.json"):
        path = episode_manifest.parent / name
        top_relative = path.relative_to(source_path.parent).as_posix()
        if (
            not path.is_file()
            or sha256_file(path) != nested_artifacts.get(name)
            or sha256_file(path) != artifacts.get(top_relative)
        ):
            raise ValueError(f"formal corpus episode artifact hash mismatch: {name}")
    return episode_manifest.parent


def _load_episode_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = {
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "object_pose",
            "object_velocity",
            "eef_position_world",
            "eef_orientation_matrix_world",
            "expert_action",
            "expert_phase",
        }
        if not required <= set(source.files):
            raise ValueError("formal episode lacks counterfactual source arrays")
        return {name: np.array(source[name], copy=True) for name in required}


def _discover_candidates(
    *,
    source_path: Path,
    split_config_path: Path,
    levels: tuple[int, ...],
    minimum_source_tick: int,
    required_real_action_count: int,
) -> tuple[BufferCausalityCandidate, ...]:
    split = load_episode_split_plan(split_config_path)
    validation = next((row for row in split.ranges if row.name == "validation"), None)
    if validation is None:
        raise ValueError("buffer counterfactual selection requires a validation split")
    rows = []
    for level in levels:
        for seed in range(validation.start, validation.stop):
            episode_dir = _verified_episode_dir(source_path=source_path, level=level, seed=seed)
            metadata = _load_json(episode_dir / "metadata.json")
            arrays = _load_episode_arrays(episode_dir / "arrays.npz")
            if (
                metadata.get("level") != level
                or metadata.get("scene_seed") != seed
                or metadata.get("record_profile") != "belief"
            ):
                raise ValueError("formal episode identity is invalid")
            rows.extend(
                phase_candidates_from_episode_arrays(
                    episode_id=str(metadata["episode_id"]),
                    level=level,
                    scene_seed=seed,
                    expert_phase=arrays["expert_phase"],
                    minimum_source_tick=minimum_source_tick,
                    required_real_action_count=required_real_action_count,
                )
            )
    return tuple(rows)


def _simulate_context(
    *,
    project_root: Path,
    source_path: Path,
    context: BufferCausalityCandidate,
    half_speed_scale: float,
    delayed_prefix_ticks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    episode_dir = _verified_episode_dir(
        source_path=source_path,
        level=context.level,
        seed=context.scene_seed,
    )
    metadata = _load_json(episode_dir / "metadata.json")
    arrays = _load_episode_arrays(episode_dir / "arrays.npz")
    expert_actions = arrays["expert_action"]
    expert_buffer = expert_actions[context.source_tick : context.source_tick + 25]
    branches = build_buffer_branches(
        expert_buffer,
        half_speed_scale=half_speed_scale,
        delayed_prefix_ticks=delayed_prefix_ticks,
    )
    spec = ExpertEpisodeSpec(
        episode_id=str(metadata["episode_id"]),
        level=context.level,
        scene_seed=context.scene_seed,
        motion_seed=int(metadata["motion_seed"]),
        expert_seed=int(metadata["expert_seed"]),
        record_profile=RecordProfile(str(metadata["record_profile"])),
        camera_width=16,
        camera_height=16,
    )
    branch_states = []
    branch_handoff = []
    replay_errors = []
    expert_reference_error = None
    for branch_id, actions in branches.items():
        runtime = build_expert_episode_runtime(project_root=project_root, spec=spec)
        try:
            snapshot = runtime.executor.initialize()
            for tick in range(context.source_tick):
                snapshot = runtime.executor.step_formal(expert_actions[tick])
            replay_error = replay_source_max_abs(
                snapshot=snapshot,
                arrays=arrays,
                source_tick=context.source_tick,
            )
            if replay_error != 0.0:
                raise RuntimeError(
                    f"counterfactual replay mismatch for {context.episode_id}: {replay_error}"
                )
            replay_errors.append(replay_error)
            states = []
            handoff = []
            for delay_index in range(20):
                if runtime.tracker.status is not OutcomeStatus.RUNNING:
                    raise RuntimeError("counterfactual branch terminated before the 20-tick target")
                snapshot = runtime.executor.step_formal(actions[delay_index])
                states.append(snapshot_return_state(snapshot))
                handoff.append(runtime.handoff.state.value)
            stacked = np.stack(states)
            if branch_id == "expert":
                recorded = np.stack(
                    [
                        recorded_return_state(arrays, context.source_tick + delay)
                        for delay in range(1, 21)
                    ]
                )
                expert_reference_error = float(np.max(np.abs(stacked - recorded)))
                if expert_reference_error > 1e-6:
                    raise RuntimeError("float32 expert branch exceeds its reference tolerance")
            branch_states.append(stacked)
            branch_handoff.append(np.asarray(handoff, dtype=str))
        finally:
            runtime.close()
    return (
        np.stack(tuple(branches.values())),
        np.stack(branch_states),
        np.stack(branch_handoff),
        max(replay_errors),
        float(expert_reference_error),
    )


def collect_buffer_counterfactual_simulation_run(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    split_config_path: Path,
    temporal_config_path: Path,
    analysis_config_path: Path,
    output_dir: Path,
    levels: tuple[int, ...],
    contexts_per_phase: int | None = None,
) -> Path:
    """Collect exact simulator counterfactuals without importing Torch."""

    root = project_root.resolve()
    source_path = source_bulk_manifest.resolve()
    split_path = split_config_path.resolve()
    temporal_path = temporal_config_path.resolve()
    analysis_path = analysis_config_path.resolve()
    for path in (source_path, split_path, temporal_path, analysis_path):
        if not path.is_file():
            raise FileNotFoundError(f"buffer counterfactual input does not exist: {path}")
    config = load_buffer_causality_config(analysis_path)
    temporal = load_temporal_contract(temporal_path)
    if (
        temporal.remaining_buffer_coverage != config.required_real_action_count
        or temporal.maximum_delay_ticks != len(config.delay_ticks)
        or temporal.history_sample_count != 6
    ):
        raise ValueError("buffer counterfactual config and temporal contract disagree")
    count = contexts_per_phase or config.contexts_per_phase
    candidates = _discover_candidates(
        source_path=source_path,
        split_config_path=split_path,
        levels=levels,
        minimum_source_tick=max(
            temporal.launch_trigger_horizon,
            temporal.history_sample_count - 1,
        ),
        required_real_action_count=config.required_real_action_count,
    )
    selected = select_buffer_causality_contexts(
        candidates=candidates,
        levels=levels,
        phases=config.phases,
        contexts_per_phase=count,
        required_real_action_count=config.required_real_action_count,
    )
    actions = []
    targets = []
    handoff = []
    replay = []
    expert_reference = []
    for index, context in enumerate(selected, start=1):
        print(
            f"[buffer-causality-sim] context {index}/{len(selected)} "
            f"L{context.level} seed={context.scene_seed} tick={context.source_tick} "
            f"phase={context.phase}",
            flush=True,
        )
        (
            row_actions,
            row_targets,
            row_handoff,
            row_replay,
            row_expert_reference,
        ) = _simulate_context(
            project_root=root,
            source_path=source_path,
            context=context,
            half_speed_scale=config.half_speed_scale,
            delayed_prefix_ticks=config.delayed_prefix_ticks,
        )
        actions.append(row_actions)
        targets.append(row_targets)
        handoff.append(row_handoff)
        replay.append(row_replay)
        expert_reference.append(row_expert_reference)
    simulation = BufferCounterfactualSimulation(
        contexts=selected,
        branch_ids=config.branch_ids,
        branch_actions=np.stack(actions),
        target_states=np.stack(targets),
        handoff_state=np.stack(handoff),
        replay_max_abs=np.asarray(replay, dtype=np.float64),
        expert_reference_max_abs=np.asarray(expert_reference, dtype=np.float64),
    )
    provenance = collect_implementation_provenance(root)
    return write_buffer_counterfactual_simulation(
        simulation=simulation,
        output_dir=output_dir,
        manifest_fields={
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "levels": list(levels),
            "contexts_per_phase": count,
            "input_sha256": {
                "source_bulk_manifest": sha256_file(source_path),
                "split_config": sha256_file(split_path),
                "temporal_config": sha256_file(temporal_path),
                "analysis_config": sha256_file(analysis_path),
            },
        },
    )
