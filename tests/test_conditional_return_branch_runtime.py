from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from latency_meta_mdp.data.expert_collection import (
    ExpertEpisodeSpec,
    build_expert_episode_runtime,
)
from latency_meta_mdp.data.recording import RecordProfile
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.motion import build_motion_profile, load_motion_config
from latency_meta_mdp.envs.outcomes import (
    EpisodeOutcomeTracker,
    OutcomeCriteria,
    OutcomeStatus,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_contracts import (
    ControlContinuationSpec,
    ExecutablePrefix,
    SourceContextIdentity,
    load_branch_corpus_config,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_runtime import (
    ReplayCertification,
    build_branch_runtime,
    execute_control_branch,
    recorded_return_states,
    replay_to_source,
    source_replay_fingerprint,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.control_continuations import (
    ControlContinuation,
    build_control_continuations,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.executable_prefix import (
    materialize_teacher_executable_prefix,
)
from latency_meta_mdp.legacy.belief.conditional_return_flow.source_corpus import (
    load_one_verified_source,
    select_source_contexts,
)
from latency_meta_mdp.runtime.temporal_contract import load_temporal_contract


def _spec(*, motion_seed: int) -> ExpertEpisodeSpec:
    return ExpertEpisodeSpec(
        episode_id=f"branch-runtime-motion-{motion_seed}",
        level=1,
        scene_seed=1011,
        motion_seed=motion_seed,
        expert_seed=1011,
        record_profile=RecordProfile.BELIEF,
        camera_width=8,
        camera_height=8,
    )


def _source_identity_for_terminal() -> SourceContextIdentity:
    return SourceContextIdentity(
        source_context_id="L1-seed001000-tick000040",
        episode_id="l1-seed-001000-attempt-000",
        level=1,
        scene_seed=1000,
        split="train",
        source_tick=40,
        source_phase="lift",
        history_start_tick=35,
        source_episode_manifest_sha256="a" * 64,
        source_arrays_sha256="b" * 64,
        source_metadata_sha256="c" * 64,
        source_observation_sha256="d" * 64,
        motion_profile_sha256="e" * 64,
    )


def test_expert_runtime_uses_exact_motion_profile_override() -> None:
    project_root = Path.cwd()
    motion_config = load_motion_config(
        project_root / "configs/tasks/moving_ball/motion/dynamic_grasp_lift_l1.yaml"
    )
    override = build_motion_profile(config=motion_config, seed=1011, workspace_z=0.833)

    runtime = build_expert_episode_runtime(
        project_root=project_root,
        spec=_spec(motion_seed=1012),
        motion_profile_override=override,
    )
    try:
        runtime_mapping = json.loads(json.dumps(dict(runtime.metadata.motion_profile)))
        assert runtime_mapping == override.to_mapping()
    finally:
        runtime.close()


def test_outcome_fingerprint_is_complete_copy_not_mutable_tracker_state() -> None:
    tracker = EpisodeOutcomeTracker(
        OutcomeCriteria(
            physics_dt_us=2_000,
            formal_tick_us=20_000,
            stable_grasp_dwell_us=40_000,
            lift_height_m=0.08,
            lift_dwell_us=100_000,
            grasp_deadline_us=3_000_000,
            lift_timeout_us=10_000_000,
        )
    )

    payload = tracker.fingerprint_payload()
    payload["events"].append({"kind": "tamper"})

    assert tracker.events == ()
    assert tracker.fingerprint_payload()["events"] == []
    assert tracker.fingerprint_payload()["last_boundary_time_us"] is None


def test_source_fingerprint_changes_with_controller_goal() -> None:
    project_root = Path.cwd()
    runtime = build_expert_episode_runtime(project_root=project_root, spec=_spec(motion_seed=1011))
    try:
        snapshot = runtime.executor.initialize()
        before = source_replay_fingerprint(runtime=runtime, snapshot=snapshot)
        arm = runtime.env.robots[0].part_controllers["right"]
        original = np.array(arm.goal_pos, copy=True)
        arm.goal_pos = np.array(original, copy=True)
        arm.goal_pos[0] += 1e-6
        after = source_replay_fingerprint(runtime=runtime, snapshot=snapshot)
        arm.goal_pos = original
    finally:
        runtime.close()

    assert before != after
    assert len(before) == 64
    assert len(after) == 64


def test_source_fingerprint_serializes_nested_piecewise_profile() -> None:
    project_root = Path.cwd()
    runtime = build_expert_episode_runtime(
        project_root=project_root,
        spec=ExpertEpisodeSpec(
            episode_id="branch-runtime-l3-profile",
            level=3,
            scene_seed=1000,
            motion_seed=1000,
            expert_seed=1000,
            record_profile=RecordProfile.BELIEF,
            camera_width=8,
            camera_height=8,
        ),
    )
    try:
        snapshot = runtime.executor.initialize()
        fingerprint = source_replay_fingerprint(runtime=runtime, snapshot=snapshot)
    finally:
        runtime.close()

    assert len(fingerprint) == 64


def test_fresh_stored_profile_replay_matches_real_l1_source() -> None:
    project_root = Path.cwd()
    source = load_one_verified_source(
        project_root=project_root,
        source_bulk_manifest=project_root
        / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json",
        split_config_path=project_root / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        level=1,
        scene_seed=1000,
    )
    contexts = select_source_contexts(
        episodes=(source,),
        config=load_branch_corpus_config(
            project_root / "configs/legacy/belief/conditional_return_flow/branch_corpus.yaml"
        ),
        temporal=load_temporal_contract(
            project_root / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"
        ),
    )
    context = next(row for row in contexts.contexts if row.identity.source_phase == "approach")
    runtime = build_branch_runtime(project_root=project_root, episode=source)
    try:
        replay = replay_to_source(runtime=runtime, episode=source, context=context)
    finally:
        runtime.close()

    assert replay.formal_tick == context.identity.source_tick
    assert replay.maximum_physical_state_abs == 0.0
    assert len(replay.fingerprint_sha256) == 64


def test_nominal_control_branch_matches_recorded_future_states() -> None:
    project_root = Path.cwd()
    source = load_one_verified_source(
        project_root=project_root,
        source_bulk_manifest=project_root
        / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json",
        split_config_path=project_root / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        level=1,
        scene_seed=1000,
    )
    config = load_branch_corpus_config(
        project_root / "configs/legacy/belief/conditional_return_flow/branch_corpus.yaml"
    )
    temporal = load_temporal_contract(
        project_root / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"
    )
    context = next(
        row
        for row in select_source_contexts(
            episodes=(source,),
            config=config,
            temporal=temporal,
        ).contexts
        if row.identity.source_phase == "approach"
    )
    expert_actions = source.load_arrays(names=("expert_action",))["expert_action"]
    prefix = materialize_teacher_executable_prefix(
        expert_actions=expert_actions,
        source_tick=context.identity.source_tick,
        temporal=temporal,
        action_contract=load_action_contract(
            project_root / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
        ),
    )
    continuation = build_control_continuations(
        nominal_prefix=prefix,
        config=config,
        action_contract=load_action_contract(
            project_root / "configs/runtime/control/panda_osc_pose_delta_v1.yaml"
        ),
        source_context_id=context.identity.source_context_id,
    )[0]
    runtime = build_branch_runtime(project_root=project_root, episode=source)
    try:
        replay = replay_to_source(runtime=runtime, episode=source, context=context)
        rollout = execute_control_branch(
            runtime=runtime,
            source=context,
            replay=replay,
            canonical_source_fingerprint=replay.fingerprint_sha256,
            continuation=continuation,
            recorded_nominal_targets=recorded_return_states(
                episode=source,
                source_tick=context.identity.source_tick,
                maximum_delay_ticks=20,
            ),
        )
    finally:
        runtime.close()

    assert rollout.source_replay_max_abs == 0.0
    assert rollout.source_fingerprint_match is True
    assert rollout.nominal_future_valid is True
    assert rollout.nominal_future_max_abs <= 1e-6


def _minimal_snapshot(value: float):
    return SimpleNamespace(
        robot_qpos=np.full(7, value),
        robot_qvel=np.full(7, value),
        robot_gripper_qpos=np.array([value, 0.0]),
        robot_gripper_qvel=np.array([value, 0.0]),
        object_qpos=np.array([value, value, value, 1.0, 0.0, 0.0, 0.0]),
        object_qvel=np.array([value, value, value, 0.0, 0.0, 0.0]),
    )


def test_terminal_boundary_begins_explicit_absorbing_suffix() -> None:
    tracker = SimpleNamespace(status=OutcomeStatus.RUNNING)
    terminal_snapshot = _minimal_snapshot(1.0)

    class Executor:
        def step_formal(self, _action):
            tracker.status = OutcomeStatus.SUCCESS
            return terminal_snapshot

    runtime = SimpleNamespace(
        tracker=tracker,
        executor=Executor(),
        handoff=SimpleNamespace(state=SimpleNamespace(value="physical")),
    )
    controls = np.zeros((20, 7), dtype=np.float32)
    controls[:, 6] = -1.0
    prefix = ExecutablePrefix(
        controls=controls,
        from_active_buffer_mask=np.zeros(20, dtype=bool),
        last_executed_gripper_command=-1.0,
    )
    continuation = ControlContinuation(
        source_context_id="L1-seed001000-tick000040",
        spec=ControlContinuationSpec(kind="hold", arm_scale=None, prefix_real_ticks=None),
        prefix=prefix,
    )
    source_identity = SimpleNamespace(identity=_source_identity_for_terminal())
    replay = ReplayCertification(
        snapshot=_minimal_snapshot(0.0),
        formal_tick=40,
        maximum_physical_state_abs=0.0,
        fingerprint_sha256="a" * 64,
    )

    rollout = execute_control_branch(
        runtime=runtime,
        source=source_identity,
        replay=replay,
        canonical_source_fingerprint="a" * 64,
        continuation=continuation,
        recorded_nominal_targets=None,
    )

    assert np.all(rollout.target_absorbing)
    np.testing.assert_array_equal(
        rollout.target_states,
        np.repeat(rollout.target_states[:1], 20, axis=0),
    )
