from __future__ import annotations

import json

import numpy as np
import pytest


def _key(index: int = 0):
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey, TaskInstanceId

    task = TaskInstanceId(1, 4000, "a" * 64, "b" * 64)
    return ExpertRealizationKey(task, index, "c" * 64)


def _candidate(index: int = 0, *, key_index: int = 0, offset: float = 0.0):
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )

    key = _key(key_index)
    timestamps = np.array([0.0, 0.02, 0.04], dtype=np.float64)
    qpos = np.tile(np.arange(7, dtype=np.float64), (3, 1)) + offset
    eef = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.1, 0.0, 1.0]]) + offset
    seed = planner_candidate_seed(key, index)
    return PlannerCandidate(
        expert_realization_key=key,
        candidate_index=index,
        requested_seed=seed,
        effective_seed=seed,
        status=PlannerCandidateStatus.SUCCESS,
        geometric_seed_qpos_path=np.array(qpos, copy=True),
        qpos_path=qpos,
        timestamps_seconds=timestamps,
        eef_positions_world=eef,
        eef_path_length_m=0.1,
        certified_clearance_lower_bound_m=0.0,
        goal_position_error_m=0.001,
        goal_rotation_error_degrees=0.1,
        planner_cost=1.0 + index,
        planning_time_seconds=0.5,
        failure_reason=None,
        deterministic_replay_verified=None,
    )


def test_candidate_seed_set_and_numeric_records_are_complete() -> None:
    """Break caught: an invocation is missing or stores status without its numerical path."""
    from latency_meta_mdp.expert_realization.planner import planner_candidate_seeds

    key = _key()
    seeds = planner_candidate_seeds(key, candidate_count=8)
    candidates = tuple(_candidate(index) for index in range(8))

    assert len(seeds) == len(set(seeds)) == 8
    assert tuple(item.requested_seed for item in candidates) == seeds
    for index, candidate in enumerate(candidates):
        assert candidate.candidate_index == index
        assert candidate.qpos_path.shape == (3, 7)
        assert candidate.geometric_seed_qpos_path.shape == (3, 7)
        assert candidate.timestamps_seconds.shape == (3,)
        assert candidate.eef_positions_world.shape == (3, 3)
        assert len(candidate.fingerprint) == 64
        assert candidate.effective_seed == candidate.requested_seed


def test_timeout_is_a_typed_candidate_and_cannot_carry_success_arrays() -> None:
    """Break caught: a timed-out invocation disappears or masquerades as a numerical plan."""
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidate,
        PlannerCandidateStatus,
        planner_candidate_seed,
    )

    key = _key()
    seed = planner_candidate_seed(key, 0)
    timeout = PlannerCandidate.failure(
        expert_realization_key=key,
        candidate_index=0,
        requested_seed=seed,
        effective_seed=seed,
        status=PlannerCandidateStatus.TIMEOUT,
        reason="planner call exceeded 5 seconds",
        planning_time_seconds=5.1,
    )
    assert timeout.status is PlannerCandidateStatus.TIMEOUT
    assert timeout.qpos_path.shape == (0, 7)
    assert timeout.fingerprint
    with pytest.raises(ValueError):
        PlannerCandidate(
            **{
                **timeout.constructor_mapping(),
                "qpos_path": np.zeros((2, 7), dtype=np.float64),
            }
        )


def test_candidate_serialization_round_trip_and_corruption_rejection(tmp_path) -> None:
    """Break caught: frozen numerical candidate bytes are not bound to their metadata."""
    from latency_meta_mdp.expert_realization.planner import (
        load_planner_candidates,
        write_planner_candidates,
    )

    candidates = tuple(_candidate(index) for index in range(8))
    root = tmp_path / "candidates"
    write_planner_candidates(root, candidates)
    assert load_planner_candidates(root, structured_expert_config_sha256="c" * 64) == candidates
    path = root / "candidate-003.npz"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="hash"):
        load_planner_candidates(root, structured_expert_config_sha256="c" * 64)


def test_hard_worker_timeout_writes_typed_candidate_record(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a killed planner process disappears instead of producing one typed row."""
    import json
    import subprocess

    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidateStatus,
        load_single_candidate_result,
        planner_candidate_seed,
        run_curobo_candidate_process,
    )

    key = _key()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "expert_realization_key": key.to_mapping(),
                "structured_expert_config_sha256": "c" * 64,
                "candidate_index": 0,
                "requested_seed": planner_candidate_seed(key, 0),
            }
        )
    )
    result = tmp_path / "result"

    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="worker", timeout=0.01)

    monkeypatch.setattr(subprocess, "run", timeout)
    run_curobo_candidate_process(
        request,
        result,
        process_timeout_seconds=0.01,
    )
    candidate = load_single_candidate_result(
        result,
        structured_expert_config_sha256="c" * 64,
    )
    assert candidate.status is PlannerCandidateStatus.TIMEOUT
    assert candidate.candidate_index == 0


def test_candidate_timestamps_bind_the_frozen_arrival_schedule() -> None:
    """Break caught: CuRobo's fastest path time replaces the sampled episode timing."""
    from latency_meta_mdp.expert_realization.planner import schedule_candidate_timestamps

    raw = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    nominal = schedule_candidate_timestamps(
        raw,
        arrival_duration_seconds=1.6,
        time_scaling_profile="nominal",
    )
    smooth = schedule_candidate_timestamps(
        raw,
        arrival_duration_seconds=1.6,
        time_scaling_profile="minimum_jerk_slow",
    )
    cubic = schedule_candidate_timestamps(
        raw,
        arrival_duration_seconds=1.6,
        time_scaling_profile="early_smooth",
    )

    assert np.allclose(nominal, [0.0, 0.4, 0.8, 1.2, 1.6], atol=1.0e-12)
    assert np.allclose(
        smooth,
        [0.0, 0.5750978636635343, 0.8, 1.0249021363364657, 1.6],
        atol=1.0e-9,
    )
    assert np.allclose(
        cubic,
        [0.0, 0.5221629157329114, 0.8, 1.0778370842670886, 1.6],
        atol=1.0e-9,
    )
    with pytest.raises(ValueError, match="positive"):
        schedule_candidate_timestamps(
            raw,
            arrival_duration_seconds=0.0,
            time_scaling_profile="nominal",
        )


def test_retimed_joint_path_must_still_respect_planning_limits() -> None:
    """Break caught: arrival retiming creates an infeasible joint-space reference."""
    from latency_meta_mdp.expert_realization.planner import validate_scheduled_joint_path

    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    lower = np.full(7, -1.0, dtype=np.float64)
    upper = np.full(7, 1.0, dtype=np.float64)
    velocity = np.full(7, 1.0, dtype=np.float64)
    acceleration = np.full(7, 5.0, dtype=np.float64)
    valid = np.zeros((3, 7), dtype=np.float64)
    valid[:, 0] = [0.0, 0.05, 0.1]
    validate_scheduled_joint_path(
        valid,
        timestamps,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity=velocity,
        joint_acceleration=acceleration,
    )

    too_fast = np.array(valid, copy=True)
    too_fast[:, 0] = [0.0, 0.2, 0.4]
    with pytest.raises(ValueError, match="velocity"):
        validate_scheduled_joint_path(
            too_fast,
            timestamps,
            joint_lower=lower,
            joint_upper=upper,
            joint_velocity=velocity,
            joint_acceleration=acceleration,
        )


def test_planner_request_serializes_curve_and_funnel_intent_not_hard_segments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: the planner protocol still exposes an ordered stop-and-go plan."""
    from expert_realization_test_support import (
        make_detached_task_instance,
        make_shared_prefix_anchor,
        strategy_config_and_keys,
    )
    from test_expert_realization_robot_bridge import _bridge

    from latency_meta_mdp.expert_realization.contracts import StrategyFamily
    from latency_meta_mdp.expert_realization.planner import write_planner_request
    from latency_meta_mdp.expert_realization.strategy import sample_strategy
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    monkeypatch.setattr(
        MaterializedTaskInstance,
        "validate_publication_consistency",
        lambda self: None,
    )
    instance = make_detached_task_instance()
    config, keys = strategy_config_and_keys(instance)
    intent = build_trajectory_intent(
        instance,
        make_shared_prefix_anchor(),
        sample_strategy(
            instance,
            keys[4],
            config,
            assigned_family=StrategyFamily.LATERAL_ARC,
        ),
    )
    path = tmp_path / "request.json"
    write_planner_request(
        expert_realization_key=keys[4],
        structured_expert_config_sha256=config.source_sha256,
        candidate_index=0,
        bridge=_bridge(),
        intent=intent,
        start_qpos=np.zeros(7, dtype=np.float64),
        timeout_seconds=5.0,
        path=path,
    )
    payload = json.loads(path.read_text())

    assert "plan" not in payload
    assert set(payload["intent"]) == {
        "task_instance_id",
        "family",
        "reference_start_tick",
        "funnel_entry_target_tick",
        "funnel_entry_deadline_tick",
        "time_scaling_profile",
        "soft_guide_regions_world",
        "soft_guide_radius_m",
        "funnel_entry_position_world",
        "funnel_entry_tangent_world",
        "fixed_orientation_world",
    }


def test_worker_parser_preserves_empty_soft_guide_matrix_shape() -> None:
    """Break caught: JSON [] is interpreted as shape [0] instead of an empty [0,3] matrix."""
    from latency_meta_mdp.expert_realization.curobo_worker import _parse_approach_geometry

    guides, radii, entry, tangent = _parse_approach_geometry(
        {
            "soft_guide_regions_world": [],
            "soft_guide_radius_m": [],
            "funnel_entry_position_world": [0.1, 0.2, 1.0],
            "funnel_entry_tangent_world": [0.0, 0.0, -1.0],
        }
    )

    assert guides.shape == (0, 3)
    assert radii.shape == (0,)
    assert entry.shape == (3,)
    assert tangent.shape == (3,)


def test_planner_timeout_is_per_pose_invocation_not_sum_over_guides() -> None:
    """Break caught: a three-target approach is rejected because total planning exceeds 5 s."""
    from latency_meta_mdp.expert_realization.curobo_worker import (
        _planner_invocation_timeout_reason,
    )

    assert _planner_invocation_timeout_reason((4.2, 4.1, 4.3), 5.0) is None
    assert _planner_invocation_timeout_reason((4.2, 5.1, 4.3), 5.0) == (
        "CuRobo pose invocation 1 exceeded 5.0 seconds"
    )
