from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _task_instance(*, level: int = 1, seed: int = 4000):
    from latency_meta_mdp.expert_realization.contracts import TaskInstanceId

    return TaskInstanceId(
        level=level,
        task_instance_seed=seed,
        motion_profile_sha256=SHA_A,
        initial_state_sha256=SHA_B,
    )


def _structured_config():
    from latency_meta_mdp.expert_realization.config import load_structured_expert_config

    return load_structured_expert_config(
        PROJECT_ROOT / "configs/expert_realization/panda_ball_structured.yaml"
    )


def _pilot_request():
    from latency_meta_mdp.expert_realization.config import load_pilot_config
    from latency_meta_mdp.expert_realization.contracts import PilotRequest

    config = load_pilot_config(PROJECT_ROOT / "configs/collection/panda_ball_structured_pilot.yaml")
    return PilotRequest(
        pilot_config=config,
        structured_expert_config_sha256=SHA_C,
        curobo_planner_config_sha256=SHA_D,
        pilot_gate_config_sha256="e" * 64,
        task_instances=tuple(
            _task_instance(level=level, seed=seed)
            for level in config.levels
            for seed in config.seeds
        ),
    )


def test_seed_derivation_has_hand_checked_stable_tagged_values() -> None:
    """Break caught: changing canonical serialization or tag mixing changes replay semantics."""
    from latency_meta_mdp.expert_realization.contracts import (
        derive_realization_seed,
        derive_subseed,
    )

    seed = derive_realization_seed(
        _task_instance(), realization_index=0, realization_namespace_sha256=SHA_C
    )

    assert seed == 13654647031461270349
    assert derive_subseed(seed, "strategy") == 5825351470875554942
    assert derive_subseed(seed, "trajectory_intent") == 5583200187228458480
    assert derive_subseed(seed, "planner") == 13716612849502852106
    assert derive_subseed(seed, "timing") == 12676960786737496095


def test_subseeds_are_tag_independent_and_worker_order_independent() -> None:
    """Break caught: worker scheduling or a tag collision changes a realization's parameters."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
    )

    task = _task_instance()
    first = ExpertRealizationKey(task, 0, SHA_C)
    second = ExpertRealizationKey(task, 1, SHA_C)

    assert first.subseeds()["strategy"] != first.subseeds()["trajectory_intent"]
    assert [key.subseeds() for key in (first, second)] == [
        key.subseeds() for key in (second, first)
    ][::-1]


def test_strategy_sampling_is_not_affected_by_process_global_rng_state() -> None:
    """Break caught: external global random calls perturb sampled semantic strategy values."""
    from latency_meta_mdp.expert_realization.contracts import sample_strategy_parameters

    config = _structured_config()
    before = sample_strategy_parameters(
        config, family="lateral_arc", realization_seed=6702884092325032541
    )
    random.seed(9234)
    _ = [random.random() for _ in range(100)]
    np.random.seed(9234)
    _ = np.random.random(100)
    after = sample_strategy_parameters(
        config, family="lateral_arc", realization_seed=6702884092325032541
    )

    assert after == before
    assert 90 <= before.close_target_tick <= 104
    assert 0.14 <= before.prediction_lead_seconds <= 0.26
    assert before.lateral_offset_m is not None
    assert 0.02 <= before.lateral_offset_m <= 0.06
    assert before.to_mapping()["family"] == "lateral_arc"


def test_realization_identity_requires_completed_task_instance_plan_set() -> None:
    """Break caught: a result becomes identifiable before task-plan provenance exists."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationId,
        ExpertRealizationKey,
    )

    task = _task_instance()
    key = ExpertRealizationKey(task, 0, SHA_C)

    with pytest.raises(ValueError):
        ExpertRealizationId(key, "")
    identity = ExpertRealizationId(key, SHA_D)
    assert identity.to_mapping()["task_instance_plan_set_sha256"] == SHA_D


def test_attempt_identity_adds_retries_without_changing_semantic_identity() -> None:
    """Break caught: retry number is accidentally treated as a new semantic realization."""
    from latency_meta_mdp.expert_realization.contracts import (
        AttemptId,
        ExpertRealizationId,
        ExpertRealizationKey,
    )

    task = _task_instance()
    key = ExpertRealizationKey(task, 0, SHA_C)
    realization = ExpertRealizationId(key, SHA_D)
    first = AttemptId(realization, 0)
    retry = AttemptId(realization, 1)

    assert first.expert_realization_id == retry.expert_realization_id
    assert first.semantic_identity() == retry.semantic_identity()
    assert first.to_mapping()["attempt_index"] == 0
    assert retry.to_mapping()["attempt_index"] == 1


def test_realization_and_attempt_ids_round_trip_with_bound_config_hash() -> None:
    """Break caught: loaders accept identity JSON without recomputing the nested key seed."""
    from latency_meta_mdp.expert_realization.contracts import (
        AttemptId,
        ExpertRealizationId,
        ExpertRealizationKey,
    )

    realization = ExpertRealizationId(ExpertRealizationKey(_task_instance(), 0, SHA_C), SHA_D)
    attempt = AttemptId(realization, 2)

    assert (
        ExpertRealizationId.from_mapping(
            realization.to_mapping(), structured_expert_config_sha256=SHA_C
        )
        == realization
    )
    assert (
        AttemptId.from_mapping(attempt.to_mapping(), structured_expert_config_sha256=SHA_C)
        == attempt
    )


def test_realization_and_attempt_id_mappings_reject_all_stored_identity_corruption() -> None:
    """Break caught: an outer ID trusts altered nested fields or ignores extra serialized data."""
    from latency_meta_mdp.expert_realization.contracts import (
        AttemptId,
        ExpertRealizationId,
        ExpertRealizationKey,
    )

    realization = ExpertRealizationId(ExpertRealizationKey(_task_instance(), 0, SHA_C), SHA_D)
    attempt = AttemptId(realization, 2)
    bad_realization = realization.to_mapping()
    bad_realization["expert_realization_key"]["realization_seed"] ^= 1
    with pytest.raises(ValueError):
        ExpertRealizationId.from_mapping(bad_realization, structured_expert_config_sha256=SHA_C)
    with pytest.raises(ValueError):
        ExpertRealizationId.from_mapping(
            {**realization.to_mapping(), "extra": 1},
            structured_expert_config_sha256=SHA_C,
        )
    bad_attempt = attempt.to_mapping()
    bad_attempt["expert_realization_id"]["task_instance_plan_set_sha256"] = "e" * 63
    with pytest.raises(ValueError):
        AttemptId.from_mapping(bad_attempt, structured_expert_config_sha256=SHA_C)
    with pytest.raises(ValueError):
        AttemptId.from_mapping(
            {**attempt.to_mapping(), "extra": 1}, structured_expert_config_sha256=SHA_C
        )


def test_failure_classes_remain_distinct_serialized_categories() -> None:
    """Break caught: a failure category is merged, losing its retry/triage semantics."""
    from latency_meta_mdp.expert_realization.contracts import FailureClass

    assert {failure.value for failure in FailureClass} == {
        "planner_failure",
        "task_failure",
        "safety_failure",
        "diversity_rejection",
        "infrastructure_failure",
    }


def test_pilot_request_expands_complete_review_only_identity_universe() -> None:
    """Break caught: pilot scope omits semantic keys or accidentally permits training use."""
    request = _pilot_request()

    assert len(request.realization_keys()) == 480
    assert request.pilot_id == "panda-ball-structured-paired-4000-4019-v1"
    assert request.bounded_review_only is True
    assert request.training_eligibility() == {
        "formal_training_authorized": False,
        "formal_dino_cache_authorized": False,
        "jepa_training_authorized": False,
        "policy_training_authorized": False,
        "meta_policy_training_authorized": False,
    }
    assert "task_instance_plan_set_sha256" not in request.to_mapping()


def test_pilot_request_rejects_permuted_task_instances_and_serializes_parent_order() -> None:
    """Break caught: worker order changes the parent pilot's canonical request identity."""
    from latency_meta_mdp.expert_realization.config import load_pilot_config
    from latency_meta_mdp.expert_realization.contracts import PilotRequest

    request = _pilot_request()
    config = load_pilot_config(PROJECT_ROOT / "configs/collection/panda_ball_structured_pilot.yaml")

    assert request.task_instances[0].to_mapping()["level"] == 1
    assert request.task_instances[0].to_mapping()["task_instance_seed"] == 4000
    assert request.canonical_json() == _pilot_request().canonical_json()
    with pytest.raises(ValueError):
        PilotRequest(
            pilot_config=config,
            structured_expert_config_sha256=SHA_C,
            curobo_planner_config_sha256=SHA_D,
            pilot_gate_config_sha256="e" * 64,
            task_instances=tuple(reversed(request.task_instances)),
        )


def test_realization_key_recomputes_a_request_bound_uint64_seed() -> None:
    """Break caught: caller-supplied key fields no longer describe the deterministic realization."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        TaskInstanceId,
        derive_realization_seed,
    )

    task = _task_instance()
    key = ExpertRealizationKey(task, 0, SHA_C)

    assert key.realization_seed == derive_realization_seed(task, 0, SHA_C)
    with pytest.raises(ValueError):
        TaskInstanceId(1, -1, SHA_A, SHA_B)
    with pytest.raises(ValueError):
        ExpertRealizationKey(task, -1, SHA_C)
    assert ExpertRealizationKey(task, 12, SHA_C).realization_index == 12
    assert derive_realization_seed(task, 12, SHA_C) >= 0
    with pytest.raises(TypeError):
        ExpertRealizationKey(task, 0, SHA_C, realization_seed=0)


def test_task_instance_and_realization_key_round_trip_through_strict_mappings() -> None:
    """Break caught: serialized identities load without validating their complete factual seed."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        TaskInstanceId,
    )

    task = _task_instance()
    key = ExpertRealizationKey(task, 0, SHA_C)

    assert TaskInstanceId.from_mapping(task.to_mapping()) == task
    assert (
        ExpertRealizationKey.from_mapping(key.to_mapping())
        == key
    )


def test_realization_key_mapping_rejects_corruption_and_noncanonical_shapes() -> None:
    """Break caught: an artifact can alter the stored seed or nested task identity unnoticed."""
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey, TaskInstanceId

    key = ExpertRealizationKey(_task_instance(), 0, SHA_C)
    corrupted = key.to_mapping()
    corrupted["realization_seed"] ^= 1
    with pytest.raises(ValueError):
        ExpertRealizationKey.from_mapping(corrupted)
    with pytest.raises(ValueError):
        ExpertRealizationKey.from_mapping({**key.to_mapping(), "unknown": "field"})
    missing = key.to_mapping()
    del missing["realization_seed"]
    with pytest.raises(ValueError):
        ExpertRealizationKey.from_mapping(missing)
    malformed = _task_instance().to_mapping()
    malformed["task_instance_seed"] = "4000"
    with pytest.raises(ValueError):
        TaskInstanceId.from_mapping(malformed)


def _hand_dwell_choice(realization_seed: int, tag: str) -> int:
    tag_payload = json.dumps(
        {"realization_seed": realization_seed, "tag": tag},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tag_seed = int.from_bytes(hashlib.sha256(tag_payload).digest()[:8], "big")
    dwell_payload = json.dumps(
        {"field": "close_dwell_ticks", "seed": tag_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(dwell_payload).digest()[:8], "big") % 5


def test_close_target_uses_the_timing_subseed_while_close_dwell_is_canonical() -> None:
    """Break caught: capture timing and canonical grasp semantics share one random variable."""
    from latency_meta_mdp.expert_realization.contracts import sample_strategy_parameters

    parameters = sample_strategy_parameters(
        _structured_config(), family="canonical_direct", realization_seed=0
    )

    assert parameters.close_target_tick == 91
    assert parameters.close_dwell_ticks == 2
    assert parameters.bilateral_contact_acquisition_ticks == 4


def test_direct_strategy_parameters_must_obey_the_bound_config() -> None:
    """Break caught: an artifact caller serializes a physically impossible sampled strategy."""
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily, StrategyParameters

    config = _structured_config()
    with pytest.raises(ValueError):
        StrategyParameters(
            family=StrategyFamily.CANONICAL_DIRECT,
            close_target_tick=-999,
            prediction_lead_seconds=-1.0,
            funnel_entry_height_m=9.0,
            high_arc_extra_height_m=9.0,
            lateral_offset_m=None,
            lateral_direction_sign=None,
            soft_guide_radius_m=9.0,
            tracking_error_clip_m=9.0,
            grasp_eef_height_offset_m=9.0,
            funnel_descent_ticks=999,
            funnel_entry_deadline_slack_ticks=999,
            close_window_half_width_ticks=999,
            handoff_window_ticks=999,
            close_dwell_ticks=999,
            bilateral_contact_acquisition_ticks=999,
            close_centering_tolerance_m=9.0,
            close_distance_tolerance_m=9.0,
            close_relative_speed_tolerance_mps=9.0,
            lift_vertical_displacement_m=9.0,
            fixed_orientation=False,
            rotation_action_variation=True,
            iid_per_tick_action_noise=True,
            config=config,
        )


def test_stage_is_constrained_to_parent_scope_and_has_no_training_override() -> None:
    """Break caught: a stage silently changes its parent's sample scope or authorization."""
    from latency_meta_mdp.expert_realization.contracts import PilotStageRequest

    request = _pilot_request()
    stage = PilotStageRequest(
        parent=request,
        stage_id="pilot-l1-first-half",
        task_instance_seed_start=4000,
        task_instance_seed_stop=4010,
        levels=(1,),
        output_root="/tmp/pilot-l1-first-half",
        resource_report_sha256="1" * 64,
        planner_worker_count=1,
        collection_worker_count=1,
    )

    assert stage.parent_pilot_id == request.pilot_id
    assert stage.training_eligibility() == request.training_eligibility()
    with pytest.raises(ValueError):
        PilotStageRequest(
            parent=request,
            stage_id="invalid",
            task_instance_seed_start=3999,
            task_instance_seed_stop=4001,
            levels=(1,),
            output_root="/tmp/invalid",
            resource_report_sha256="1" * 64,
            planner_worker_count=1,
            collection_worker_count=1,
        )
