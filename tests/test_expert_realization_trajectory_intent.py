from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
from expert_realization_test_support import (
    make_detached_task_instance,
    make_shared_prefix_anchor,
    strategy_config_and_keys,
)

_anchor = make_shared_prefix_anchor
_config_and_keys = strategy_config_and_keys
_task_instance = make_detached_task_instance


def _sample(instance, key, config):
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    families = tuple(family for family in StrategyFamily for _ in range(2))
    return sample_strategy(
        instance,
        key,
        config,
        assigned_family=families[key.realization_index],
    )


@pytest.fixture(autouse=True)
def _validated_task_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    monkeypatch.setattr(
        MaterializedTaskInstance,
        "validate_publication_consistency",
        lambda self: None,
    )


def test_intent_is_one_curve_followed_by_one_canonical_funnel() -> None:
    """Break caught: semantic approach regions become executable stop-and-go segments."""
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    instance = _task_instance(level=3)
    anchor = _anchor()
    config, keys = _config_and_keys(instance)
    intents = tuple(
        build_trajectory_intent(instance, anchor, _sample(instance, key, config))
        for key in keys
    )

    assert [intent.family.value for intent in intents] == [
        "canonical_direct",
        "canonical_direct",
        "early_high_arc",
        "early_high_arc",
        "lateral_arc",
        "lateral_arc",
        "time_shifted_smooth",
        "time_shifted_smooth",
    ]
    for intent in intents:
        curve = intent.approach
        funnel = intent.grasp_funnel
        assert "segments" not in {field.name for field in fields(curve)}
        assert curve.soft_guide_regions_world.shape[0] <= 2
        assert curve.soft_guide_regions_world.shape[1:] == (3,)
        assert curve.soft_guide_radius_m.shape == (len(curve.soft_guide_regions_world),)
        assert 5 < curve.funnel_entry_target_tick <= curve.funnel_entry_deadline_tick
        assert (
            curve.funnel_entry_deadline_tick
            + intent.strategy.funnel_descent_ticks
            + funnel.close_dwell_ticks
            < funnel.handoff_deadline_tick
        )
        assert funnel.close_earliest_tick <= funnel.close_target_tick <= funnel.close_deadline_tick
        assert funnel.close_deadline_tick < funnel.handoff_deadline_tick < 150
        assert funnel.handoff_deadline_tick == min(
            funnel.close_target_tick + intent.strategy.handoff_window_ticks,
            149,
        )
        assert np.isclose(np.linalg.norm(curve.funnel_entry_tangent_world), 1.0)
        assert abs(curve.funnel_entry_tangent_world[2]) < 1.0e-12
        assert np.array_equal(curve.fixed_orientation_world, funnel.fixed_orientation_world)
        np.testing.assert_array_equal(
            funnel.centered_pad_axis_world,
            curve.fixed_orientation_world[:, 0],
        )
        assert funnel.requires_physical_handoff is True
        assert funnel.symmetric_close_command is True
        assert funnel.centering_tolerance_m == 0.004
        assert funnel.distance_tolerance_m == 0.016
        assert funnel.relative_speed_tolerance_mps == 0.13
        assert funnel.lift_relative_displacement_world.tolist() == [0.0, 0.0, 0.16]
        assert not curve.soft_guide_regions_world.flags.writeable
        assert not curve.funnel_entry_position_world.flags.writeable

    assert intents[0].approach.soft_guide_regions_world.shape == (0, 3)
    assert intents[2].approach.soft_guide_regions_world.shape == (1, 3)
    assert intents[4].approach.soft_guide_regions_world.shape == (1, 3)
    assert intents[6].approach.soft_guide_regions_world.shape == (0, 3)


def test_all_families_share_identical_task_specific_funnel_geometry() -> None:
    """Break caught: diversity leaks into close/lift instead of staying in approach."""
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    instance = _task_instance(level=2)
    anchor = _anchor()
    config, keys = _config_and_keys(instance)
    funnels = tuple(
        build_trajectory_intent(instance, anchor, _sample(instance, key, config)).grasp_funnel
        for key in keys
    )

    for funnel in funnels[1:]:
        np.testing.assert_array_equal(
            funnel.ball_relative_entry_offset_world,
            funnels[0].ball_relative_entry_offset_world,
        )
        np.testing.assert_array_equal(
            funnel.centered_pad_axis_world,
            funnels[0].centered_pad_axis_world,
        )
        np.testing.assert_array_equal(
            funnel.lift_relative_displacement_world,
            funnels[0].lift_relative_displacement_world,
        )
        assert funnel.close_dwell_ticks == funnels[0].close_dwell_ticks
        assert (
            funnel.bilateral_contact_acquisition_ticks
            == funnels[0].bilateral_contact_acquisition_ticks
        )


def test_trajectory_intent_rejects_an_anchor_from_another_task() -> None:
    """Break caught: a smooth plan is built from a valid but different K6 source state."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.task_instance import TaskInstanceReplayMismatch
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    instance = _task_instance()
    config, keys = _config_and_keys(instance)
    wrong_anchor = replace(
        instance.expected_anchor,
        anchor_robot_qpos=np.ones(7, dtype=np.float64),
    )

    with pytest.raises(TaskInstanceReplayMismatch, match="anchor.robot_qpos"):
        build_trajectory_intent(
            instance,
            wrong_anchor,
            _sample(instance, keys[0], config),
        )


def test_prediction_lead_cannot_set_funnel_entry_timing() -> None:
    """Break caught: the causal prediction horizon is reused as a phase clock."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    instance = _task_instance()
    anchor = _anchor()
    config, keys = _config_and_keys(instance)
    original = _sample(instance, keys[0], config)
    changed = replace(
        original,
        prediction_lead_seconds=0.25,
        config=config.expert,
    )

    original_intent = build_trajectory_intent(instance, anchor, original)
    changed_intent = build_trajectory_intent(instance, anchor, changed)

    assert (
        original_intent.approach.funnel_entry_target_tick
        == changed_intent.approach.funnel_entry_target_tick
    )
    assert (
        original_intent.approach.funnel_entry_deadline_tick
        == changed_intent.approach.funnel_entry_deadline_tick
    )
