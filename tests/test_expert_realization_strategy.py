from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest


def _task_instance(level: int = 1, seed: int = 4000):
    from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    instance = object.__new__(MaterializedTaskInstance)
    object.__setattr__(
        instance,
        "task_instance_id",
        TaskInstanceId(level, seed, "a" * 64, "b" * 64),
    )
    return instance


def _strategy_config():
    from latency_meta_mdp.expert_realization.strategy import StructuredStrategyConfig

    return StructuredStrategyConfig.from_path(
        Path.cwd() / "configs/expert_realization/panda_ball_structured.yaml"
    )


def _keys(instance: object, config: object):
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey

    return tuple(
        ExpertRealizationKey(
            instance.task_instance_id,
            index,
            config.source_sha256,
        )
        for index in range(8)
    )


@pytest.fixture(autouse=True)
def _validated_task_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    from latency_meta_mdp.expert_realization.task_instance import MaterializedTaskInstance

    monkeypatch.setattr(
        MaterializedTaskInstance,
        "validate_publication_consistency",
        lambda self: None,
    )


def test_strategy_sampling_has_bounded_approach_diversity_and_canonical_grasp() -> None:
    """Break caught: randomness leaks out of approach intent into grasp or lift."""
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    instance = _task_instance()
    config = _strategy_config()
    families = tuple(family for family in StrategyFamily for _ in range(2))
    strategies = [
        sample_strategy(instance, key, config, assigned_family=family)
        for key, family in zip(_keys(instance, config), families, strict=True)
    ]

    assert [item.family.value for item in strategies] == [
        "canonical_direct",
        "canonical_direct",
        "early_high_arc",
        "early_high_arc",
        "lateral_arc",
        "lateral_arc",
        "time_shifted_smooth",
        "time_shifted_smooth",
    ]
    for strategy in strategies:
        bounds = config.expert.close_target_tick_ranges[strategy.family.value]
        assert bounds[0] <= strategy.close_target_tick <= bounds[1]
        assert 0.14 <= strategy.prediction_lead_seconds <= 0.26
        assert strategy.tracking_error_clip_m == 0.040
        assert strategy.grasp_eef_height_offset_m == 0.005
        assert strategy.funnel_entry_height_m == 0.10
        assert strategy.funnel_descent_ticks == 30
        assert strategy.close_dwell_ticks == 2
        assert strategy.bilateral_contact_acquisition_ticks == 4
        assert strategy.close_centering_tolerance_m == 0.004
        assert strategy.close_distance_tolerance_m == 0.016
        assert strategy.close_relative_speed_tolerance_mps == 0.13
        assert strategy.lift_vertical_displacement_m == 0.16
        if strategy.family.value == "early_high_arc":
            assert strategy.high_arc_extra_height_m is not None
            assert 0.025 <= strategy.high_arc_extra_height_m <= 0.065
        else:
            assert strategy.high_arc_extra_height_m is None
        if strategy.family.value == "lateral_arc":
            assert strategy.lateral_offset_m is not None
            assert strategy.lateral_direction_sign in (-1, 1)
        else:
            assert strategy.lateral_offset_m is None
            assert strategy.lateral_direction_sign is None
        assert strategy.fixed_orientation is True
        assert strategy.rotation_action_variation is False
        assert strategy.iid_per_tick_action_noise is False


def test_strategy_sampling_is_replayable_and_independent_of_global_rng() -> None:
    """Break caught: collection order or process-global RNG changes a realization strategy."""
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    instance = _task_instance()
    config = _strategy_config()
    key = _keys(instance, config)[5]
    from latency_meta_mdp.expert_realization.contracts import StrategyFamily

    expected = sample_strategy(instance, key, config, assigned_family=StrategyFamily.LATERAL_ARC)

    random.seed(91)
    _ = [random.random() for _ in range(100)]
    np.random.seed(91)
    _ = np.random.random(100)

    assert (
        sample_strategy(
            instance,
            key,
            config,
            assigned_family=StrategyFamily.LATERAL_ARC,
        )
        == expected
    )
    mapping = expected.to_mapping()
    assert mapping["lateral_direction_sign"] in (-1, 1)
    assert "lift_lateral_direction_sign" not in mapping
    assert mapping["iid_per_tick_action_noise"] is False


def test_strategy_rejects_key_from_another_task_or_config() -> None:
    """Break caught: a realization key can be reinterpreted under another task/config."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        StrategyFamily,
    )
    from latency_meta_mdp.expert_realization.strategy import (
        StructuredStrategyConfig,
        sample_strategy,
    )

    instance = _task_instance()
    config = _strategy_config()
    wrong_task = _task_instance(seed=4001)
    wrong_task_key = ExpertRealizationKey(
        wrong_task.task_instance_id,
        0,
        config.source_sha256,
    )
    with pytest.raises(ValueError, match="task instance"):
        sample_strategy(
            instance,
            wrong_task_key,
            config,
            assigned_family=StrategyFamily.CANONICAL_DIRECT,
        )
    raw = Path.cwd() / "configs/expert_realization/panda_ball_structured.yaml"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == config.source_sha256
    with pytest.raises(ValueError, match="source_sha256"):
        StructuredStrategyConfig(config.expert, "c" * 64, raw.read_bytes())
    wrong_key = ExpertRealizationKey(
        instance.task_instance_id,
        0,
        "c" * 64,
    )
    with pytest.raises(ValueError, match="config"):
        sample_strategy(
            instance,
            wrong_key,
            config,
            assigned_family=StrategyFamily.CANONICAL_DIRECT,
        )


def test_formal_request_is_the_only_source_of_formal_family_and_strategy_seed() -> None:
    """Break caught: formal collection reconstructs family or seed outside its request universe."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config
    from latency_meta_mdp.expert_realization.contracts import (
        TaskInstanceId,
        build_formal_realization_requests,
        build_formal_request_universe,
    )
    from latency_meta_mdp.expert_realization.strategy import sample_requested_strategy

    config = _strategy_config()
    formal = replace(
        load_formal_corpus_config(Path("configs/collection/panda_ball_structured_formal.yaml")),
        task_instance_count=1,
        realizations_per_task=3,
        reserve_task_instance_count=0,
    )
    universe = build_formal_request_universe(
        formal,
        corpus_config_sha256="a" * 64,
        structured_expert_config_sha256=config.source_sha256,
    )
    task_id = TaskInstanceId(
        level=1,
        task_instance_seed=universe.primary_tasks[0].master_task_seed,
        motion_profile_sha256="b" * 64,
        initial_state_sha256="c" * 64,
    )
    request = build_formal_realization_requests(universe, task_id)[0]
    task = _task_instance(seed=task_id.task_instance_seed)
    object.__setattr__(task, "task_instance_id", task_id)

    strategy = sample_requested_strategy(task, request, config, universe=universe)

    assert strategy.family is request.assigned_family
    with pytest.raises(ValueError, match="universe"):
        sample_requested_strategy(
            task,
            request,
            config,
            universe=replace(universe, structured_expert_config_sha256="d" * 64),
        )
