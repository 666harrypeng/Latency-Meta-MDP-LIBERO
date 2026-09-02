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


def test_strategy_sampling_has_two_realizations_per_family_and_exact_bounds() -> None:
    """Break caught: canonical index mapping or bounded episode-level diversity drifts."""
    from latency_meta_mdp.expert_realization.strategy import sample_strategy

    instance = _task_instance()
    config = _strategy_config()
    strategies = [sample_strategy(instance, key, config) for key in _keys(instance, config)]

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
        bounds = config.expert.interception_tick_ranges[strategy.family.value]
        assert bounds[0] <= strategy.interception_tick <= bounds[1]
        assert 0.14 <= strategy.interception_lead_seconds <= 0.26
        assert 0.08 <= strategy.pregrasp_height_m <= 0.13
        assert 0.022 <= strategy.tracking_error_clip_m <= 0.038
        assert strategy.close_dwell_ticks in (0, 1, 2, 3, 4)
        assert 0.0 <= strategy.lift_lateral_offset_m <= 0.02
        assert 0.14 <= strategy.lift_vertical_offset_m <= 0.19
        assert strategy.lift_lateral_direction_sign in (-1, 1)
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
    expected = sample_strategy(instance, key, config)

    random.seed(91)
    _ = [random.random() for _ in range(100)]
    np.random.seed(91)
    _ = np.random.random(100)

    assert sample_strategy(instance, key, config) == expected
    mapping = expected.to_mapping()
    assert mapping["lateral_direction_sign"] in (-1, 1)
    assert mapping["lift_lateral_direction_sign"] in (-1, 1)
    assert mapping["iid_per_tick_action_noise"] is False


def test_strategy_rejects_key_from_another_task_or_config() -> None:
    """Break caught: a realization key can be reinterpreted under another task/config."""
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
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
        sample_strategy(instance, wrong_task_key, config)

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
        sample_strategy(instance, wrong_key, config)
