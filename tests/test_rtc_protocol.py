from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def _plan(**updates):
    from latency_meta_mdp.rtc_protocol import TimedActionPlan

    args = dict(
        origin_tick=100,
        actions=np.repeat(np.arange(50, dtype=np.float32)[:, None] / 100, 7, axis=1),
        valid_mask=np.ones(50, dtype=bool),
        request_id=2,
        buffer_version=3,
    )
    return TimedActionPlan(**(args | updates))


@pytest.mark.parametrize(("tick", "length", "first"), [(104, 46, 0.04), (120, 30, 0.2)])
def test_plan_slices_expired_actions_by_observation_time(tick, length, first):
    plan = _plan()
    result = plan.remaining_from(formal_tick=tick)
    assert result.shape == (length, 7)
    np.testing.assert_allclose(result[0], first)
    assert not result.flags.writeable


def test_plan_padding_is_not_executable_and_input_mutation_cannot_change_it():
    actions = np.zeros((50, 7))
    plan = _plan(actions=actions, valid_mask=np.arange(50) < 6)
    actions[:] = 0.8
    np.testing.assert_array_equal(plan.remaining_from(formal_tick=104), np.zeros((2, 7)))
    assert plan.remaining_from(formal_tick=106).shape == (0, 7)
    assert plan.remaining_from(formal_tick=200).shape == (0, 7)
    with pytest.raises(ValueError):
        plan.remaining_from(formal_tick=99)


@pytest.mark.parametrize(
    "updates",
    [
        {"protocol_id": "sharp_return_time_chunk_v2"},
        {"origin_tick": True},
        {"request_id": -1},
        {"buffer_version": -1},
        {"actions": np.zeros((50, 32))},
        {"actions": np.full((50, 7), np.nan)},
        {"valid_mask": np.arange(50) != 4},
        {"valid_mask": np.zeros(50, dtype=bool)},
        {"valid_mask": np.ones(50, dtype=int)},
    ],
)
def test_plan_rejects_wrong_alignment_shapes_and_noncontiguous_validity(updates):
    with pytest.raises((TypeError, ValueError)):
        _plan(**updates)


def test_history_uses_only_completed_observed_delays_and_evicts_old_values():
    from latency_meta_mdp.rtc_protocol import RollingDelayHistory

    history = RollingDelayHistory(capacity=2, initial_delays=(4, 8))
    assert history.estimate_ticks() == 8
    history.record_completed(request_id=0, origin_tick=100, completion_tick=103)
    assert history.estimate_ticks() == 8
    history.record_completed(request_id=1, origin_tick=103, completion_tick=108)
    assert history.delays == (3, 5)
    assert history.estimate_ticks() == 5
    with pytest.raises(ValueError):
        history.record_completed(request_id=1, origin_tick=108, completion_tick=110)
    assert history.delays == (3, 5)
    with pytest.raises(ValueError):
        history.record_completed(request_id=2, origin_tick=111, completion_tick=110)
    assert history.delays == (3, 5)
    history.record_completed(request_id=2, origin_tick=108, completion_tick=108)
    assert history.delays == (5, 0)


def test_rtc_config_rejects_incompatible_protocol_and_unusable_history():
    from latency_meta_mdp.rtc_protocol import load_rtc_client_config

    config = load_rtc_client_config(Path("configs/client/rtc_observation_time_h50_v1.yaml"))
    assert config.prediction_horizon == 50
    assert config.maximum_delay_ticks == 20
    with pytest.raises(ValueError):
        replace(config, initial_delay_ticks=())
    with pytest.raises(ValueError):
        replace(config, delay_history_capacity=0)
    with pytest.raises(ValueError):
        replace(config, protocol_id="sharp_return_time_chunk_v2")
