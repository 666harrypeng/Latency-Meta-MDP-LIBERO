import dataclasses

import numpy as np
import pytest

from latency_meta_mdp.policy_data import PolicyEpisode


def _episode():
    return PolicyEpisode(
        episode_id="episode",
        task_id="ball",
        instruction="Grasp the ball.",
        level=3,
        action_horizon=50,
        source_formal_tick=np.arange(24),
        source_time_us=np.arange(24) * 20000,
        agentview_rgb=np.zeros((24, 8, 8, 3), np.uint8),
        wrist_rgb=np.ones((24, 8, 8, 3), np.uint8),
        state=np.zeros((24, 16), np.float32),
        actions=np.repeat(np.arange(24, dtype=np.float32)[:, None] / 30, 7, axis=1),
        valid_action_chunk_sources=np.arange(24),
        state_contract="joint_qpos_qvel_gripper_width_velocity",
    )


def test_last_real_source_retains_target_without_inventing_forecast_controls():
    from latency_meta_mdp.policy_forecast import materialize_forecast_policy_source

    ep = _episode()
    sample = materialize_forecast_policy_source(ep, source_tick=23, query_ticks=20)
    assert sample.history_ticks == (15, 19, 23)
    assert sample.recorded_control_mask.sum() == 1
    assert sample.gt_endpoint_recorded is False
    assert sample.executable_controls is None
    assert not sample.forecast_ready
    assert sample.policy_inputs["forecast"]["rgb"] is None
    with pytest.raises(ValueError, match="lacks real"):
        sample.with_forecast(rgb=None, proprio=None, source_tick=23, target_tick=43)
    assert (~sample.policy_inputs["actions_is_pad"]).sum() == 1
    np.testing.assert_array_equal(sample.policy_inputs["actions"][1:], 0)
    short = materialize_forecast_policy_source(ep, source_tick=23, query_ticks=1)
    assert short.forecast_ready
    with pytest.raises(ValueError, match="identity"):
        short.with_forecast(rgb=None, proprio=None, source_tick=22, target_tick=23)
    np.testing.assert_array_equal(short.executable_controls[0], ep.actions[23])
    assert not short.executable_controls.flags.writeable


def test_query_changes_forecast_horizon_not_expert_target_origin():
    from latency_meta_mdp.policy_forecast import materialize_forecast_policy_source

    ep = _episode()
    short = materialize_forecast_policy_source(ep, source_tick=10, query_ticks=1)
    long = materialize_forecast_policy_source(ep, source_tick=10, query_ticks=20)
    np.testing.assert_array_equal(short.policy_inputs["actions"], long.policy_inputs["actions"])
    np.testing.assert_array_equal(short.executed_controls, ep.actions[2:10].reshape(2, 4, 7))
    assert short.gt_endpoint_recorded and not long.gt_endpoint_recorded
    for tick in range(24):
        sample = materialize_forecast_policy_source(ep, source_tick=tick, query_ticks=20)
        assert not sample.policy_inputs["actions_is_pad"][0]
        assert (sample.history_ticks is None) == (tick < 10)


def test_source_rejects_legacy_state_and_invalid_queries():
    from latency_meta_mdp.policy_forecast import materialize_forecast_policy_source

    with pytest.raises(ValueError, match="16D"):
        materialize_forecast_policy_source(
            dataclasses.replace(_episode(), state=np.zeros((24, 8))), source_tick=10, query_ticks=4
        )
    for q in (-1, 21, True, 1.5):
        with pytest.raises(ValueError, match="query"):
            materialize_forecast_policy_source(_episode(), source_tick=10, query_ticks=q)


def test_direct_query_accepts_declared_tail_buffer_without_inventing_targets():
    from types import SimpleNamespace

    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
        materialize_direct_query,
        materialize_direct_sample,
    )
    from latency_meta_mdp.policy_forecast import materialize_forecast_policy_source

    ep = _episode()
    record = SimpleNamespace(
        level=3,
        terminal_tick=24,
        controls=ep.actions,
        proprio_physical=np.zeros((25, 16), np.float32),
        cache=SimpleNamespace(features=np.zeros((25, 2, 196, 384), np.float16)),
    )
    norm = SimpleNamespace(level=3, normalize=lambda x: x)
    source = materialize_forecast_policy_source(ep, source_tick=23, query_ticks=20)
    assert not source.forecast_ready
    # An actual deployment plan may extend beyond this recording. It is supplied
    # explicitly here, unlike constructing invented controls in the source view.
    deployment_buffer = np.full((20, 7), 0.2, np.float32)
    query = materialize_direct_query(
        record,
        source_tick=23,
        query_ticks=20,
        normalization=norm,
        executable_controls=deployment_buffer,
    )
    assert query.query_ticks.item() == 20
    assert query.control_mask.sum().item() == 20
    torch.testing.assert_close(query.executable_controls[0], torch.tensor(deployment_buffer))
    with pytest.raises(ValueError, match="real history, controls and future endpoint"):
        materialize_direct_sample(record, source_tick=23, query_ticks=20, normalization=norm)


def test_clocked_history_rejects_gaps_and_resets_between_episodes():
    from pathlib import Path

    import torch
    from test_action_conditioned_jepa_rollout import _normalization

    from latency_meta_mdp.belief.action_conditioned_jepa.config import load_jepa_temporal_sampling
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    history = JepaRuntimeHistory(
        _normalization(),
        temporal_sampling=load_jepa_temporal_sampling(
            Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
        ),
    )
    visual = torch.zeros(2, 196, 384, dtype=torch.float16)
    proprio = torch.zeros(16)
    action = torch.zeros(7)
    for tick in range(11):
        history.append_boundary(
            vision_features=visual,
            proprio=proprio,
            executed_control_from_previous=action if tick else None,
            formal_tick=tick,
            episode_id="a",
        )
    query = history.build_forecast_query(torch.zeros(20, 7), query_ticks=7)
    assert query.source_ticks.item() == 10 and query.query_ticks.item() == 7
    assert query.control_mask.sum().item() == 7
    with pytest.raises(ValueError, match="consecutive"):
        history.append_boundary(
            vision_features=visual,
            proprio=proprio,
            executed_control_from_previous=action,
            formal_tick=12,
            episode_id="a",
        )
    with pytest.raises(ValueError, match="episode"):
        history.append_boundary(
            vision_features=visual,
            proprio=proprio,
            executed_control_from_previous=action,
            formal_tick=11,
            episode_id="b",
        )
    history.reset()
    assert not history.ready
    history.append_boundary(
        vision_features=visual,
        proprio=proprio,
        executed_control_from_previous=None,
        formal_tick=0,
        episode_id="b",
    )
    with pytest.raises(RuntimeError, match="history"):
        history.build_forecast_query(torch.zeros(20, 7), query_ticks=7)
