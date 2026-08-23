from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.temporal_contract import load_temporal_contract

_CONFIG = Path("configs/temporal/h50_e25_d20_k6_v1.yaml")


def test_checked_in_temporal_contract_derives_locked_durations_and_coverage() -> None:
    contract = load_temporal_contract(_CONFIG)

    assert contract.contract_id == "h50_e25_d20_k6_v1"
    assert contract.formal_tick_us == 20_000
    assert contract.control_frequency_hz == 50
    assert contract.prediction_horizon == 50
    assert contract.launch_trigger_horizon == 25
    assert contract.maximum_delay_ticks == 20
    assert contract.history_sample_count == 6
    assert contract.chunk_duration_us == 1_000_000
    assert contract.launch_trigger_duration_us == 500_000
    assert contract.remaining_buffer_coverage == 25
    assert contract.remaining_buffer_duration_us == 500_000
    assert contract.history_span_us == 100_000
    assert contract.no_starvation_guaranteed


def test_temporal_contract_derives_episode_source_intervals_without_magic_numbers() -> None:
    contract = load_temporal_contract(_CONFIG)

    belief = contract.belief_source_interval(episode_action_count=117)
    sft = contract.sft_source_interval(episode_action_count=117)
    branch_1 = contract.return_chunk_source_interval(
        episode_action_count=117,
        delay_tick=1,
    )
    branch_20 = contract.return_chunk_source_interval(
        episode_action_count=117,
        delay_tick=20,
    )

    assert belief.to_mapping() == {"count": 68, "maximum": 92, "minimum": 25}
    assert sft.to_mapping() == {"count": 68, "maximum": 67, "minimum": 0}
    assert branch_1.to_mapping() == {"count": 42, "maximum": 66, "minimum": 25}
    assert branch_20.to_mapping() == {"count": 23, "maximum": 47, "minimum": 25}
    assert contract.teacher_chunk_bounds(source_tick=25, episode_action_count=117) == (
        0,
        50,
    )
    assert contract.teacher_remaining_bounds(
        source_tick=25,
        episode_action_count=117,
    ) == (25, 50)


def test_temporal_contract_rejects_latency_scope_beyond_launch_buffer_coverage() -> None:
    contract = load_temporal_contract(_CONFIG)

    with pytest.raises(ValueError, match="maximum delay exceeds launch-time buffer coverage"):
        replace(
            contract,
            contract_id="h50_e25_d26_k6_v1",
            maximum_delay_ticks=26,
        )


def test_temporal_contract_identifier_encodes_values_without_preventing_new_profiles() -> None:
    contract = load_temporal_contract(_CONFIG)

    with pytest.raises(ValueError, match="identifier does not match"):
        replace(contract, launch_trigger_horizon=24)

    alternate = replace(
        contract,
        contract_id="h16_e8_d4_k6_v1",
        prediction_horizon=16,
        launch_trigger_horizon=8,
        maximum_delay_ticks=4,
    )
    assert alternate.remaining_buffer_coverage == 8

    with pytest.raises(ValueError, match="50 Hz formal clock"):
        replace(contract, formal_tick_us=10_000, control_frequency_hz=100)


def test_temporal_contract_rejects_episode_too_short_for_requested_interval() -> None:
    contract = load_temporal_contract(_CONFIG)

    with pytest.raises(ValueError, match="episode has no valid belief source"):
        contract.belief_source_interval(episode_action_count=40)
    with pytest.raises(ValueError, match="return-time chunk branch"):
        contract.return_chunk_source_interval(
            episode_action_count=70,
            delay_tick=20,
        )
