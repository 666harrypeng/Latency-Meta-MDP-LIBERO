from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.outcomes import OutcomeStatus


def _module():
    return importlib.import_module("latency_meta_mdp.action_chunk_parity")


def test_real_l1_replay_chunks_exactly_match_direct_expert_trace() -> None:
    module = _module()

    result = module.run_action_chunk_zero_delay_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=32,
        camera_height=32,
    )

    result.validate_exact()
    assert result.direct.terminal_status is OutcomeStatus.SUCCESS
    assert result.chunk.terminal_status is OutcomeStatus.SUCCESS
    assert result.bootstrap.simulation_time_before_us == 0
    assert result.bootstrap.simulation_time_after_us == 0
    assert result.chunk_launch_ticks[:3] == (8, 16, 24)
    assert result.starvation_ticks == ()
    assert result.generator_kind == "reference_action_replay_oracle"


def test_chunk_parity_records_sharp_zero_delay_install_and_index_zero_execution() -> None:
    module = _module()
    result = module.run_action_chunk_zero_delay_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )

    installs = [
        event
        for event in result.chunk_events
        if event.kind.value == "chunk_install"
    ]
    assert [event.formal_tick for event in installs[:3]] == [8, 16, 24]
    assert all(event.installed_chunk_index == 0 for event in installs)
    assert all(event.discarded_action_count == 8 for event in installs)
    executed = [
        event
        for event in result.chunk_events
        if event.kind.value == "chunk_action_execute"
    ]
    assert executed[8].formal_tick == 8
    assert executed[8].executed_chunk_index == 0


def test_chunk_parity_diagnostic_names_mutated_action_transition() -> None:
    module = _module()
    result = module.run_action_chunk_zero_delay_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )
    changed_actions = result.chunk.executed_action.copy()
    changed_actions[0, 0] += 0.01
    mutated = replace(
        result,
        chunk=replace(result.chunk, executed_action=changed_actions),
    )

    with pytest.raises(
        ValueError,
        match="zero-delay parity mismatch: executed_action at transition 0",
    ):
        mutated.validate_exact()
