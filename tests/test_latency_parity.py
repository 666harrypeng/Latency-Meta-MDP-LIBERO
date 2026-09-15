from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.envs.outcomes import OutcomeStatus


def _module():
    return importlib.import_module("latency_meta_mdp.runtime.latency_parity")


def test_real_l1_direct_and_zero_delay_harness_are_exactly_identical() -> None:
    module = _module()

    result = module.run_direct_zero_latency_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=32,
        camera_height=32,
    )

    result.validate_exact()
    assert result.direct.terminal_status is OutcomeStatus.SUCCESS
    assert result.harness.terminal_status is OutcomeStatus.SUCCESS
    assert result.direct.terminal_time_us == result.harness.terminal_time_us
    assert result.direct.executed_action.shape[0] == result.direct.boundary_time_us.shape[0] - 1
    assert len(result.harness_events) == result.harness.executed_action.shape[0] * 7
    assert {
        event.realized_delay_ticks
        for event in result.harness_events
        if event.realized_delay_ticks is not None
    } == {0}
    assert all(event.kind.value != "starvation" for event in result.harness_events)


def test_exact_parity_diagnostic_names_the_first_mutated_field_and_index() -> None:
    module = _module()
    result = module.run_direct_zero_latency_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )
    mutated_action = result.harness.executed_action.copy()
    mutated_action[0, 0] += 0.01
    mutated = replace(
        result,
        harness=replace(result.harness, executed_action=mutated_action),
    )

    with pytest.raises(
        ValueError,
        match="zero-delay parity mismatch: executed_action at transition 0",
    ):
        mutated.validate_exact()


def test_parity_trace_arrays_are_immutable_copies() -> None:
    module = _module()
    result = module.run_direct_zero_latency_parity(
        project_root=Path.cwd(),
        seed=10,
        camera_width=8,
        camera_height=8,
    )

    for trace in (result.direct, result.harness):
        for field_name in module.ParityLaneTrace.array_field_names():
            value = getattr(trace, field_name)
            assert isinstance(value, np.ndarray)
            assert value.flags.writeable is False
