from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.legacy.belief.flow.rolling_config import (
    load_flow_belief_rolling_config,
)
from latency_meta_mdp.legacy.belief.flow.rolling_selection import (
    select_rolling_source_ticks,
)

_CONFIG = Path("configs/legacy/analysis/flow_belief_rolling_inspection_v1.yaml")


def test_rolling_config_locks_critical_window_contract() -> None:
    config = load_flow_belief_rolling_config(_CONFIG)

    assert config.report_id == "flow_belief_rolling_inspection_v1"
    assert config.evaluation_split == "validation"
    assert config.source_phases == ("pregrasp", "approach")
    assert config.history_sample_count == 6
    assert config.inspection_stride_ticks == 10
    assert config.display_delay_ticks == (1, 5, 10, 15, 20)
    assert config.sample_count == 32
    assert config.solver == "heun"
    assert config.solver_step_count == 16
    assert config.include_approach_anchor is True
    assert config.include_last_full_window_anchor is True
    assert config.video_fps == 1

    with pytest.raises(ValueError, match="stride"):
        replace(config, inspection_stride_ticks=0)


def test_rolling_selection_uses_stride_and_required_anchors() -> None:
    config = load_flow_belief_rolling_config(_CONFIG)
    selected = select_rolling_source_ticks(
        legal_source_ticks=tuple(range(5, 61)),
        source_phase_by_tick={
            **{tick: "pregrasp" for tick in range(5, 25)},
            **{tick: "approach" for tick in range(25, 61)},
        },
        critical_end_tick=60,
        config=config,
    )

    assert selected == (5, 15, 25, 35, 40)
    assert all(tick + 20 <= 60 for tick in selected)


def test_rolling_selection_adds_non_grid_approach_and_end_anchors() -> None:
    config = load_flow_belief_rolling_config(_CONFIG)
    selected = select_rolling_source_ticks(
        legal_source_ticks=tuple(range(5, 69)),
        source_phase_by_tick={
            **{tick: "pregrasp" for tick in range(5, 56)},
            **{tick: "approach" for tick in range(56, 69)},
        },
        critical_end_tick=88,
        config=config,
    )

    assert selected == (5, 15, 25, 35, 45, 55, 56, 65, 68)


def test_rolling_selection_rejects_missing_full_critical_window() -> None:
    with pytest.raises(ValueError, match="full pregrasp/approach window"):
        select_rolling_source_ticks(
            legal_source_ticks=(45, 55),
            source_phase_by_tick={45: "approach", 55: "approach"},
            critical_end_tick=60,
            config=load_flow_belief_rolling_config(_CONFIG),
        )


def test_rolling_selection_rejects_unmapped_or_out_of_scope_source_ticks() -> None:
    config = load_flow_belief_rolling_config(_CONFIG)
    with pytest.raises(ValueError, match="phase mapping"):
        select_rolling_source_ticks(
            legal_source_ticks=(5, 15, 25),
            source_phase_by_tick={5: "pregrasp", 15: "pregrasp"},
            critical_end_tick=45,
            config=config,
        )
    with pytest.raises(ValueError, match="configured critical phases"):
        select_rolling_source_ticks(
            legal_source_ticks=(5, 15, 25),
            source_phase_by_tick={5: "pregrasp", 15: "close", 25: "close"},
            critical_end_tick=45,
            config=config,
        )
