from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.flow.buffer_causality import (
    BufferCausalityCandidate,
    build_buffer_branches,
    load_buffer_causality_config,
    select_buffer_causality_contexts,
)


def _candidates(*, per_phase: int = 6) -> tuple[BufferCausalityCandidate, ...]:
    rows = []
    for level in (1, 2, 3):
        for phase_index, phase in enumerate(("pregrasp", "approach", "close", "lift")):
            for index in range(per_phase):
                rows.append(
                    BufferCausalityCandidate(
                        episode_id=f"l{level}-seed-{1180 + index:06d}-attempt-000",
                        level=level,
                        scene_seed=1180 + index,
                        source_tick=10 + 20 * phase_index + index,
                        phase=phase,
                        real_transition_count=160,
                    )
                )
    return tuple(reversed(rows))


def test_formal_buffer_causality_config_is_locked() -> None:
    config = load_buffer_causality_config(
        Path("configs/analysis/dinov3_flow_belief_buffer_causality_v1.yaml")
    )

    assert config.contexts_per_phase == 4
    assert config.phases == ("pregrasp", "approach", "close", "lift")
    assert config.branch_ids == ("expert", "hold", "half_speed", "delayed_prefix_5")
    assert config.delay_ticks == tuple(range(1, 21))
    assert config.sample_count == 32
    assert config.solver == "heun"
    assert config.solver_step_count == 16


def test_selection_is_stratified_distinct_and_deterministic() -> None:
    selected = select_buffer_causality_contexts(
        candidates=_candidates(),
        levels=(1, 2, 3),
        phases=("pregrasp", "approach", "close", "lift"),
        contexts_per_phase=4,
        required_real_action_count=25,
    )

    assert len(selected) == 3 * 4 * 4
    assert selected == tuple(
        sorted(selected, key=lambda row: (row.level, row.phase, row.scene_seed))
    )
    for level in (1, 2, 3):
        for phase in ("pregrasp", "approach", "close", "lift"):
            rows = [row for row in selected if row.level == level and row.phase == phase]
            assert len(rows) == 4
            assert len({row.episode_id for row in rows}) == 4
            assert [row.scene_seed for row in rows] == [1180, 1182, 1183, 1185]
            assert all(row.source_tick + 25 <= row.real_transition_count for row in rows)


def test_selection_rejects_insufficient_real_horizon() -> None:
    candidates = tuple(
        row
        for row in _candidates(per_phase=4)
        if not (row.level == 3 and row.phase == "lift" and row.scene_seed > 1181)
    )

    with pytest.raises(ValueError, match="L3 lift"):
        select_buffer_causality_contexts(
            candidates=candidates,
            levels=(1, 2, 3),
            phases=("pregrasp", "approach", "close", "lift"),
            contexts_per_phase=4,
            required_real_action_count=25,
        )


def test_action_buffer_branches_preserve_controller_semantics() -> None:
    expert = np.arange(25 * 7, dtype=np.float32).reshape(25, 7) / 100.0
    expert[:, 6] = np.where(np.arange(25) < 12, -1.0, 1.0)

    branches = build_buffer_branches(expert)

    assert tuple(branches) == ("expert", "hold", "half_speed", "delayed_prefix_5")
    np.testing.assert_array_equal(branches["expert"], expert)
    np.testing.assert_array_equal(branches["hold"][:, :6], 0.0)
    np.testing.assert_array_equal(branches["hold"][:, 6], -1.0)
    np.testing.assert_allclose(branches["half_speed"][:, :6], 0.5 * expert[:, :6])
    np.testing.assert_array_equal(branches["half_speed"][:, 6], expert[:, 6])
    np.testing.assert_array_equal(branches["delayed_prefix_5"][:5, :6], 0.0)
    np.testing.assert_array_equal(branches["delayed_prefix_5"][:5, 6], -1.0)
    np.testing.assert_array_equal(branches["delayed_prefix_5"][5:], expert[:20])
    assert all(value.shape == (25, 7) and value.dtype == np.float32 for value in branches.values())
    assert all(not value.flags.writeable for value in branches.values())


def test_action_buffer_branches_reject_invalid_input() -> None:
    with pytest.raises(ValueError, match=r"\[25, 7\]"):
        build_buffer_branches(np.zeros((20, 7), dtype=np.float32))
