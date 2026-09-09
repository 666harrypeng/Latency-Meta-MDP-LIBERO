from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchCorpusConfig,
    ExecutablePrefix,
)
from latency_meta_mdp.belief.conditional_return_flow.control_continuations import (
    build_control_continuations,
)
from latency_meta_mdp.control import load_action_contract


def _contract():
    return load_action_contract(Path("configs/control/panda_osc_pose_delta_v1.yaml"))


def _config() -> BranchCorpusConfig:
    return BranchCorpusConfig(
        schema_version=1,
        config_id="conditional_return_control_branch_corpus",
        contexts_per_phase_per_episode=1,
        phases=("pregrasp", "approach", "close", "lift"),
        arm_scale_factors=(0.5, 0.8),
        prefix_hold_ticks=(5, 10),
    )


def _nominal_prefix() -> ExecutablePrefix:
    controls = np.zeros((20, 7), dtype=np.float32)
    controls[:, 0] = np.linspace(0.02, 0.4, 20, dtype=np.float32)
    controls[:3, 6] = -1.0
    controls[3:, 6] = 1.0
    return ExecutablePrefix(
        controls=controls,
        from_active_buffer_mask=np.ones(20, dtype=bool),
        last_executed_gripper_command=-1.0,
    )


def test_control_continuation_inventory_is_deterministic_and_d20() -> None:
    rows = build_control_continuations(
        nominal_prefix=_nominal_prefix(),
        config=_config(),
        action_contract=_contract(),
        source_context_id="L3-seed001000-tick000040",
    )

    assert [row.spec.kind for row in rows] == [
        "nominal",
        "hold",
        "arm_scale_0.5",
        "arm_scale_0.8",
        "prefix_hold_5",
        "prefix_hold_10",
    ]
    assert all(row.prefix.controls.shape == (20, 7) for row in rows)
    assert all(row.source_context_id == "L3-seed001000-tick000040" for row in rows)


def test_scaled_branches_change_only_arm_coordinates() -> None:
    nominal = _nominal_prefix()
    rows = build_control_continuations(
        nominal_prefix=nominal,
        config=_config(),
        action_contract=_contract(),
        source_context_id="source",
    )
    half = next(row for row in rows if row.spec.kind == "arm_scale_0.5")

    np.testing.assert_allclose(half.prefix.controls[:, :6], nominal.controls[:, :6] * 0.5)
    np.testing.assert_array_equal(half.prefix.controls[:, 6], nominal.controls[:, 6])
    np.testing.assert_array_equal(
        half.prefix.from_active_buffer_mask,
        nominal.from_active_buffer_mask,
    )


def test_prefix_hold_preserves_last_prefix_gripper() -> None:
    rows = build_control_continuations(
        nominal_prefix=_nominal_prefix(),
        config=_config(),
        action_contract=_contract(),
        source_context_id="source",
    )
    row = next(branch for branch in rows if branch.spec.kind == "prefix_hold_5")

    assert np.all(row.prefix.controls[5:, :6] == 0)
    assert np.all(row.prefix.controls[5:, 6] == 1.0)
    assert row.prefix.from_active_buffer_mask.tolist() == [True] * 5 + [False] * 15


def test_all_hold_uses_source_gripper_and_no_active_rows() -> None:
    rows = build_control_continuations(
        nominal_prefix=_nominal_prefix(),
        config=_config(),
        action_contract=_contract(),
        source_context_id="source",
    )
    row = next(branch for branch in rows if branch.spec.kind == "hold")

    assert np.all(row.prefix.controls[:, :6] == 0)
    assert np.all(row.prefix.controls[:, 6] == -1.0)
    assert not np.any(row.prefix.from_active_buffer_mask)
