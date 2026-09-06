from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _record


def test_j4_control_branches_preserve_direction_and_prefix_causality() -> None:
    """Catches arbitrary action noise, changed gripper semantics, or a wrong prefix boundary."""

    from latency_meta_mdp.belief.action_conditioned_jepa.action_effect import (
        build_j4_control_branches,
    )

    nominal = np.linspace(-0.5, 0.5, num=20 * 7, dtype=np.float32).reshape(20, 7)
    last = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    branches = build_j4_control_branches(
        nominal_controls=nominal,
        last_executed_control=last,
    )

    assert tuple(branches) == ("nominal", "hold", "scale_0.5", "prefix4_then_hold")
    np.testing.assert_array_equal(branches["nominal"], nominal)
    np.testing.assert_array_equal(branches["hold"][:, :6], 0.0)
    np.testing.assert_array_equal(branches["hold"][:, 6], -1.0)
    np.testing.assert_allclose(branches["scale_0.5"][:, :6], nominal[:, :6] * 0.5)
    np.testing.assert_array_equal(branches["scale_0.5"][:, 6], nominal[:, 6])
    np.testing.assert_array_equal(branches["prefix4_then_hold"][:4], nominal[:4])
    np.testing.assert_array_equal(branches["prefix4_then_hold"][4:, :6], 0.0)
    np.testing.assert_array_equal(branches["prefix4_then_hold"][4:, 6], nominal[3, 6])
    assert not any(value.flags.writeable for value in branches.values())


def _phase_episode(tmp_path: Path):
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_signal_audit import (
        TemporalSignalEpisode,
    )

    record = _record(tmp_path, episode_id="formal-episode", split="validation", terminal_tick=80)
    phases = tuple(
        "smooth_approach"
        if tick < 20
        else "grasp_funnel"
        if tick < 40
        else "close_stabilize"
        if tick < 50
        else "lift"
        for tick in range(81)
    )
    record = replace(record, logical_master_task_index=13, phases=phases)
    zeros3 = np.zeros((81, 3), dtype=np.float32)
    zeros = np.zeros(81, dtype=np.bool_)
    return TemporalSignalEpisode(
        record=record,
        object_position=zeros3,
        object_linear_velocity=zeros3,
        eef_position=zeros3,
        left_pad_contact=zeros,
        right_pad_contact=zeros,
        handoff_state=tuple("driven" if tick < 50 else "physical" for tick in range(81)),
        motion_segment_index=np.zeros(81, dtype=np.int64),
    )


def test_j4_context_selection_uses_one_d20_ready_center_per_phase(tmp_path: Path) -> None:
    """Catches selecting edge/terminal ticks or overweighting a long phase."""

    from latency_meta_mdp.belief.action_conditioned_jepa.action_effect import (
        select_j4_source_contexts,
    )

    contexts = select_j4_source_contexts((_phase_episode(tmp_path),))

    assert [value.category for value in contexts] == [
        "smooth_approach",
        "grasp_funnel",
        "close_stabilize",
        "post_handoff_lift",
    ]
    assert [value.source_tick for value in contexts] == [14, 30, 45, 55]
    assert all(value.logical_master_task_index == 13 for value in contexts)
    assert all(value.source_tick + 20 <= 80 for value in contexts)


def test_j4_collection_cli_exposes_only_formal_l3_bank_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches adding training mixtures or arbitrary branch knobs to formal J4 collection."""

    import latency_meta_mdp.belief.action_conditioned_jepa.action_effect_run as module
    from latency_meta_mdp.cli.collect_action_conditioned_jepa_l3_action_effect import main

    output = tmp_path / "bank"
    observed = {}

    def collect(**kwargs):
        observed.update(kwargs)
        output.mkdir()
        manifest = output / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(module, "collect_l3_j4_bank", collect)
    status = main(
        [
            "--project-root",
            ".",
            "--device",
            "cuda:0",
            "--output-dir",
            str(output),
        ]
    )

    assert status == 0
    assert observed == {
        "project_root": Path("."),
        "device": "cuda:0",
        "output_dir": output,
    }
    assert capsys.readouterr().out.strip() == str(output / "manifest.json")


def test_j4_source_replay_gate_uses_lossless_source_precision() -> None:
    """Catches comparing float64 simulator replay against a quantized float32 model view."""

    from latency_meta_mdp.belief.action_conditioned_jepa.action_effect_run import (
        _source_replay_max_abs,
    )

    value = np.float64(0.123456789123)
    snapshot = SimpleNamespace(
        robot_qpos=np.full(7, value),
        robot_qvel=np.full(7, value),
        robot_gripper_qpos=np.asarray([value, -value]),
        robot_gripper_qvel=np.asarray([value, -value]),
        object_qpos=np.full(7, value),
        object_qvel=np.full(6, value),
        eef_pos=np.full(3, value),
    )
    reference = {
        "robot_qpos": np.full((1, 7), value),
        "robot_qvel": np.full((1, 7), value),
        "gripper_qpos": np.asarray([[value, -value]]),
        "gripper_qvel": np.asarray([[value, -value]]),
        "object_pose": np.full((1, 7), value),
        "object_velocity": np.full((1, 6), value),
        "eef_position_world": np.full((1, 3), value),
    }

    assert _source_replay_max_abs(snapshot=snapshot, reference=reference, source_tick=0) == 0.0


def test_j4_replay_equivalence_allows_only_float64_scale_drift() -> None:
    """Catches both a bitwise-only gate and a physically meaningful replay mismatch."""

    from latency_meta_mdp.belief.action_conditioned_jepa.action_effect_run import (
        replay_is_numerically_equivalent,
    )

    assert replay_is_numerically_equivalent(1.4e-9)
    assert replay_is_numerically_equivalent(1.0e-8)
    assert not replay_is_numerically_equivalent(1.0e-7)


def test_j4_nominal_future_equivalence_is_bounded_at_float32_scale() -> None:
    """Catches requiring bitwise float32 replay or accepting physical future drift."""

    from latency_meta_mdp.belief.action_conditioned_jepa.action_effect_run import (
        nominal_future_is_numerically_equivalent,
    )

    assert nominal_future_is_numerically_equivalent(1.5e-8)
    assert nominal_future_is_numerically_equivalent(1.0e-7)
    assert not nominal_future_is_numerically_equivalent(1.0e-6)
