from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_j4_summary_separates_robot_onset_and_object_coupling_regimes() -> None:
    """Catches mixing pre-handoff invariance with post-handoff coupled dynamics."""

    from latency_meta_mdp.belief.jepa.diagnostics.action_effect_evaluation import (
        summarize_j4_effects,
    )

    shape = (2, 4, 2)
    gt_proprio = np.zeros((*shape, 16), dtype=np.float32)
    predicted_proprio = np.zeros_like(gt_proprio)
    gt_object = np.zeros((*shape, 6), dtype=np.float32)
    predicted_object = np.zeros_like(gt_object)
    for branch in (1, 2):
        gt_proprio[:, branch, :, 0] = 1.0
        predicted_proprio[:, branch, :, 0] = 1.0
    gt_proprio[:, 3, 1, 0] = 1.0
    predicted_proprio[:, 3, 1, 0] = 1.0
    predicted_object[0, 1:, :, 0] = 0.001
    gt_object[1, 1:, :, 0] = 0.01
    predicted_object[1, 1:, :, 0] = 0.01
    handoff = np.zeros(shape, dtype=np.bool_)
    handoff[1] = True

    summary = summarize_j4_effects(
        predicted_proprio=predicted_proprio,
        predicted_object=predicted_object,
        gt_proprio=gt_proprio,
        gt_object=gt_object,
        handoff_physical=handoff,
        context_categories=("smooth_approach", "post_handoff_lift"),
        branch_names=("nominal", "hold", "scale_0.5", "prefix4_then_hold"),
        native_delay_ticks=(4, 8),
    )

    assert summary["branches"]["hold"]["robot_qpos_effect_cosine_mean"] == 1.0
    assert summary["branches"]["hold"]["robot_qpos_effect_cosine_per_anchor"] == [1.0, 1.0]
    assert summary["prefix4_first_anchor_predicted_robot_effect_max"] == 0.0
    assert summary["pre_handoff_predicted_object_spurious_effect_mean_m"] == pytest.approx(0.001)
    assert summary[
        "pre_handoff_predicted_object_spurious_effect_mean_per_anchor_m"
    ] == pytest.approx([0.001, 0.001])
    assert summary["post_handoff_object_effect_cosine_mean"] == 1.0
    assert summary["post_handoff_object_effect_cosine_per_anchor"] == [1.0, 1.0]

    pre_only = summarize_j4_effects(
        predicted_proprio=predicted_proprio[:1],
        predicted_object=predicted_object[:1],
        gt_proprio=gt_proprio[:1],
        gt_object=gt_object[:1],
        handoff_physical=handoff[:1],
        context_categories=("smooth_approach",),
        branch_names=("nominal", "hold", "scale_0.5", "prefix4_then_hold"),
        native_delay_ticks=(4, 8),
    )
    post_only = summarize_j4_effects(
        predicted_proprio=predicted_proprio[1:],
        predicted_object=predicted_object[1:],
        gt_proprio=gt_proprio[1:],
        gt_object=gt_object[1:],
        handoff_physical=handoff[1:],
        context_categories=("post_handoff_lift",),
        branch_names=("nominal", "hold", "scale_0.5", "prefix4_then_hold"),
        native_delay_ticks=(4, 8),
    )
    assert pre_only["post_handoff_object_effect_cosine_mean"] is None
    assert post_only["pre_handoff_predicted_object_spurious_effect_mean_m"] is None


def test_j4_evaluation_cli_runs_all_final_seeds_on_one_fixed_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches per-seed branch recollection or arbitrary evaluation configuration."""

    import latency_meta_mdp.belief.jepa.diagnostics.action_effect_evaluation_run as module
    from latency_meta_mdp.belief.jepa.diagnostics.evaluate_action_effect import (
        main,
    )

    output = tmp_path / "report.json"
    observed = {}

    def evaluate(**kwargs):
        observed.update(kwargs)
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(module, "evaluate_l3_j4_bank", evaluate)
    status = main(
        [
            "--project-root",
            ".",
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert observed == {
        "project_root": Path("."),
        "device": "cuda:0",
        "output_path": output,
    }
    assert capsys.readouterr().out.strip() == str(output)
