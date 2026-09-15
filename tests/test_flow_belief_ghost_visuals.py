from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.legacy.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.legacy.belief.flow.ghost_visuals import (
    assemble_context_panel,
    compose_agentview_ghost,
    joint_quantile_bands,
    render_joint_band_plot,
    render_state_cloud_plot,
    write_rgb_video,
)


def _config():
    return load_flow_belief_ghost_config(
        Path("configs/legacy/analysis/flow_belief_agentview_ghost_v1.yaml")
    )


def test_ghost_composition_uses_translucent_overlap_and_distinct_contours() -> None:
    background = np.full((41, 41, 3), 100, dtype=np.uint8)
    gt_robot = np.zeros((41, 41), dtype=np.bool_)
    gt_ball = np.zeros((41, 41), dtype=np.bool_)
    pred_robot = np.zeros((41, 41), dtype=np.bool_)
    pred_ball = np.zeros((41, 41), dtype=np.bool_)
    gt_robot[4:37, 3:34] = True
    pred_robot[4:37, 7:38] = True

    composed = compose_agentview_ghost(
        background_rgb=background,
        ground_truth_robot_mask=gt_robot,
        ground_truth_ball_mask=gt_ball,
        prediction_robot_mask=pred_robot,
        prediction_ball_mask=pred_ball,
        config=_config(),
    )

    assert composed.dtype == np.uint8
    assert composed.shape == background.shape
    np.testing.assert_array_equal(composed[40, 40], background[40, 40])
    np.testing.assert_array_equal(composed[20, 22], [134, 131, 92])
    np.testing.assert_array_equal(composed[20, 20], [181, 174, 80])
    np.testing.assert_array_equal(composed[20, 3], _config().ground_truth_rgb)
    np.testing.assert_array_equal(composed[20, 37], _config().prediction_rgb)


def test_joint_quantiles_include_68_and_95_percent_bands() -> None:
    values = np.broadcast_to(
        np.arange(100, dtype=np.float64)[None, :, None],
        (5, 100, 7),
    ).copy()

    lower95, lower68, median, upper68, upper95 = joint_quantile_bands(values)

    assert lower95.shape == (5, 7)
    np.testing.assert_allclose(lower95, 2.475)
    np.testing.assert_allclose(lower68, 15.84)
    np.testing.assert_allclose(median, 49.5)
    np.testing.assert_allclose(upper68, 83.16)
    np.testing.assert_allclose(upper95, 96.525)


def test_trajectory_joint_and_context_panels_have_fixed_rgb_shapes() -> None:
    rng = np.random.default_rng(7)
    delays = np.asarray([1, 5, 10, 15, 20])
    object_samples = rng.normal(size=(5, 32, 3)).astype(np.float32) * 0.01
    object_targets = rng.normal(size=(5, 3)).astype(np.float32) * 0.01
    eef_samples = rng.normal(size=(5, 32, 3)).astype(np.float32) * 0.01
    eef_targets = rng.normal(size=(5, 3)).astype(np.float32) * 0.01
    joint_samples = rng.normal(size=(5, 32, 7)).astype(np.float32) * 0.1
    joint_targets = rng.normal(size=(5, 7)).astype(np.float32) * 0.1

    object_plot = render_state_cloud_plot(
        delay_ticks=delays,
        state_samples=object_samples,
        state_targets=object_targets,
        title="Object future distribution",
        state_label="Ball center",
        config=_config(),
    )
    eef_plot = render_state_cloud_plot(
        delay_ticks=delays,
        state_samples=eef_samples,
        state_targets=eef_targets,
        title="EEF future distribution",
        state_label="EEF",
        config=_config(),
    )
    joints = render_joint_band_plot(
        delay_ticks=delays,
        joint_samples=joint_samples,
        joint_targets=joint_targets,
        config=_config(),
    )
    current = np.full((256, 256, 3), 20, dtype=np.uint8)
    overlays = np.full((5, 256, 256, 3), 120, dtype=np.uint8)
    panel = assemble_context_panel(
        current_rgb=current,
        ghost_overlays=overlays,
        object_plot=object_plot,
        eef_plot=eef_plot,
        joint_plot=joints,
        delay_ticks=delays,
        delay_summaries=(
            "ball 1.2 mm | EEF 2.3 mm | q 3.4 mrad | 32/32 valid",
            "ball 1.4 mm | EEF 2.5 mm | q 3.6 mrad | 32/32 valid",
            "ball 1.6 mm | EEF 2.7 mm | q 3.8 mrad | 32/32 valid",
            "ball 1.8 mm | EEF 2.9 mm | q 4.0 mrad | 32/32 valid",
            "ball 2.0 mm | EEF 3.1 mm | q 4.2 mrad | 31/32 valid",
        ),
        title="Flow Belief future-state quality",
        subtitle="L1 | seed 1180 | source tick 25 | phase approach | role typical",
    )

    assert object_plot.shape == (500, 500, 3)
    assert eef_plot.shape == (500, 500, 3)
    assert joints.shape == (500, 820, 3)
    assert panel.shape == (1200, 1920, 3)
    assert panel.dtype == np.uint8


def test_rgb_video_encodes_without_overwrite(tmp_path: Path) -> None:
    frames = np.stack(
        [
            np.full((64, 96, 3), 20, dtype=np.uint8),
            np.full((64, 96, 3), 220, dtype=np.uint8),
        ]
    )
    output = tmp_path / "review.mp4"

    write_rgb_video(frames=frames, output_path=output, fps=5)

    assert output.is_file()
    assert output.stat().st_size > 0
