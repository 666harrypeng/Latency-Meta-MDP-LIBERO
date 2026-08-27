from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.belief.flow.ghost_visuals import (
    assemble_context_panel,
    compose_agentview_ghost,
    render_joint_band_plot,
    render_trajectory_plot,
    write_rgb_video,
)


def _config():
    return load_flow_belief_ghost_config(
        Path("configs/analysis/flow_belief_agentview_ghost_v1.yaml")
    )


def test_ghost_composition_tints_masks_and_preserves_background() -> None:
    background = np.full((3, 4, 3), 100, dtype=np.uint8)
    gt_robot = np.zeros((3, 4), dtype=np.bool_)
    gt_ball = np.zeros((3, 4), dtype=np.bool_)
    pred_robot = np.zeros((3, 4), dtype=np.bool_)
    pred_ball = np.zeros((3, 4), dtype=np.bool_)
    gt_robot[0, 0] = True
    pred_robot[0, 1] = True
    gt_ball[1, 0] = True
    pred_ball[1, 0] = True

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
    np.testing.assert_array_equal(composed[2, 3], background[2, 3])
    assert tuple(composed[0, 0]) != tuple(background[0, 0])
    assert tuple(composed[0, 1]) != tuple(background[0, 1])
    assert tuple(composed[1, 0]) == _config().overlap_rgb


def test_trajectory_joint_and_context_panels_have_fixed_rgb_shapes() -> None:
    rng = np.random.default_rng(7)
    delays = np.asarray([1, 5, 10, 15, 20])
    object_samples = rng.normal(size=(5, 32, 3)).astype(np.float32) * 0.01
    object_targets = rng.normal(size=(5, 3)).astype(np.float32) * 0.01
    eef_samples = rng.normal(size=(5, 32, 3)).astype(np.float32) * 0.01
    eef_targets = rng.normal(size=(5, 3)).astype(np.float32) * 0.01
    joint_samples = rng.normal(size=(5, 32, 7)).astype(np.float32) * 0.1
    joint_targets = rng.normal(size=(5, 7)).astype(np.float32) * 0.1

    trajectory = render_trajectory_plot(
        delay_ticks=delays,
        object_samples=object_samples,
        object_targets=object_targets,
        eef_samples=eef_samples,
        eef_targets=eef_targets,
        config=_config(),
    )
    joints = render_joint_band_plot(
        delay_ticks=delays,
        joint_samples=joint_samples,
        joint_targets=joint_targets,
        config=_config(),
    )
    current = np.full((256, 256, 3), 20, dtype=np.uint8)
    gt = np.full((5, 256, 256, 3), 80, dtype=np.uint8)
    overlays = np.full((5, 256, 256, 3), 120, dtype=np.uint8)
    panel = assemble_context_panel(
        current_rgb=current,
        ground_truth_rgb=gt,
        ghost_overlays=overlays,
        trajectory_plot=trajectory,
        joint_plot=joints,
        delay_ticks=delays,
        title="L1 seed 1180 tick 25",
    )

    assert trajectory.shape == (512, 512, 3)
    assert joints.shape == (512, 512, 3)
    assert panel.shape == (1280, 1280, 3)
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
