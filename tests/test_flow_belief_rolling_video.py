from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.belief.flow.rolling_video import (
    build_progressive_reveal_schedule,
    iter_progressive_context_frames,
    write_progressive_review_video,
)
from latency_meta_mdp.belief.flow.rolling_visuals import (
    derive_rolling_joint_limits,
    derive_rolling_plot_limits,
)


def test_progressive_schedule_uses_only_real_delay_queries_and_final_hold() -> None:
    schedule = build_progressive_reveal_schedule(
        delay_ticks=(1, 5, 10, 15, 20),
        fade_alphas=(0.35, 0.7, 1.0),
        final_hold_frames=4,
    )

    assert tuple(stage.delay_tick for stage in schedule[:15:3]) == (1, 5, 10, 15, 20)
    assert {stage.delay_tick for stage in schedule} == {1, 5, 10, 15, 20}
    assert tuple(stage.alpha for stage in schedule[:3]) == (0.35, 0.7, 1.0)
    assert len(schedule) == 19
    assert all(stage.delay_tick == 20 and stage.alpha == 1.0 for stage in schedule[-4:])


def test_progressive_context_frames_keep_fixed_layout_and_change_reveal_state() -> None:
    rng = np.random.default_rng(29)
    delays = np.asarray([1, 5, 10, 15, 20])
    object_targets = rng.normal(size=(5, 3)) * 0.01
    object_samples = object_targets[:, None, :] + rng.normal(size=(5, 32, 3)) * 0.002
    eef_targets = rng.normal(size=(5, 3)) * 0.01
    eef_samples = eef_targets[:, None, :] + rng.normal(size=(5, 32, 3)) * 0.002
    joint_targets = rng.normal(size=(5, 7)) * 0.1
    joint_samples = joint_targets[:, None, :] + rng.normal(size=(5, 32, 7)) * 0.005
    overlays = np.stack(
        [np.full((256, 256, 3), 30 + index * 35, dtype=np.uint8) for index in range(5)]
    )
    config = load_flow_belief_ghost_config(
        Path("configs/analysis/flow_belief_agentview_ghost_v1.yaml")
    )
    frames = list(
        iter_progressive_context_frames(
            current_rgb=np.full((256, 256, 3), 20, dtype=np.uint8),
            ghost_overlays=overlays,
            object_samples=object_samples,
            object_targets=object_targets,
            eef_samples=eef_samples,
            eef_targets=eef_targets,
            joint_samples=joint_samples,
            joint_targets=joint_targets,
            delay_ticks=delays,
            delay_summaries=tuple(f"delay {delay}" for delay in delays),
            config=config,
            object_limits=derive_rolling_plot_limits(object_samples[None], object_targets[None]),
            eef_limits=derive_rolling_plot_limits(eef_samples[None], eef_targets[None]),
            joint_limits=derive_rolling_joint_limits(joint_samples[None], joint_targets[None]),
            title="Flow Belief future-state quality",
            subtitle_prefix="L1 | seed 1180 | source tick 25 | rolling window 1/7",
            fade_alphas=(1.0,),
            final_hold_frames=2,
        )
    )

    assert len(frames) == 7
    assert all(frame.shape == (1200, 1920, 3) for frame in frames)
    assert all(frame.dtype == np.uint8 for frame in frames)
    assert not np.array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames[-1], frames[-2])


def test_progressive_review_video_streams_multiple_window_groups(tmp_path: Path) -> None:
    frame_a = np.full((64, 96, 3), 20, dtype=np.uint8)
    frame_b = np.full((64, 96, 3), 220, dtype=np.uint8)
    output = tmp_path / "rolling.mp4"

    frame_count = write_progressive_review_video(
        frame_groups=((frame_a, frame_b), (frame_b,)),
        output_path=output,
        fps=10,
    )

    assert frame_count == 3
    assert output.is_file()
    assert output.stat().st_size > 0
