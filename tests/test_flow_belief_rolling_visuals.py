from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.belief.flow.ghost_visuals import (
    render_joint_band_plot,
    render_state_cloud_plot,
)
from latency_meta_mdp.belief.flow.rolling_visuals import (
    derive_rolling_joint_limits,
    derive_rolling_plot_limits,
    render_rolling_metric_heatmaps,
    render_rolling_trajectory_overview,
)


def _state_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    targets = rng.normal(size=(3, 5, 3)).astype(np.float64) * 0.02
    samples = targets[:, :, None, :] + rng.normal(size=(3, 5, 32, 3)) * 0.004
    return samples, targets


def test_rolling_plot_limits_cover_every_window_with_equal_xy_scale() -> None:
    samples, targets = _state_data()

    limits = derive_rolling_plot_limits(samples, targets)
    relative_samples = (samples[..., :2] - targets[:, :1, None, :2]) * 1_000.0
    relative_targets = (targets[..., :2] - targets[:, :1, :2]) * 1_000.0

    assert limits.x_max_mm - limits.x_min_mm == pytest.approx(limits.y_max_mm - limits.y_min_mm)
    assert relative_samples[..., 0].min() >= limits.x_min_mm
    assert relative_samples[..., 0].max() <= limits.x_max_mm
    assert relative_samples[..., 1].min() >= limits.y_min_mm
    assert relative_samples[..., 1].max() <= limits.y_max_mm
    assert relative_targets[..., 0].min() >= limits.x_min_mm
    assert relative_targets[..., 0].max() <= limits.x_max_mm


def test_shared_state_and_joint_limits_render_fixed_shapes() -> None:
    state_samples, state_targets = _state_data()
    rng = np.random.default_rng(19)
    joint_targets = rng.normal(size=(3, 5, 7)) * 0.2
    joint_samples = joint_targets[:, :, None, :] + rng.normal(size=(3, 5, 32, 7)) * 0.01
    state_limits = derive_rolling_plot_limits(state_samples, state_targets)
    joint_limits = derive_rolling_joint_limits(joint_samples, joint_targets)
    config = load_flow_belief_ghost_config(
        Path("configs/analysis/flow_belief_agentview_ghost_v1.yaml")
    )

    cloud = render_state_cloud_plot(
        delay_ticks=np.asarray([1, 5, 10, 15, 20]),
        state_samples=state_samples[1],
        state_targets=state_targets[1],
        title="Object future distribution",
        state_label="Ball center",
        config=config,
        xy_limits_mm=state_limits,
    )
    joints = render_joint_band_plot(
        delay_ticks=np.asarray([1, 5, 10, 15, 20]),
        joint_samples=joint_samples[1],
        joint_targets=joint_targets[1],
        config=config,
        joint_limits=joint_limits,
    )

    assert cloud.shape == (500, 500, 3)
    assert joints.shape == (500, 820, 3)
    assert joint_limits.radians.shape == (7, 2)


def test_rolling_overview_and_heatmaps_have_review_shapes() -> None:
    boundary_ticks = np.arange(25, 89, dtype=np.int64)
    phase = np.asarray(["pregrasp"] * 31 + ["approach"] * 33)
    object_position = np.stack(
        (
            np.linspace(-0.10, 0.08, len(boundary_ticks)),
            0.03 * np.sin(np.linspace(0.0, np.pi, len(boundary_ticks))),
            np.full(len(boundary_ticks), 0.82),
        ),
        axis=1,
    )
    source_ticks = np.asarray([25, 35, 45, 55, 56, 65, 68])
    delays = np.asarray([1, 5, 10, 15, 20])
    values = np.arange(35, dtype=np.float64).reshape(7, 5) / 10.0

    overview = render_rolling_trajectory_overview(
        boundary_ticks=boundary_ticks,
        object_position=object_position,
        boundary_phase=phase,
        selected_source_ticks=source_ticks,
        critical_start_tick=25,
        critical_end_tick=88,
        maximum_delay_ticks=20,
        level=2,
        scene_seed=1199,
    )
    heatmaps = render_rolling_metric_heatmaps(
        source_ticks=source_ticks,
        delay_ticks=delays,
        object_error_mm=values,
        eef_error_mm=values + 1.0,
        joint_rmse_mrad=values + 2.0,
        invalid_sample_fraction=values / 100.0,
        level=2,
        scene_seed=1199,
    )

    assert overview.shape == (900, 1440, 3)
    assert heatmaps.shape == (900, 1440, 3)
    assert overview.dtype == np.uint8
    assert heatmaps.dtype == np.uint8
