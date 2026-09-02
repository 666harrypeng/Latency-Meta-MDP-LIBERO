from __future__ import annotations

import numpy as np
import pytest


def _limits() -> dict[str, np.ndarray]:
    return {
        "joint_lower": np.full(7, -2.0, dtype=np.float64),
        "joint_upper": np.full(7, 2.0, dtype=np.float64),
        "joint_velocity": np.full(7, 4.0, dtype=np.float64),
        "joint_acceleration": np.full(7, 20.0, dtype=np.float64),
    }


def test_whole_path_smoothing_removes_stop_and_corner_at_internal_guide() -> None:
    """Break caught: independently planned segments are only concatenated and retimed."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import smooth_joint_approach

    raw = np.zeros((7, 7), dtype=np.float64)
    raw[:, :2] = np.array(
        [
            [0.00, 0.00],
            [0.10, 0.00],
            [0.20, 0.00],
            [0.20, 0.00],
            [0.20, 0.10],
            [0.20, 0.20],
            [0.20, 0.30],
        ],
        dtype=np.float64,
    )
    result = smooth_joint_approach(
        raw,
        duration_seconds=1.6,
        sample_period_seconds=0.02,
        **_limits(),
    )

    assert result.qpos.shape == (81, 7)
    assert np.array_equal(result.qpos[0], raw[0])
    np.testing.assert_allclose(result.qpos[-1], raw[-1], atol=1.0e-12, rtol=0.0)
    assert np.linalg.norm(result.velocity[-1]) > 1.0e-3
    assert np.min(np.linalg.norm(result.velocity[5:-5], axis=1)) > 1.0e-3
    assert np.max(np.linalg.norm(np.diff(result.velocity, axis=0), axis=1)) < 0.2
    assert np.max(np.linalg.norm(np.diff(result.acceleration, axis=0), axis=1)) < 4.0
    assert not result.qpos.flags.writeable


def test_smoothing_rejects_a_timing_that_violates_joint_limits() -> None:
    """Break caught: spatial smoothing is accepted despite infeasible execution timing."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import smooth_joint_approach

    raw = np.zeros((3, 7), dtype=np.float64)
    raw[:, 0] = [0.0, 0.5, 1.0]
    limits = _limits()
    limits["joint_velocity"] = np.full(7, 0.1, dtype=np.float64)

    with pytest.raises(ValueError, match="velocity"):
        smooth_joint_approach(
            raw,
            duration_seconds=0.4,
            sample_period_seconds=0.02,
            **limits,
        )


def test_smoothing_rejects_degenerate_or_nonfinite_geometry() -> None:
    """Break caught: a missing geometric path is silently converted into a hold trajectory."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import smooth_joint_approach

    limits = _limits()
    with pytest.raises(ValueError, match="distinct"):
        smooth_joint_approach(
            np.zeros((4, 7), dtype=np.float64),
            duration_seconds=1.0,
            sample_period_seconds=0.02,
            **limits,
        )
    invalid = np.zeros((3, 7), dtype=np.float64)
    invalid[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        smooth_joint_approach(
            invalid,
            duration_seconds=1.0,
            sample_period_seconds=0.02,
            **limits,
        )


def test_corner_rounding_handles_a_local_path_reversal_without_duplicate_knots() -> None:
    """Break caught: corner cutting creates equal adjacent knots for an A-B-A proposal."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import smooth_joint_approach

    raw = np.zeros((5, 7), dtype=np.float64)
    raw[:, 0] = [0.0, 0.1, 0.0, 0.1, 0.2]
    limits = _limits()
    limits["joint_acceleration"] = np.full(7, 1_000.0, dtype=np.float64)
    result = smooth_joint_approach(
        raw,
        duration_seconds=2.0,
        sample_period_seconds=0.02,
        **limits,
    )

    assert np.array_equal(result.qpos[0], raw[0])
    np.testing.assert_allclose(result.qpos[-1], raw[-1], atol=1.0e-12, rtol=0.0)
    assert np.all(np.isfinite(result.qpos))


def test_smoothing_filters_planner_knot_jitter_instead_of_interpolating_every_point() -> None:
    """Break caught: tiny optimizer oscillations become large spline accelerations."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import smooth_joint_approach

    progress = np.linspace(0.0, 1.0, 61, dtype=np.float64)
    raw = np.zeros((61, 7), dtype=np.float64)
    raw[:, 0] = 0.3 * progress
    raw[:, 1] = 0.02 * np.sin(30.0 * np.pi * progress)
    result = smooth_joint_approach(
        raw,
        duration_seconds=1.2,
        sample_period_seconds=0.02,
        **_limits(),
    )

    assert np.max(np.abs(result.qpos[:, 1])) < 0.01
    assert np.max(np.abs(result.acceleration)) <= 20.0


def test_cartesian_reference_is_one_smooth_curve_through_soft_approach_geometry() -> None:
    """Break caught: semantic guides are executed as separate point-to-point robot segments."""
    from latency_meta_mdp.expert_realization.trajectory_smoothing import (
        build_cartesian_approach_reference,
    )

    start = np.array([-0.10, -0.20, 1.05], dtype=np.float64)
    guide = np.array([[0.02, -0.08, 1.10]], dtype=np.float64)
    entry = np.array([0.08, 0.04, 0.93], dtype=np.float64)
    tangent = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    result = build_cartesian_approach_reference(
        start_position_world=start,
        soft_guide_regions_world=guide,
        funnel_entry_position_world=entry,
        funnel_entry_tangent_world=tangent,
        duration_seconds=1.4,
        sample_period_seconds=0.02,
    )

    assert result.positions_world.shape == (71, 3)
    np.testing.assert_array_equal(result.positions_world[0], start)
    np.testing.assert_allclose(result.positions_world[-1], entry, atol=1.0e-12, rtol=0.0)
    assert np.linalg.norm(result.velocity_world[0]) < 1.0e-12
    terminal_direction = result.velocity_world[-1] / np.linalg.norm(result.velocity_world[-1])
    np.testing.assert_allclose(terminal_direction, tangent, atol=1.0e-12, rtol=0.0)
    assert np.linalg.norm(result.positions_world - guide[0], axis=1).min() < 0.04
    assert np.max(np.linalg.norm(np.diff(result.velocity_world, axis=0), axis=1)) < 0.08
    assert not result.positions_world.flags.writeable
