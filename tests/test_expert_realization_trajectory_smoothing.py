from __future__ import annotations

import numpy as np


def test_cartesian_reference_is_one_smooth_curve_through_soft_approach_geometry() -> None:
    """Break caught: semantic guides are executed as separate point-to-point robot segments."""
    from latency_meta_mdp.data.collection.trajectory_smoothing import (
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
