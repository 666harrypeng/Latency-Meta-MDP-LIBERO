from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_curobo_and_robosuite_policy_eef_fk_match_across_joint_domain() -> None:
    """Gate: Panda base/TCP bridge matches beyond home before planning is allowed."""
    from latency_meta_mdp.data.collection.planner_protocol import (
        request_curobo_fk_subprocess,
    )
    from latency_meta_mdp.data.collection.robot_bridge import (
        build_panda_planning_bridge,
        compute_robosuite_fk,
        qualify_fk_parity,
        sample_fk_parity_set,
    )

    root = Path.cwd()
    bridge = build_panda_planning_bridge(root)
    samples = sample_fk_parity_set(bridge, interior_count=16)
    robosuite_fk = compute_robosuite_fk(bridge, samples, project_root=root)
    curobo_fk = request_curobo_fk_subprocess(bridge, samples)
    report = qualify_fk_parity(bridge, robosuite_fk, curobo_fk)

    assert report.sample_count == 31
    assert report.joint_order_matches is True
    assert report.base_transform_matches is True
    assert report.tcp_transform_matches is True
    assert report.max_translation_error_m <= 0.001
    assert report.max_rotation_error_degrees <= 0.5
    assert report.eligible is True
    assert np.all(np.isfinite(curobo_fk.positions_world))
    assert np.all(np.isfinite(curobo_fk.rotations_world))
