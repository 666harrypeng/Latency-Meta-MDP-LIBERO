from __future__ import annotations

import numpy as np


def test_swept_clearance_detects_between_tick_collision() -> None:
    """Break caught: safe 50 Hz endpoints hide a collision at an intermediate 2 ms point."""
    from latency_meta_mdp.expert_realization.safety import (
        evaluate_moving_target_swept_clearance,
    )

    report = evaluate_moving_target_swept_clearance(
        reference_positions_world=np.array(
            [[-0.10, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float64
        ),
        object_positions_world=np.zeros((2, 3), dtype=np.float64),
        reference_start_time_us=100_000,
        minimum_center_distance_m=0.05,
    )

    assert report.safe is False
    assert report.minimum_distance_m == 0.0
    assert report.minimum_time_us == 110_000


def test_contact_admission_is_phase_and_link_aware() -> None:
    """Break caught: any ball contact is accepted, regardless of phase or robot link."""
    from latency_meta_mdp.expert_realization.executor import StructuredExpertPhase
    from latency_meta_mdp.expert_realization.safety import ContactKind, contact_is_allowed

    assert (
        contact_is_allowed(ContactKind.PAD_BALL, StructuredExpertPhase.CLOSE_STABILIZE) is True
    )
    assert contact_is_allowed(ContactKind.PAD_BALL, StructuredExpertPhase.LIFT) is True
    assert contact_is_allowed(ContactKind.PAD_BALL, StructuredExpertPhase.GRASP_FUNNEL) is False
    assert (
        contact_is_allowed(ContactKind.OTHER_ROBOT_BALL, StructuredExpertPhase.CLOSE_STABILIZE)
        is False
    )
    assert (
        contact_is_allowed(ContactKind.ROBOT_ENVIRONMENT, StructuredExpertPhase.SMOOTH_APPROACH)
        is False
    )
