from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.control_calibration import run_control_calibration

_CONTROL_CONFIG = Path("configs/control/panda_osc_pose_delta_v1.yaml")
_CALIBRATION_CONFIG = Path("configs/control/panda_control_calibration_v1.yaml")
_TASK_CONFIG = Path("configs/task/dynamic_grasp_lift_l0.yaml")


def test_control_calibration_certifies_clock_tracking_gripper_and_safety() -> None:
    report = run_control_calibration(
        control_config_path=_CONTROL_CONFIG,
        calibration_config_path=_CALIBRATION_CONFIG,
        task_config_path=_TASK_CONFIG,
    )

    assert report["eligible"] is True
    assert report["blockers"] == []
    assert report["contract_id"] == "panda_osc_pose_delta_v1"
    assert report["calibration_id"] == "panda_osc_pose_delta_calibration_v1"
    assert report["physics_step_count"] == 2_100
    assert report["formal_tick_count"] == 210
    assert report["physics_control_sample_count"] == 2_100
    assert report["final_time_us"] == 4_200_000
    assert report["hold_eef_drift_m"] < 1e-8
    assert report["hold_joint_drift_rad"] < 1e-8
    assert report["hold_eef_drift_m"] == report["phase_reports"]["hold_open"][
        "max_eef_displacement_m"
    ]
    assert report["hold_joint_drift_rad"] == report["phase_reports"]["hold_open"][
        "max_joint_displacement_rad"
    ]
    assert report["positive_x_average_velocity_mps"] > 0.20
    assert report["negative_x_average_velocity_mps"] < -0.20
    assert report["arm_saturated_physics_step_fraction"] < 0.01
    assert report["arm_saturated_actuator_step_fraction"] < 0.01
    assert report["minimum_joint_limit_margin_rad"] > 0.4
    assert report["closed_gripper_width_m"] < 0.01
    assert report["reopened_gripper_width_m"] > 0.07
    assert report["return_position_error_m"] < 0.02
    assert len(report["implementation_revision"]) == 40
    assert len(report["implementation_source_sha256"]) == 64
    assert len(report["trace_sha256"]) == 64
    assert len(report["physics_trace"]) == 2_100
    assert report["physics_trace"][0]["time_us"] == 0
    assert report["physics_trace"][-1]["time_us"] == 4_198_000
    assert len(report["boundary_trace"]) == 211
    assert report["boundary_trace"][0]["time_us"] == 0
    assert report["boundary_trace"][-1]["time_us"] == 4_200_000
    recomputed_saturation = float(
        np.mean([any(row["saturated_arm_joints"]) for row in report["physics_trace"]])
    )
    assert recomputed_saturation == report["arm_saturated_physics_step_fraction"]
