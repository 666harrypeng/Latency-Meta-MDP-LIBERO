from __future__ import annotations

from latency_meta_mdp.calibration import run_g1_calibration, validate_g1_report

_G0_REFERENCE = {
    "manifest": "certification/g0/g0_runtime/manifest.json",
    "sha256": "a" * 64,
}


def test_real_g1_calibration_certifies_one_second_and_boundary_coherence() -> None:
    report = run_g1_calibration(
        seed=7,
        camera_width=64,
        camera_height=64,
        prerequisites={"g0": _G0_REFERENCE},
    )
    validated = validate_g1_report(report)

    assert validated["eligible"] is True
    assert validated["blockers"] == []
    assert validated["integrator"] == "Euler"
    assert validated["external_control_callback_present"] is False
    assert validated["prerequisites"] == {"g0": _G0_REFERENCE}
    assert validated["stock_step"]["step1_count"] == 10
    assert validated["stock_step"]["step2_count"] == 10
    assert validated["stock_step"]["goal_refresh_count"] == 1
    assert validated["stock_step"]["control_refresh_count"] == 10
    assert validated["stock_step"]["returned_observation_age_us"] == 0
    for lane in validated["lanes"]:
        assert lane["physics_step_count"] == 500
        assert lane["formal_tick_count"] == 50
        assert lane["compatibility_tick_count"] == 10
        assert lane["snapshot_count"] == 51
        assert lane["world_write_count"] == 501
        assert lane["step1_count"] == 501
        assert lane["step2_count"] == 500
        assert lane["goal_refresh_count"] == 50
        assert lane["control_refresh_count"] == 500
        assert lane["camera_names"] == ["agentview", "robot0_eye_in_hand"]
        assert lane["camera_source_mismatch_count"] == 0
        assert lane["marker_missing_frame_count"] == 0
        assert lane["marker_centroid_span_pixels"] > 1.0
        assert lane["marker_largest_jump_tick"] == 25
        assert lane["marker_expected_jump_centroid_pixels"] >= 3.0
        assert lane["marker_command_max_abs_error"] <= 1e-12
        assert lane["robot_hold_max_abs_drift_rad"] <= 1e-3

    comparison = validated["repeat_comparison"]
    assert comparison["state_max_abs_error"] <= 1e-9
    assert comparison["shared_100ms_state_max_abs_error"] <= 1e-9
    assert comparison["segmentation_centroid_max_error_pixels"] <= 1.0
    assert comparison["rgb_max_abs_error"] == 0


def test_g1_validator_rejects_a_false_completed_clock() -> None:
    report = run_g1_calibration(
        seed=7,
        camera_width=32,
        camera_height=32,
        prerequisites={"g0": _G0_REFERENCE},
    )
    report["lanes"][0]["physics_step_count"] = 499

    validated = validate_g1_report(report)

    assert validated["eligible"] is False
    assert "lane_0_physics_step_count_mismatch" in validated["blockers"]
