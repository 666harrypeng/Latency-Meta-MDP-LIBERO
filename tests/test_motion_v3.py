from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.motion import (
    CubicPolynomialProfile,
    PiecewisePolynomialProfile,
    build_motion_profile,
    load_motion_config,
    sample_shared_geometry,
)

_CONFIG_ROOT = Path("configs/motion")


def _config(level: int):
    return load_motion_config(_CONFIG_ROOT / f"dynamic_grasp_lift_l{level}.yaml")


def test_dynamic_configs_share_one_calibrated_v3_geometry_contract() -> None:
    configs = [_config(level) for level in (1, 2, 3)]

    assert [config.profile_id for config in configs] == [
        "dynamic_grasp_lift_l1_v3",
        "dynamic_grasp_lift_l2_v3",
        "dynamic_grasp_lift_l3_v3",
    ]
    assert all(config.schema_version == 2 for config in configs)
    assert all(config.path_bounds_xy == (-0.12, 0.12, -0.20, 0.20) for config in configs)
    shared_fields = (
        "path_bounds_xy",
        "anchor_time_us",
        "min_chord_length_m",
        "max_chord_length_m",
        "min_path_length_m",
        "max_path_length_m",
        "max_speed_mps",
        "max_continuous_acceleration_mps2",
        "curve_deviation_range_m",
        "segment_count_three_probability",
        "cubic_segment_probability",
        "two_segment_change_time_us_range",
        "three_segment_first_change_time_us_range",
        "three_segment_second_change_time_us_range",
        "turn_angle_degrees_range",
        "velocity_jump_range_mps",
        "max_retries",
    )
    for field in shared_fields:
        assert len({getattr(config, field) for config in configs}) == 1


def test_shared_geometry_is_level_independent_and_seeded() -> None:
    configs = [_config(level) for level in (1, 2, 3)]

    same_seed = [sample_shared_geometry(config=config, seed=1_007) for config in configs]
    different_seed = sample_shared_geometry(config=configs[0], seed=1_008)

    for geometry in same_seed[1:]:
        np.testing.assert_array_equal(geometry.start_xy, same_seed[0].start_xy)
        np.testing.assert_array_equal(geometry.end_xy, same_seed[0].end_xy)
    assert not np.array_equal(different_seed.start_xy, same_seed[0].start_xy)
    assert not np.array_equal(different_seed.end_xy, same_seed[0].end_xy)
    chord_length = np.linalg.norm(same_seed[0].end_xy - same_seed[0].start_xy)
    assert configs[0].min_chord_length_m < chord_length < configs[0].max_chord_length_m
    assert configs[0].contains(same_seed[0].start_xy)
    assert configs[0].contains(same_seed[0].end_xy)


def test_same_master_seed_gives_all_levels_identical_start_and_endpoint() -> None:
    for seed in range(1_000, 1_025):
        profiles = [
            build_motion_profile(config=_config(level), seed=seed, workspace_z=0.833)
            for level in (1, 2, 3)
        ]
        starts = [profile.sample(0).position[:2] for profile in profiles]
        ends = [profile.sample(3_000_000).position[:2] for profile in profiles]
        for start in starts[1:]:
            np.testing.assert_array_equal(start, starts[0])
        for end in ends[1:]:
            np.testing.assert_array_equal(end, ends[0])


def test_level1_is_the_exact_straight_reference_for_shared_geometry() -> None:
    config = _config(1)
    geometry = sample_shared_geometry(config=config, seed=1_007)
    profile = build_motion_profile(config=config, seed=1_007, workspace_z=0.833)
    times = range(0, config.anchor_time_us + 1, 20_000)
    samples = [profile.sample(time_us) for time_us in times]

    np.testing.assert_array_equal(samples[0].position[:2], geometry.start_xy)
    np.testing.assert_array_equal(samples[-1].position[:2], geometry.end_xy)
    velocities = np.stack([sample.velocity[:2] for sample in samples])
    accelerations = np.stack([sample.acceleration[:2] for sample in samples])
    np.testing.assert_allclose(
        velocities,
        np.repeat(velocities[0][None, :], len(velocities), axis=0),
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(accelerations, np.zeros_like(accelerations))


def test_level2_is_one_four_waypoint_degree_three_polynomial() -> None:
    config = _config(2)
    profile = build_motion_profile(config=config, seed=1_007, workspace_z=0.833)

    assert isinstance(profile, CubicPolynomialProfile)
    np.testing.assert_array_equal(profile.waypoint_times_us, [0, 1_000_000, 2_000_000, 3_000_000])
    for time_us, waypoint in zip(
        profile.waypoint_times_us,
        profile.waypoint_positions_xy,
        strict=True,
    ):
        np.testing.assert_allclose(
            profile.sample(int(time_us)).position[:2], waypoint, atol=1e-12, rtol=0
        )
    assert np.linalg.norm(profile.coefficients_xy[3]) > 1e-4

    samples = [
        profile.sample(time_us) for time_us in range(0, config.anchor_time_us + 1, 2_000)
    ]
    positions = np.stack([sample.position[:2] for sample in samples])
    speeds = np.array([np.linalg.norm(sample.velocity[:2]) for sample in samples])
    accelerations = np.array([np.linalg.norm(sample.acceleration[:2]) for sample in samples])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    chord = positions[-1] - positions[0]
    relative = positions - positions[0]
    deviation = np.abs(
        chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
    ) / np.linalg.norm(chord)

    assert config.min_path_length_m <= path_length <= config.max_path_length_m
    assert deviation.max() >= config.curve_deviation_range_m[0]
    assert speeds.max() <= config.max_speed_mps + 1e-12
    assert accelerations.max() <= config.max_continuous_acceleration_mps2 + 1e-12
    assert all(config.contains(position) for position in positions)


def test_level2_train_bank_has_balanced_obvious_curve_directions() -> None:
    config = _config(2)
    signs = []
    deviations = []
    for seed in range(1_000, 1_200):
        profile = build_motion_profile(config=config, seed=seed, workspace_z=0.833)
        start = profile.sample(0).position[:2]
        midpoint = profile.sample(config.anchor_time_us // 2).position[:2]
        end = profile.sample(config.anchor_time_us).position[:2]
        chord = end - start
        displacement = midpoint - 0.5 * (start + end)
        signed = chord[0] * displacement[1] - chord[1] * displacement[0]
        signs.append(int(np.sign(signed)))
        deviations.append(abs(signed) / np.linalg.norm(chord))

    assert set(signs) == {-1, 1}
    assert signs.count(-1) >= 70
    assert signs.count(1) >= 70
    assert min(deviations) >= config.curve_deviation_range_m[0]


def test_level3_has_early_position_continuous_bounded_turns() -> None:
    config = _config(3)
    profile = build_motion_profile(config=config, seed=1_007, workspace_z=0.833)

    assert isinstance(profile, PiecewisePolynomialProfile)
    assert profile.segment_count in {2, 3}
    assert {segment.kind for segment in profile.segments} <= {"line", "cubic"}
    assert len(profile.change_times_us) == profile.segment_count - 1
    if profile.segment_count == 2:
        lower, upper = config.two_segment_change_time_us_range
        assert lower <= profile.change_times_us[0] <= upper
    else:
        first_lower, first_upper = config.three_segment_first_change_time_us_range
        second_lower, second_upper = config.three_segment_second_change_time_us_range
        assert first_lower <= profile.change_times_us[0] <= first_upper
        assert second_lower <= profile.change_times_us[1] <= second_upper

    for change_time_us in profile.change_times_us:
        left = profile.sample_side(change_time_us, side="left")
        right = profile.sample_side(change_time_us, side="right")
        np.testing.assert_allclose(left.position, right.position, atol=1e-12, rtol=0)
        left_velocity = left.velocity[:2]
        right_velocity = right.velocity[:2]
        cosine = np.clip(
            np.dot(left_velocity, right_velocity)
            / (np.linalg.norm(left_velocity) * np.linalg.norm(right_velocity)),
            -1.0,
            1.0,
        )
        angle_degrees = float(np.degrees(np.arccos(cosine)))
        jump = float(np.linalg.norm(right_velocity - left_velocity))
        assert config.turn_angle_degrees_range[0] <= angle_degrees <= (
            config.turn_angle_degrees_range[1]
        )
        assert config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]

    samples = [
        profile.sample(time_us) for time_us in range(0, config.anchor_time_us + 1, 2_000)
    ]
    positions = np.stack([sample.position[:2] for sample in samples])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    chord = positions[-1] - positions[0]
    relative = positions - positions[0]
    deviations = np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0])
    assert config.min_path_length_m <= path_length <= config.max_path_length_m
    assert deviations.max() / np.linalg.norm(chord) >= config.curve_deviation_range_m[0]
    assert all(config.contains(position) for position in positions)


def test_level3_train_bank_contains_both_segment_counts_and_segment_kinds() -> None:
    config = _config(3)
    counts = []
    kinds = set()
    for seed in range(1_000, 1_200):
        profile = build_motion_profile(config=config, seed=seed, workspace_z=0.833)
        counts.append(profile.segment_count)
        kinds.update(segment.kind for segment in profile.segments)

    assert set(counts) == {2, 3}
    assert 120 <= counts.count(2) <= 180
    assert 20 <= counts.count(3) <= 80
    assert kinds == {"line", "cubic"}
