from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.motion import (
    MotionConfig,
    build_motion_profile,
    load_motion_config,
    motion_profile_from_mapping,
    sample_shared_geometry,
)

_CONFIG_ROOT = Path("configs/motion")


def config_for(level: int) -> MotionConfig:
    return load_motion_config(_CONFIG_ROOT / f"dynamic_grasp_lift_l{level}.yaml")


def sampled_times(config: MotionConfig, dt_us: int = 2_000) -> range:
    return range(0, config.anchor_time_us + 1, dt_us)


def test_motion_configs_are_strict_and_level_specific() -> None:
    configs = [config_for(level) for level in range(4)]

    assert [config.level for config in configs] == [0, 1, 2, 3]
    assert [config.profile_id for config in configs] == [
        "dynamic_grasp_lift_l0",
        "dynamic_grasp_lift_l1_v3",
        "dynamic_grasp_lift_l2_v3",
        "dynamic_grasp_lift_l3_v3",
    ]
    assert all(config.anchor_time_us % 20_000 == 0 for config in configs)


def test_level0_is_stationary() -> None:
    config = config_for(0)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)

    samples = [profile.sample(time_us) for time_us in (0, 1_000_000, 3_000_000, 4_000_000)]
    assert all(sample.segment_index == 0 for sample in samples)
    for sample in samples:
        np.testing.assert_allclose(sample.position, [0.0, -0.12, 0.833], atol=0, rtol=0)
        np.testing.assert_allclose(sample.velocity, np.zeros(3), atol=0, rtol=0)
        np.testing.assert_allclose(sample.acceleration, np.zeros(3), atol=0, rtol=0)
        assert sample.terminal is False


def test_level1_is_constant_velocity_and_hits_anchor() -> None:
    config = config_for(1)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)

    samples = [profile.sample(time_us) for time_us in sampled_times(config)]
    velocities = np.stack([sample.velocity for sample in samples])
    accelerations = np.stack([sample.acceleration for sample in samples])
    np.testing.assert_allclose(
        velocities, np.repeat(velocities[0][None, :], len(velocities), axis=0), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(accelerations, 0.0, atol=0, rtol=0)
    speed = np.linalg.norm(velocities[0, :2])
    chord_length = speed * config.anchor_time_us / 1_000_000
    assert config.min_chord_length_m < chord_length < config.max_chord_length_m
    geometry = sample_shared_geometry(config=config, seed=7)
    np.testing.assert_array_equal(samples[-1].position[:2], geometry.end_xy)
    assert samples[-1].terminal is True
    assert all(config.contains(sample.position[:2]) for sample in samples)


def test_level2_is_smooth_curved_and_bounded() -> None:
    config = config_for(2)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)

    samples = [profile.sample(time_us) for time_us in sampled_times(config)]
    speeds = np.array([np.linalg.norm(sample.velocity[:2]) for sample in samples])
    accelerations = np.array([np.linalg.norm(sample.acceleration[:2]) for sample in samples])
    assert speeds.max() <= config.max_speed_mps + 1e-12
    assert accelerations.max() <= config.max_continuous_acceleration_mps2 + 1e-12
    assert np.ptp(speeds) >= 1e-3
    chord = samples[-1].position[:2] - samples[0].position[:2]
    chord_length = np.linalg.norm(chord)
    offsets = []
    for sample in samples:
        relative = sample.position[:2] - samples[0].position[:2]
        cross_magnitude = abs(chord[0] * relative[1] - chord[1] * relative[0])
        offsets.append(cross_magnitude / chord_length)
    assert config.curve_deviation_range_m[0] <= max(offsets) <= (
        config.curve_deviation_range_m[1]
    )
    geometry = sample_shared_geometry(config=config, seed=7)
    np.testing.assert_array_equal(samples[-1].position[:2], geometry.end_xy)
    assert all(config.contains(sample.position[:2]) for sample in samples)


def test_level3_has_position_continuity_and_bounded_velocity_jumps() -> None:
    config = config_for(3)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)

    assert 2 <= profile.segment_count <= 3
    assert len(profile.change_times_us) == profile.segment_count - 1
    for change_time_us in profile.change_times_us:
        left = profile.sample_side(change_time_us, side="left")
        right = profile.sample_side(change_time_us, side="right")
        np.testing.assert_allclose(left.position, right.position, atol=1e-12, rtol=0)
        jump = np.linalg.norm(right.velocity[:2] - left.velocity[:2])
        assert config.velocity_jump_range_mps[0] <= jump <= config.velocity_jump_range_mps[1]
    samples = [profile.sample(time_us) for time_us in sampled_times(config)]
    assert max(np.linalg.norm(sample.velocity[:2]) for sample in samples) <= (
        config.max_speed_mps + 1e-12
    )
    assert all(config.contains(sample.position[:2]) for sample in samples)
    geometry = sample_shared_geometry(config=config, seed=7)
    np.testing.assert_array_equal(samples[-1].position[:2], geometry.end_xy)


def test_profiles_are_seed_deterministic() -> None:
    config = config_for(3)
    first = build_motion_profile(config=config, seed=17, workspace_z=0.833)
    repeat = build_motion_profile(config=config, seed=17, workspace_z=0.833)
    different = build_motion_profile(config=config, seed=18, workspace_z=0.833)

    times = range(0, config.anchor_time_us + 1, 20_000)
    first_positions = np.stack([first.sample(time_us).position for time_us in times])
    repeat_positions = np.stack([repeat.sample(time_us).position for time_us in times])
    different_positions = np.stack([different.sample(time_us).position for time_us in times])
    np.testing.assert_array_equal(first_positions, repeat_positions)
    assert not np.array_equal(first_positions, different_positions)


def test_train_seed_bank_has_both_level2_curve_directions() -> None:
    config = config_for(2)
    signs = []
    for seed in range(1_000, 1_200):
        profile = build_motion_profile(config=config, seed=seed, workspace_z=0.833)
        start = profile.sample(0).position[:2]
        midpoint = profile.sample(config.anchor_time_us // 2).position[:2]
        end = profile.sample(config.anchor_time_us).position[:2]
        chord = end - start
        deviation = midpoint - 0.5 * (start + end)
        signs.append(int(np.sign(chord[0] * deviation[1] - chord[1] * deviation[0])))

    assert set(signs) == {-1, 1}
    assert signs.count(-1) >= 70
    assert signs.count(1) >= 70


def test_train_seed_bank_chord_lengths_have_no_boundary_atoms() -> None:
    config = config_for(1)
    chord_lengths = []
    for seed in range(1_000, 1_200):
        geometry = sample_shared_geometry(config=config, seed=seed)
        chord_lengths.append(float(np.linalg.norm(geometry.end_xy - geometry.start_xy)))

    assert all(
        config.min_chord_length_m < length < config.max_chord_length_m
        for length in chord_lengths
    )
    assert all(
        abs(length - config.min_chord_length_m) > 1e-12
        and abs(length - config.max_chord_length_m) > 1e-12
        for length in chord_lengths
    )


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_motion_profile_round_trip_preserves_exact_samples(level: int) -> None:
    config = config_for(level)
    profile = build_motion_profile(config=config, seed=23, workspace_z=0.833)

    mapping = profile.to_mapping()
    json.dumps(mapping)
    restored = motion_profile_from_mapping(mapping, config=config)

    comparison_times = set(range(0, config.anchor_time_us + 1, 20_000))
    comparison_times.update(profile.change_times_us)
    comparison_times.add(config.anchor_time_us + 400_000)
    for time_us in sorted(comparison_times):
        expected = profile.sample(time_us)
        actual = restored.sample(time_us)
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.velocity, expected.velocity)
        np.testing.assert_array_equal(actual.acceleration, expected.acceleration)
        assert actual.segment_index == expected.segment_index
        assert actual.terminal == expected.terminal


@pytest.mark.parametrize("level", [1, 2, 3])
def test_dynamic_profiles_hold_anchor_after_terminal_deadline(level: int) -> None:
    config = config_for(level)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)

    at_deadline = profile.sample(config.anchor_time_us)
    after_deadline = profile.sample(config.anchor_time_us + 400_000)
    assert at_deadline.terminal is True
    assert after_deadline.terminal is True
    np.testing.assert_array_equal(after_deadline.position[:2], at_deadline.position[:2])
    np.testing.assert_allclose(after_deadline.velocity, np.zeros(3), atol=0, rtol=0)
    np.testing.assert_allclose(after_deadline.acceleration, np.zeros(3), atol=0, rtol=0)


def test_deserialization_rejects_discontinuous_or_off_grid_segments() -> None:
    config = config_for(3)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)
    discontinuous = profile.to_mapping()
    discontinuous["segments"][1]["start_xy"][0] += 0.01
    with pytest.raises(ValueError, match="position continuity"):
        motion_profile_from_mapping(discontinuous, config=config)

    off_grid = profile.to_mapping()
    off_grid["segments"][0]["duration_us"] += 1
    with pytest.raises(ValueError, match="20 ms formal-grid"):
        motion_profile_from_mapping(off_grid, config=config)


def test_deserialization_rejects_out_of_bounds_profile() -> None:
    config = config_for(1)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)
    mapping = profile.to_mapping()
    mapping["start_xy"] = [100.0, 100.0]
    mapping["end_xy"] = (
        np.asarray(mapping["start_xy"])
        + np.asarray(mapping["velocity_xy"]) * config.anchor_time_us / 1_000_000
    ).tolist()

    with pytest.raises(ValueError, match="physical contract"):
        motion_profile_from_mapping(mapping, config=config)


def test_deserialization_rejects_inconsistent_level1_endpoint_velocity() -> None:
    config = config_for(1)
    profile = build_motion_profile(config=config, seed=7, workspace_z=0.833)
    mapping = profile.to_mapping()
    mapping["end_xy"] = mapping["start_xy"]

    with pytest.raises(ValueError, match="endpoint and velocity"):
        motion_profile_from_mapping(mapping, config=config)


def test_motion_config_rejects_nonfinite_scalar(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.yaml"
    path.write_text(
        (_CONFIG_ROOT / "dynamic_grasp_lift_l1.yaml")
        .read_text()
        .replace("max_speed_mps: 0.18", "max_speed_mps: .inf")
    )

    with pytest.raises(ValueError, match="max_speed_mps must be finite"):
        load_motion_config(path)


def test_motion_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text((_CONFIG_ROOT / "dynamic_grasp_lift_l1.yaml").read_text() + "extra: 1\n")

    with pytest.raises(ValueError, match="unknown motion config fields"):
        load_motion_config(path)
