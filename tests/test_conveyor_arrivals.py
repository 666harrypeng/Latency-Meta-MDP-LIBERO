from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def spec():
    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec

    return load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml"))


def test_seeded_feed_is_external_bounded_and_has_no_nominal_overlap():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    config = spec()
    a = sample_arrivals(config, seed=7)
    np.random.seed(9)
    np.random.random(100)
    assert a == sample_arrivals(config, seed=7)
    assert a != sample_arrivals(config, seed=17)
    assert all(0 <= e.tick < config.supply_ticks for e in a)
    assert [e.parcel_id for e in a] == list(range(len(a)))
    assert all(e.color in config.colors for e in a)
    for i, event in enumerate(a):
        for old in a[:i]:
            # The conservative schedule screen is not a runtime position override.
            p = np.array(old.xy) + [0, config.spacing_speed_mps * 0.02 * (event.tick - old.tick)]
            assert (
                np.linalg.norm(p - event.xy) >= 2 * config.scene.ball_radius_m + config.clearance_m
            )
    assert not any(x.kind == y.kind == "burst" for x, y in zip(a, a[1:]))


def test_invalid_geometry_is_rejected_before_environment_construction():
    config = spec()
    with pytest.raises(ValueError):
        replace(config, belt_speed_mps=-1).validate()
    with pytest.raises(ValueError):
        replace(config, spawn_x=(3.0, 4.0)).validate()


def test_palette_changes_do_not_change_birth_times_or_positions():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    original = spec()
    recolored = replace(original, colors=((0.1, 0.8, 0.5, 1.0),))
    a = sample_arrivals(original, seed=27)
    b = sample_arrivals(recolored, seed=27)
    assert [(e.tick, e.xy, e.kind) for e in a] == [(e.tick, e.xy, e.kind) for e in b]


def test_close_pairs_have_bounded_spacing_and_a_recovery_gap():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    config = replace(
        spec(),
        spawn_count=None,
        spacing_speed_mps=0.102,
        pair_max_separation_m=0.13,
        intervals={"moderate": (200, 250), "sparse": (325, 425), "burst": (40, 50)},
        interval_probabilities=(0.68, 0.20, 0.12),
    )
    events = sample_arrivals(config, seed=0)
    pairs = [i for i, e in enumerate(events) if e.kind == "burst"]
    assert pairs
    for i in pairs:
        previous, current = events[i - 1 : i + 1]
        dt = (current.tick - previous.tick) * 0.02
        spacing = np.array(previous.xy) + [0, dt * config.belt_speed_mps] - current.xy
        assert 0.8 <= dt <= 1.0
        assert (
            2 * config.scene.ball_radius_m + config.clearance_m <= np.linalg.norm(spacing) <= 0.13
        )
        if i + 1 < len(events):
            assert events[i + 1].kind == "sparse"
            assert events[i + 1].tick - current.tick >= config.intervals["sparse"][0]


def test_pair_groups_have_recovery_before_as_well_as_after():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    config = replace(spec(), spawn_count=None)
    checked = 0
    for seed in range(32):
        events = sample_arrivals(config, seed=seed)
        for i, event in enumerate(events):
            if event.kind == "burst" and i >= 2:
                checked += 1
                assert events[i - 1].tick - events[i - 2].tick >= config.intervals["sparse"][0]
                assert event.group_id == events[i - 1].group_id
                assert event.group_id != events[i - 2].group_id
    assert checked > 0


def test_parallel_view_changes_only_main_camera_translation():
    from dataclasses import asdict

    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals
    from latency_meta_mdp.envs.conveyor.config import apply_view_profile

    original = spec()
    overview = apply_view_profile(original, Path("configs/legacy/conveyor_sort/overview-view.yaml"))
    a, b = asdict(original), asdict(overview)
    assert {k for k in a if a[k] != b[k]} <= {"agentview_retreat_m", "agentview_upstream_shift_m"}
    assert overview.agentview_retreat_m > original.agentview_retreat_m
    assert sample_arrivals(original, seed=27) == sample_arrivals(overview, seed=27)


def test_fixed_count_preserves_pairs_and_fills_the_supply_window():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    config = replace(spec(), spawn_count=10)
    for seed in range(32):
        events = sample_arrivals(config, seed=seed)
        assert len(events) == 10
        assert events[0].tick == config.first_spawn_tick
        assert events[-1].tick == config.supply_ticks - 1
        assert all(a.tick < b.tick < config.supply_ticks for a, b in zip(events, events[1:]))
        for a, b in zip(events, events[1:]):
            if b.kind == "burst":
                assert (
                    config.intervals["burst"][0] <= b.tick - a.tick <= config.intervals["burst"][1]
                )
                assert a.group_id == b.group_id
        assert events == sample_arrivals(config, seed=seed)


def test_fixed_count_rejects_unrepresentable_schedule_and_supports_one_ball():
    from latency_meta_mdp.envs.conveyor.arrivals import sample_arrivals

    assert len(sample_arrivals(replace(spec(), spawn_count=1), seed=0)) == 1
    with pytest.raises(ValueError):
        sample_arrivals(replace(spec(), spawn_count=10, supply_ticks=40), seed=0)
    for count in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            replace(spec(), spawn_count=count).validate()
