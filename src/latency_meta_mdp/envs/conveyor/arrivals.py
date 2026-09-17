"""Exogenous seed-controlled arrival plans; never passed to the deployment actor."""

from dataclasses import dataclass

import numpy as np

ARRIVAL_PROTOCOL_ID = "count_or_duration_groups_v5"


@dataclass(frozen=True)
class SpawnEvent:
    parcel_id: int
    tick: int
    xy: tuple[float, float]
    color: tuple[float, float, float, float]
    kind: str
    group_id: int


def _duration_schedule(spec, timing):
    """Sample whole one/two-parcel groups with exogenous recovery on both sides."""
    modes = ("moderate", "sparse", "burst")
    mode = str(timing.choice(modes, p=spec.interval_probabilities))
    tick, group_id, kind = spec.first_spawn_tick, 0, "moderate"
    while tick < spec.supply_ticks:
        yield tick, kind, group_id
        if mode == "burst":
            low, high = spec.intervals["burst"]
            second_tick = tick + int(timing.integers(low, high + 1))
            if second_tick < spec.supply_ticks:
                yield second_tick, "burst", group_id
                tick = second_tick
        following = str(timing.choice(modes, p=spec.interval_probabilities))
        gap_kind = "sparse" if mode == "burst" or following == "burst" else mode
        low, high = spec.intervals[gap_kind]
        tick += int(timing.integers(low, high + 1))
        group_id, kind, mode = group_id + 1, gap_kind, following


def _fixed_count_schedule(spec, timing):
    """Keep pair gaps literal; fit relative inter-group gap weights to the window.

    Fixing both count and supply duration changes the absolute inter-group
    intervals. Their sampled values specify relative busy/quiet spacing here.
    """
    modes, remaining = [], spec.spawn_count
    while remaining:
        mode = str(timing.choice(("moderate", "sparse", "burst"), p=spec.interval_probabilities))
        if mode == "burst" and remaining == 1:
            mode = "moderate"
        modes.append(mode)
        remaining -= 2 if mode == "burst" else 1
    gaps, kinds = [], []
    for i, mode in enumerate(modes):
        if i:
            kind = "sparse" if mode == "burst" or modes[i - 1] == "burst" else modes[i - 1]
            low, high = spec.intervals[kind]
            gaps.append(int(timing.integers(low, high + 1)))
            kinds.append(kind)
        if mode == "burst":
            low, high = spec.intervals["burst"]
            gaps.append(int(timing.integers(low, high + 1)))
            kinds.append("burst")
    gaps = np.asarray(gaps, dtype=np.int64)
    free = np.flatnonzero(np.asarray(kinds) != "burst")
    budget = spec.supply_ticks - 1 - spec.first_spawn_tick
    fixed_sum = int(gaps[np.asarray(kinds) == "burst"].sum())
    minimum = int(
        np.ceil((2 * spec.scene.ball_radius_m + spec.clearance_m) / (0.02 * spec.spacing_speed_mps))
    )
    if fixed_sum + len(free) * minimum > budget:
        raise ValueError("spawn_count cannot fit the supply window with non-overlapping gaps")
    if len(free):
        quota = minimum + (budget - fixed_sum - len(free) * minimum) * gaps[free] / gaps[free].sum()
        allocated = np.floor(quota).astype(np.int64)
        remainder = budget - fixed_sum - int(allocated.sum())
        allocated[np.argsort(-(quota - allocated), kind="stable")[:remainder]] += 1
        gaps[free] = allocated
    tick, group_id = spec.first_spawn_tick, 0
    yield tick, "moderate", group_id
    for gap, kind in zip(gaps, kinds):
        tick += int(gap)
        group_id += int(kind != "burst")
        yield tick, kind, group_id


def sample_arrivals(spec, *, seed: int) -> tuple[SpawnEvent, ...]:
    if type(seed) is not int or seed < 0:
        raise ValueError("arrival seed must be a nonnegative integer")
    spec.validate()
    timing_seed, object_seed, color_seed = np.random.SeedSequence(seed).spawn(3)
    timing, objects = np.random.default_rng(timing_seed), np.random.default_rng(object_seed)
    colors = np.random.default_rng(color_seed)
    events = []
    separation = 2 * spec.scene.ball_radius_m + spec.clearance_m
    schedule = (
        _duration_schedule(spec, timing)
        if spec.spawn_count is None
        else _fixed_count_schedule(spec, timing)
    )
    for tick, kind, group_id in schedule:
        for _ in range(128):
            xy = (float(objects.uniform(*spec.spawn_x)), float(objects.uniform(*spec.spawn_y)))
            if kind == "burst":
                previous = events[-1]
                nominal = np.array(previous.xy) + [
                    0,
                    spec.belt_speed_mps * 0.02 * (tick - previous.tick),
                ]
                if np.linalg.norm(nominal - xy) > spec.pair_max_separation_m:
                    continue
            if all(
                np.linalg.norm(
                    np.array(old.xy) + [0, spec.spacing_speed_mps * 0.02 * (tick - old.tick)] - xy
                )
                >= separation
                for old in events
            ):
                break
        else:
            raise ValueError("arrival configuration cannot provide non-overlapping births")
        color = spec.colors[int(colors.integers(len(spec.colors)))]
        events.append(SpawnEvent(len(events), tick, xy, color, kind, group_id))
    return tuple(events)
