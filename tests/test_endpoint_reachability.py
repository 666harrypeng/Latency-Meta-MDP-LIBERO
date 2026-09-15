from __future__ import annotations

from pathlib import Path

import pytest


def _valid_mapping() -> dict:
    return {
        "schema_version": 1,
        "calibration_id": "panda_endpoint_reachability_v1",
        "scene_seed": 7,
        "x_values_m": [-0.08, -0.04, 0.0, 0.04, 0.08],
        "y_values_m": [-0.20, -0.16, -0.12, -0.08, -0.04],
        "interior_margin_cells": 1,
        "camera_size_px": 8,
        "handoff_deadline_us": 3_000_000,
    }


def _result(point, *, succeeded: bool):
    from latency_meta_mdp.envs.endpoint_reachability import EndpointReachabilityResult

    return EndpointReachabilityResult(
        point=point,
        succeeded=succeeded,
        terminal_status="success" if succeeded else "timeout",
        terminal_reason="lift_succeeded" if succeeded else "calibration_timeout",
        terminal_time_us=2_000_000 if succeeded else 4_000_000,
        first_contact_us=1_000_000 if succeeded else None,
        stable_grasp_us=1_040_000 if succeeded else None,
        handoff_us=1_060_000 if succeeded else None,
        final_lift_height_m=0.10 if succeeded else 0.0,
        event_times_us={"success": 2_000_000} if succeeded else {},
    )


def test_reachability_spec_builds_a_stable_y_major_grid() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import EndpointReachabilitySpec

    spec = EndpointReachabilitySpec.from_mapping(_valid_mapping())

    points = spec.grid_points()
    assert len(points) == 25
    first_six = [
        (point.point_id, point.x_index, point.y_index, point.x_m, point.y_m) for point in points[:6]
    ]
    assert first_six == [
        ("x00-y00", 0, 0, -0.08, -0.20),
        ("x01-y00", 1, 0, -0.04, -0.20),
        ("x02-y00", 2, 0, 0.0, -0.20),
        ("x03-y00", 3, 0, 0.04, -0.20),
        ("x04-y00", 4, 0, 0.08, -0.20),
        ("x00-y01", 0, 1, -0.08, -0.16),
    ]
    assert len({point.point_id for point in points}) == len(points)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x_values_m", [-0.04, -0.04, 0.0], "strictly increasing"),
        ("y_values_m", [-0.12], "at least three"),
        ("interior_margin_cells", 3, "leaves no candidate interior"),
        ("camera_size_px", 0, "positive integer"),
    ],
)
def test_reachability_spec_rejects_ambiguous_or_empty_grids(
    field: str,
    value: object,
    message: str,
) -> None:
    from latency_meta_mdp.envs.endpoint_reachability import EndpointReachabilitySpec

    raw = _valid_mapping()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        EndpointReachabilitySpec.from_mapping(raw)


def test_largest_safe_rectangle_uses_only_cells_with_a_success_margin() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import (
        EndpointReachabilitySpec,
        derive_safe_rectangle,
    )

    spec = EndpointReachabilitySpec.from_mapping(_valid_mapping())
    results = tuple(
        _result(point, succeeded=point.x_index >= 1 and point.y_index >= 1)
        for point in spec.grid_points()
    )

    rectangle = derive_safe_rectangle(spec=spec, results=results)

    assert rectangle is not None
    assert rectangle.to_mapping() == {
        "cell_count": 4,
        "x_index_max": 3,
        "x_index_min": 2,
        "x_max_m": 0.04,
        "x_min_m": 0.0,
        "y_index_max": 3,
        "y_index_min": 2,
        "y_max_m": -0.08,
        "y_min_m": -0.12,
    }


def test_safe_rectangle_rejects_missing_or_duplicate_grid_results() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import (
        EndpointReachabilitySpec,
        derive_safe_rectangle,
    )

    spec = EndpointReachabilitySpec.from_mapping(_valid_mapping())
    complete = [_result(point, succeeded=True) for point in spec.grid_points()]

    with pytest.raises(ValueError, match="exactly one result"):
        derive_safe_rectangle(spec=spec, results=tuple(complete[:-1]))
    with pytest.raises(ValueError, match="exactly one result"):
        derive_safe_rectangle(spec=spec, results=tuple(complete + [complete[0]]))


def test_safe_rectangle_excludes_physical_success_after_the_handoff_deadline() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import (
        EndpointReachabilityResult,
        EndpointReachabilitySpec,
        derive_safe_rectangle,
    )

    raw = _valid_mapping()
    raw["interior_margin_cells"] = 0
    spec = EndpointReachabilitySpec.from_mapping(raw)
    results = [_result(point, succeeded=True) for point in spec.grid_points()]
    late = results[-1]
    results[-1] = EndpointReachabilityResult(
        point=late.point,
        succeeded=True,
        terminal_status="success",
        terminal_reason="lift_succeeded",
        terminal_time_us=3_800_000,
        first_contact_us=3_100_000,
        stable_grasp_us=3_140_000,
        handoff_us=3_160_000,
        final_lift_height_m=0.10,
        event_times_us={"success": 3_800_000},
    )

    rectangle = derive_safe_rectangle(spec=spec, results=tuple(results))

    assert rectangle is not None
    assert rectangle.cell_count == 20
    assert rectangle.y_index_max == 3


def test_checked_in_reachability_config_is_valid() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import load_endpoint_reachability_spec

    spec = load_endpoint_reachability_spec(
        Path("configs/runtime/control/panda_endpoint_reachability_v1.yaml")
    )

    assert spec.calibration_id == "panda_endpoint_reachability_v1"
    assert spec.scene_seed == 7
    assert len(spec.grid_points()) >= 25


def test_canonical_endpoint_completes_the_real_grasp_and_lift_stack() -> None:
    from latency_meta_mdp.envs.endpoint_reachability import (
        EndpointGridPoint,
        load_endpoint_reachability_spec,
        run_endpoint_attempt,
    )

    spec = load_endpoint_reachability_spec(
        Path("configs/runtime/control/panda_endpoint_reachability_v1.yaml")
    )
    result = run_endpoint_attempt(
        project_root=Path.cwd(),
        spec=spec,
        point=EndpointGridPoint(
            point_id="canonical",
            x_index=0,
            y_index=0,
            x_m=0.0,
            y_m=-0.12,
        ),
    )

    assert result.succeeded
    assert result.terminal_status == "success"
    assert result.terminal_reason == "lift_succeeded"
    assert result.terminal_time_us < 4_000_000
    assert result.first_contact_us is not None
    assert result.stable_grasp_us is not None
    assert result.handoff_us is not None
    assert result.event_times_us["success"] == result.terminal_time_us
    assert all(time_us % 2_000 == 0 for time_us in result.event_times_us.values())
