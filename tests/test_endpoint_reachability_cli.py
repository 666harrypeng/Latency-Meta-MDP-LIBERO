from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_meta_mdp.endpoint_reachability import (
    EndpointReachabilityResult,
    EndpointReachabilitySpec,
    build_endpoint_reachability_report,
    render_endpoint_reachability_svg,
)


def _spec() -> EndpointReachabilitySpec:
    return EndpointReachabilitySpec.from_mapping(
        {
            "schema_version": 1,
            "calibration_id": "panda_endpoint_reachability_v1",
            "scene_seed": 7,
            "x_values_m": [-0.04, 0.0, 0.04],
            "y_values_m": [-0.16, -0.12, -0.08],
            "interior_margin_cells": 1,
            "camera_size_px": 8,
            "handoff_deadline_us": 3_000_000,
        }
    )


def _successful_results(spec: EndpointReachabilitySpec) -> tuple[EndpointReachabilityResult, ...]:
    return tuple(
        EndpointReachabilityResult(
            point=point,
            succeeded=True,
            terminal_status="success",
            terminal_reason="lift_succeeded",
            terminal_time_us=2_000_000,
            first_contact_us=1_000_000,
            stable_grasp_us=1_040_000,
            handoff_us=1_060_000,
            final_lift_height_m=0.11,
            event_times_us={"success": 2_000_000},
        )
        for point in spec.grid_points()
    )


def test_report_preserves_every_raw_cell_and_derives_the_interior() -> None:
    spec = _spec()
    report = build_endpoint_reachability_report(
        spec=spec,
        results=_successful_results(spec),
        implementation_revision="a" * 40,
        implementation_source_sha256="b" * 64,
        implementation_dirty=False,
    )

    assert report["point_count"] == 9
    assert report["successful_point_count"] == 9
    assert report["within_handoff_deadline_point_count"] == 9
    assert report["failed_point_count"] == 0
    assert report["eligible"]
    assert len(report["results"]) == 9
    assert report["safe_rectangle"]["cell_count"] == 1
    assert report["safe_rectangle"]["x_min_m"] == 0.0
    assert report["safe_rectangle"]["y_min_m"] == -0.12


def test_svg_contains_one_labeled_cell_per_raw_result() -> None:
    spec = _spec()
    report = build_endpoint_reachability_report(
        spec=spec,
        results=_successful_results(spec),
        implementation_revision="a" * 40,
        implementation_source_sha256="b" * 64,
        implementation_dirty=False,
    )

    svg = render_endpoint_reachability_svg(report)

    assert svg.startswith("<svg")
    assert svg.count('class="grid-cell success"') == 9
    assert svg.count("data-point-id=") == 9
    assert 'class="safe-outline"' in svg


def test_publish_is_atomic_and_refuses_to_overwrite(tmp_path: Path) -> None:
    from latency_meta_mdp.cli.calibrate_endpoint_reachability import _publish

    spec = _spec()
    report = build_endpoint_reachability_report(
        spec=spec,
        results=_successful_results(spec),
        implementation_revision="a" * 40,
        implementation_source_sha256="b" * 64,
        implementation_dirty=False,
    )
    run_root = tmp_path / "calibration" / "endpoint_reachability" / "unit-run"

    manifest_path = _publish(run_root=run_root, report=report)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["eligible"]
    assert set(manifest["artifacts"]) == {
        "endpoint_reachability_report.json",
        "endpoint_reachability_heatmap.svg",
    }
    assert all(len(digest) == 64 for digest in manifest["artifacts"].values())
    with pytest.raises(FileExistsError, match="already exists"):
        _publish(run_root=run_root, report=report)


def test_dirty_implementation_is_preserved_as_an_explicit_blocker() -> None:
    spec = _spec()

    report = build_endpoint_reachability_report(
        spec=spec,
        results=_successful_results(spec),
        implementation_revision="a" * 40,
        implementation_source_sha256="b" * 64,
        implementation_dirty=True,
    )

    assert not report["eligible"]
    assert report["implementation_dirty"]
    assert report["blockers"] == ["implementation_dirty"]
