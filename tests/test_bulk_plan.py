from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from latency_meta_mdp.bulk_coverage import write_bulk_motion_coverage
from latency_meta_mdp.bulk_plan import load_bulk_collection_plan
from latency_meta_mdp.recording import RecordProfile


def test_bulk_plan_locks_level_specific_train_and_evaluation_seed_banks() -> None:
    plan = load_bulk_collection_plan(
        Path("configs/collection/panda_ball_bulk_v1.yaml")
    )

    assert plan.collection_id == "panda_ball_bulk_v1"
    assert plan.levels == (1, 2, 3)
    assert plan.record_profile is RecordProfile.BELIEF
    assert plan.train.seeds == tuple(range(1_000, 1_200))
    assert plan.development.seeds == tuple(range(2_000, 2_050))
    assert plan.test.seeds == tuple(range(3_000, 3_100))
    assert plan.train.collect_expert is True
    assert plan.development.collect_expert is False
    assert plan.test.collect_expert is False
    assert plan.excluded_pilot_seeds == (10, 11, 12)
    assert plan.first_tranche_count == 25
    assert plan.minimum_first_attempt_success_rate == 0.98
    assert plan.same_master_seeds_across_levels is True


def test_bulk_plan_rejects_overlapping_seed_banks() -> None:
    plan = load_bulk_collection_plan(
        Path("configs/collection/panda_ball_bulk_v1.yaml")
    )

    with pytest.raises(ValueError, match="seed banks must be disjoint"):
        replace(plan, development=replace(plan.development, start=1_100))


def test_bulk_coverage_cli_parses_explicit_paths(tmp_path: Path) -> None:
    cli = importlib.import_module("latency_meta_mdp.cli.audit_bulk_motion")
    args = cli._parser().parse_args(
        [
            "--project-root",
            str(Path.cwd()),
            "--plan",
            "configs/collection/panda_ball_bulk_v1.yaml",
            "--output-dir",
            str(tmp_path / "coverage"),
        ]
    )

    assert args.output_dir == tmp_path / "coverage"


def test_bulk_motion_coverage_passes_all_three_seed_banks(tmp_path: Path) -> None:
    plan_path = Path("configs/collection/panda_ball_bulk_v1.yaml")

    output = tmp_path / "coverage"
    manifest = write_bulk_motion_coverage(
        project_root=Path.cwd(),
        plan_path=plan_path,
        output_dir=output,
    )
    coverage = json.loads(manifest.read_text(encoding="utf-8"))

    assert coverage["coverage_passed"] is True
    assert [row["name"] for row in coverage["splits"]] == [
        "train",
        "development",
        "test",
    ]
    train = coverage["splits"][0]
    assert train["seed_count"] == 200
    assert train["shared_geometry_matches_across_levels"] is True
    level2 = train["levels"][1]
    assert level2["curve_sign_counts"]["negative"] >= 70
    assert level2["curve_sign_counts"]["positive"] >= 70
    assert level2["cubic_degree_counts"] == {"3": 200}
    level3 = train["levels"][2]
    assert 120 <= level3["segment_count_counts"]["2"] <= 180
    assert 20 <= level3["segment_count_counts"]["3"] <= 80
    assert level3["segment_kind_counts"]["line"] > 0
    assert level3["segment_kind_counts"]["cubic"] > 0
    assert all(
        level["chord_boundary_hit_count"] == 0
        for split in coverage["splits"]
        for level in split["levels"]
    )

    assert coverage["format_id"] == "panda_ball_bulk_motion_coverage_v2"
    assert coverage["eligible"] is (not coverage["implementation_dirty"])
