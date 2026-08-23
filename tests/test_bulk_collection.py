from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from latency_meta_mdp.bulk_plan import load_bulk_collection_plan
from latency_meta_mdp.outcomes import OutcomeStatus, TerminalReason


def test_first_tranche_selects_exactly_25_train_seeds_per_level() -> None:
    from latency_meta_mdp.bulk_collection import select_first_tranche

    plan = load_bulk_collection_plan(Path("configs/collection/panda_ball_bulk_v1.yaml"))

    attempts = select_first_tranche(plan)

    assert len(attempts) == 75
    assert [(attempt.level, attempt.seed) for attempt in attempts[:27]] == [
        *((1, seed) for seed in range(1_000, 1_025)),
        (2, 1_000),
        (2, 1_001),
    ]
    assert {(attempt.level, attempt.seed) for attempt in attempts} == {
        (level, seed) for level in (1, 2, 3) for seed in range(1_000, 1_025)
    }


def test_first_tranche_gate_is_applied_independently_per_level() -> None:
    from latency_meta_mdp.bulk_collection import BulkAttemptResult, evaluate_first_tranche_gate

    results = []
    for level in (1, 2, 3):
        for seed in range(1_000, 1_025):
            succeeded = not (level == 2 and seed == 1_024)
            results.append(
                BulkAttemptResult(
                    level=level,
                    seed=seed,
                    terminal_status=(
                        OutcomeStatus.SUCCESS if succeeded else OutcomeStatus.FAILURE
                    ),
                    terminal_reason=(
                        TerminalReason.LIFT_SUCCEEDED
                        if succeeded
                        else TerminalReason.GRASP_DEADLINE_MISSED
                    ),
                    handoff_time_us=2_000_000 if succeeded else None,
                    terminal_time_us=2_800_000 if succeeded else 3_000_000,
                    episode_manifest=f"episodes/L{level}/seed_{seed:06d}/manifest.json",
                )
            )

    gate = evaluate_first_tranche_gate(results=tuple(results), minimum_success_rate=0.98)

    assert not gate["passed"]
    assert gate["levels"]["1"]["required_success_count"] == 25
    assert gate["levels"]["1"]["passed"]
    assert gate["levels"]["2"] == {
        "attempt_count": 25,
        "failure_count": 1,
        "passed": False,
        "required_success_count": 25,
        "success_count": 24,
        "success_rate": 0.96,
    }
    assert gate["levels"]["3"]["passed"]


def test_review_video_selection_uses_first_three_successes_per_level() -> None:
    from latency_meta_mdp.bulk_collection import BulkAttemptResult, select_review_attempts

    results = tuple(
        BulkAttemptResult(
            level=level,
            seed=seed,
            terminal_status=(
                OutcomeStatus.FAILURE
                if seed == 1_000
                else OutcomeStatus.SUCCESS
            ),
            terminal_reason=(
                TerminalReason.GRASP_DEADLINE_MISSED
                if seed == 1_000
                else TerminalReason.LIFT_SUCCEEDED
            ),
            handoff_time_us=None if seed == 1_000 else 2_000_000,
            terminal_time_us=3_000_000 if seed == 1_000 else 2_800_000,
            episode_manifest=f"episodes/L{level}/seed_{seed:06d}/manifest.json",
        )
        for level in (1, 2, 3)
        for seed in range(1_000, 1_005)
    )

    selected = select_review_attempts(results=results, count_per_level=3)

    assert [(result.level, result.seed) for result in selected] == [
        (1, 1_001),
        (1, 1_002),
        (1, 1_003),
        (2, 1_001),
        (2, 1_002),
        (2, 1_003),
        (3, 1_001),
        (3, 1_002),
        (3, 1_003),
    ]


def test_tiny_first_tranche_run_writes_belief_attempts_and_review_videos(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.bulk_collection import collect_first_tranche

    plan = load_bulk_collection_plan(Path("configs/collection/panda_ball_bulk_v1.yaml"))
    tiny = replace(
        plan,
        camera_width=8,
        camera_height=8,
        first_tranche_count=1,
        minimum_first_attempt_success_rate=1.0,
        review_video_count_per_level=1,
    )
    tiny_plan_path = tmp_path / "tiny_bulk_plan.yaml"
    raw = yaml.safe_load(Path("configs/collection/panda_ball_bulk_v1.yaml").read_text())
    raw.update(
        {
            "camera_height": tiny.camera_height,
            "camera_width": tiny.camera_width,
            "first_tranche_count": tiny.first_tranche_count,
            "minimum_first_attempt_success_rate": tiny.minimum_first_attempt_success_rate,
            "review_video_count_per_level": tiny.review_video_count_per_level,
        }
    )
    tiny_plan_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    progress = []

    manifest_path = collect_first_tranche(
        project_root=Path.cwd(),
        plan_path=tiny_plan_path,
        output_root=tmp_path,
        run_id="tiny-first-tranche",
        on_attempt=lambda index, total, result: progress.append(
            (index, total, result.level, result.seed, result.succeeded)
        ),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "panda_ball_bulk_first_tranche_v1"
    assert manifest["attempt_count"] == 3
    assert manifest["success_count"] == 3
    assert manifest["gate"]["passed"]
    assert manifest["eligible"] is (not manifest["implementation_dirty"])
    assert len(manifest["admitted_episode_manifests"]) == 3
    assert len(manifest["review_videos"]) == 3
    assert progress == [
        (1, 3, 1, 1_000, True),
        (2, 3, 2, 1_000, True),
        (3, 3, 3, 1_000, True),
    ]
    for relative in manifest["admitted_episode_manifests"]:
        episode_root = manifest_path.parent / Path(relative).parent
        with np.load(episode_root / "arrays.npz", allow_pickle=False) as arrays:
            assert "object_pose" in arrays
            assert "commanded_motion_position" in arrays
            assert "actuator_ctrl" not in arrays
    with pytest.raises(FileExistsError, match="already exists"):
        collect_first_tranche(
            project_root=Path.cwd(),
            plan_path=tiny_plan_path,
            output_root=tmp_path,
            run_id="tiny-first-tranche",
        )


def test_bulk_collection_cli_parses_explicit_paths(tmp_path: Path) -> None:
    from latency_meta_mdp.cli.collect_expert_bulk import _parser

    args = _parser().parse_args(
        [
            "--project-root",
            str(Path.cwd()),
            "--plan",
            "configs/collection/panda_ball_bulk_v1.yaml",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "first-tranche",
        ]
    )

    assert args.output_root == tmp_path
    assert args.run_id == "first-tranche"
