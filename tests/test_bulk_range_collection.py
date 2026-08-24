from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from latency_meta_mdp.bulk_plan import load_bulk_collection_plan


def _plan():
    return load_bulk_collection_plan(Path("configs/collection/panda_ball_bulk_v1.yaml"))


def test_continuation_range_selects_exact_unique_level_seed_identities() -> None:
    from latency_meta_mdp.bulk_collection import select_seed_range

    attempts = select_seed_range(
        _plan(),
        levels=(1, 2, 3),
        seed_start=1025,
        seed_count=175,
    )

    assert len(attempts) == 525
    assert len({(attempt.level, attempt.seed) for attempt in attempts}) == 525
    assert attempts[0].level == 1 and attempts[0].seed == 1025
    assert attempts[174].level == 1 and attempts[174].seed == 1199
    assert attempts[175].level == 2 and attempts[175].seed == 1025
    assert attempts[-1].level == 3 and attempts[-1].seed == 1199


def test_first_tranche_selection_is_the_same_seed_range_contract() -> None:
    from latency_meta_mdp.bulk_collection import select_first_tranche, select_seed_range

    plan = _plan()

    assert select_seed_range(
        plan,
        levels=plan.levels,
        seed_start=1000,
        seed_count=25,
    ) == select_first_tranche(plan)


@pytest.mark.parametrize(
    ("levels", "seed_start", "seed_count"),
    (
        ((1, 2, 3), 999, 1),
        ((1, 2, 3), 1199, 2),
        ((3, 2, 1), 1025, 1),
        ((1, 1), 1025, 1),
        ((), 1025, 1),
    ),
)
def test_seed_range_rejects_out_of_bank_or_invalid_levels(
    levels: tuple[int, ...],
    seed_start: int,
    seed_count: int,
) -> None:
    from latency_meta_mdp.bulk_collection import select_seed_range

    with pytest.raises(ValueError):
        select_seed_range(
            _plan(),
            levels=levels,
            seed_start=seed_start,
            seed_count=seed_count,
        )


def test_tiny_explicit_range_run_is_atomic_and_preserves_range_identity(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.bulk_collection import collect_expert_range

    raw = yaml.safe_load(Path("configs/collection/panda_ball_bulk_v1.yaml").read_text())
    raw["camera_width"] = 8
    raw["camera_height"] = 8
    tiny_plan = tmp_path / "tiny_plan.yaml"
    tiny_plan.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    progress = []

    manifest_path = collect_expert_range(
        project_root=Path.cwd(),
        plan_path=tiny_plan,
        output_root=tmp_path,
        run_id="tiny-range",
        levels=(1, 2, 3),
        seed_start=1025,
        seed_count=1,
        on_attempt=lambda index, total, result: progress.append(
            (index, total, result.level, result.seed, result.succeeded)
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "panda_ball_bulk_range_v1"
    assert manifest["levels"] == [1, 2, 3]
    assert manifest["seed_start"] == 1025
    assert manifest["seed_count_per_level"] == 1
    assert manifest["attempt_count"] == 3
    assert manifest["success_count"] == 3
    assert len(manifest["review_videos"]) == 3
    assert progress == [
        (1, 3, 1, 1025, True),
        (2, 3, 2, 1025, True),
        (3, 3, 3, 1025, True),
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        collect_expert_range(
            project_root=Path.cwd(),
            plan_path=tiny_plan,
            output_root=tmp_path,
            run_id="tiny-range",
            levels=(1, 2, 3),
            seed_start=1025,
            seed_count=1,
        )


def test_range_collection_cli_parses_explicit_range(tmp_path: Path) -> None:
    from latency_meta_mdp.cli.collect_expert_range import _parser

    args = _parser().parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--run-id",
            "continuation",
            "--levels",
            "1",
            "2",
            "3",
            "--seed-start",
            "1025",
            "--seed-count",
            "175",
        ]
    )

    assert args.levels == [1, 2, 3]
    assert args.seed_start == 1025
    assert args.seed_count == 175
