from __future__ import annotations

from pathlib import Path

import pytest

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
