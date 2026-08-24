from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_formal_split_config_uses_train_and_validation_only() -> None:
    from latency_meta_mdp.episode_split import load_episode_split_plan

    plan = load_episode_split_plan(
        Path("configs/data/formal_belief_train_val_v1.yaml")
    )

    assert plan.source_seed_start == 1000
    assert plan.source_seed_count == 200
    assert plan.split_names == ("train", "validation")
    assert plan.split_for_seed(1000) == "train"
    assert plan.split_for_seed(1179) == "train"
    assert plan.split_for_seed(1180) == "validation"
    assert plan.split_for_seed(1199) == "validation"
    assert plan.counts == {"train": 180, "validation": 20}


def test_split_plan_is_range_general_not_tied_to_formal_seed_values(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.episode_split import load_episode_split_plan

    path = tmp_path / "split.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "synthetic_split",
                "source_seed_start": 50,
                "source_seed_count": 6,
                "splits": {
                    "train": {"start": 50, "count": 4},
                    "validation": {"start": 54, "count": 2},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = load_episode_split_plan(path)

    assert plan.split_for_seed(53) == "train"
    assert plan.split_for_seed(54) == "validation"
    with pytest.raises(ValueError, match="outside"):
        plan.split_for_seed(56)


@pytest.mark.parametrize(
    "splits",
    (
        {
            "train": {"start": 100, "count": 5},
            "validation": {"start": 104, "count": 6},
        },
        {
            "train": {"start": 100, "count": 4},
            "validation": {"start": 105, "count": 5},
        },
    ),
)
def test_split_plan_rejects_overlap_or_gap(tmp_path: Path, splits: dict) -> None:
    from latency_meta_mdp.episode_split import load_episode_split_plan

    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "split_id": "bad",
                "source_seed_start": 100,
                "source_seed_count": 10,
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_episode_split_plan(path)
