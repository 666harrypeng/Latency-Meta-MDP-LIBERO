from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FORMAL_CONFIG = Path("configs/collection/panda_ball_structured_formal.yaml")


def _valid_mapping() -> dict[str, object]:
    return {
        "schema_version": 2,
        "corpus_id": "panda-ball-structured-formal-v1",
        "logical_task_index_start": 0,
        "task_instance_count": 100,
        "levels": [1, 2, 3],
        "realizations_per_task": 4,
        "families": [
            "canonical_direct",
            "early_high_arc",
            "lateral_arc",
            "time_shifted_smooth",
        ],
        "family_allocation": "iid_uniform_seeded",
        "reserve_task_instance_count": 20,
        "require_complete_realization_block": True,
        "group_unit": "logical_master_task_index",
    }


def _write(tmp_path: Path, mapping: dict[str, object]) -> Path:
    path = tmp_path / "formal.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


def test_default_formal_config_expands_to_400_per_level_and_1200_total() -> None:
    """Break caught: pilot cardinality leaks into the scalable formal request."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    config = load_formal_corpus_config(FORMAL_CONFIG)

    assert config.primary_task_indices == tuple(range(100))
    assert config.reserve_task_indices == tuple(range(100, 120))
    assert config.trajectories_per_level == 400
    assert config.primary_trajectory_count == 1200
    assert config.to_mapping() == _valid_mapping()


def test_formal_scale_dimensions_are_independent_and_zero_reserve_is_valid(
    tmp_path: Path,
) -> None:
    """Break caught: task count changes realization count or requires a reserve pool."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    mapping = _valid_mapping()
    mapping.update(
        logical_task_index_start=100,
        task_instance_count=25,
        realizations_per_task=8,
        reserve_task_instance_count=0,
    )
    config = load_formal_corpus_config(_write(tmp_path, mapping))

    assert config.primary_task_indices == tuple(range(100, 125))
    assert config.reserve_task_indices == ()
    assert config.trajectories_per_level == 200
    assert config.primary_trajectory_count == 600


def test_three_realizations_per_task_is_a_valid_general_request(tmp_path: Path) -> None:
    """Break caught: collection cardinality is incorrectly tied to strategy-family count."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    mapping = _valid_mapping()
    mapping.update(task_instance_count=3, realizations_per_task=3, reserve_task_instance_count=0)
    config = load_formal_corpus_config(_write(tmp_path, mapping))

    assert config.trajectories_per_level == 9
    assert config.primary_trajectory_count == 27


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(extra_field=1), "unknown formal corpus"),
        (lambda row: row.pop("corpus_id"), "missing formal corpus"),
        (lambda row: row.update(schema_version=True), "schema_version"),
        (lambda row: row.update(corpus_id=""), "corpus_id"),
        (lambda row: row.update(logical_task_index_start=-1), "logical_task_index_start"),
        (lambda row: row.update(task_instance_count=0), "task_instance_count"),
        (lambda row: row.update(reserve_task_instance_count=-1), "reserve_task_instance_count"),
        (lambda row: row.update(levels=[1, 1, 3]), "levels"),
        (lambda row: row.update(levels=[1, 2, 4]), "levels"),
        (
            lambda row: row.update(
                families=[
                    "canonical_direct",
                    "canonical_direct",
                    "lateral_arc",
                    "time_shifted_smooth",
                ]
            ),
            "families",
        ),
        (lambda row: row.update(realizations_per_task=0), "realizations_per_task"),
        (lambda row: row.update(family_allocation="balanced_seeded"), "family_allocation"),
        (
            lambda row: row.update(require_complete_realization_block=False),
            "complete_realization",
        ),
        (lambda row: row.update(group_unit="trajectory"), "group_unit"),
    ],
)
def test_formal_config_rejects_semantic_and_scalar_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    """Break caught: malformed scale settings silently redefine the formal corpus."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    mapping = _valid_mapping()
    mutation(mapping)
    with pytest.raises((TypeError, ValueError), match=message):
        load_formal_corpus_config(_write(tmp_path, mapping))
