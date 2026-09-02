from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

SOURCE_CONFIG = Path("configs/source_corpus/panda_ball_source_parquet.yaml")
FORMAL_CONFIG = Path("configs/source_corpus/panda_ball_formal_source_pilot.yaml")
EXECUTION_CONFIG = Path("configs/source_corpus/panda_ball_formal_source_execution.yaml")
FORMAL_SPLIT_CONFIG = Path("configs/source_corpus/panda_ball_formal_source_pilot_split.yaml")


def _valid_source_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "format_id": "structured_expert_source_parquet_v1",
        "image_encoding": "lossless_png",
        "png_compress_level": 6,
        "parquet_compression": "zstd",
        "parquet_compression_level": 3,
        "target_shard_bytes": 268_435_456,
        "episode_row_group": True,
        "split_unit": "master_task_index",
    }


def _valid_split_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "split_id": "panda-ball-source-pilot-split",
        "corpus_id": "panda-ball-structured-source-pilot",
        "train_master_task_indices": [0, 1, 3, 4],
        "validation_master_task_indices": [2, 5],
    }


def _valid_execution_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_id": "panda-ball-formal-source-sequential-first-qualified-v1",
        "candidate_policy": "sequential_first_qualified",
        "maximum_candidate_attempts_per_realization": 8,
        "infrastructure_retry_limit": 2,
        "determinism_canaries_per_level": 1,
        "maximum_formal_ticks": 220,
        "admission_unit": "paired_master_block",
    }


def _write(tmp_path: Path, name: str, mapping: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


def test_source_config_loads_the_canonical_lossless_parquet_contract() -> None:
    """Break caught: source storage silently becomes lossy or model-specific."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )

    config = load_source_corpus_config(SOURCE_CONFIG)

    assert config.to_mapping() == _valid_source_mapping()
    assert config.target_shard_bytes == 256 * 1024 * 1024
    assert config.sha256 == "29cdad2739059d24d9ab978dfa6a20430265baaf0dfd9a23ab046ea1651a8e38"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(extra=True), "unknown source corpus"),
        (lambda row: row.pop("format_id"), "missing source corpus"),
        (lambda row: row.update(schema_version=True), "schema_version"),
        (lambda row: row.update(format_id="lerobot_v3"), "format_id"),
        (lambda row: row.update(image_encoding="h264"), "image_encoding"),
        (lambda row: row.update(png_compress_level=10), "png_compress_level"),
        (lambda row: row.update(parquet_compression="snappy"), "parquet_compression"),
        (lambda row: row.update(parquet_compression_level=0), "parquet_compression_level"),
        (lambda row: row.update(target_shard_bytes=64 * 1024 * 1024 - 1), "target_shard"),
        (lambda row: row.update(target_shard_bytes=1024 * 1024 * 1024 + 1), "target_shard"),
        (lambda row: row.update(episode_row_group=False), "episode_row_group"),
        (lambda row: row.update(split_unit="episode"), "split_unit"),
    ],
)
def test_source_config_rejects_storage_semantic_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    """Break caught: a config edit changes source meaning without a schema revision."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )

    mapping = _valid_source_mapping()
    mutation(mapping)

    with pytest.raises((TypeError, ValueError), match=message):
        load_source_corpus_config(_write(tmp_path, "source.yaml", mapping))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("png_compress_level", 6.0),
        ("parquet_compression_level", True),
        ("target_shard_bytes", 268_435_456.0),
        ("episode_row_group", 1),
    ],
)
def test_source_config_rejects_yaml_scalar_and_container_coercion(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    """Break caught: bool/int/float or list/tuple coercion weakens an exact file contract."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )

    mapping = _valid_source_mapping()
    mapping[field] = bad_value

    with pytest.raises((TypeError, ValueError), match=field):
        load_source_corpus_config(_write(tmp_path, "source.yaml", mapping))


def test_master_task_split_plan_round_trips_and_covers_a_declared_universe(
    tmp_path: Path,
) -> None:
    """Break caught: reserve tasks or one level can drift into another split."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_master_task_split_plan,
    )

    plan = load_master_task_split_plan(
        _write(tmp_path, "split.yaml", _valid_split_mapping())
    )

    assert plan.to_mapping() == _valid_split_mapping()
    assert plan.sha256 == "7e7af9fbae0708f8ab761638e9c76567d65d330871b23a8d0cc69c4c677bcd68"
    assert plan.split_for(0) == "train"
    assert plan.split_for(5) == "validation"
    plan.require_exact_indices(tuple(range(6)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(extra=True), "unknown master task split"),
        (lambda row: row.pop("split_id"), "missing master task split"),
        (lambda row: row.update(schema_version=True), "schema_version"),
        (lambda row: row.update(split_id=""), "split_id"),
        (lambda row: row.update(corpus_id=" panda"), "corpus_id"),
        (lambda row: row.update(train_master_task_indices=[]), "train"),
        (lambda row: row.update(validation_master_task_indices=[]), "validation"),
        (lambda row: row.update(train_master_task_indices=[0, 0, 1]), "unique"),
        (lambda row: row.update(validation_master_task_indices=[2, 2]), "unique"),
        (lambda row: row.update(validation_master_task_indices=[1, 2]), "disjoint"),
        (lambda row: row.update(train_master_task_indices=[0, -1]), "non-negative"),
    ],
)
def test_master_task_split_plan_rejects_identity_and_partition_drift(
    tmp_path: Path, mutation, message: str
) -> None:
    """Break caught: malformed split metadata permits group leakage or ambiguous identity."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_master_task_split_plan,
    )

    mapping = deepcopy(_valid_split_mapping())
    mutation(mapping)

    with pytest.raises((TypeError, ValueError), match=message):
        load_master_task_split_plan(_write(tmp_path, "split.yaml", mapping))


def test_master_task_split_plan_rejects_missing_or_extra_universe_indices(
    tmp_path: Path,
) -> None:
    """Break caught: a formally requested reserve identity has no stable split assignment."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_master_task_split_plan,
    )

    plan = load_master_task_split_plan(
        _write(tmp_path, "split.yaml", _valid_split_mapping())
    )

    with pytest.raises(ValueError, match="exact requested master-task universe"):
        plan.require_exact_indices((0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="exact requested master-task universe"):
        plan.require_exact_indices(tuple(range(7)))


def test_formal_collection_configs_lock_36_successes_and_sequential_planning() -> None:
    """Break caught: pilot size, retry policy, or paired admission silently drifts."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_master_task_split_plan,
        load_source_execution_config,
    )

    formal = load_formal_corpus_config(FORMAL_CONFIG)
    execution = load_source_execution_config(EXECUTION_CONFIG)
    split = load_master_task_split_plan(FORMAL_SPLIT_CONFIG)

    assert formal.task_instance_count == 3
    assert formal.reserve_task_instance_count == 6
    assert formal.levels == (1, 2, 3)
    assert formal.realizations_per_task == 4
    assert formal.primary_trajectory_count == 36
    assert execution.to_mapping() == _valid_execution_mapping()
    assert split.train_master_task_indices == (0, 1, 3, 4, 6, 7)
    assert split.validation_master_task_indices == (2, 5, 8)
    assert split.corpus_id == formal.corpus_id
    split.require_exact_indices(formal.primary_task_indices + formal.reserve_task_indices)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", True),
        ("candidate_policy", "best_of_eight"),
        ("maximum_candidate_attempts_per_realization", 0),
        ("maximum_candidate_attempts_per_realization", 8.0),
        ("infrastructure_retry_limit", -1),
        ("determinism_canaries_per_level", 0),
        ("maximum_formal_ticks", 5),
        ("admission_unit", "level"),
    ],
)
def test_source_execution_config_rejects_semantic_or_type_drift(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    """Break caught: invalid retry or admission semantics enter a formal collection identity."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_execution_config,
    )

    mapping = _valid_execution_mapping()
    mapping[field] = bad_value
    with pytest.raises((TypeError, ValueError), match=field):
        load_source_execution_config(_write(tmp_path, "execution.yaml", mapping))


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_source_execution_config_rejects_missing_or_unknown_fields(
    tmp_path: Path, mutation: str
) -> None:
    """Break caught: an execution default changes without changing serialized identity."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_execution_config,
    )

    mapping = _valid_execution_mapping()
    if mutation == "missing":
        mapping.pop("candidate_policy")
    else:
        mapping["candidate_count"] = 8
    with pytest.raises(ValueError, match="source execution"):
        load_source_execution_config(_write(tmp_path, "execution.yaml", mapping))
