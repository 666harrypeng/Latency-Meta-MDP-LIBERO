from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_source_storage_config_has_no_split_semantics() -> None:
    """Break caught: collection-time partition policy remains part of source identity."""
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_source_corpus_config,
    )

    config = load_source_corpus_config(
        ROOT / "configs/source_corpus/panda_ball_source_parquet.yaml"
    )
    assert config.format_id == "structured_expert_source_parquet_v2"
    assert "split_unit" not in config.to_mapping()


def test_formal_collection_config_names_only_the_paired_group_unit() -> None:
    """Break caught: the raw collection request still declares a dataset partition policy."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    config = load_formal_corpus_config(
        ROOT / "configs/source_corpus/panda_ball_formal_source_pilot.yaml"
    )
    assert config.group_unit == "logical_master_task_index"
    assert "split_unit" not in config.to_mapping()


def test_source_episode_contract_has_no_split_plan_identity() -> None:
    """Break caught: one raw trajectory is semantically tied to a train/validation plan."""
    from expert_realization_test_support import make_formal_source_metadata

    mapping = make_formal_source_metadata().to_mapping()
    assert mapping["schema_version"] == 2
    assert "master_task_split_plan_sha256" not in mapping


def test_source_metadata_schemas_have_no_split_column() -> None:
    """Break caught: raw task or episode tables still carry a partition label."""
    from latency_meta_mdp.expert_realization.source_corpus.schema import (
        EPISODE_SCHEMA,
        TASK_INSTANCE_SCHEMA,
        source_schema_document,
    )

    assert "split" not in TASK_INSTANCE_SCHEMA.names
    assert "split" not in EPISODE_SCHEMA.names
    assert source_schema_document()["format_id"] == "structured_expert_source_parquet_v2"


def test_collection_dry_run_needs_no_split_config(
    tmp_path: Path,
) -> None:
    """Break caught: an unsplit collection cannot start without a partition policy."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "--project-root",
                str(ROOT),
                "--formal-config",
                "configs/source_corpus/panda_ball_formal_source_pilot.yaml",
                "--execution-config",
                "configs/source_corpus/panda_ball_formal_source_execution.yaml",
                "--source-config",
                "configs/source_corpus/panda_ball_source_parquet.yaml",
                "--work-root",
                str(tmp_path / "work"),
                "--output-root",
                str(tmp_path / "output"),
                "--planner-python",
                str(planner),
                "--dry-run",
            ]
        )
        == 0
    )


def test_fresh_unsplit_smoke_config_is_one_complete_paired_block() -> None:
    """Break caught: the smoke silently grows into a formal collection or loses one level."""
    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config

    config = load_formal_corpus_config(
        ROOT / "configs/source_corpus/panda_ball_formal_source_smoke.yaml"
    )
    assert config.corpus_id == "panda-ball-structured-source-smoke-1x4-v2"
    assert config.task_instance_count == 1
    assert config.reserve_task_instance_count == 4
    assert config.levels == (1, 2, 3)
    assert config.realizations_per_task == 4
    assert config.primary_trajectory_count == 12
