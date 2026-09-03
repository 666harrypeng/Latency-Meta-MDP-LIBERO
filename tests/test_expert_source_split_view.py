from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

MANIFEST_BYTES = b'{"demo":true}\n'
MANIFEST_SHA256 = "c7a295e9b687af1e8faf5eb8250306fc625d60307a2c0f7567636a446a34d9ac"


def _corpus(tmp_path: Path, *, omit: tuple[int, int, int] | None = None):
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        SourceCorpusManifest,
        VerifiedSourceCorpus,
    )

    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.json").write_bytes(MANIFEST_BYTES)
    rows = tuple(
        {
            "episode_id": f"source-L{level}-task{master:06d}-r{realization:04d}",
            "logical_master_task_index": master,
            "level": level,
            "realization_index": realization,
            "split": "train" if master != 12 else "validation",
        }
        for master in (10, 11, 12)
        for level in (1, 2, 3)
        for realization in range(4)
        if (master, level, realization) != omit
    )
    manifest = SourceCorpusManifest(
        corpus_id="demo-source",
        request_sha256="a" * 64,
        source_config_sha256="b" * 64,
        split_plan_sha256="c" * 64,
        admitted_master_task_indices=(10, 11, 12),
        master_task_count=3,
        level_task_instance_count=9,
        episode_count=36,
        episodes_by_level=MappingProxyType({"1": 12, "2": 12, "3": 12}),
        frame_count=5_400,
        shard_count=3,
        artifacts=MappingProxyType({}),
    )
    return VerifiedSourceCorpus(root=root, manifest=manifest, episode_rows=rows)


def test_external_split_is_deterministic_complete_and_master_grouped(tmp_path: Path) -> None:
    """Break caught: related levels or realizations leak across train and validation."""
    from latency_meta_mdp.expert_realization.source_corpus.split_view import (
        build_source_split,
    )

    corpus = _corpus(tmp_path)
    split = build_source_split(
        corpus,
        split_id="demo-2train-1validation-v1",
        validation_master_count=1,
        split_seed=7,
    )

    assert split.source_corpus_id == "demo-source"
    assert split.source_manifest_sha256 == MANIFEST_SHA256
    assert split.train_master_task_indices == (10, 11)
    assert split.validation_master_task_indices == (12,)
    assert len(split.train_episode_ids) == 24
    assert len(split.validation_episode_ids) == 12
    assert set(split.train_episode_ids).isdisjoint(split.validation_episode_ids)
    assert all("task000012" not in value for value in split.train_episode_ids)
    assert all("task000012" in value for value in split.validation_episode_ids)
    assert (
        build_source_split(
            corpus,
            split_id="demo-2train-1validation-v1",
            validation_master_count=1,
            split_seed=7,
        )
        == split
    )


def test_external_split_rejects_incomplete_master_block(tmp_path: Path) -> None:
    """Break caught: a split is generated after one level/realization silently disappears."""
    from latency_meta_mdp.expert_realization.source_corpus.split_view import (
        build_source_split,
    )

    corpus = _corpus(tmp_path, omit=(11, 3, 3))
    with pytest.raises(ValueError, match="complete master-task blocks"):
        build_source_split(
            corpus,
            split_id="invalid",
            validation_master_count=1,
            split_seed=7,
        )


def test_split_publication_round_trips_and_rejects_tamper_or_overwrite(
    tmp_path: Path,
) -> None:
    """Break caught: a split can detach from its exact source or be replaced in place."""
    from latency_meta_mdp.expert_realization.source_corpus.split_view import (
        build_source_split,
        load_verified_source_split,
        write_source_split,
    )

    corpus = _corpus(tmp_path)
    split = build_source_split(
        corpus,
        split_id="demo-2train-1validation-v1",
        validation_master_count=1,
        split_seed=7,
    )
    target = tmp_path / "derived" / "split.json"

    assert write_source_split(target, split) == target
    assert load_verified_source_split(target, corpus) == split
    with pytest.raises(FileExistsError):
        write_source_split(target, split)

    mapping = json.loads(target.read_text(encoding="utf-8"))
    mapping["validation_master_task_indices"] = [11]
    target.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError):
        load_verified_source_split(target, corpus)


def test_split_loader_rejects_source_manifest_drift(tmp_path: Path) -> None:
    """Break caught: a valid split is reused after the source manifest changes."""
    from latency_meta_mdp.expert_realization.source_corpus.split_view import (
        build_source_split,
        load_verified_source_split,
        write_source_split,
    )

    corpus = _corpus(tmp_path)
    target = tmp_path / "split.json"
    write_source_split(
        target,
        build_source_split(
            corpus,
            split_id="demo-2train-1validation-v1",
            validation_master_count=1,
            split_seed=7,
        ),
    )
    (corpus.root / "manifest.json").write_bytes(b'{"demo":false}\n')
    assert hashlib.sha256((corpus.root / "manifest.json").read_bytes()).hexdigest() != (
        MANIFEST_SHA256
    )
    with pytest.raises(ValueError, match="source manifest"):
        load_verified_source_split(target, corpus)


def test_split_cli_writes_only_the_final_path_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: operational logs pollute the derived split artifact path contract."""
    from latency_meta_mdp.cli.split_structured_expert_source import main

    corpus = _corpus(tmp_path)
    target = tmp_path / "derived" / "split.json"
    assert (
        main(
            [
                "--source-root",
                str(corpus.root),
                "--output",
                str(target),
                "--split-id",
                "demo-2train-1validation-v1",
                "--validation-master-count",
                "1",
                "--split-seed",
                "7",
            ],
            load_fn=lambda _root: corpus,
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == f"{target}\n"
    assert captured.err == ""
    assert json.loads(target.read_text(encoding="utf-8"))["source_manifest_sha256"] == (
        MANIFEST_SHA256
    )
