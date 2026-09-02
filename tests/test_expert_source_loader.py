from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_expert_source_collection import _admitted, _publication_fixture, _summary


def _publish(tmp_path: Path, *, boundary_count: int = 2) -> Path:
    from latency_meta_mdp.expert_realization.source_corpus.collection import publish_source_corpus

    request, split, config, task, episode = _publication_fixture(
        boundary_count=boundary_count
    )
    target = tmp_path / "source"
    publish_source_corpus(
        target=target,
        request=request,
        source_config=config,
        split_plan=split,
        task_entries=(task,),
        admitted_episodes=(_admitted(episode),),
        collection_summary=_summary(),
    )
    return target


def _refresh_manifest_entry(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / relative
    manifest["artifacts"][relative] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_verified_loader_indexes_and_reads_one_episode_row_group(tmp_path: Path) -> None:
    """Break caught: centralized metadata cannot recover the logical episode abstraction."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    corpus = load_verified_source_corpus(_publish(tmp_path))

    assert corpus.manifest.episode_count == 1
    assert corpus.episode_ids(level=1, split="train") == ("source-l1-task000-r000",)
    episode = corpus.read_episode("source-l1-task000-r000")
    assert episode.frames.num_rows == 2
    assert episode.frames["formal_tick"].to_pylist() == [0, 1]
    assert episode.frames["expert_action"][1].as_py() is None
    assert episode.metadata["data_shard"] == "data/level-1/shard-00000.parquet"


def test_verified_loader_enforces_typed_field_role_allowlists(tmp_path: Path) -> None:
    """Break caught: simulator GT leaks through a generic read-fields call."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )
    from latency_meta_mdp.expert_realization.source_corpus.schema import SourceFieldRole

    corpus = load_verified_source_corpus(_publish(tmp_path))
    table = corpus.read_fields(
        "source-l1-task000-r000",
        fields=("formal_tick", "agentview_rgb", "robot_qpos", "expert_action"),
        allowed_roles=frozenset(
            {SourceFieldRole.IDENTITY, SourceFieldRole.DEPLOYMENT_INPUT}
        ),
    )
    assert table.column_names == ["formal_tick", "agentview_rgb", "robot_qpos", "expert_action"]
    with pytest.raises(PermissionError, match="object_pose"):
        corpus.read_fields(
            "source-l1-task000-r000",
            fields=("object_pose",),
            allowed_roles=frozenset({SourceFieldRole.DEPLOYMENT_INPUT}),
        )
    with pytest.raises(TypeError, match="SourceFieldRole"):
        corpus.read_fields(
            "source-l1-task000-r000",
            fields=("robot_qpos",),
            allowed_roles=frozenset({"deployment_input"}),  # type: ignore[arg-type]
        )


def test_loader_rejects_extra_missing_and_hash_drift_files(tmp_path: Path) -> None:
    """Break caught: a partial or contaminated corpus is accepted as complete."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    extra = _publish(tmp_path / "extra")
    (extra / "failed-attempt.json").write_text("{}")
    with pytest.raises(ValueError, match="file inventory"):
        load_verified_source_corpus(extra)

    missing = _publish(tmp_path / "missing")
    (missing / "meta/events.parquet").unlink()
    with pytest.raises(ValueError, match="file inventory"):
        load_verified_source_corpus(missing)

    changed = _publish(tmp_path / "changed")
    with (changed / "README.md").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="(size|hash) mismatch"):
        load_verified_source_corpus(changed)


def test_loader_rejects_metadata_schema_split_and_row_group_drift(tmp_path: Path) -> None:
    """Break caught: self-consistent file hashes hide relational/schema corruption."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    schema_root = _publish(tmp_path / "schema")
    episodes_path = schema_root / "meta/episodes.parquet"
    table = pq.read_table(episodes_path)
    index = table.schema.get_field_index("frame_count")
    table = table.set_column(index, "frame_count", pa.array([2], type=pa.int64()))
    pq.write_table(table, episodes_path)
    _refresh_manifest_entry(schema_root, "meta/episodes.parquet")
    with pytest.raises(ValueError, match="episode metadata schema"):
        load_verified_source_corpus(schema_root)

    split_root = _publish(tmp_path / "split")
    episodes_path = split_root / "meta/episodes.parquet"
    table = pq.read_table(episodes_path)
    index = table.schema.get_field_index("split")
    table = table.set_column(index, table.schema.field(index), pa.array(["validation"]))
    pq.write_table(table, episodes_path)
    _refresh_manifest_entry(split_root, "meta/episodes.parquet")
    with pytest.raises(ValueError, match="split join"):
        load_verified_source_corpus(split_root)

    group_root = _publish(tmp_path / "row-group")
    episodes_path = group_root / "meta/episodes.parquet"
    table = pq.read_table(episodes_path)
    index = table.schema.get_field_index("row_group_index")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array([1], type=pa.int32()),
    )
    pq.write_table(table, episodes_path)
    _refresh_manifest_entry(group_root, "meta/episodes.parquet")
    with pytest.raises(ValueError, match="row group"):
        load_verified_source_corpus(group_root)


def test_loader_rejects_non_png_embedded_camera_bytes(tmp_path: Path) -> None:
    """Break caught: a file with the right Arrow schema contains lossy or invalid image payload."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    root = _publish(tmp_path)
    shard = root / "data/level-1/shard-00000.parquet"
    table = pq.read_table(shard)
    images = table["agentview_rgb"].to_pylist()
    images[0] = {"bytes": b"not-a-png", "path": images[0]["path"]}
    index = table.schema.get_field_index("agentview_rgb")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(images, type=table.schema[index].type),
    )
    pq.write_table(table, shard)
    _refresh_manifest_entry(root, "data/level-1/shard-00000.parquet")

    with pytest.raises(ValueError, match="PNG"):
        load_verified_source_corpus(root)


def test_loader_rejects_wrong_length_nullable_vectors(tmp_path: Path) -> None:
    """Break caught: variable-list Parquet storage weakens the declared fixed source shape."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    root = _publish(tmp_path)
    shard = root / "data/level-1/shard-00000.parquet"
    table = pq.read_table(shard)
    actions = table["expert_action"].to_pylist()
    actions[0] = actions[0][:6]
    index = table.schema.get_field_index("expert_action")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(actions, type=table.schema[index].type),
    )
    pq.write_table(table, shard)
    _refresh_manifest_entry(root, "data/level-1/shard-00000.parquet")

    with pytest.raises(ValueError, match="expert_action.*shape"):
        load_verified_source_corpus(root)


def test_loader_rehashes_embedded_task_source_payloads(tmp_path: Path) -> None:
    """Break caught: task identity trusts adjacent hash text instead of the stored source bytes."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    root = _publish(tmp_path)
    tasks_path = root / "meta/task_instances.parquet"
    table = pq.read_table(tasks_path)
    index = table.schema.get_field_index("motion_profile_json")
    table = table.set_column(
        index,
        table.schema.field(index),
        pa.array(['{"changed":true}\n']),
    )
    pq.write_table(table, tasks_path)
    _refresh_manifest_entry(root, "meta/task_instances.parquet")

    with pytest.raises(ValueError, match="motion profile payload hash"):
        load_verified_source_corpus(root)
