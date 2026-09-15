from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from expert_realization_test_support import make_formal_source_episode


def _config():
    from latency_meta_mdp.data.source.config import (
        load_source_corpus_config,
    )

    return load_source_corpus_config(
        Path("configs/data/source_corpus/panda_ball_source_parquet.yaml")
    )


def test_png_serialization_is_deterministic_and_pixel_identical() -> None:
    """Break caught: source image serialization changes pixels or depends on process state."""
    from latency_meta_mdp.data.source.parquet import (
        decode_png,
        encode_png,
    )

    row, col = np.indices((256, 256))
    rgb = np.stack([row % 256, col % 256, (row + col) % 256], axis=-1).astype(np.uint8)
    first = encode_png(rgb, compress_level=6)
    second = encode_png(rgb, compress_level=6)

    assert first == second
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    np.testing.assert_array_equal(decode_png(first), rgb)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((256, 256, 3), dtype=np.float32),
        np.zeros((256, 256), dtype=np.uint8),
        np.zeros((255, 256, 3), dtype=np.uint8),
    ],
)
def test_png_serialization_rejects_noncanonical_source_frames(bad: np.ndarray) -> None:
    """Break caught: resized, grayscale, or coerced pixels enter canonical source storage."""
    from latency_meta_mdp.data.source.parquet import encode_png

    with pytest.raises((TypeError, ValueError), match=r"uint8\[256,256,3\]"):
        encode_png(bad, compress_level=6)


def test_episode_table_preserves_fields_and_uses_null_terminal_transition() -> None:
    """Break caught: terminal padding masquerades as a real action or source fields disappear."""
    from latency_meta_mdp.data.source.parquet import (
        decode_png,
        episode_to_frame_table,
    )
    from latency_meta_mdp.data.source.schema import SOURCE_FRAME_SCHEMA

    episode = make_formal_source_episode(camera_height=256, camera_width=256)
    table = episode_to_frame_table(episode, config=_config())

    assert table.schema == SOURCE_FRAME_SCHEMA
    assert table.num_rows == 2
    assert table["formal_tick"].to_pylist() == [0, 1]
    assert table["expert_action"][0].as_py() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert table["expert_action"][1].as_py() is None
    assert table["action_mask"][1].as_py() is None
    assert table["phase_id"][1].as_py() is None
    assert table["eef_orientation_matrix_world"][0].as_py() == np.eye(3).reshape(-1).tolist()
    image = table["agentview_rgb"][1].as_py()
    assert image["path"] == "source-l1-task000-r000/agentview/frame_000001.png"
    np.testing.assert_array_equal(
        decode_png(image["bytes"]),
        episode.boundaries[1].deployment.agentview_rgb,
    )


def test_episode_table_rejects_failed_or_noncanonical_camera_episode() -> None:
    """Break caught: an arbitrary typed-looking object bypasses source admission/shape checks."""
    from latency_meta_mdp.data.source.parquet import episode_to_frame_table

    with pytest.raises(TypeError, match="FormalSourceSynchronizedEpisode"):
        episode_to_frame_table(object(), config=_config())  # type: ignore[arg-type]
    small = make_formal_source_episode()
    with pytest.raises(ValueError, match="256x256"):
        episode_to_frame_table(small, config=_config())


def test_shard_writer_places_each_episode_in_one_row_group_and_publishes_once(
    tmp_path: Path,
) -> None:
    """Break caught: an episode crosses row groups or a partial Parquet file looks complete."""
    from latency_meta_mdp.data.source.parquet import (
        SourceParquetShardWriter,
    )

    first = make_formal_source_episode(camera_height=256, camera_width=256)
    second = replace(
        first,
        metadata=replace(first.metadata, episode_id="source-l1-task000-r001"),
    )
    target = tmp_path / "level-1" / "shard-00000.parquet"
    writer = SourceParquetShardWriter(target=target, level=1, config=_config())
    first_location = writer.add_episode(first)
    second_location = writer.add_episode(second)

    assert not target.exists()
    assert writer.current_byte_count > 0
    published = writer.close()
    parquet = pq.ParquetFile(target)

    assert published.relative_name == "shard-00000.parquet"
    assert published.row_count == 4
    assert published.row_group_count == 2
    assert published.byte_count == target.stat().st_size
    assert first_location.row_group_index == 0
    assert first_location.row_offset == 0
    assert second_location.row_group_index == 1
    assert second_location.row_offset == 2
    assert parquet.metadata.num_row_groups == 2
    assert parquet.metadata.row_group(0).num_rows == 2
    assert parquet.metadata.row_group(1).num_rows == 2
    with pytest.raises(RuntimeError, match="closed"):
        writer.add_episode(first)
    with pytest.raises(RuntimeError, match="closed"):
        writer.close()
