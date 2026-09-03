from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest
from test_expert_source_loader import _publish


def _index_corpus(tmp_path: Path):
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        SourceCorpusManifest,
        VerifiedSourceCorpus,
    )

    rows = tuple(
        {
            "episode_id": f"source-L{level}-task{master:06d}-r{realization:04d}",
            "logical_master_task_index": master,
            "level": level,
            "realization_index": realization,
        }
        for level in (1, 2, 3)
        for master in (4, 2, 3)
        for realization in range(4)
    )
    manifest = SourceCorpusManifest(
        corpus_id="review-source",
        request_sha256="a" * 64,
        source_config_sha256="b" * 64,
        split_plan_sha256=None,
        admitted_master_task_indices=(2, 3, 4),
        master_task_count=3,
        level_task_instance_count=9,
        episode_count=36,
        episodes_by_level=MappingProxyType({"1": 12, "2": 12, "3": 12}),
        frame_count=5_400,
        shard_count=3,
        artifacts=MappingProxyType({}),
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    return VerifiedSourceCorpus(root=root, manifest=manifest, episode_rows=rows)


def test_review_selection_uses_dataset_level_and_canonical_episode_range(
    tmp_path: Path,
) -> None:
    """Break caught: review order follows Parquet rows instead of task/realization identity."""
    from latency_meta_mdp.expert_realization.source_corpus.review_video import (
        select_review_episodes,
    )

    corpus = _index_corpus(tmp_path)
    all_levels = select_review_episodes(corpus, level=None, episode_range=None)
    assert set(all_levels) == {1, 2, 3}
    assert all_levels[1][:5] == (
        "source-L1-task000002-r0000",
        "source-L1-task000002-r0001",
        "source-L1-task000002-r0002",
        "source-L1-task000002-r0003",
        "source-L1-task000003-r0000",
    )
    assert select_review_episodes(corpus, level=2, episode_range=(4, 7)) == {
        2: (
            "source-L2-task000003-r0000",
            "source-L2-task000003-r0001",
            "source-L2-task000003-r0002",
        )
    }


def test_review_selection_rejects_ambiguous_or_out_of_bounds_ranges(tmp_path: Path) -> None:
    """Break caught: one range is silently applied to every level or clips out of bounds."""
    from latency_meta_mdp.expert_realization.source_corpus.review_video import (
        select_review_episodes,
    )

    corpus = _index_corpus(tmp_path)
    with pytest.raises(ValueError, match="requires one level"):
        select_review_episodes(corpus, level=None, episode_range=(0, 2))
    with pytest.raises(ValueError, match="outside"):
        select_review_episodes(corpus, level=3, episode_range=(11, 13))
    with pytest.raises(ValueError, match="ordered"):
        select_review_episodes(corpus, level=3, episode_range=(4, 4))


@pytest.mark.parametrize(
    ("frame_count", "speed", "expected"),
    [
        (7, 1, (0, 1, 2, 3, 4, 5, 6)),
        (7, 2, (0, 2, 4, 6)),
        (8, 3, (0, 3, 6, 7)),
        (11, 5, (0, 5, 10)),
        (12, 10, (0, 10, 11)),
    ],
)
def test_speed_stride_always_retains_terminal_frame(
    frame_count: int, speed: int, expected: tuple[int, ...]
) -> None:
    """Break caught: fast review drops the successful terminal boundary."""
    from latency_meta_mdp.expert_realization.source_corpus.review_video import (
        review_frame_indices,
    )

    assert review_frame_indices(frame_count=frame_count, speed=speed) == expected


def test_review_cli_dry_run_selects_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: dry-run decodes images, calls FFmpeg, or creates review paths."""
    from latency_meta_mdp.cli.render_structured_source_video import main

    corpus = _index_corpus(tmp_path)
    output = tmp_path / "review"
    assert (
        main(
            [
                "--source-root",
                str(corpus.root),
                "--output-root",
                str(output),
                "--level",
                "2",
                "--episode-range",
                "4",
                "7",
                "--speed",
                "5",
                "--dry-run",
            ],
            load_fn=lambda _root: corpus,
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["levels"]["2"] == {
        "episode_count": 3,
        "episode_ids": [
            "source-L2-task000003-r0000",
            "source-L2-task000003-r0001",
            "source-L2-task000003-r0002",
        ],
        "start": 4,
        "stop": 7,
    }
    assert summary["speed"] == 5
    assert not output.exists()


def test_real_review_video_is_h264_50fps_and_manifest_bound(
    tmp_path: Path,
) -> None:
    """Break caught: the renderer emits an unplayable video or loses source provenance."""
    from latency_meta_mdp.expert_realization.source_corpus.review_video import (
        render_source_review_videos,
    )

    source = _publish(tmp_path / "fixture", boundary_count=61)
    output = tmp_path / "review"
    manifest_path = render_source_review_videos(
        source_root=source,
        output_root=output,
        level=1,
        episode_range=(0, 1),
        speed=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_manifest_sha256"]
    assert manifest["videos"][0]["source_frame_count"] == 61
    assert manifest["videos"][0]["output_frame_count"] == 31
    video = output / manifest["videos"][0]["path"]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {
        "avg_frame_rate": "50/1",
        "codec_name": "h264",
        "height": 288,
        "nb_read_frames": "31",
        "width": 512,
    }
    with pytest.raises(FileExistsError):
        render_source_review_videos(
            source_root=source,
            output_root=output,
            level=1,
            episode_range=(0, 1),
            speed=2,
        )
