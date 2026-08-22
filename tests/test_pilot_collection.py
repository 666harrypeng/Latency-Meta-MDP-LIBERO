from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from latency_meta_mdp.recording import RecordProfile


def test_load_pilot_run_spec_uses_the_approved_three_by_three_campaign() -> None:
    pilot = importlib.import_module("latency_meta_mdp.pilot_collection")

    spec = pilot.load_pilot_run_spec(
        Path("configs/collection/panda_ball_pilot_v1.yaml")
    )

    assert spec.levels == (1, 2, 3)
    assert spec.seeds == (10, 11, 12)
    assert spec.record_profile is RecordProfile.PILOT_DEBUG
    assert (spec.camera_width, spec.camera_height) == (256, 256)
    assert spec.review_video_fps == 50


def test_collect_expert_pilot_run_writes_three_level_artifacts_and_review_videos(
    tmp_path: Path,
) -> None:
    pilot = importlib.import_module("latency_meta_mdp.pilot_collection")
    spec = pilot.PilotRunSpec(
        levels=(1, 2, 3),
        seeds=(10,),
        record_profile=RecordProfile.PILOT_DEBUG,
        camera_width=8,
        camera_height=8,
        review_video_fps=50,
    )

    manifest_path = pilot.collect_expert_pilot_run(
        project_root=Path.cwd(),
        output_root=tmp_path,
        run_id="pilot-test",
        spec=spec,
    )

    assert manifest_path == tmp_path / "pilot-test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "expert_pilot_run_v1"
    assert manifest["episode_count"] == 3
    assert len(manifest["implementation_revision"]) == 40
    assert len(manifest["implementation_source_sha256"]) == 64
    assert type(manifest["implementation_dirty"]) is bool
    assert [(row["level"], row["seed"]) for row in manifest["episodes"]] == [
        (1, 10),
        (2, 10),
        (3, 10),
    ]
    assert all(row["terminal_time_us"] < 4_000_000 for row in manifest["episodes"])
    assert all(row["handoff_time_us"] < 3_000_000 for row in manifest["episodes"])
    for row in manifest["episodes"]:
        episode_manifest = manifest_path.parent / row["episode_manifest"]
        review_video = manifest_path.parent / row["review_video"]
        assert episode_manifest.is_file()
        assert review_video.is_file()
        assert review_video.stat().st_size > 0
    assert all(len(value) == 64 for value in manifest["artifacts"].values())

    with pytest.raises(FileExistsError, match="already exists"):
        pilot.collect_expert_pilot_run(
            project_root=Path.cwd(),
            output_root=tmp_path,
            run_id="pilot-test",
            spec=spec,
        )


def test_collect_expert_pilot_cli_runs_a_real_tiny_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = importlib.import_module("latency_meta_mdp.cli.collect_expert_pilot")
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "collection_id: panda_ball_pilot_v1",
                "levels: [1]",
                "seeds: [10]",
                "record_profile: pilot_debug",
                "camera_width: 8",
                "camera_height: 8",
                "review_video_fps: 50",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = cli.main(
        [
            "--project-root",
            str(Path.cwd()),
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "tiny-cli",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert Path(output["manifest"]).is_file()
