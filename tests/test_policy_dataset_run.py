from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.policy_dataset_run import (
    convert_formal_corpus_to_lerobot,
    convert_pilot_run_to_lerobot,
)
from latency_meta_mdp.recording import RecordProfile
from latency_meta_mdp.sft_certification import certify_lerobot_pilot_run
from latency_meta_mdp.sft_profile import load_sft_profile


class _FakeLeRobotDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames: list[dict] = []
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}\n", encoding="utf-8")

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        data_dir = self.root / "data"
        data_dir.mkdir(exist_ok=True)
        episode_index = len(list(data_dir.glob("episode_*.txt")))
        (data_dir / f"episode_{episode_index:06d}.txt").write_text(
            f"{len(self.frames)}\n",
            encoding="utf-8",
        )
        self.frames = []


class _FakeDatasetFactory:
    def __init__(self) -> None:
        self.roots: list[Path] = []

    def __call__(self, **kwargs) -> _FakeLeRobotDataset:
        root = Path(kwargs["root"])
        self.roots.append(root)
        return _FakeLeRobotDataset(root)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_pilot(tmp_path: Path) -> Path:
    root = tmp_path / "raw-pilot"
    rows = []
    artifacts = {}
    for level in (1, 2, 3):
        episode = collect_expert_episode(
            project_root=Path.cwd(),
            spec=ExpertEpisodeSpec(
                episode_id=f"l{level}-seed-000010-attempt-000",
                level=level,
                scene_seed=10,
                motion_seed=10,
                expert_seed=10,
                record_profile=RecordProfile.SFT,
                camera_width=8,
                camera_height=8,
            ),
        )
        episode_root = root / "episodes" / f"L{level}" / "seed_000010"
        manifest = write_synchronized_episode_artifact(
            episode=episode,
            output_dir=episode_root,
        )
        relative_manifest = manifest.relative_to(root).as_posix()
        artifacts[relative_manifest] = sha256_file(manifest)
        rows.append(
            {
                "episode_id": episode.metadata.episode_id,
                "level": level,
                "seed": 10,
                "boundary_count": len(episode.boundaries),
                "transition_count": len(episode.transitions),
                "handoff_time_us": 0,
                "terminal_time_us": episode.boundaries[-1].time_us,
                "episode_manifest": relative_manifest,
                "review_video": f"review/L{level}.mp4",
            }
        )
    source_manifest = root / "manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "format_id": "expert_pilot_run_v1",
            "run_id": "test-pilot",
            "implementation_revision": "1" * 40,
            "implementation_source_sha256": "2" * 64,
            "implementation_dirty": False,
            "record_profile": "sft",
            "camera_width": 8,
            "camera_height": 8,
            "review_video_fps": 50,
            "episode_count": 3,
            "episodes": rows,
            "artifacts": artifacts,
        },
    )
    return source_manifest


def _source_formal(tmp_path: Path) -> Path:
    pilot_manifest = _source_pilot(tmp_path)
    root = pilot_manifest.parent
    pilot = json.loads(pilot_manifest.read_text(encoding="utf-8"))
    admitted = []
    artifacts = {}
    for row in pilot["episodes"]:
        relative_manifest = row["episode_manifest"]
        episode_manifest = root / relative_manifest
        admitted.append(relative_manifest)
        artifacts[relative_manifest] = sha256_file(episode_manifest)
        nested = json.loads(episode_manifest.read_text(encoding="utf-8"))
        for name in nested["artifacts"]:
            relative = (Path(relative_manifest).parent / name).as_posix()
            artifacts[relative] = sha256_file(root / relative)
    source_manifest = root / "formal-manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "format_id": "panda_ball_formal_corpus_v1",
            "eligible": True,
            "blockers": [],
            "implementation_revision": "1" * 40,
            "implementation_source_sha256": "2" * 64,
            "implementation_dirty": False,
            "levels": [1, 2, 3],
            "seed_start": 10,
            "seed_count_per_level": 1,
            "episode_count": 3,
            "source_runs": [],
            "admitted_episode_manifests": admitted,
            "artifacts": artifacts,
        },
    )
    return source_manifest


def test_convert_pilot_run_writes_three_atomic_level_specific_datasets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_manifest = _source_pilot(tmp_path)
    profile_path = Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml")
    profile = load_sft_profile(profile_path)
    output = tmp_path / "derived"
    factory = _FakeDatasetFactory()

    manifest_path = convert_pilot_run_to_lerobot(
        source_manifest=source_manifest,
        output_dir=output,
        profile_path=profile_path,
        dataset_factory=factory,
    )

    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "metamdp_lerobot_pilot_run_v1"
    assert manifest["source_manifest_sha256"] == sha256_file(source_manifest)
    assert manifest["sft_profile_sha256"] == sha256_file(profile_path)
    assert manifest["episode_count"] == 3
    assert set(row["level"] for row in manifest["datasets"]) == {1, 2, 3}
    assert len(factory.roots) == 3
    for row in manifest["datasets"]:
        level = row["level"]
        assert row["repo_id"] == profile.levels[level].repo_id
        assert row["episode_count"] == 1
        assert row["frame_count"] > row["valid_action_chunk_source_count"]
        dataset_manifest = output / row["dataset_manifest"]
        assert dataset_manifest.is_file()
        assert row["dataset_manifest_sha256"] == sha256_file(dataset_manifest)

    cli = importlib.import_module("latency_meta_mdp.cli.convert_sft_pilot")
    cli_output = tmp_path / "derived-cli"
    result = cli.main(
        [
            "--source-manifest",
            str(source_manifest),
            "--output-dir",
            str(cli_output),
            "--profile",
            str(profile_path),
        ],
        dataset_factory=_FakeDatasetFactory(),
    )
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert Path(printed["manifest"]) == cli_output / "manifest.json"

    def probe(**kwargs) -> dict:
        expected = kwargs["expected_source_count"]
        return {
            "metadata_fps": 50,
            "episode_count": 1,
            "frame_count": expected + 49,
            "source_count": expected,
            "norm_source_count": expected,
            "norm_batch_sizes": [expected],
            "no_action_padding": True,
            "train_state_shape": [4, 32],
            "train_action_shape": [4, 50, 32],
        }

    certification_dir = tmp_path / "certification"
    certification_path = certify_lerobot_pilot_run(
        project_root=Path.cwd(),
        derived_manifest=manifest_path,
        output_dir=certification_dir,
        profile_path=profile_path,
        level_probe=probe,
    )
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["format_id"] == "metamdp_openpi_pilot_certification_v1"
    assert certification["derived_manifest_sha256"] == sha256_file(manifest_path)
    assert certification["eligible"] is (not certification["implementation_dirty"])
    assert [row["level"] for row in certification["levels"]] == [1, 2, 3]

    certification_cli = importlib.import_module(
        "latency_meta_mdp.cli.certify_sft_pilot"
    )
    cli_certification_dir = tmp_path / "certification-cli"
    result = certification_cli.main(
        [
            "--project-root",
            str(Path.cwd()),
            "--derived-manifest",
            str(manifest_path),
            "--output-dir",
            str(cli_certification_dir),
            "--profile",
            str(profile_path),
        ],
        level_probe=probe,
    )
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert Path(printed["manifest"]) == cli_certification_dir / "manifest.json"


def test_convert_formal_corpus_streams_three_verified_level_datasets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_manifest = _source_formal(tmp_path)
    profile_path = Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml")
    output = tmp_path / "formal-derived"
    factory = _FakeDatasetFactory()

    manifest_path = convert_formal_corpus_to_lerobot(
        source_manifest=source_manifest,
        output_dir=output,
        profile_path=profile_path,
        dataset_factory=factory,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "metamdp_lerobot_formal_corpus_v1"
    assert manifest["source_format_id"] == "panda_ball_formal_corpus_v1"
    assert manifest["source_manifest_sha256"] == sha256_file(source_manifest)
    assert manifest["episode_count"] == 3
    assert [row["episode_count"] for row in manifest["datasets"]] == [1, 1, 1]
    assert len(factory.roots) == 3

    cli = importlib.import_module("latency_meta_mdp.cli.convert_sft_formal")
    cli_output = tmp_path / "formal-derived-cli"
    result = cli.main(
        [
            "--source-manifest",
            str(source_manifest),
            "--output-dir",
            str(cli_output),
            "--profile",
            str(profile_path),
        ],
        dataset_factory=_FakeDatasetFactory(),
    )
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert Path(printed["manifest"]) == cli_output / "manifest.json"


def test_convert_formal_corpus_rejects_a_bad_episode_inventory_hash(
    tmp_path: Path,
) -> None:
    source_manifest = _source_formal(tmp_path)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    first = source["admitted_episode_manifests"][0]
    source["artifacts"][first] = "0" * 64
    _write_json(source_manifest, source)
    output = tmp_path / "formal-derived"

    with pytest.raises(ValueError, match="episode manifest hash mismatch"):
        convert_formal_corpus_to_lerobot(
            source_manifest=source_manifest,
            output_dir=output,
            profile_path=Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"),
            dataset_factory=_FakeDatasetFactory(),
        )

    assert not output.exists()
