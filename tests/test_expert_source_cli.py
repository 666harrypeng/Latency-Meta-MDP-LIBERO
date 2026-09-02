from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).parents[1]


def _collection_args(tmp_path: Path, *, planner: Path) -> list[str]:
    return [
        "--project-root",
        str(ROOT),
        "--formal-config",
        "configs/source_corpus/panda_ball_formal_source_pilot.yaml",
        "--execution-config",
        "configs/source_corpus/panda_ball_formal_source_execution.yaml",
        "--source-config",
        "configs/source_corpus/panda_ball_source_parquet.yaml",
        "--split-config",
        "configs/source_corpus/panda_ball_formal_source_pilot_split.yaml",
        "--work-root",
        str(tmp_path / "work"),
        "--output-root",
        str(tmp_path / "output"),
        "--planner-python",
        str(planner),
    ]


def test_dry_run_expands_exact_request_without_creating_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: dry-run changes the reviewed 3x3x4 request or mutates disk."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    work = tmp_path / "work"
    output = tmp_path / "output"

    assert main([*_collection_args(tmp_path, planner=planner), "--dry-run"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    summary = json.loads(captured.err)
    assert summary["mode"] == "dry_run"
    assert summary["corpus_id"] == "panda-ball-structured-source-pilot-3x4-v2"
    assert summary["primary_master_task_indices"] == [0, 1, 2]
    assert summary["reserve_master_task_indices"] == [3, 4, 5]
    assert summary["levels"] == [1, 2, 3]
    assert summary["realizations_per_task"] == 4
    assert summary["target_success_count"] == 36
    assert summary["predeclared_realization_count"] == 72
    assert summary["expected_planner_calls_if_candidate_zero_qualifies"] == 39
    assert set(summary["family_assignments"]) == {"0", "1", "2", "3", "4", "5"}
    assert all(len(rows) == 4 for rows in summary["family_assignments"].values())
    assert len(summary["formal_config_sha256"]) == 64
    assert len(summary["formal_request_sha256"]) == 64
    assert len(summary["source_config_sha256"]) == 64
    assert len(summary["split_config_sha256"]) == 64
    assert len(summary["execution_config_sha256"]) == 64
    assert not work.exists()
    assert not output.exists()


def test_dry_run_subprocess_does_not_import_simulator_or_planner_stacks(
    tmp_path: Path,
) -> None:
    """Break caught: CLI module import eagerly initializes heavy online runtimes."""
    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        """
import builtins
_real_import = builtins.__import__
def _guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'curobo', 'mujoco', 'robosuite'}:
        raise RuntimeError('heavy runtime imported during dry-run: ' + name)
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _guarded
""".lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(guard), str(ROOT / "src")))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "latency_meta_mdp.cli.collect_structured_expert_source",
            *_collection_args(tmp_path, planner=planner),
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout == ""
    assert json.loads(process.stderr)["mode"] == "dry_run"
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "output").exists()


def test_actual_mode_preserves_lexical_launcher_and_stream_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: CLI resolves the venv symlink or pollutes stdout with progress."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    environment_python = tmp_path / "runtime" / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    environment_python.write_text("", encoding="utf-8")
    symlink = tmp_path / "planner-python"
    symlink.symlink_to(environment_python)
    received = {}
    final_manifest = tmp_path / "output" / "manifest.json"

    def collect(**kwargs):
        received.update(kwargs)
        kwargs["on_progress"]("candidate level=1 index=0 started")
        return final_manifest

    assert main(_collection_args(tmp_path, planner=symlink), collect_fn=collect) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{final_manifest}\n"
    assert captured.err == "candidate level=1 index=0 started\n"
    assert received["planner_python"] == symlink.absolute()
    assert received["planner_python"] != symlink.resolve()
    assert received["resume"] is False


def test_resume_flag_is_explicit_and_forwarded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: an existing run is resumed implicitly or --resume is discarded."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    received = {}

    def collect(**kwargs):
        received.update(kwargs)
        return tmp_path / "output" / "manifest.json"

    assert main([*_collection_args(tmp_path, planner=planner), "--resume"], collect_fn=collect) == 0
    capsys.readouterr()
    assert received["resume"] is True


def test_collection_cli_rejects_missing_launcher_and_nested_roots(tmp_path: Path) -> None:
    """Break caught: a malformed operational request reaches the collection runtime."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    with pytest.raises(FileNotFoundError, match="launcher"):
        main([*_collection_args(tmp_path, planner=tmp_path / "missing"), "--dry-run"])

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    args = _collection_args(tmp_path, planner=planner)
    output_index = args.index("--output-root") + 1
    args[output_index] = str(tmp_path / "work" / "output")
    with pytest.raises(ValueError, match="non-nested"):
        main([*args, "--dry-run"])


def test_collection_cli_strictly_rejects_unknown_config_fields(tmp_path: Path) -> None:
    """Break caught: misspelled scientific configuration fields are silently ignored."""
    from latency_meta_mdp.cli.collect_structured_expert_source import main

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    bad = tmp_path / "bad-execution.yaml"
    source = ROOT / "configs/source_corpus/panda_ball_formal_source_execution.yaml"
    bad.write_text(source.read_text(encoding="utf-8") + "unexpected: true\n", encoding="utf-8")
    args = _collection_args(tmp_path, planner=planner)
    config_index = args.index("--execution-config") + 1
    args[config_index] = str(bad)
    with pytest.raises(ValueError, match="unknown source execution fields"):
        main([*args, "--dry-run"])


def test_inspection_cli_emits_verified_corpus_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: inspection omits the admitted block and per-level inventory."""
    from latency_meta_mdp.cli.inspect_structured_expert_source import main
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        SourceCorpusManifest,
        VerifiedSourceCorpus,
    )

    manifest = SourceCorpusManifest(
        corpus_id="demo",
        request_sha256="a" * 64,
        source_config_sha256="b" * 64,
        split_plan_sha256="c" * 64,
        admitted_master_task_indices=(0, 1, 3),
        master_task_count=3,
        level_task_instance_count=9,
        episode_count=36,
        episodes_by_level=MappingProxyType({"1": 12, "2": 12, "3": 12}),
        frame_count=5_760,
        shard_count=3,
        artifacts=MappingProxyType(
            {
                "data/level-1/shard-00000.parquet": MappingProxyType(
                    {"sha256": "d" * 64, "bytes": 123}
                )
            }
        ),
    )
    rows = tuple(
        {
            "episode_id": f"episode-{level}-{split}",
            "level": level,
            "split": split,
        }
        for level in (1, 2, 3)
        for split in ("train", "validation")
    )
    corpus = VerifiedSourceCorpus(root=tmp_path / "corpus", manifest=manifest, episode_rows=rows)

    assert main(["--source-root", str(tmp_path / "corpus")], load_fn=lambda _root: corpus) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "admitted_master_task_indices": [0, 1, 3],
        "artifact_bytes": 123,
        "corpus_id": "demo",
        "episode_count": 36,
        "episodes_by_level": {"1": 12, "2": 12, "3": 12},
        "frame_count": 5760,
        "level_task_instance_count": 9,
        "master_task_count": 3,
        "root": str(tmp_path / "corpus"),
        "shard_count": 3,
        "split_episode_counts": {
            "level_1_train": 1,
            "level_1_validation": 1,
            "level_2_train": 1,
            "level_2_validation": 1,
            "level_3_train": 1,
            "level_3_validation": 1,
        },
    }


def test_formal_config_hash_is_raw_file_sha256(tmp_path: Path) -> None:
    """Break caught: CLI request identity hashes parsed values rather than immutable bytes."""
    from latency_meta_mdp.cli.collect_structured_expert_source import load_collection_inputs

    planner = tmp_path / "planner-python"
    planner.write_text("", encoding="utf-8")
    inputs = load_collection_inputs(_collection_args(tmp_path, planner=planner))
    path = ROOT / "configs/source_corpus/panda_ball_formal_source_pilot.yaml"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert inputs.formal_request.corpus_config_sha256 == expected
