from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from latency_meta_mdp.artifacts import sha256_file


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_source(
    root: Path,
    *,
    run_id: str,
    seeds: tuple[int, ...],
    format_id: str,
    control_hash: str = "control-v1",
) -> Path:
    artifacts = {}
    admitted = []
    for level in (1, 2, 3):
        for seed in seeds:
            episode = root / run_id / "attempts" / f"L{level}" / f"seed_{seed:06d}"
            episode.mkdir(parents=True)
            for name in ("arrays.npz", "events.json"):
                (episode / name).write_bytes(f"{level}:{seed}:{name}".encode())
            metadata = {
                "schema_version": 3,
                "episode_id": f"l{level}-seed-{seed:06d}-attempt-000",
                "level": level,
                "scene_seed": seed,
                "motion_seed": seed,
                "expert_seed": seed,
                "formal_tick_us": 20_000,
                "physics_dt_us": 2_000,
                "record_profile": "belief",
                "task_id": "dynamic_grasp_lift",
                "action_contract_id": "panda_osc_pose_delta_v1",
                "action_dim": 7,
                "actuator_dim": 9,
                "expert_id": "panda_ball_feedback_v1",
                "instruction": "grasp and lift the moving ball",
                "config_sha256": {
                    "control": control_hash,
                    "expert": "expert-v1",
                    "motion": f"motion-L{level}",
                    "runtime": "runtime-v1",
                    "task": "task-v1",
                },
                "motion_profile": {"type": f"L{level}"},
                "terminal_status": "success",
                "terminal_reason": "lift_succeeded",
            }
            _write_json(episode / "metadata.json", metadata)
            nested = {
                "schema_version": 3,
                "format_id": "synchronized_episode_npz_v3",
                "record_profile": "belief",
                "episode_id": metadata["episode_id"],
                "boundary_count": 2,
                "transition_count": 1,
                "physical_event_count": 0,
                "terminal_status": "success",
                "terminal_reason": "lift_succeeded",
                "artifacts": {
                    name: sha256_file(episode / name)
                    for name in ("arrays.npz", "events.json", "metadata.json")
                },
            }
            _write_json(episode / "manifest.json", nested)
            relative = (episode / "manifest.json").relative_to(root / run_id).as_posix()
            admitted.append(relative)
            artifacts[relative] = sha256_file(episode / "manifest.json")
    manifest = {
        "schema_version": 1,
        "format_id": format_id,
        "run_id": run_id,
        "collection_id": "panda_ball_bulk_v1",
        "implementation_revision": "1" * 40,
        "implementation_source_sha256": "2" * 64,
        "implementation_dirty": False,
        "bulk_plan_sha256": "3" * 64,
        "levels": [1, 2, 3],
        "record_profile": "belief",
        "camera_width": 256,
        "camera_height": 256,
        "eligible": True,
        "blockers": [],
        "admitted_episode_manifests": admitted,
        "artifacts": artifacts,
    }
    _write_json(root / run_id / "manifest.json", manifest)
    return root / run_id / "manifest.json"


def test_formal_merge_hardlinks_exact_complete_seed_coverage(tmp_path: Path) -> None:
    from latency_meta_mdp.formal_corpus import materialize_formal_corpus

    first = _write_source(
        tmp_path,
        run_id="first",
        seeds=(1000,),
        format_id="panda_ball_bulk_first_tranche_v1",
    )
    continuation = _write_source(
        tmp_path,
        run_id="continuation",
        seeds=(1001,),
        format_id="panda_ball_bulk_range_v1",
    )

    manifest_path = materialize_formal_corpus(
        project_root=Path.cwd(),
        source_manifests=(first, continuation),
        output_dir=tmp_path / "formal",
        levels=(1, 2, 3),
        seed_start=1000,
        seed_count=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format_id"] == "panda_ball_formal_corpus_v1"
    assert manifest["episode_count"] == 6
    assert manifest["seed_start"] == 1000
    assert manifest["seed_count_per_level"] == 2
    linked = manifest_path.parent / "episodes/L1/seed_001000/arrays.npz"
    source = first.parent / "attempts/L1/seed_001000/arrays.npz"
    assert not linked.is_symlink()
    assert os.stat(linked).st_ino == os.stat(source).st_ino


@pytest.mark.parametrize("mode", ("overlap", "gap", "semantic_mismatch"))
def test_formal_merge_rejects_invalid_sources_without_partial_output(
    tmp_path: Path,
    mode: str,
) -> None:
    from latency_meta_mdp.formal_corpus import materialize_formal_corpus

    first = _write_source(
        tmp_path,
        run_id="first",
        seeds=(1000,),
        format_id="panda_ball_bulk_first_tranche_v1",
    )
    second_seed = 1000 if mode == "overlap" else 1002 if mode == "gap" else 1001
    continuation = _write_source(
        tmp_path,
        run_id="continuation",
        seeds=(second_seed,),
        format_id="panda_ball_bulk_range_v1",
        control_hash="different" if mode == "semantic_mismatch" else "control-v1",
    )
    output = tmp_path / "formal"

    with pytest.raises(ValueError):
        materialize_formal_corpus(
            project_root=Path.cwd(),
            source_manifests=(first, continuation),
            output_dir=output,
            levels=(1, 2, 3),
            seed_start=1000,
            seed_count=2,
        )

    assert not output.exists()
    assert list(tmp_path.glob("formal.building-*")) == []


def test_formal_corpus_cli_parses_two_sources_and_expected_coverage(tmp_path: Path) -> None:
    from latency_meta_mdp.cli.materialize_formal_corpus import _parser

    args = _parser().parse_args(
        [
            "--source-manifest",
            "first/manifest.json",
            "--source-manifest",
            "continuation/manifest.json",
            "--output-dir",
            str(tmp_path / "formal"),
            "--seed-start",
            "1000",
            "--seed-count",
            "200",
        ]
    )

    assert len(args.source_manifests) == 2
    assert args.levels == [1, 2, 3]
    assert args.seed_start == 1000
    assert args.seed_count == 200
