import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from latency_meta_mdp.data.conveyor.source import ConveyorRecorder
from latency_meta_mdp.runtime.policy_execution import PolicyObservation


def job(tmp_path, splits=None):
    path = tmp_path / "job.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "task_id": "conveyor_sort",
                "corpus_id": "test-corpus",
                "purpose": "development_smoke",
                "scene": str(Path("configs/tasks/conveyor_sort/surface.yaml").resolve()),
                "expert": str(Path("configs/data/expert/conveyor_surface.yaml").resolve()),
                "splits": splits or {"train": [1000, 1001], "validation": [2000], "test": [3000]},
                "collect_splits": ["train"],
            }
        )
    )
    return path


def episode_runner(scene, expert, output, *, seed, **kwargs):
    from dataclasses import asdict

    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.expert import load_expert_spec

    output = Path(output)
    output.mkdir(parents=True)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    summary = dict(spawned=10, successes=10, misses=0, timeouts=0, active=0, end_tick=1)
    with ConveyorRecorder(
        output / "source",
        seed=seed,
        action_contract_id="panda_osc_pose_delta_conveyor_v2",
        purpose=kwargs.get("source_purpose", "development_smoke"),
        context={
            "scene": asdict(load_conveyor_spec(scene)),
            "expert": asdict(load_expert_spec(expert)),
        },
    ) as recorder:
        recorder.start(PolicyObservation(0, image, image, np.zeros(16)))
        recorder.append(
            np.zeros(7),
            PolicyObservation(1, image, image, np.ones(16)),
            success_delta=10,
            done=True,
        )
        recorder.finish(summary)
    status = dict(
        status="completed",
        seed=seed,
        expert_admitted=True,
        all_scheduled_spawned=True,
        source_manifest="source/manifest.json",
        **summary,
    )
    (output / "status.json").write_text(json.dumps(status))
    return status


def test_collection_manifest_and_resume(tmp_path):
    from latency_meta_mdp.data.conveyor.collection import collect

    config, output = job(tmp_path), tmp_path / "corpus"
    manifest = collect(config, output, run_episode=episode_runner)
    data = json.loads(manifest.read_text())
    assert data["status"] == "completed"
    assert data["requested_count"] == data["admitted_count"] == 2
    assert {e["seed"] for e in data["episodes"]} == {1000, 1001}
    assert all(e["split"] == "train" for e in data["episodes"])

    def forbidden(*args, **kwargs):
        raise AssertionError("resume reran a completed episode")

    assert collect(config, output, resume=True, run_episode=forbidden) == manifest
    config.write_text(config.read_text().replace("test-corpus", "changed-corpus"))
    with pytest.raises(ValueError, match="identity"):
        collect(config, output, resume=True, run_episode=forbidden)


def test_collection_rejects_split_overlap(tmp_path):
    from latency_meta_mdp.data.conveyor.collection import load_collection_job

    config = job(tmp_path, {"train": [1], "validation": [1], "test": []})
    with pytest.raises(ValueError, match="overlap"):
        load_collection_job(config)


def test_collection_keeps_failed_seed_in_denominator(tmp_path):
    from latency_meta_mdp.data.conveyor.collection import collect

    def failed(scene, expert, output, *, seed, **kwargs):
        Path(output).mkdir(parents=True)
        status = dict(
            status="completed",
            seed=seed,
            expert_admitted=False,
            spawned=10,
            successes=9,
            misses=1,
            timeouts=0,
        )
        (Path(output) / "status.json").write_text(json.dumps(status))
        return status

    manifest = collect(job(tmp_path), tmp_path / "failed", run_episode=failed)
    data = json.loads(manifest.read_text())
    assert data["status"] == "needs_attention"
    assert data["requested_count"] == data["rejected_count"] == 2
    assert data["admitted_count"] == 0


def test_training_source_purpose_and_seed_ranges_are_explicit(tmp_path):
    from latency_meta_mdp.data.conveyor.collection import collect, load_collection_job

    config = job(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["purpose"] = "training_source"
    value["splits"]["train"] = {"start": 1000, "stop": 1002}
    config.write_text(yaml.safe_dump(value))
    assert load_collection_job(config)["splits"]["train"] == [1000, 1001]
    manifest = collect(config, tmp_path / "formal", run_episode=episode_runner)
    data = json.loads(manifest.read_text())
    for row in data["episodes"]:
        source = json.loads((manifest.parent / row["source_root"] / "manifest.json").read_text())
        assert source["purpose"] == data["purpose"] == "training_source"
