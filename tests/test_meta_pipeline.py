import json
from pathlib import Path

import pytest
import yaml
from test_meta_success_data import manifest_fixture


def test_initial_replay_audits_grouping_and_preserves_sources(tmp_path):
    from latency_meta_mdp.meta.success_data import create_success_replay

    original = manifest_fixture(tmp_path)
    entries = json.loads(original.read_text())["episodes"]
    before = (tmp_path / "1.npz").read_bytes()
    target = tmp_path / "snapshot.json"
    create_success_replay(entries, target)
    data = json.loads(target.read_text())
    assert data["train_masters"] == [1]
    assert data["validation_masters"] == [2]
    assert data["expected_episodes"] == 2
    assert (tmp_path / "1.npz").read_bytes() == before
    with pytest.raises(FileExistsError):
        create_success_replay(entries, target)
    with pytest.raises(ValueError, match="duplicate"):
        create_success_replay(entries + entries, tmp_path / "duplicate.json")


def test_candidate_selection_uses_validation_and_marks_infeasible_fallback():
    from latency_meta_mdp.meta.pipeline import select_candidate

    def candidate(phase, cost, successes):
        return {
            "phase": phase,
            "mean_cost": cost,
            "validation": {
                "nominal": {"successes": successes, "episodes": 10},
                "family": {"successes": successes, "episodes": 10},
            },
        }

    candidates = [candidate(0, 12, 10), candidate(1, 8, 8), candidate(2, 7, 7)]
    selected = select_candidate(candidates, budget=10)
    assert selected["phase"] == 1
    assert selected["feasible_candidate_found"] is True
    fallback = select_candidate(candidates, budget=5)
    assert fallback["feasible_candidate_found"] is False


def test_cohort_split_rejects_feedback_leakage_and_duplicate_cases(tmp_path):
    from latency_meta_mdp.meta.cycle_config import validate_cohorts

    def cohort(name, master, part):
        path = tmp_path / (name + ".json")
        path.write_text(
            json.dumps(
                {
                    "partition": "train_pool_development",
                    "level": 2,
                    "cases": [{"master_index": master, "policy_seed": 0, "meta_partition": part}],
                }
            )
        )
        return str(path)

    paths = {
        "train": {"nominal": cohort("train", 1, "train")},
        "validation": {"nominal": cohort("validation", 2, "validation")},
        "feedback": {"nominal": cohort("feedback", 1, "train")},
    }
    validate_cohorts(paths)
    paths["feedback"]["nominal"] = cohort("leak", 2, "train")
    with pytest.raises(ValueError, match="feedback"):
        validate_cohorts(paths)
    paths["feedback"]["nominal"] = paths["train"]["nominal"]
    p = Path(paths["train"]["nominal"])
    data = json.loads(p.read_text())
    data["cases"] *= 2
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        validate_cohorts(paths)


def test_cycle_resumes_completed_work_and_updates_budget_only_from_feedback(tmp_path, monkeypatch):
    from latency_meta_mdp.meta import pipeline

    manifest = manifest_fixture(tmp_path)
    config = {
        "phases": 2,
        "budget": 10.0,
        "initial_multiplier": 0.1,
        "multiplier_step": 0.1,
        "initial_replay": str(manifest),
        "initial_checkpoint": None,
        "exploration_epsilon": 0.2,
        "replica_start": 100,
        "cost_profile": {},
        "identity": "test",
    }
    monkeypatch.setattr(pipeline, "load_cycle_config", lambda _: config)
    fits, collections = [], []
    fail = [True]

    def fit(cfg, replay, multiplier, init, output):
        fits.append(multiplier)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "config.json", "training-summary.json"):
            (output / name).write_text("{}")
        return output

    def collect(cfg, partition, output, **kwargs):
        collections.append((partition, str(output)))
        if partition == "validation" and fail[0]:
            fail[0] = False
            raise RuntimeError("interrupted worker")
        return [{"result": "unused", "replay": "unused"}]

    def summary(entries, profile):
        # Feedback cost drives lambda; validation measures selection only.
        return {"nominal": {"episodes": 2, "successes": 1, "mean_cost": 12.0}}

    def append(parent, entries, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(Path(parent).read_text())

    monkeypatch.setattr(pipeline, "fit_phase", fit)
    monkeypatch.setattr(pipeline, "run_rollouts", collect)
    monkeypatch.setattr(pipeline, "summarize_results", summary)
    monkeypatch.setattr(pipeline, "append_success_replay", append)
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1}))
    work = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.run_cycle(config_path, work)
    assert len(fits) == 1
    pipeline.run_cycle(config_path, work, resume=True)
    assert fits == pytest.approx([0.1, 0.12])
    assert len([p for p, _ in collections if p == "feedback"]) == 2
    state = json.loads((work / "cycle.json").read_text())
    assert state["status"] == "completed"
    assert state["selected"]["feasible_candidate_found"] is False
    assert len([p for p, _ in collections if p == "train"]) == 1
    pipeline.run_cycle(config_path, work, resume=True)
    assert len(fits) == 2
    config["identity"] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        pipeline.run_cycle(config_path, work, resume=True)


def test_rollout_command_keeps_shared_forecast_and_replay_collection_separate():
    from latency_meta_mdp.meta.rollouts import rollout_command, worker_environment

    cfg = {
        "evaluation_python": "/env/bin/python",
        "record_video": True,
        "evaluation": {"checkpoint": "/assets/policy", "decision_interval_ticks": 4},
    }
    common = dict(
        cohort="cohort.json",
        regime="family",
        output="result",
        worker=1,
        workers=4,
        scheduler="learned",
        checkpoint="meta",
        epsilon=0.0,
        replica=3,
    )
    validation = rollout_command(cfg, collect=False, **common)
    collection = rollout_command(cfg, collect=True, **common)
    assert "--prepare-forecast-before-decision" in validation
    assert "--collect-meta-transitions" not in validation
    assert "--collect-meta-transitions" in collection
    assert "--record-video" in validation
    assert validation[validation.index("--worker-index") + 1] == "1"
    env = worker_environment(5)
    assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == "5"


def test_cycle_config_binds_inputs_and_keeps_cost_profile_path(tmp_path, monkeypatch):
    import sys

    from test_meta_cost import profile

    from latency_meta_mdp.meta.cycle_config import EVALUATION_PATHS, load_cycle_config

    monkeypatch.chdir(tmp_path)
    evaluation = {}
    for key in EVALUATION_PATHS:
        file = tmp_path / (key + ".json")
        file.write_text("{}")
        evaluation[key] = file.name
    cohorts = {}
    for group, master in (("train", 1), ("feedback", 1), ("validation", 2)):
        file = tmp_path / (group + ".json")
        file.write_text(
            json.dumps(
                {
                    "partition": "train_pool_development",
                    "level": 2,
                    "cases": [
                        {
                            "master_index": master,
                            "policy_seed": 0,
                            "meta_partition": "validation" if group == "validation" else "train",
                        }
                    ],
                }
            )
        )
        cohorts[group] = {"nominal": file.name}
    (tmp_path / "training.yaml").write_text(
        yaml.safe_dump({"objective": "rtc_budget_success_v1", "recoverable_training": True})
    )
    (tmp_path / "cost.json").write_text(json.dumps(profile()))
    job = dict(
        schema_version=1,
        training_config="training.yaml",
        cost_profile="cost.json",
        budget=10,
        phases=2,
        evaluation_python=sys.executable,
        gpu_ids=[0, 1],
        evaluation=evaluation,
        cohorts=cohorts,
    )
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job))
    loaded = load_cycle_config(path)
    # Resolve asset paths, but keep venv executable symlinks: dereferencing them
    # invokes the base Python and silently loses the environment's packages.
    env_python = tmp_path / "venv/bin/python"
    env_python.parent.mkdir(parents=True)
    env_python.symlink_to(sys.executable)
    job["evaluation_python"] = "venv/bin/python"
    path.write_text(yaml.safe_dump(job))
    assert load_cycle_config(path)["evaluation_python"] == str(env_python)
    assert loaded["cost_profile_path"] == str(tmp_path / "cost.json")
    assert loaded["cost_profile"]["weights"]["vla_conditioned"] == 1
    (tmp_path / "training.yaml").write_text(
        yaml.safe_dump(
            {"objective": "rtc_budget_success_v1", "recoverable_training": True, "updates": 10}
        )
    )
    assert load_cycle_config(path)["identity"] != loaded["identity"]
