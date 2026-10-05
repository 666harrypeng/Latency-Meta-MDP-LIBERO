from pathlib import Path

import pytest

from latency_meta_mdp.runtime.evaluate import evaluation_jobs


def test_fixed_grid_keeps_cases_and_conditioning_but_separates_realized_delay():
    cases = [{"master_index": 1}, {"master_index": 2}]
    identity = {"regime": "zero", "rtc_calibration": {"delay_ticks": [4] * 5}}
    jobs = list(
        evaluation_jobs(
            cases,
            regime="zero",
            output_root=Path("eval"),
            identity=identity,
            fixed_delay_ticks=[0, 1, 20],
        )
    )
    assert len(jobs) == 6
    assert [job[1] for job in jobs] == [
        "zero",
        "zero",
        "fixed20",
        "fixed20",
        "fixed400",
        "fixed400",
    ]
    assert [job[2] for job in jobs[::2]] == [
        Path("eval/fixed0"),
        Path("eval/fixed20"),
        Path("eval/fixed400"),
    ]
    assert [job[3]["actual_fixed_delay_ticks"] for job in jobs[::2]] == [0, 1, 20]
    assert all(job[3]["rtc_calibration"] == identity["rtc_calibration"] for job in jobs)
    assert [job[0] for job in jobs] == cases * 3
    assert "actual_fixed_delay_ticks" not in identity


@pytest.mark.parametrize("ticks", [[-1], [21], [4, 4], [], [True]])
def test_fixed_grid_rejects_invalid_ticks(ticks):
    with pytest.raises(ValueError):
        list(
            evaluation_jobs(
                [{}], regime="zero", output_root=Path("out"), identity={}, fixed_delay_ticks=ticks
            )
        )


def test_single_regime_keeps_existing_output_contract():
    identity = {"regime": "family"}
    assert list(
        evaluation_jobs([{}], regime="family", output_root=Path("out"), identity=identity)
    ) == [({}, "family", Path("out"), identity)]


def test_validation_cohort_rejects_training_masters_and_duplicate_trials():
    import copy
    import json

    from latency_meta_mdp.data.collection.contracts import _seed
    from latency_meta_mdp.io.artifacts import sha256_file
    from latency_meta_mdp.runtime.evaluate import validate_evaluation_cohort

    split_path = Path("configs/data/structured_source_split_v1.json")
    split = json.loads(split_path.read_text())
    master = split["validation_master_task_indices"][0]
    cohort = {
        "level": 2,
        "partition": "validation",
        "split_manifest_sha256": sha256_file(split_path),
        "cases": [
            {
                "master_index": master,
                "policy_seed": 0,
                "source_episode_id": f"source-L2-task{master:06d}-s00-d0000",
                "task_instance_id": {
                    "level": 2,
                    "task_instance_seed": _seed(
                        {"corpus_id": split["source_corpus_id"], "logical_task_index": master}
                    ),
                },
            }
        ],
    }
    validate_evaluation_cohort(cohort, project_root=Path.cwd())
    bad = copy.deepcopy(cohort)
    bad["cases"][0]["master_index"] = split["train_master_task_indices"][0]
    with pytest.raises(ValueError, match="validation"):
        validate_evaluation_cohort(bad, project_root=Path.cwd())
    bad = copy.deepcopy(cohort)
    bad["cases"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_evaluation_cohort(bad, project_root=Path.cwd())
    bad = copy.deepcopy(cohort)
    bad["split_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="split"):
        validate_evaluation_cohort(bad, project_root=Path.cwd())

    bad = copy.deepcopy(cohort)
    bad["cases"][0]["task_instance_id"]["task_instance_seed"] += 1
    with pytest.raises(ValueError, match="corpus/task identity"):
        validate_evaluation_cohort(bad, project_root=Path.cwd())
    bad = copy.deepcopy(cohort)
    bad["cases"][0]["task_instance_id"]["level"] = 3
    with pytest.raises(ValueError, match="level"):
        validate_evaluation_cohort(bad, project_root=Path.cwd())
