from pathlib import Path

import pytest

from latency_meta_mdp.cli.evaluate_policy import evaluation_jobs


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
