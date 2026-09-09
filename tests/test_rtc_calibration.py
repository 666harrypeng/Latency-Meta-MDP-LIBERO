import hashlib
import json

import pytest


def _files(tmp_path, *, requests=None):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "protocol_id": "rtc_observation_time_h50_v1",
                "latency_mode": "controlled_logical_policy_delay",
                "request_ledger": requests
                if requests is not None
                else [
                    {
                        "request_id": 0,
                        "source_formal_tick": 25,
                        "launch_formal_tick": 25,
                        "arrival_formal_tick": 29,
                        "realized_delay_ticks": 4,
                    },
                    {
                        "request_id": 1,
                        "source_formal_tick": 50,
                        "launch_formal_tick": 50,
                        "arrival_formal_tick": 56,
                        "realized_delay_ticks": 6,
                    },
                    {
                        "request_id": 2,
                        "source_formal_tick": 75,
                        "launch_formal_tick": 75,
                        "arrival_formal_tick": None,
                        "realized_delay_ticks": None,
                    },
                ],
            }
        )
    )
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "format_id": "rtc_delay_calibration_v1",
                "clock": "controlled_logical",
                "source_result": "source.json",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "request_ids": [0, 1],
            }
        )
    )
    return path, source


def test_calibration_extracts_only_bound_completed_requests(tmp_path):
    from latency_meta_mdp.rtc_calibration import load_rtc_calibration

    path, _ = _files(tmp_path)
    assert load_rtc_calibration(path, project_root=tmp_path).delay_ticks == (4, 6)
    data = json.loads(path.read_text())
    data["request_ids"] = [2]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="completed"):
        load_rtc_calibration(path, project_root=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("clock", "wall"),
        ("source_result", "../source.json"),
        ("request_ids", [1, 0]),
        ("request_ids", [0, 0]),
        ("request_ids", []),
    ],
)
def test_calibration_rejects_wrong_clock_path_or_selection(tmp_path, field, value):
    from latency_meta_mdp.rtc_calibration import load_rtc_calibration

    path, _ = _files(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_rtc_calibration(path, project_root=tmp_path)


def test_calibration_rejects_source_tamper_and_inconsistent_duration(tmp_path):
    from latency_meta_mdp.rtc_calibration import load_rtc_calibration

    path, source = _files(tmp_path)
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        load_rtc_calibration(path, project_root=tmp_path)
    data = json.loads(source.read_text())
    data["request_ledger"][0]["realized_delay_ticks"] = 9
    source.write_text(json.dumps(data))
    config = json.loads(path.read_text())
    config["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="duration"):
        load_rtc_calibration(path, project_root=tmp_path)
