import copy

import pytest
from test_meta_cost import episode


def process_rows(*, conditioned, q):
    rows = []
    for index in range(4):
        row = copy.deepcopy(episode(q=q))
        row["identity"].update(
            profile_sha256="profile",
            rtc_max_guidance_weight=5,
            client_config_sha256="client",
            decision_interval_ticks=4,
        )
        if not conditioned:
            row["identity"]["conditioning"] = None
            row["forecast_calls"] = row["forecast_decodes"] = 0
            row["stage_events"] = [
                e for e in row["stage_events"] if not e["stage"].startswith("forecast")
            ]
        for j, event in enumerate(row["stage_events"]):
            event["wall_start_ns"] = index * 100_000 + j * 100
            event["wall_duration_ns"] = 20 if index else 10000
        rows.append(row)
    return rows


def test_calibration_discards_cold_calls_and_binds_policy():
    from latency_meta_mdp.meta.calibration import build_cost_profile

    groups = [process_rows(conditioned=True, q=True), process_rows(conditioned=False, q=False)]
    profile = build_cost_profile(groups, hardware="test")
    assert profile["weights"]["vla_conditioned"] == 1.0
    assert profile["calibration"]["samples"]["vla_conditioned"]["n"] == 3
    assert profile["calibration"]["samples"]["decode"]["median_seconds"] == 20 / 1e9
    groups[0][1]["identity"]["client_config_sha256"] = "changed"
    with pytest.raises(ValueError, match="binding"):
        build_cost_profile(groups, hardware="test")
