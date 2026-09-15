import copy
import json

import pytest


def profile():
    return {
        "schema": 1,
        "unit": "conditioned_vla_rtc_equivalent",
        "reference_seconds": 0.2,
        "binding": {"protocol_id": "rtc-test"},
        "weights": {
            "vla_conditioned": 1.0,
            "vla_clean": 0.8,
            "forecast": 0.2,
            "decode": 0.03,
            "meta_q": 0.01,
            "rule": 0.001,
            "bootstrap_clean": 0.5,
        },
    }


def episode(*, combined=False, q=True):
    def event(stage, tick=None, **kw):
        return {"stage": stage, "formal_tick": tick, "wall_duration_ns": 1, **kw}

    events = [event("bootstrap_launch"), event("bootstrap_return")]
    if not combined:
        events.append(event("forecast_prepare", 10, available=True))
    events.append(event("meta_decision", 10, q_values=[0.0, 1.0] if q else None, launch=False))
    if not combined:
        events.append(event("forecast_prepare", 14, available=True))
    events += [
        event("meta_decision", 14, q_values=[0.0, 1.0] if q else None, launch=True),
        event("forecast" if combined else "forecast_decode", 14, available=True, request_id=0),
        event("policy_launch", 14, request_id=0),
        event("policy_return", 14, request_id=0),
    ]
    return {
        "identity": {"protocol_id": "rtc-test", "conditioning": "native_rtc_forecast_rgb_v1"},
        "stage_events": events,
        "bootstrap_calls": 1,
        "policy_calls": 1,
        "forecast_calls": 1 if combined else 2,
        "forecast_decodes": 1,
        "pending_at_terminal": True,
        "terminated": True,
        "truncated": False,
        "decision_transitions": [
            {"start_tick": 10, "end_tick": 14, "action": "wait"},
            {"start_tick": 14, "end_tick": 15, "action": "launch"},
        ],
    }


def test_cost_assigns_boundary_forecast_to_new_decision_and_charges_pending():
    from latency_meta_mdp.meta.cost import episode_cost_components

    r = episode_cost_components(episode(), profile())
    assert r["initial_cost"] == 0.5
    assert r["transition_costs"] == pytest.approx([0.21, 1.24])
    assert r["episode_cost"] == pytest.approx(1.95)
    assert r["counts"]["vla_conditioned"] == 1  # No chunk installation occurred.


def test_combined_forecast_is_not_double_charged_and_rule_is_not_q():
    from latency_meta_mdp.meta.cost import episode_cost_components

    r = episode_cost_components(episode(combined=True, q=False), profile())
    assert r["counts"]["meta_q"] == 0 and r["counts"]["rule"] == 2
    assert r["transition_costs"] == pytest.approx([0.001, 1.231])
    assert r["episode_cost"] == pytest.approx(1.732)


def test_unavailable_forecast_does_not_charge_full_prediction_or_decoder():
    from latency_meta_mdp.meta.cost import episode_cost_components

    d = episode()
    for e in d["stage_events"]:
        if "available" in e:
            e["available"] = False
    d["forecast_decodes"] = 0
    r = episode_cost_components(d, profile())
    assert r["counts"]["forecast"] == r["counts"]["decode"] == 0
    assert r["episode_cost"] == pytest.approx(1.52)


def test_clean_baseline_uses_same_unit_without_forecast_or_replay():
    from latency_meta_mdp.meta.cost import episode_cost_components

    d = episode(combined=True, q=False)
    d["identity"]["conditioning"] = None
    d["stage_events"] = [e for e in d["stage_events"] if e["stage"] != "forecast"]
    d["forecast_calls"] = d["forecast_decodes"] = 0
    del d["decision_transitions"]
    r = episode_cost_components(d, profile())
    assert r["transition_costs"] is None
    assert r["counts"]["vla_clean"] == 1 and r["episode_cost"] == pytest.approx(1.302)


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing_return", "binding", "gap", "double_forecast"]
)
def test_cost_rejects_ambiguous_or_incompatible_records(fault):
    from latency_meta_mdp.meta.cost import episode_cost_components

    d = episode()
    if fault == "duplicate":
        d["stage_events"].append(copy.deepcopy(d["stage_events"][-2]))
    elif fault == "missing_return":
        d["stage_events"].pop()
    elif fault == "binding":
        d["identity"]["protocol_id"] = "different"
    elif fault == "gap":
        d["decision_transitions"][0]["end_tick"] = 13
    else:
        d["stage_events"].append(
            {"stage": "forecast", "formal_tick": 14, "request_id": 0, "available": True}
        )
    with pytest.raises(ValueError):
        episode_cost_components(d, profile())


def test_profile_validation_and_budget_update(tmp_path):
    from latency_meta_mdp.meta.cost import budget_multiplier_step, load_cost_profile

    p = tmp_path / "cost.json"
    p.write_text(json.dumps(profile()))
    assert load_cost_profile(p)["weights"]["forecast"] == 0.2
    bad = profile()
    bad["weights"]["forecast"] = float("nan")
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_cost_profile(p)
    assert budget_multiplier_step(0.1, 12.0, 10.0, 0.1) == pytest.approx(0.12)
    assert budget_multiplier_step(0.01, 0.0, 10.0, 0.1) == 0
    with pytest.raises(ValueError):
        budget_multiplier_step(0.1, 1.0, 0.0, 0.1)
