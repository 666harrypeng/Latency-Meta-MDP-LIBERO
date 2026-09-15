"""Fixed cost weights from isolated, warm model-service measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

from latency_meta_mdp.meta.cost import COMPONENTS, episode_cost_components, validate_cost_profile

SHARED_KEYS = (
    "protocol_id",
    "profile_sha256",
    "rtc_max_guidance_weight",
    "client_config_sha256",
    "decision_interval_ticks",
)
POLICY_KEYS = (
    "checkpoint_step",
    "checkpoint_verification_sha256",
    "conditioning",
    "forecast_identity",
    "runtime_patch_sha256",
)


def build_cost_profile(process_groups, *, hardware):
    """Each group contains ordered episode results from ONE isolated worker process."""
    samples = {key: [] for key in COMPONENTS}
    binding, policies = None, {}
    for rows in process_groups:
        lane = {key: [] for key in COMPONENTS}
        for row in rows:
            identity = row["identity"]
            shared = {k: identity[k] for k in SHARED_KEYS}
            if binding is not None and binding != shared:
                raise ValueError("calibration changes runtime binding")
            binding = shared
            kind = "conditioned" if identity.get("conditioning") is not None else "clean"
            policy = {k: identity.get(k) for k in POLICY_KEYS}
            if kind in policies and policies[kind] != policy:
                raise ValueError("calibration changes policy binding")
            policies[kind] = policy
            events = row["stage_events"]
            launches = {e["request_id"]: e for e in events if e["stage"] == "policy_launch"}
            for event in events:
                stage = event["stage"]
                if stage == "policy_return":
                    start = launches[event["request_id"]]["wall_start_ns"]
                    lane["vla_" + kind].append((event["wall_start_ns"] - start) / 1e9)
                elif stage in ("forecast_prepare", "forecast_decode") and event["available"]:
                    key = "forecast" if stage == "forecast_prepare" else "decode"
                    lane[key].append(event["wall_duration_ns"] / 1e9)
                elif stage == "meta_decision":
                    key = "meta_q" if event.get("q_values") is not None else "rule"
                    lane[key].append(event["wall_duration_ns"] / 1e9)
            starts = [e for e in events if e["stage"] == "bootstrap_launch"]
            ends = [e for e in events if e["stage"] == "bootstrap_return"]
            if len(starts) != 1 or len(ends) != 1:
                raise ValueError("calibration needs paired bootstrap timestamps")
            lane["bootstrap_clean"].append(
                (ends[0]["wall_start_ns"] - starts[0]["wall_start_ns"]) / 1e9
            )
        for key, values in lane.items():
            samples[key].extend(values[1:])
    if any(len(v) < 3 or not np.isfinite(v).all() or min(v) <= 0 for v in samples.values()):
        raise ValueError("need at least three positive warm samples per cost component")
    medians = {key: float(np.median(v)) for key, v in samples.items()}
    profile = {
        "schema": 1,
        "unit": "conditioned_vla_rtc_equivalent",
        "reference_seconds": medians["vla_conditioned"],
        "weights": {key: v / medians["vla_conditioned"] for key, v in medians.items()},
        "binding": binding,
        "policy_bindings": policies,
        "calibration": {
            "hardware": hardware,
            "workers": 1,
            "cold_policy": "exclude first invocation of each measured stage per process",
            "samples": {
                key: {
                    "n": len(v),
                    "median_seconds": medians[key],
                    "p95_seconds": float(np.quantile(v, 0.95)),
                }
                for key, v in samples.items()
            },
            "scope": "reference model-service cost, not logical delay or concurrent wall time",
        },
    }
    validate_cost_profile(profile)
    for rows in process_groups:
        for row in rows:
            episode_cost_components(row, profile)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-results",
        type=Path,
        action="append",
        required=True,
        help="One result directory per isolated worker process; repeat as needed",
    )
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups = []
    for directory in args.process_results:
        rows = [json.loads(p.read_text()) for p in directory.glob("master*.json")]
        rows.sort(
            key=lambda r: next(
                e["wall_start_ns"] for e in r["stage_events"] if e["stage"] == "bootstrap_launch"
            )
        )
        groups.append(rows)
    profile = build_cost_profile(groups, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(profile, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
