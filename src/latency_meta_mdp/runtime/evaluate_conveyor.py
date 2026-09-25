"""Evaluate a full clean conveyor policy with controlled nominal/family latency."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.runtime.latency_harness import FixedDelaySampler
from latency_meta_mdp.runtime.latency_law import load_latency_law
from latency_meta_mdp.runtime.latency_law_family import load_episode_latency_law_family


def conveyor_delay_sampler(root, *, regime, scene_seed, policy_seed):
    """Seeded request-index stream; law metadata is for reporting, never the actor."""
    if regime == "zero":
        return FixedDelaySampler(0), {"regime": "zero"}
    if regime == "nominal":
        law = load_latency_law(root / "configs/runtime/latency/truncated_beta_8_65_400ms_v1.yaml")
    elif regime == "family":
        family = load_episode_latency_law_family(
            root / "configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml"
        )
        law = family.sample_for_key(assignment_key=4 * scene_seed + policy_seed)
    else:
        raise ValueError("unsupported conveyor delay regime")
    probabilities = np.asarray(law.probabilities)
    rng = np.random.default_rng(np.random.SeedSequence([scene_seed, policy_seed, 20260907]))
    return (
        lambda: int(rng.choice(np.arange(1, 21), p=probabilities)),
        {
            "regime": regime,
            "probabilities": probabilities.tolist(),
            "seed_components": [scene_seed, policy_seed, 20260907],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--normalization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("rtc", "sharp"), default="rtc")
    parser.add_argument("--regime", choices=("zero", "nominal", "family"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2000, 2020)))
    parser.add_argument("--policy-seed", type=int, default=0)
    args = parser.parse_args()
    root = repository_root()
    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime
    from latency_meta_mdp.io.policy_video import DualCameraVideoWriter
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
    from latency_meta_mdp.policy.profile import load_sft_profile
    from latency_meta_mdp.runtime.action_chunk_client import load_action_chunk_client_config
    from latency_meta_mdp.runtime.conveyor_evaluation import run_conveyor_policy_episode
    from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config

    checkpoint = args.checkpoint.resolve()
    norms = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(norms) != 1 or sha256_file(norms[0]) != args.normalization_sha256:
        raise ValueError("checkpoint normalization differs from the declared training data")
    spec = load_conveyor_spec(root / "configs/tasks/conveyor_sort/surface.yaml")
    profile = load_sft_profile(root / "configs/contracts/policy/pi05_state16_h50.yaml")
    rtc = args.protocol == "rtc"
    client = (
        load_rtc_client_config(root / "configs/runtime/client/rtc_observation_time_h50_v1.yaml")
        if rtc
        else load_action_chunk_client_config(
            root / "configs/runtime/client/sharp_return_time_h50_e25_v1.yaml"
        )
    )
    patches = tuple(sorted((root / "patches/openpi").glob("000[1-7]-*.patch")))
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_revision": args.checkpoint_revision,
        "normalization_sha256": args.normalization_sha256,
        "finetune_mode": "full",
        "scope": "development_validation_not_independent_test",
        "protocol": args.protocol,
        "regime": args.regime,
        "policy_seed": args.policy_seed,
        "seeds": args.seeds,
        "request_cursor": 25,
        "scheduler_clock": "observation_origin_plan_age"
        if rtc
        else "actions_consumed_since_arrival",
        "task": dataclasses.asdict(spec),
        "rtc_guidance": 5.0 if rtc else None,
        "initial_delay_ticks": list(client.initial_delay_ticks) if rtc else None,
        "patches": {p.name: sha256_file(p) for p in patches},
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != json.loads(
        json.dumps(identity)
    ):
        raise ValueError("existing evaluation identity differs")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        import jax
        from openpi.policies.policy_config import create_trained_policy

        from latency_meta_mdp.policy.openpi.training import build_task_train_config
        from latency_meta_mdp.runtime.policy_execution import (
            InProcessOpenpiPolicy,
            InProcessRtcOpenpiPolicy,
        )

        config = build_task_train_config(
            profile,
            config_name="conveyor_clean_eval",
            repo_id="metamdp/conveyor_sort_surface",
            task_id="conveyor_sort",
            action_contract_id=spec.action_contract_path.rsplit("/", 1)[-1].removesuffix(".yaml"),
        )
        print("Restoring full clean policy", flush=True)
        policy = create_trained_policy(
            config, checkpoint, sample_kwargs={"rtc_max_guidance_weight": 5.0} if rtc else None
        )
        results = []
        for seed in args.seeds:
            result_path = output / f"seed-{seed}.json"
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                continue
            sampler, delay_identity = conveyor_delay_sampler(
                root, regime=args.regime, scene_seed=seed, policy_seed=args.policy_seed
            )
            policy._rng = jax.random.key(0)
            actor_type = InProcessRtcOpenpiPolicy if rtc else InProcessOpenpiPolicy
            actor = actor_type(policy, noise_rng=np.random.default_rng(args.policy_seed))
            runtime = make_conveyor_runtime(spec, seed=seed)
            with DualCameraVideoWriter(output / f"seed-{seed}.mp4") as video:

                def record(observation):
                    video.write(observation)
                    if observation.formal_tick % 500 == 0:
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "tick": observation.formal_tick,
                                    "parcels": runtime.world.ledger.summary(),
                                }
                            ),
                            flush=True,
                        )

                result = run_conveyor_policy_episode(
                    runtime=runtime,
                    policy=actor,
                    client_config=client,
                    delay_sampler=sampler,
                    maximum_steps=spec.supply_ticks + spec.drain_ticks,
                    record_observation=record,
                    policy_alignment="observation_time" if rtc else None,
                )
            result.update(
                seed=seed, delay_identity=delay_identity, goal_events=runtime.world.goal.events
            )
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            progress = {
                "completed": len(results),
                "planned": len(args.seeds),
                "successes": sum(r["parcels"]["successes"] for r in results),
                "spawned": sum(r["parcels"]["spawned"] for r in results),
            }
            (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
        (output / "completed.json").write_text(
            json.dumps(
                {
                    "identity": identity,
                    "episodes": [
                        {k: r[k] for k in ("seed", "parcels", "policy_calls", "starvation_ticks")}
                        for r in results
                    ],
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
