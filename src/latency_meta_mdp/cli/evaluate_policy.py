"""Run a deterministic shard of clean-policy closed-loop development episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from latency_meta_mdp.action_chunk_client import load_action_chunk_client_config
from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
from latency_meta_mdp.expert_realization.task_instance import (
    _build_task_instance_runtime,
    materialize_task_instance,
)
from latency_meta_mdp.latency_harness import FixedDelaySampler
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.latency_law_family import load_episode_latency_law_family
from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy
from latency_meta_mdp.policy_evaluation import run_native_policy_episode
from latency_meta_mdp.sft_launch import SFTLaunchRequest
from latency_meta_mdp.sft_profile import load_sft_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-verification", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--maximum-steps", type=int, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--regime",
        choices=(
            "zero",
            "fixed80",
            "fixed160",
            "fixed240",
            "fixed320",
            "fixed400",
            "nominal",
            "family",
        ),
        default="zero",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must belong to the positive worker count")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("max-cases must be positive")
    root = Path.cwd()
    cohort = json.loads(args.cohort.read_text())
    verified = json.loads(args.checkpoint_verification.read_text())
    if (
        cohort["partition"] != "train_pool_development"
        or not verified["all_downloaded_hashes_match"]
    ):
        raise ValueError("evaluation requires a development cohort and verified checkpoint")
    level = cohort["level"]
    if args.checkpoint.resolve() != Path(verified["checkpoint_root"]).resolve():
        raise ValueError("checkpoint path does not match the verified download")
    preparation = json.loads((args.preparation_root / "preparation.json").read_text())
    profile_path = root / "configs/policy/pi05_structured_state16_h50_v1.yaml"
    profile = load_sft_profile(profile_path)
    if preparation["profile_sha256"] != sha256_file(profile_path) or preparation["level"] != level:
        raise ValueError("policy preparation identity differs from the evaluation profile")
    patches = tuple(root / "patches/openpi" / name for name in preparation["patches"])
    if any(sha256_file(path) != preparation["patches"][path.name] for path in patches):
        raise ValueError("policy patches differ from the verified preparation")
    identity = {
        "cohort_sha256": sha256_file(args.cohort),
        "checkpoint_verification_sha256": sha256_file(args.checkpoint_verification),
        "checkpoint_repo": verified["repo_id"],
        "checkpoint_revision": verified["repo_sha"],
        "checkpoint_step": verified["checkpoint_step"],
        "profile_sha256": sha256_file(profile_path),
        "regime": args.regime,
        "maximum_steps": args.maximum_steps,
    }
    cases = cohort["cases"][args.worker_index :: args.worker_count]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    args.output_root.mkdir(parents=True, exist_ok=True)
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        import jax
        from openpi.policies.policy_config import create_trained_policy

        from latency_meta_mdp.openpi_sft import build_level_train_config
        from latency_meta_mdp.policy_execution import InProcessOpenpiPolicy

        config = build_level_train_config(
            profile=profile,
            request=SFTLaunchRequest(
                level=level,
                experiment_name="evaluation",
                mode="formal",
                resume=False,
                device_count=1,
                batch_size_override=128,
            ),
            assets_root=args.preparation_root / "assets",
            checkpoint_root=args.checkpoint.parent,
            wandb_enabled=False,
        )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "identity": identity,
                        "case_count": len(cases),
                        "state_tokens": config.model.discrete_state_input,
                        "action_horizon": config.model.action_horizon,
                    }
                )
            )
            return
        policy = create_trained_policy(config, args.checkpoint)
        client_config = load_action_chunk_client_config(
            root / "configs/client/sharp_return_time_h50_e25_v1.yaml"
        )
        for case in cases:
            target = (
                args.output_root
                / f"master-{case['master_index']:03d}-seed-{case['policy_seed']}-{args.regime}.json"
            )
            if target.exists():
                previous = json.loads(target.read_text())
                if previous["identity"] != identity or previous["case"] != case:
                    raise ValueError("existing episode result has a different identity")
                continue
            task_id = TaskInstanceId.from_mapping(case["task_instance_id"])
            if task_id.level != level:
                raise ValueError("case level differs from the selected policy")
            task = materialize_task_instance(
                project_root=root, level=level, task_instance_seed=task_id.task_instance_seed
            )
            if task.task_instance_id != task_id:
                raise ValueError("materialized task differs from source cohort identity")
            rng = np.random.default_rng(
                np.random.SeedSequence([case["master_index"], case["policy_seed"], 20260907])
            )
            if args.regime == "zero":
                probabilities = None
                sampler = FixedDelaySampler(0)
            elif args.regime.startswith("fixed"):
                delay = int(args.regime.removeprefix("fixed")) // 20
                probabilities = None
                sampler = FixedDelaySampler(delay)
            else:
                if args.regime == "nominal":
                    probabilities = load_latency_law(
                        root / "configs/latency/truncated_beta_8_65_400ms_v1.yaml"
                    ).probabilities
                else:
                    family = load_episode_latency_law_family(
                        root / "configs/latency/truncated_beta_family_8_65_400ms_v1.yaml"
                    )
                    probabilities = family.sample_for_key(
                        assignment_key=4 * case["master_index"] + case["policy_seed"]
                    ).probabilities

                def sampler():
                    return int(rng.choice(np.arange(1, 21), p=probabilities))

            actor = InProcessOpenpiPolicy(
                policy, noise_rng=np.random.default_rng(case["policy_seed"])
            )
            print(
                f"START master={case['master_index']} seed={case['policy_seed']} "
                f"regime={args.regime}",
                flush=True,
            )
            result = run_native_policy_episode(
                runtime=_build_task_instance_runtime(task),
                policy=actor,
                client_config=client_config,
                delay_sampler=sampler,
                maximum_steps=args.maximum_steps,
            )
            result.update(
                identity=identity,
                case=case,
                latency_probabilities=None if probabilities is None else probabilities.tolist(),
                jax_memory_stats=jax.devices()[0].memory_stats(),
            )
            with target.open("x") as file:
                json.dump(result, file, indent=2, allow_nan=False)
                file.write("\n")
            print(
                f"DONE master={case['master_index']} seed={case['policy_seed']} "
                f"success={result['success']} steps={result['executed_steps']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
