"""Run explicitly selected information-scheduler checks in the existing environment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "smoke",
            "diagnose",
            "bootstrap",
            "collect",
            "train",
            "cycle-bootstrap",
            "cycle-collect",
            "online-train",
            "monitor-series",
            "parallel-collect",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-steps", type=int, default=80)
    parser.add_argument("--regime", choices=["nominal", "family"], default="nominal")
    parser.add_argument(
        "--behavior", choices=["periodic", "active", "lazy", "exploration"], default="periodic"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scene-audit", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cohort-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-iterations", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phase", choices=["smoke", "validation", "online", "monitor"])
    parser.add_argument("--offline-dir", type=Path)
    parser.add_argument("--compatibility", type=Path)
    parser.add_argument("--meta-only", action="store_true")
    parser.add_argument("--bootstrap-dir", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gpus", help="Physical GPU indices for independent rollout workers")
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--allow-shared-pids", default="")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--train-gpus", help="Physical GPUs for synchronous FQI updates")
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--replay-cache-gib", type=int, default=128)
    parser.add_argument("--replay-cache-dir", type=Path)
    parser.add_argument("--replay-prepare-workers", type=int, default=4)
    parser.add_argument(
        "--cpu-threads", type=int, default=4, help="CPU threads per rollout/learner worker"
    )
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.allow_shared_pids:
        import os

        os.environ["CONVEYOR_SHARED_GPU_PIDS"] = args.allow_shared_pids
    root = Path(__file__).resolve().parents[1]
    if args.command == "parallel-collect":
        if any(
            v is None
            for v in (args.gpus, args.checkpoint, args.phase, args.scene_audit, args.bootstrap_dir)
        ):
            parser.error(
                "parallel-collect requires gpus, checkpoint, phase, scene-audit, bootstrap-dir"
            )
        from latency_meta_mdp.meta.information.parallel import run_parallel

        print(
            json.dumps(
                run_parallel(
                    args.config,
                    args.output_dir,
                    root=root,
                    audit_path=args.scene_audit,
                    checkpoint=args.checkpoint,
                    phase=args.phase,
                    bootstrap_dir=args.bootstrap_dir,
                    gpus=args.gpus,
                    worker_python=args.worker_python,
                    cpu_threads=args.cpu_threads,
                    baselines=not args.meta_only,
                    source_manifest=args.offline_dir / "cohort.json" if args.offline_dir else None,
                    enabled=not args.no_wandb,
                ),
                indent=2,
            )
        )
    elif args.command == "monitor-series":
        if any(v is None for v in (args.checkpoint, args.bootstrap_dir, args.scene_audit)):
            parser.error("monitor-series requires checkpoint directory, bootstrap-dir, scene-audit")
        from latency_meta_mdp.meta.information.monitor import run_monitor_series

        run_monitor_series(
            args.config,
            args.checkpoint,
            args.output_dir,
            root=root,
            audit_path=args.scene_audit,
            bootstrap_dir=args.bootstrap_dir,
            source_manifest=args.offline_dir / "cohort.json" if args.offline_dir else None,
            worker_python=args.worker_python,
            gpus=args.gpus,
            cpu_threads=args.cpu_threads,
        )
    elif args.command in ("cycle-bootstrap", "cycle-collect"):
        if args.checkpoint is None or args.phase is None or args.scene_audit is None:
            parser.error("cycle commands require checkpoint, phase, and scene-audit")
        from latency_meta_mdp.meta.information.cohort import collect_cohort, prepare_bootstraps
        from latency_meta_mdp.meta.information.cycle import prepare_cycle

        cycle = dict(
            checkpoint=args.checkpoint,
            phase=args.phase,
            baselines=not args.meta_only,
            source_manifest=args.offline_dir / "cohort.json" if args.offline_dir else None,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        if args.command == "cycle-bootstrap":
            prepare_bootstraps(
                args.config, args.output_dir, root=root, audit_path=args.scene_audit, cycle=cycle
            )
        else:
            from latency_meta_mdp.meta.cycle_tracking import CycleTracker

            prepare_cycle(
                args.config, args.output_dir, root=root, audit_path=args.scene_audit, **cycle
            )
            tracker = CycleTracker(
                args.output_dir, name=args.output_dir.name, enabled=not args.no_wandb
            )
            success = False
            try:
                result = collect_cohort(
                    args.config,
                    args.output_dir,
                    root=root,
                    audit_path=args.scene_audit,
                    cycle=cycle,
                    limit=args.limit,
                    on_episode=tracker,
                )
                print(json.dumps(result, indent=2))
                success = True
            finally:
                tracker.close(success=success)
    elif args.command == "online-train":
        if any(
            v is None
            for v in (
                args.offline_dir,
                args.cohort_dir,
                args.checkpoint,
                args.compatibility,
                args.scene_audit,
            )
        ):
            parser.error(
                "online-train requires offline-dir, cohort-dir, checkpoint, "
                "compatibility, scene-audit"
            )
        from latency_meta_mdp.meta.information.online_training import run_online_training

        print(
            json.dumps(
                run_online_training(
                    args.config,
                    args.offline_dir,
                    args.cohort_dir,
                    args.output_dir,
                    root=root,
                    checkpoint=args.checkpoint,
                    compatibility_path=args.compatibility,
                    audit_path=args.scene_audit,
                    device=args.device,
                    seed=0 if args.seed is None else args.seed,
                    validation_bootstrap=args.bootstrap_dir,
                    worker_python=args.worker_python,
                    validation_gpus=args.gpus,
                    training_gpus=args.train_gpus,
                    global_batch=args.global_batch,
                    replay_cache_gib=args.replay_cache_gib,
                    replay_cache_dir=args.replay_cache_dir,
                    replay_prepare_workers=args.replay_prepare_workers,
                    cpu_threads=args.cpu_threads,
                    stop_after_iterations=args.stop_after_iterations,
                    warm_start_checkpoint=args.warm_start_checkpoint,
                    total_iterations=args.iterations,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        if args.cohort_dir is None:
            parser.error("train requires --cohort-dir")
        from latency_meta_mdp.meta.information.training import run_training

        print(
            json.dumps(
                run_training(
                    args.config,
                    args.cohort_dir,
                    args.output_dir,
                    root=root,
                    device=args.device,
                    seed=0 if args.seed is None else args.seed,
                    stop_after_iterations=args.stop_after_iterations,
                ),
                indent=2,
            )
        )
    elif args.command == "preflight":
        from latency_meta_mdp.meta.information.preflight import preflight

        print(json.dumps(preflight(args.config, args.output_dir, root=root), indent=2))
    elif args.command in ("bootstrap", "collect"):
        if args.scene_audit is None:
            parser.error("bootstrap/collect requires --scene-audit")
        from latency_meta_mdp.meta.information.cohort import collect_cohort, prepare_bootstraps

        if args.command == "bootstrap":
            prepare_bootstraps(args.config, args.output_dir, root=root, audit_path=args.scene_audit)
        else:
            print(
                json.dumps(
                    collect_cohort(
                        args.config,
                        args.output_dir,
                        root=root,
                        audit_path=args.scene_audit,
                        limit=args.limit,
                    ),
                    indent=2,
                )
            )
    else:
        from latency_meta_mdp.meta.information.collection import run_smoke

        run_smoke(
            args.config,
            args.output_dir,
            root=root,
            maximum_steps=args.maximum_steps,
            diagnose=args.command == "diagnose",
            regime=args.regime,
            behavior=args.behavior,
            seed=2000 if args.seed is None else args.seed,
        )


if __name__ == "__main__":
    main()
