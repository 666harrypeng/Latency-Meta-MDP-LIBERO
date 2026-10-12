"""Local real-model integration smoke for the frozen conveyor scheduler stack."""

import json
from pathlib import Path


def resolve_scheduler(behavior, seed, scheduler):
    from latency_meta_mdp.meta.information.behaviors import RuleScheduler
    from latency_meta_mdp.meta.information.fqi import FqiScheduler

    if behavior == "fqi":
        if not isinstance(scheduler, FqiScheduler):
            raise ValueError("fqi requires an explicitly restored scheduler")
        return scheduler
    if scheduler is not None:
        raise ValueError("learned scheduler cannot be labeled as a rule")
    return RuleScheduler(behavior, period=25, seed=seed)


def run_smoke(
    config_path,
    output_dir,
    *,
    root,
    maximum_steps=80,
    diagnose=False,
    regime="nominal",
    behavior="periodic",
    seed=2000,
):
    # Import model stacks only after the CPU preflight; never replace environments.
    from latency_meta_mdp.meta.information.config import load_config
    from latency_meta_mdp.meta.information.preflight import preflight

    root, output_dir = Path(root), Path(output_dir)
    report = preflight(config_path, output_dir / "preflight", root=root)
    if report["compute_processes"]:
        raise RuntimeError("GPU already has a compute process; do not overlap model jobs")
    config = load_config(config_path)
    if seed not in config["excluded_scenes"]:
        raise ValueError("smoke requires a declared historical scene")
    from latency_meta_mdp.meta.information.services import frozen_services

    with frozen_services(root, config) as services:
        return run_episode(
            root,
            config,
            output_dir,
            services,
            code_hash=report["code_hash"],
            maximum_steps=maximum_steps,
            diagnose=diagnose,
            regime=regime,
            behavior=behavior,
            seed=seed,
            scheduler_seed=0,
            bootstrap_dir=root / "outputs/analysis/conveyor-sort/conditioned-eval-v1/bootstrap",
            scope="historical-scene integration smoke; never admitted to training",
        )


def run_episode(
    root,
    config,
    output_dir,
    services,
    *,
    code_hash,
    maximum_steps,
    diagnose=False,
    regime,
    behavior,
    seed,
    scheduler_seed,
    bootstrap_dir,
    scope,
    scheduler=None,
    learned_binding=None,
):
    import jax
    import numpy as np

    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime
    from latency_meta_mdp.io.policy_video import DualCameraVideoWriter
    from latency_meta_mdp.meta.information.dataset import save_episode
    from latency_meta_mdp.meta.information.perception import PerceptionAccess
    from latency_meta_mdp.meta.information.records import StageRecorder
    from latency_meta_mdp.runtime.bootstrap_cache import CachedBootstrapPolicy
    from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler
    from latency_meta_mdp.runtime.information_execution import run_information_episode
    from latency_meta_mdp.runtime.policy_execution import InProcessRtcOpenpiPolicy
    from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config

    scheduler = resolve_scheduler(behavior, scheduler_seed, scheduler)
    if behavior == "fqi":
        from latency_meta_mdp.meta.information.deployment import scheduler_identity

        expected = dict(scheduler_identity(scheduler), epsilon=scheduler.epsilon)
        if learned_binding != expected:
            raise ValueError("learned policy binding differs")
    elif learned_binding is not None:
        raise ValueError("rule cannot carry learned policy binding")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy, engine, encoder, norm = (
        services.policy,
        services.engine,
        services.encoder,
        services.normalization,
    )
    rng = np.random.default_rng(0)
    policy._rng = jax.random.key(0)
    actor = InProcessRtcOpenpiPolicy(policy, noise_rng=rng)
    bootstrap_identity = bootstrap_binding(root, config)
    bootstrap = CachedBootstrapPolicy(
        bootstrap_dir / f"seed-{seed}.npz",
        identity=bootstrap_identity,
        noise_rng=rng,
    )
    recorder = StageRecorder(f"smoke-{seed}")
    from latency_meta_mdp.meta.information.archive import SourceArchive

    perception = PerceptionAccess(
        engine, encoder, norm, recorder, source_archive=SourceArchive(output_dir / "sources")
    )
    if diagnose:
        from latency_meta_mdp.meta.information.diagnostics import SnapshotProbe
        from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext

        def infer_probe(context, forecast):
            # Isolate both policy PRNG and explicit action noise from the live rollout.
            saved_key = policy._rng
            try:
                policy._rng = jax.random.key(917)
                probe_actor = InProcessRtcOpenpiPolicy(policy, noise_rng=np.random.default_rng(917))
                probe_context = RtcInferenceContext(
                    0,
                    context.origin_tick,
                    context.observation,
                    context.buffer_version,
                    context.previous_actions,
                    context.previous_action_mask,
                    context.estimated_delay_ticks,
                    forecast,
                )
                return probe_actor(context.observation, probe_context)["actions"]
            finally:
                policy._rng = saved_key

        perception = SnapshotProbe(
            perception,
            engine=engine,
            encoder=encoder,
            normalization=norm,
            output_dir=output_dir / "fixed-source",
            infer=infer_probe,
            calibrate=True,
        )
    sampler, delay_identity = conveyor_delay_sampler(
        root, regime=regime, scene_seed=seed, policy_seed=0
    )
    spec = load_conveyor_spec(root / config["task_config"])
    print(f"Starting episode {seed}/{regime}/{behavior}", flush=True)
    from latency_meta_mdp.envs.conveyor.meta_audit import ProgressAudit, audit_episode

    with DualCameraVideoWriter(output_dir / "smoke.mp4") as video:
        runtime = make_conveyor_runtime(spec, seed=seed)
        progress_audit = ProgressAudit(
            output_dir / "privileged-current.jsonl",
            goal_center=spec.goal_center,
            goal_radius=spec.goal_radius_m + spec.scene.ball_radius_m,
            minimum_z=spec.belt_top_z + spec.scene.ball_radius_m + spec.minimum_lift_m,
            grasp_ticks=spec.grasp_ticks,
        )

        def record(obs):
            video.write(obs)
            with recorder.measure("audit"):
                progress_audit.observe(runtime.world, obs.formal_tick)
            if obs.formal_tick % 25 == 0:
                print(f"tick={obs.formal_tick}", flush=True)

        result = run_information_episode(
            runtime=runtime,
            policy=lambda obs, ctx: actor(obs, ctx)["actions"],
            bootstrap_policy=lambda obs: bootstrap(obs)["actions"],
            perception=perception,
            scheduler=scheduler,
            recorder=recorder,
            client_config=load_rtc_client_config(root / config["rtc_config"]),
            delay_sampler=sampler,
            maximum_steps=maximum_steps,
            mode=config["mode"],
            record_observation=record,
        )
    from collections import Counter

    result["component_calls"] = dict(Counter(op["component"] for op in recorder.operations))
    result["forced_stages"] = sum(row["forced"] for row in recorder.rows)
    result["scheduler_diagnostics"] = getattr(scheduler, "diagnostics", [])
    result["progress_events"] = progress_audit.events
    result["delivery_events"] = runtime.world.goal.events
    result["conservation"] = audit_episode(recorder, result)
    if diagnose:
        result["diagnostic_extra_calls"] = True
        result["cost_qualification"] = "Do not use this probe rollout for cost comparisons"
    result["bootstrap_cache"] = bootstrap.last_record
    result["scope"] = scope
    save_episode(
        output_dir / "episode",
        recorder,
        report=result,
        binding=dict(
            assets=config["assets"]["identity"],
            mode=config["mode"],
            scene=seed,
            delay=delay_identity,
            scope=result["scope"],
            code_hash=code_hash,
            scheduler_seed=scheduler_seed,
            behavior=behavior,
            policy_seed=0,
            **({"learned_policy": learned_binding} if learned_binding is not None else {}),
        ),
    )
    summary = {k: v for k, v in result.items() if k not in ("executed_actions", "parcel_events")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def bootstrap_binding(root, config):
    from latency_meta_mdp.meta.information.config import sha256

    identity = config["assets"]["identity"]
    patches = tuple(sorted((root / "patches/openpi").glob("000[1-9]-*.patch")))
    return dict(
        checkpoint_revision=identity["clean_baseline"]["revision"],
        checkpoint_step=identity["clean_baseline"]["step"],
        normalization_sha256=identity["normalization_sha256"],
        policy_seed=0,
        task_spec_sha256=sha256(root / config["task_config"]),
        patches={p.name: sha256(p) for p in patches},
    )
