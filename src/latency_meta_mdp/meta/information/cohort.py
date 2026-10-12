"""Explicit scene-grouped starter cohort and immutable completion receipts."""

import json
from pathlib import Path

from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding

BEHAVIORS = ("lazy", "active", "periodic", "exploration")


def build_manifest(config, audit, *, root, code_hash):
    if (
        not audit["local_inventory_pass"]
        or audit["candidate_splits"] != config["splits"]
        or audit["excluded_scenes"] != config["excluded_scenes"]
    ):
        raise ValueError("scene audit does not match the collection configuration")
    for source in audit["sources"]:
        if sha256(Path(root) / source["path"]) != source["sha256"]:
            raise ValueError("scene audit source changed")
    if config["mode"] != "bound" or len(config["splits"]["train"]) < 16:
        raise ValueError("starter cohort requires 16 bound-mode training scenes")
    episodes = []
    for scene in config["splits"]["train"][:16]:
        for delay_index, regime in enumerate(("nominal", "family")):
            for behavior_index, behavior in enumerate(BEHAVIORS):
                episodes.append(
                    dict(
                        episode_id=f"{scene}-{regime}-{behavior}",
                        scene=scene,
                        regime=regime,
                        behavior=behavior,
                        policy_seed=0,
                        scheduler_seed=scene * 100 + delay_index * 10 + behavior_index,
                        split="train",
                    )
                )
    binding = dict(
        config_hash=identity_hash(config),
        assets=config["assets"],
        mode=config["mode"],
        code_hash=code_hash,
        scene_audit_hash=identity_hash(audit),
        schema_version=1,
        task_id="conveyor_sort",
        scope="starter_training",
    )
    return dict(binding=binding, episodes=episodes)


def next_attempt(root, episode_id):
    parent = Path(root) / episode_id
    parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while (parent / f"attempt-{index:03d}").exists():
        index += 1
    attempt = parent / f"attempt-{index:03d}"
    attempt.mkdir()
    return attempt


def verify_complete(path, *, binding):
    path = Path(path)
    if not (path / "complete.json").exists():
        return False
    receipt = json.loads((path / "complete.json").read_text())
    validate_binding(binding, receipt["binding"])
    for relative, expected in receipt["files"].items():
        file = path / relative
        if not file.resolve().is_relative_to(path.resolve()):
            raise ValueError("completion file escapes episode")
        if not file.is_file() or sha256(file) != expected:
            raise ValueError("completed episode hash mismatch")
    return True


def complete_attempt(path, *, binding):
    path = Path(path)
    files = {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and str(p.relative_to(path)) not in ("complete.json", "complete.tmp")
    }
    if not files:
        raise ValueError("empty collection attempt")
    receipt = dict(binding=binding, files=files)
    temporary = path / "complete.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path / "complete.json")


def prepare_cohort(config_path, output_dir, *, root, audit_path):
    from latency_meta_mdp.meta.information.config import load_config
    from latency_meta_mdp.meta.information.preflight import preflight

    config = load_config(config_path)
    report = preflight(config_path, Path(output_dir) / "preflight", root=root)
    if report["compute_processes"]:
        raise RuntimeError("GPU compute process already active")
    audit = json.loads(Path(audit_path).read_text())
    manifest = build_manifest(config, audit, root=root, code_hash=report["code_hash"])
    path = Path(output_dir) / "cohort.json"
    if path.exists():
        validate_binding(json.loads(path.read_text()), manifest)
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return config, manifest


def prepare_bootstraps(config_path, output_dir, *, root, audit_path, cycle=None):
    from latency_meta_mdp.meta.information.collection import bootstrap_binding
    from latency_meta_mdp.meta.information.services import frozen_services

    if cycle is None:
        config, manifest = prepare_cohort(config_path, output_dir, root=root, audit_path=audit_path)
    else:
        from latency_meta_mdp.meta.information.cycle import prepare_cycle

        config, manifest = prepare_cycle(
            config_path, output_dir, root=root, audit_path=audit_path, **cycle
        )
    # The process exits before conditioned collection starts; no two VLAs coexist.
    with frozen_services(root, config, bootstrap_only=True) as services:
        import jax
        import numpy as np

        from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
        from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime
        from latency_meta_mdp.runtime.bootstrap_cache import CachedBootstrapPolicy
        from latency_meta_mdp.runtime.policy_execution import InProcessOpenpiPolicy

        spec = load_conveyor_spec(Path(root) / config["task_config"])
        for seed in sorted({e["scene"] for e in manifest["episodes"]}):
            runtime = make_conveyor_runtime(spec, seed=seed)
            try:
                obs = runtime.executor.initialize()
                services.policy._rng = jax.random.key(0)
                rng = np.random.default_rng(0)
                actor = InProcessOpenpiPolicy(services.policy, noise_rng=rng)
                cache = CachedBootstrapPolicy(
                    Path(output_dir) / f"bootstrap/seed-{seed}.npz",
                    identity=bootstrap_binding(root, config),
                    noise_rng=rng,
                    native_policy=actor,
                )
                cache(obs)
                print(f"BOOTSTRAP_COMPLETE {seed}", flush=True)
            finally:
                runtime.env.close()


def collect_cohort(
    config_path, output_dir, *, root, audit_path, limit=None, cycle=None, on_episode=None
):
    import shutil
    from time import perf_counter

    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.meta.information.collection import run_episode
    from latency_meta_mdp.meta.information.services import frozen_services

    if cycle is None:
        config, manifest = prepare_cohort(config_path, output_dir, root=root, audit_path=audit_path)
    else:
        from latency_meta_mdp.meta.information.cycle import prepare_cycle

        config, manifest = prepare_cycle(
            config_path, output_dir, root=root, audit_path=audit_path, **cycle
        )
    root, output_dir = Path(root), Path(output_dir)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("collection limit must be positive")
    spec = load_conveyor_spec(root / config["task_config"])
    episodes_dir = output_dir / "episodes"
    completed = []
    pending = []
    for episode in manifest["episodes"]:
        binding = dict(cohort=manifest["binding"], episode=episode)
        found = []
        for attempt in sorted((episodes_dir / episode["episode_id"]).glob("attempt-*")):
            if verify_complete(attempt, binding=binding):
                found.append(attempt)
        if len(found) > 1:
            raise ValueError("duplicate completed scene/regime/behavior")
        if found:
            completed.append(str(found[0].relative_to(output_dir)))
        else:
            pending.append(episode)
    if not pending:
        return dict(completed=len(completed), remaining=0)
    if limit is not None:
        pending = pending[:limit]
    with frozen_services(root, config) as services:
        for episode in pending:
            if shutil.disk_usage(output_dir).free < 40 * 1024**3:
                raise RuntimeError("less than 40GiB free: stop new episode admissions")
            attempt = next_attempt(episodes_dir, episode["episode_id"])
            started = perf_counter()
            scheduler, learned = None, None
            if episode["behavior"] == "fqi":
                from latency_meta_mdp.meta.information.cycle import policy_binding
                from latency_meta_mdp.meta.information.deployment import (
                    deployment_binding,
                    load_scheduler,
                )

                learned = policy_binding(manifest["binding"], episode)
                scheduler = load_scheduler(
                    learned["checkpoint"],
                    deployment_binding(config),
                    device="cuda:0",
                    epsilon=episode["epsilon"],
                    seed=episode["scheduler_seed"],
                    source_manifest=learned.get("source_manifest"),
                )
            try:
                result = run_episode(
                    root,
                    config,
                    attempt,
                    services,
                    code_hash=manifest["binding"]["code_hash"],
                    maximum_steps=spec.supply_ticks + spec.drain_ticks,
                    regime=episode["regime"],
                    behavior=episode["behavior"],
                    seed=episode["scene"],
                    scheduler_seed=episode["scheduler_seed"],
                    bootstrap_dir=output_dir / "bootstrap",
                    scope=manifest["binding"].get("scope", "starter_training"),
                    scheduler=scheduler,
                    learned_binding=learned,
                )
                if not result["terminated"] or result["truncated"] or result["starvation_ticks"]:
                    raise ValueError(
                        "episode failed collection contract; preserve attempt for audit"
                    )
                complete_attempt(attempt, binding=dict(cohort=manifest["binding"], episode=episode))
                completed.append(str(attempt.relative_to(output_dir)))
                progress = dict(
                    completed=len(completed),
                    remaining=len(manifest["episodes"]) - len(completed),
                    complete_attempts=completed,
                    last_episode=episode,
                    last_seconds=perf_counter() - started,
                    last_bytes=sum(p.stat().st_size for p in attempt.rglob("*") if p.is_file()),
                )
                temporary = output_dir / "progress.tmp"
                temporary.write_text(json.dumps(progress, indent=2) + "\n")
                temporary.replace(output_dir / "progress.json")
                print("EPISODE_COMPLETE " + json.dumps(progress), flush=True)
                if on_episode is not None:
                    on_episode(episode, result, attempt, progress)
            except BaseException as error:
                (attempt / "failure.json").write_text(
                    json.dumps(
                        dict(error=repr(error), scope="interrupted attempt, excluded from training")
                    )
                    + "\n"
                )
                raise
    return dict(completed=len(completed), remaining=len(manifest["episodes"]) - len(completed))
