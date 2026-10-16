"""Explicit scene-grouped starter cohort and immutable completion receipts."""

import json
import shutil
import tempfile
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
        schema_version=config.get("schema_version", 1),
        task_id="conveyor_sort",
        scope="starter_training",
    )
    if config.get("schema_version", 1) == 2:
        binding["actions"] = config["actions"]
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


def publish_attempt(source, destination, *, binding):
    """Copy a finished scratch attempt; publish completion only after durable hashes match."""
    source, destination = Path(source), Path(destination)
    if any(destination.iterdir()):
        raise FileExistsError("attempt publication requires an empty destination")
    receipt = json.loads((source / "complete.json").read_text())
    validate_binding(binding, receipt["binding"])
    for relative, expected in receipt["files"].items():
        origin, target = source / relative, destination / relative
        if not origin.resolve().is_relative_to(
            source.resolve()
        ) or not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("completion file escapes episode")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if sha256(target) != expected:
            raise ValueError("published episode hash mismatch")
    from latency_meta_mdp.meta.information.validation import atomic_json

    atomic_json(destination / "complete.json", receipt)


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
    from latency_meta_mdp.meta.information.collection import (
        bootstrap_binding,
        bootstrap_observation,
    )
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
                    native_on_observation_miss=True,
                )
                cache(bootstrap_observation(obs))
                print(f"BOOTSTRAP_COMPLETE {seed}", flush=True)
            finally:
                runtime.env.close()


def collect_cohort(
    config_path,
    output_dir,
    *,
    root,
    audit_path,
    limit=None,
    cycle=None,
    on_episode=None,
    scratch_dir=None,
):
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
    if scratch_dir is not None:
        scratch_dir = Path(scratch_dir).resolve()
        scratch_dir.mkdir(parents=True, exist_ok=True)
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
            work = attempt
            if scratch_dir is not None:
                if shutil.disk_usage(scratch_dir).free < 40 * 1024**3:
                    raise RuntimeError("less than 40GiB scratch free: stop new episode admissions")
                work = Path(tempfile.mkdtemp(prefix="conveyor-episode-", dir=scratch_dir))
                print(
                    "EPISODE_STAGING " + json.dumps(dict(attempt=str(attempt), scratch=str(work))),
                    flush=True,
                )
            started = perf_counter()
            scheduler, learned = None, None
            if episode["behavior"] in ("fqi", "actor"):
                from latency_meta_mdp.meta.information.cycle import policy_binding
                from latency_meta_mdp.meta.information.deployment import (
                    deployment_binding,
                    load_scheduler,
                )

                learned = policy_binding(manifest["binding"], episode)
                scheduler = load_scheduler(
                    learned["checkpoint"],
                    deployment_binding(
                        config, algorithm=("actor" if episode["behavior"] == "actor" else "fqi")
                    ),
                    device="cuda:0",
                    epsilon=episode["epsilon"],
                    seed=episode["scheduler_seed"],
                    source_manifest=learned.get("source_manifest"),
                    deterministic=episode.get("deterministic", True),
                )
            try:
                result = run_episode(
                    root,
                    config,
                    work,
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
                    episode_id=episode["episode_id"],
                    pattern=episode.get("pattern"),
                    delay_seed=episode.get("delay_seed", 0),
                )
                allow_hold = config.get("actions", {}).get("starvation_mode") == "allow_hold"
                if (
                    not result["terminated"]
                    or result["truncated"]
                    or (result["starvation_ticks"] and not allow_hold)
                ):
                    raise ValueError(
                        "episode failed collection contract; preserve attempt for audit"
                    )
                binding = dict(cohort=manifest["binding"], episode=episode)
                complete_attempt(work, binding=binding)
                if work != attempt:
                    publish_started = perf_counter()
                    publish_attempt(work, attempt, binding=binding)
                    try:
                        shutil.rmtree(work)
                    except OSError as error:
                        print(
                            "EPISODE_SCRATCH_RETAINED "
                            + json.dumps(dict(scratch=str(work), error=repr(error))),
                            flush=True,
                        )
                    print(
                        "EPISODE_PUBLISH "
                        + json.dumps(
                            dict(attempt=str(attempt), seconds=perf_counter() - publish_started)
                        ),
                        flush=True,
                    )
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
                        dict(
                            error=repr(error),
                            scratch=str(work),
                            scope="interrupted attempt, excluded from training",
                        )
                    )
                    + "\n"
                )
                raise
    return dict(completed=len(completed), remaining=len(manifest["episodes"]) - len(completed))
