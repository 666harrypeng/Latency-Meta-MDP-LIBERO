"""Explicit learned-policy cohorts for one bounded offline-to-online cycle."""

import json
import re
from pathlib import Path

from latency_meta_mdp.meta.information.cohort import build_manifest
from latency_meta_mdp.meta.information.config import validate_binding


def build_cycle_manifest(
    config, audit, *, root, code_hash, policy, phase, baselines=True, generation=None
):
    base = build_manifest(config, audit, root=root, code_hash=code_hash)
    if generation is not None and config.get("schema_version", 1) == 2:
        return _attention_generation(base, config, generation, policy, phase, baselines)
    if generation is not None:
        if phase != "online" or set(generation) != {
            "name",
            "scenes",
            "behaviors",
            "regimes",
            "epsilon",
        }:
            raise ValueError("explicit generations require an online recipe")
        scenes, behaviors, regimes = (generation[k] for k in ("scenes", "behaviors", "regimes"))
        if (
            not isinstance(generation["name"], str)
            or not generation["name"]
            or not scenes
            or len(scenes) != len(set(scenes))
            or any(type(s) is not int or s not in config["splits"]["train"] for s in scenes)
            or len(behaviors) != len(scenes)
            or set(behaviors) - {"fqi", "periodic", "active"}
            or not regimes
            or len(regimes) != len(set(regimes))
            or set(regimes) - {"nominal", "family", "zero"}
            or not 0 <= generation["epsilon"] <= 1
        ):
            raise ValueError("invalid online generation recipe")
        episodes = [
            dict(
                episode_id=f"{scene}-{regime}-{behavior}",
                scene=scene,
                regime=regime,
                behavior=behavior,
                policy_seed=0,
                scheduler_seed=scene * 100 + {"nominal": 0, "family": 10, "zero": 20}[regime],
                split="train",
                epsilon=generation["epsilon"] if behavior == "fqi" else 0.0,
            )
            for scene, behavior in zip(scenes, behaviors, strict=True)
            for regime in regimes
        ]
        return dict(
            binding=dict(
                base["binding"],
                scope="online_training",
                policy=policy,
                phase=phase,
                baselines=baselines,
                generation=generation,
            ),
            episodes=episodes,
        )
    if phase == "smoke":
        scenes, scope, split = config["splits"]["train"][:2], "diagnostic", "train"
    elif phase == "validation":
        scenes, scope, split = config["splits"]["validation"][:4], "evaluation", "validation"
    elif phase == "monitor":
        scenes, scope, split = config["splits"]["validation"][:3], "evaluation", "validation"
        if len(scenes) != 3 or baselines:
            raise ValueError("monitor requires three validation scenes and meta-only")
    elif phase == "online":
        scenes, scope, split = config["splits"]["train"][16:32], "online_training", "train"
        if len(scenes) != 16:
            raise ValueError("online cycle requires sixteen additional audited train scenes")
    else:
        raise ValueError("unsupported cycle phase")
    episodes = []
    for index, scene in enumerate(scenes):
        behaviors = (
            ("fqi", "periodic", "active") if phase == "validation" and baselines else ("fqi",)
        )
        if phase == "online" and index >= 12:
            behaviors = ("periodic" if index < 14 else "active",)
        regimes = ("nominal", "family", "zero") if phase == "monitor" else ("nominal", "family")
        for d, regime in enumerate(regimes):
            for behavior in behaviors:
                episodes.append(
                    dict(
                        episode_id=f"{scene}-{regime}-{behavior}",
                        scene=scene,
                        regime=regime,
                        behavior=behavior,
                        policy_seed=0,
                        scheduler_seed=scene * 100 + d * 10,
                        split=split,
                        epsilon=0.2 if phase == "online" and behavior == "fqi" else 0.0,
                    )
                )
    binding = dict(base["binding"], scope=scope, policy=policy, phase=phase, baselines=baselines)
    return dict(binding=binding, episodes=episodes)


def grid_generation(name, scenes):
    return dict(
        name=name,
        scenes=list(scenes),
        regimes=["zero", "nominal", "family"],
        epsilon=0.0,
        trial=0,
        patterns=[
            dict(behavior="grid", rho=rho, refresh=refresh)
            for rho in (0, 0.25, 0.5, 0.75, 1)
            for refresh in ("on_request", "every_4_ticks")
        ]
        + [dict(behavior="periodic"), dict(behavior="exploration")],
    )


def _attention_generation(base, config, generation, policy, phase, baselines):
    required = {"name", "scenes", "regimes", "epsilon"}
    keys = set(generation)
    if (
        phase != "online"
        or not required <= keys
        or keys - required - {"trial", "behaviors", "patterns"}
        or ("behaviors" in keys) == ("patterns" in keys)
    ):
        raise ValueError("invalid attention generation fields")
    name, scenes, regimes = (generation[k] for k in ("name", "scenes", "regimes"))
    trial = generation.get("trial", 0)
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        or type(trial) is not int
        or trial < 0
        or not scenes
        or len(set(scenes)) != len(scenes)
        or any(type(s) is not int or s not in config["splits"]["train"] for s in scenes)
        or not regimes
        or len(set(regimes)) != len(regimes)
        or set(regimes) - {"zero", "nominal", "family"}
        or not 0 <= generation["epsilon"] <= 1
    ):
        raise ValueError("invalid attention generation identity or seed split")
    if "behaviors" in generation and len(generation["behaviors"]) != len(scenes):
        raise ValueError("one behavior required per scene")
    from latency_meta_mdp.meta.information.behaviors import AttentionRuleScheduler
    from latency_meta_mdp.meta.information.config import identity_hash
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec

    episodes = []
    for i, scene in enumerate(scenes):
        patterns = generation.get(
            "patterns",
            [dict(behavior=generation.get("behaviors", [])[i])]
            if "behaviors" in generation
            else [],
        )
        if not patterns:
            raise ValueError("empty generation patterns")
        for p, pattern in enumerate(patterns):
            behavior = pattern["behavior"]
            options = {k: v for k, v in pattern.items() if k != "behavior"}
            if behavior != "fqi":
                AttentionRuleScheduler(
                    behavior, action_spec=ActionSpec(**config["actions"]), **options
                )
            elif options:
                raise ValueError("learned behavior cannot carry rule options")
            for regime in regimes:
                episode_id = f"{name}-t{trial}-{scene}-{regime}-p{p}-{behavior}"
                episode = dict(
                    episode_id=episode_id,
                    scene=scene,
                    regime=regime,
                    behavior=behavior,
                    policy_seed=0,
                    scheduler_seed=int(identity_hash(episode_id)[:8], 16),
                    split="train",
                    epsilon=generation["epsilon"] if behavior == "fqi" else 0.0,
                    trial=trial,
                    pattern_id=p,
                )
                if options:
                    episode["pattern"] = options
                episodes.append(episode)
    return dict(
        binding=dict(
            base["binding"],
            scope="online_training",
            policy=policy,
            phase=phase,
            baselines=baselines,
            generation=generation,
        ),
        episodes=episodes,
    )


def prepare_cycle(
    config_path,
    output_dir,
    *,
    root,
    audit_path,
    checkpoint,
    phase,
    baselines=True,
    source_manifest=None,
    shard_index=None,
    num_shards=None,
    generation=None,
):
    from latency_meta_mdp.meta.information.config import load_config
    from latency_meta_mdp.meta.information.deployment import (
        deployment_binding,
        load_scheduler,
        scheduler_identity,
    )
    from latency_meta_mdp.meta.information.preflight import preflight

    output_dir = Path(output_dir)
    config = load_config(config_path)
    report = preflight(config_path, output_dir / "preflight", root=root)
    if report["compute_processes"]:
        raise RuntimeError("GPU compute process already active")
    scheduler = load_scheduler(
        checkpoint, deployment_binding(config), source_manifest=source_manifest
    )
    policy = scheduler_identity(scheduler)
    manifest = build_cycle_manifest(
        config,
        json.loads(Path(audit_path).read_text()),
        root=root,
        code_hash=report["code_hash"],
        policy=policy,
        phase=phase,
        baselines=baselines,
        generation=generation,
    )
    if shard_index is not None or num_shards is not None:
        from latency_meta_mdp.meta.information.parallel import partition_manifest

        manifest = partition_manifest(manifest, shard_index, num_shards)
    path = output_dir / "cohort.json"
    if path.exists():
        validate_binding(json.loads(path.read_text()), manifest)
    else:
        from latency_meta_mdp.meta.information.validation import atomic_json

        atomic_json(path, manifest)
    return config, manifest


def policy_binding(cohort, episode):
    return (
        dict(cohort["policy"], epsilon=episode["epsilon"]) if episode["behavior"] == "fqi" else None
    )
