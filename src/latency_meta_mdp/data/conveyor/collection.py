"""Collect a seed-grouped conveyor corpus with resumable, explicit admission."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from latency_meta_mdp.data.conveyor.source import ConveyorSource
from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
from latency_meta_mdp.envs.conveyor.expert import load_expert_spec
from latency_meta_mdp.io.paths import repository_root


def load_collection_job(path):
    path = Path(path).resolve()
    job = yaml.safe_load(path.read_text())
    if (
        job.get("schema_version") != 1
        or job.get("task_id") != "conveyor_sort"
        or job.get("purpose") not in {"development_smoke", "training_source"}
    ):
        raise ValueError("unsupported conveyor corpus purpose")
    splits = job["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("declare train, validation and test seed groups")
    seen = set()
    for split, seeds in list(splits.items()):
        if isinstance(seeds, dict):
            if (
                set(seeds) != {"start", "stop"}
                or any(type(v) is not int for v in seeds.values())
                or not 0 <= seeds["start"] < seeds["stop"]
            ):
                raise ValueError("seed ranges require integer start/stop (stop exclusive)")
            seeds = splits[split] = list(range(seeds["start"], seeds["stop"]))
        if not isinstance(seeds, list) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("split seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
            raise ValueError("seed groups overlap")
        seen.update(seeds)
    selected = job["collect_splits"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(splits)
        or not any(splits[s] for s in selected)
    ):
        raise ValueError("select nonempty declared collection splits")
    for key in ("scene", "expert"):
        job[key] = str((path.parent / job[key]).resolve())
    return job


def collection_identity(job):
    root = repository_root()
    scene = load_conveyor_spec(Path(job["scene"]))
    expert = load_expert_spec(Path(job["expert"]))
    control = root / scene.action_contract_path
    paths = [
        *sorted((root / "src/latency_meta_mdp/envs/conveyor").glob("*.py")),
        root / "src/latency_meta_mdp/envs/backend.py",
        root / "src/latency_meta_mdp/envs/control.py",
        root / "src/latency_meta_mdp/runtime/timing.py",
        root / "src/latency_meta_mdp/runtime/policy_execution.py",
        Path(__file__).resolve(),
        root / "src/latency_meta_mdp/data/conveyor/source.py",
    ]
    payload = dict(
        job=job,
        scene=asdict(scene),
        expert=asdict(expert),
        controller_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
        implementation={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload, load_action_contract(control).contract_id


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def collect(config, output_dir, *, resume=False, run_episode=None):
    job = load_collection_job(config)
    identity, bindings, contract = collection_identity(job)
    output = Path(output_dir).resolve()
    manifest = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError("collection exists; use --resume")
        previous = json.loads(manifest.read_text())
        if previous["identity"] != identity:
            raise ValueError("collection identity changed; use a new output directory")
    else:
        output.mkdir(parents=True)
    if run_episode is None:
        from latency_meta_mdp.envs.conveyor.expert_rollout import run_expert

        run_episode = run_expert
    requested = [(split, seed) for split in job["collect_splits"] for seed in job["splits"][split]]
    report = dict(
        format_id="conveyor_expert_corpus_v1",
        schema_version=1,
        corpus_id=job["corpus_id"],
        purpose=job["purpose"],
        task_id="conveyor_sort",
        variant="surface",
        identity=identity,
        bindings=bindings,
        action_contract=contract,
        splits=job["splits"],
        collect_splits=job["collect_splits"],
        implementation_revision=subprocess.check_output(
            ["git", "-C", str(repository_root()), "rev-parse", "HEAD"], text=True
        ).strip(),
        requested_count=len(requested),
        admitted_count=0,
        rejected_count=0,
        status="running",
        episodes=[],
    )
    _write(manifest, report)
    for split, seed in requested:
        folder = output / "episodes" / f"seed-{seed}"
        status_path = folder / "status.json"
        if folder.exists():
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") != "completed":
                raise RuntimeError(f"incomplete episode {seed}; inspect it before retrying")
        else:
            status = run_episode(
                Path(job["scene"]),
                Path(job["expert"]),
                folder,
                seed=seed,
                video=False,
                record_source=True,
                source_purpose=job["purpose"],
            )
        admitted = bool(status.get("expert_admitted"))
        row = dict(
            seed=seed,
            split=split,
            episode_id=f"conveyor-surface-seed-{seed}",
            group_id=f"conveyor_sort/surface/seed-{seed}",
            admitted=admitted,
            status_path=str(status_path.relative_to(output)),
            outcomes={k: status.get(k) for k in ("spawned", "successes", "misses", "timeouts")},
        )
        if admitted:
            source = ConveyorSource(folder / "source")
            if (
                source.manifest["seed"] != seed
                or source.manifest["purpose"] != job["purpose"]
                or source.manifest["action_contract"] != contract
                or not status.get("all_scheduled_spawned")
                or source.manifest["summary"]["spawned"] != bindings["scene"]["spawn_count"]
            ):
                raise ValueError("recorded episode identity or admission differs from the request")
            row.update(
                source_root=str((folder / "source").relative_to(output)),
                frame_count=source.manifest["frame_count"],
                transition_count=len(source.actions),
            )
        report["episodes"].append(row)
        report["admitted_count"] += int(admitted)
        report["rejected_count"] += int(not admitted)
        _write(manifest, report)
        print(
            json.dumps(
                {
                    "completed": len(report["episodes"]),
                    "requested": len(requested),
                    "seed": seed,
                    "admitted": admitted,
                }
            ),
            flush=True,
        )
    report["status"] = "completed" if report["rejected_count"] == 0 else "needs_attention"
    _write(manifest, report)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    path = collect(args.config, args.output_dir, resume=args.resume)
    print(path, flush=True)
    return 0 if json.loads(path.read_text())["status"] == "completed" else 2
