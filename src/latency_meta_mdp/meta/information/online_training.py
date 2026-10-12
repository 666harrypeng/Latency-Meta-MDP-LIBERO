"""Source-balanced replay and explicit parent-initialized FQI generations."""

from latency_meta_mdp.meta.information.training import VerifiedReplay


class CumulativeReplay(VerifiedReplay):
    """Concatenate explicitly admitted online cohorts without scanning for new data."""

    def __init__(self, streams):
        streams = list(streams)
        if not streams:
            raise ValueError("cumulative replay requires admitted cohorts")
        self.entries = [entry for stream in streams for entry in stream.entries]
        self.receipts = {
            f"{i}/{key}": digest
            for i, stream in enumerate(streams)
            for key, digest in stream.receipts.items()
        }
        self.cache = streams[0].cache


class MixedReplay:
    def __init__(self, offline, online):
        if not len(offline) or not len(online):
            raise ValueError("both replay sources must be nonempty")
        self.offline, self.online = offline, online
        self.cache = getattr(offline, "cache", None)
        self.sources = ["offline"] * len(offline) + ["online"] * len(online)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.offline):
            return self.offline[index]
        return self.online[index - len(self.offline)]

    def __iter__(self):
        yield from self.offline
        yield from self.online


def start_generation(parent, output_dir, expected, *, device, source_manifest=None):
    import json
    from pathlib import Path

    from latency_meta_mdp.meta.information.deployment import load_scheduler, scheduler_identity

    scheduler = load_scheduler(parent, expected, device=device, source_manifest=source_manifest)
    identity = scheduler_identity(scheduler)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "initial-parent.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("generation parent differs")
    else:
        if any(output_dir.glob("iteration-*")):
            raise ValueError("existing generation has no parent identity")
        path.write_text(json.dumps(identity, indent=2) + "\n")
    return scheduler.model.requires_grad_(True)


def validate_mixture(config, manifests, compatibility):
    from latency_meta_mdp.meta.information.config import identity_hash, validate_binding

    if compatibility["config_hash"] != identity_hash(config) or set(
        compatibility["collection_code_hashes"]
    ) != {m["binding"]["code_hash"] for m in manifests}:
        raise ValueError("explicit collector compatibility differs")
    if len(manifests) < 2 or [m["binding"]["scope"] for m in manifests] != ["starter_training"] + [
        "online_training"
    ] * (len(manifests) - 1):
        raise ValueError("expected offline and online training sources")
    identities = [e["episode_id"] for m in manifests for e in m["episodes"]]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate training episode identity across cohorts")
    for manifest in manifests:
        binding = manifest["binding"]
        for expected, actual in (
            (config["assets"], binding["assets"]),
            (config["mode"], binding["mode"]),
            (1, binding["schema_version"]),
            (identity_hash(config), binding["config_hash"]),
        ):
            validate_binding(expected, actual)
        if any(
            e["split"] != "train" or e["scene"] not in config["splits"]["train"]
            for e in manifest["episodes"]
        ):
            raise ValueError("non-training scene in mixture")


def run_online_training(
    config_path,
    offline_dir,
    online_dir,
    output_dir,
    *,
    root,
    checkpoint,
    compatibility_path,
    audit_path,
    device="cuda:0",
    seed=0,
    validation_bootstrap=None,
    worker_python=None,
    validation_gpus=None,
    training_gpus=None,
    global_batch=4,
    replay_cache_gib=128,
    replay_cache_dir=None,
    replay_prepare_workers=4,
    replay_storage="archive",
    cpu_threads=4,
    stop_after_iterations=None,
    warm_start_checkpoint=None,
    total_iterations=10,
    additional_online_dirs=(),
    optimizer_updates=None,
    validate_initial=True,
):
    import json
    import os
    import random
    from pathlib import Path

    import numpy as np
    import torch

    from latency_meta_mdp.meta.information.cohort import build_manifest
    from latency_meta_mdp.meta.information.config import (
        identity_hash,
        load_config,
        sha256,
        validate_binding,
    )
    from latency_meta_mdp.meta.information.cycle import build_cycle_manifest
    from latency_meta_mdp.meta.information.preflight import (
        verify_assets,
        verify_runtime_dependencies,
    )
    from latency_meta_mdp.meta.information.training import VerifiedReplay, train_iterations
    from latency_meta_mdp.meta.online_tracking import OnlineFitTracker

    root, offline_dir, online_dir, output_dir = map(
        Path, (root, offline_dir, online_dir, output_dir)
    )
    directories = [offline_dir, online_dir, *map(Path, additional_online_dirs)]
    if additional_online_dirs and warm_start_checkpoint is None:
        raise ValueError("cumulative replay requires an explicit warm-start checkpoint")
    if optimizer_updates is not None and training_gpus is None:
        raise ValueError("exact update budgets require explicit train-gpus")
    config = load_config(config_path)
    if replay_storage not in ("archive", "mmap"):
        raise ValueError("unknown replay storage")
    if replay_storage == "mmap" and training_gpus is None:
        raise ValueError("mmap replay requires explicit train-gpus, including for one GPU")
    if training_gpus is not None:
        devices = training_gpus.split(",")
        if (
            not devices
            or any(not d.isdigit() for d in devices)
            or len(set(devices)) != len(devices)
        ):
            raise ValueError("training GPUs must be distinct physical indices")
        if global_batch < 1 or replay_cache_gib < 0:
            raise ValueError("invalid distributed batch/cache budget")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = training_gpus
        device = "cpu"  # Coordinator never owns CUDA tensors or contexts.
    manifests = [json.loads((d / "cohort.json").read_text()) for d in directories]
    compatibility = json.loads(Path(compatibility_path).read_text())
    validate_mixture(config, manifests, compatibility)
    audit = json.loads(Path(audit_path).read_text())
    expected_old = build_manifest(
        config, audit, root=root, code_hash=manifests[0]["binding"]["code_hash"]
    )
    validate_binding(expected_old, manifests[0])
    for manifest in manifests[1:]:
        collector = manifest["binding"]
        expected_new = build_cycle_manifest(
            config,
            audit,
            root=root,
            code_hash=collector["code_hash"],
            policy=collector["policy"],
            phase="online",
            baselines=collector["baselines"],
            generation=collector.get("generation"),
        )
        validate_binding(expected_new, manifest)
    from latency_meta_mdp.meta.information.resources import gpu_processes

    if (training_gpus is not None or torch.device(device).type == "cuda") and gpu_processes()[0]:
        raise RuntimeError("GPU compute process already active")
    verify_runtime_dependencies(root, config)
    verify_assets(root, config["assets"])
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    cache = None if training_gpus is not None else EpisodeCache()

    def admission(source):
        return lambda value: print(json.dumps(dict(source=source, **value)), flush=True)

    admitted = [
        VerifiedReplay(directory, manifest, cache=cache, progress=admission(f"cohort-{i}"))
        for i, (directory, manifest) in enumerate(zip(directories, manifests, strict=True))
    ]
    replay = MixedReplay(admitted[0], CumulativeReplay(admitted[1:]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from latency_meta_mdp.meta.information.deployment import deployment_binding

    expected = deployment_binding(config)
    from latency_meta_mdp.meta.information.deployment import load_scheduler, scheduler_identity

    for i, manifest in enumerate(manifests[1:]):
        policy = manifest["binding"]["policy"]
        behavior = load_scheduler(
            checkpoint if i == 0 else policy["checkpoint"],
            expected,
            source_manifest=policy.get("source_manifest"),
        )
        validate_binding(scheduler_identity(behavior), policy)
        del behavior
    model = start_generation(
        warm_start_checkpoint or checkpoint,
        output_dir,
        expected,
        device=device,
        source_manifest=manifests[1]["binding"]["policy"].get("source_manifest"),
    )
    parent = json.loads((output_dir / "initial-parent.json").read_text())
    if warm_start_checkpoint is None:
        validate_binding(parent, manifests[1]["binding"]["policy"])
    code = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted((root / "src/latency_meta_mdp/meta/information").glob("*.py"))
    }
    code["scripts/run_conveyor_meta.py"] = sha256(root / "scripts/run_conveyor_meta.py")
    binding = dict(
        expected,
        seed=seed,
        device=device,
        parent=parent,
        normalization="fixed-physical-scales-and-learned-LayerNorm",
        collector_compatibility=compatibility,
        data=[
            dict(manifest_sha256=sha256(d / "cohort.json"), receipts_hash=identity_hash(r.receipts))
            for d, r in zip(directories, admitted, strict=True)
        ],
        offline_training_episodes=len(admitted[0]),
        online_training_episodes=len(replay.online),
        online_learned_episodes=sum(
            e["behavior"] == "fqi" for m in manifests[1:] for e in m["episodes"]
        ),
        behavior_policies=[m["binding"]["policy"] for m in manifests[1:]],
        sampling="0.5 mean offline transitions + 0.5 mean online transitions",
        trainer_code_hash=identity_hash(code),
    )
    if warm_start_checkpoint is not None:
        binding["behavior_policy"] = manifests[1]["binding"]["policy"]
    fitter, fitter_options = None, None
    if training_gpus is not None:
        from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed
        from latency_meta_mdp.meta.information.replay_preparation import prepare_replay

        if replay_storage == "mmap":
            from latency_meta_mdp.meta.information.packed_replay import prepare_packed_replay

            prepare_replay = prepare_packed_replay
        cache_dir = replay_cache_dir or output_dir / f"prepared-{replay_storage}"
        replay = prepare_replay(
            replay,
            cache_dir,
            workers=replay_prepare_workers,
            progress=admission("mixed"),
        )
        fitter = fit_distributed
        fitter_options = dict(
            devices=[f"cuda:{i}" for i in range(len(devices))],
            global_batch=global_batch,
            cache_bytes=replay_cache_gib * 2**30,
            cpu_threads=cpu_threads,
            **({"optimizer_updates": optimizer_updates} if optimizer_updates is not None else {}),
        )
        binding["parallel_training"] = dict(
            recipe="synchronous-episode-minibatch-v1",
            global_batch=global_batch,
        )
        # Hardware affects execution, not the objective or replay identity.
        import time

        with (output_dir / "execution-history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        started_at=time.time(),
                        host=os.uname().nodename,
                        physical_gpus=devices,
                        replay_cache_dir=str(cache_dir),
                        replay_storage=replay_storage,
                        replay_prepare_workers=replay_prepare_workers,
                        **fitter_options,
                    )
                )
                + "\n"
            )
    from latency_meta_mdp.meta.information.monitor import IterationValidator

    validator = None
    if validation_bootstrap is not None:
        validator = IterationValidator(
            config_path,
            output_dir / "validation",
            root=root,
            audit_path=audit_path,
            bootstrap_dir=validation_bootstrap,
            model=model,
            worker_python=worker_python,
            gpus=validation_gpus,
            cpu_threads=cpu_threads,
        )
        binding["validation_protocol"] = validator.protocol
    updates_per_pass = (
        (len(replay) + global_batch - 1) // global_batch
        if training_gpus is not None
        else len(replay)
    )
    tracker = OnlineFitTracker(
        output_dir,
        binding=binding,
        episodes=updates_per_pass,
        epochs=2
        if optimizer_updates is None
        else (optimizer_updates + updates_per_pass - 1) // updates_per_pass,
    )
    if validator is not None:
        validator.progress = tracker
    success = False
    try:
        result = train_iterations(
            model,
            replay,
            output_dir,
            binding=binding,
            total_iterations=total_iterations,
            epochs=2,
            learning_rate=config["training"]["learning_rate"],
            conservative_weight=config["training"]["conservative_weight"],
            progress=tracker,
            validate_checkpoint=validator,
            fitter=fitter,
            fitter_options=fitter_options,
            stop_after_iterations=stop_after_iterations,
            validate_initial=validate_initial,
        )
        success = True
        return result
    finally:
        tracker.close(success=success)
