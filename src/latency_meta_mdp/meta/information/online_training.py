"""Source-balanced replay and explicit parent-initialized FQI generations."""


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
    if [m["binding"]["scope"] for m in manifests] != ["starter_training", "online_training"]:
        raise ValueError("expected offline and online training sources")
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
):
    import json
    import os
    import random
    import subprocess
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
    config = load_config(config_path)
    manifests = [json.loads((d / "cohort.json").read_text()) for d in (offline_dir, online_dir)]
    compatibility = json.loads(Path(compatibility_path).read_text())
    validate_mixture(config, manifests, compatibility)
    audit = json.loads(Path(audit_path).read_text())
    expected_old = build_manifest(
        config, audit, root=root, code_hash=manifests[0]["binding"]["code_hash"]
    )
    expected_new = build_cycle_manifest(
        config,
        audit,
        root=root,
        code_hash=manifests[1]["binding"]["code_hash"],
        policy=manifests[1]["binding"]["policy"],
        phase="online",
    )
    validate_binding(expected_old, manifests[0])
    validate_binding(expected_new, manifests[1])
    if (
        torch.device(device).type == "cuda"
        and subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
    ):
        raise RuntimeError("GPU compute process already active")
    verify_runtime_dependencies(root, config)
    verify_assets(root, config["assets"])
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    cache = EpisodeCache()

    def admission(source):
        return lambda value: print(json.dumps(dict(source=source, **value)), flush=True)

    replay = MixedReplay(
        VerifiedReplay(offline_dir, manifests[0], cache=cache, progress=admission("offline")),
        VerifiedReplay(online_dir, manifests[1], cache=cache, progress=admission("online")),
    )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from latency_meta_mdp.meta.information.deployment import deployment_binding

    expected = deployment_binding(config)
    model = start_generation(
        checkpoint,
        output_dir,
        expected,
        device=device,
        source_manifest=manifests[1]["binding"]["policy"].get("source_manifest"),
    )
    parent = json.loads((output_dir / "initial-parent.json").read_text())
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
            for d, r in zip((offline_dir, online_dir), (replay.offline, replay.online), strict=True)
        ],
        online_training_episodes=32,
        online_learned_episodes=24,
        sampling="0.5 mean offline transitions + 0.5 mean online transitions",
        trainer_code_hash=identity_hash(code),
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
        )
        binding["validation_protocol"] = validator.protocol
    tracker = OnlineFitTracker(output_dir, binding=binding, episodes=len(replay), epochs=2)
    if validator is not None:
        validator.progress = tracker
    success = False
    try:
        result = train_iterations(
            model,
            replay,
            output_dir,
            binding=binding,
            total_iterations=10,
            epochs=2,
            learning_rate=config["training"]["learning_rate"],
            conservative_weight=config["training"]["conservative_weight"],
            progress=tracker,
            validate_checkpoint=validator,
        )
        success = True
        return result
    finally:
        tracker.close(success=success)
