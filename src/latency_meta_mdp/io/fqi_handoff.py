"""Portable scratch-FQI recipes, pinned model restoration and selected-state export."""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from latency_meta_mdp.meta.information.config import identity_hash, load_config, sha256


def build_recipe(config, starter, *, root, acceptance=False):
    root, starter = Path(root).resolve(), Path(starter).resolve()
    manifest_path = starter / "cohort.json"
    binding = json.loads(manifest_path.read_text())["binding"]
    from latency_meta_mdp.meta.information.attention_replay import admitted_source_hashes

    if binding.get("config_hash") not in admitted_source_hashes(config, root=root):
        raise ValueError("starter recording config is not explicitly admitted")
    if config["schema_version"] != 3 or any(
        binding.get(k) != v
        for k, v in dict(
            schema_version=3, scope="starter_training"
        ).items()
    ):
        raise ValueError("fresh optional-protocol starter data required")
    scenes = config["splits"]["train"][16:]
    if len(scenes) < 8:
        raise ValueError("at least eight additional training scenes required")
    cycles = []
    for cycle in range(1, 2 if acceptance else 101):
        selected = [scenes[((cycle - 1) * 8 + i) % len(scenes)] for i in range(8)]
        cycles.append(
            dict(
                fitted_iterations=2,
                optimizer_updates=64 if acceptance else 1000,
                validation_every=2,
                generation=dict(
                    name=f"online-{cycle:04d}",
                    trial=cycle,
                    scenes=selected,
                    regimes=["zero", "nominal", "family"],
                    behaviors=["fqi"] * 8,
                    epsilon=0.2,
                ),
            )
        )
    return dict(
        schema_version=2,
        startup_sources=[
            dict(
                path=str(starter.relative_to(root)),
                manifest_sha256=sha256(manifest_path),
                group="startup",
            )
        ],
        initial_checkpoint=None,
        validation_bootstrap=None,
        global_batch=256,
        seed=0,
        startup=dict(
            fitted_iterations=2 if acceptance else 40,
            optimizer_updates=64 if acceptance else 1000,
            validation_every=2 if acceptance else 10,
        ),
        cycles=cycles,
        reward_regression_policy="report_only",
        replay_curriculum=dict(
            total_cycles=100,
            start=0.1,
            end=0.5,
            curvature=1.5,
            latest_fraction_cap=0.25,
            latest_online_fraction=0.5,
            oversampling_cap=4.0,
        ),
        retention=dict(
            recent_batches=20,
            history_episodes=96,
            max_online_bytes=96 * 2**30,
            max_cache_bytes=192 * 2**30,
            max_run_bytes=360 * 2**30,
            min_free_bytes=40 * 2**30,
            episode_cache_size=64,
        ),
    )


def install_asset(source, target, digest):
    source, target = Path(source), Path(target)
    if sha256(source) != digest:
        raise ValueError("downloaded asset hash differs")
    if target.exists():
        if sha256(target) != digest:
            raise ValueError("existing asset destination differs")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def restore_assets(manifest, root):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(root).resolve()
    cache = root / "outputs/cache/huggingface/hub"
    items = json.loads(Path(manifest).read_text())

    def restore(item):
        target = (root / item["destination"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("asset destination escapes project")
        source = hf_hub_download(
            item["repo"], item["filename"], revision=item["revision"], cache_dir=cache
        )
        install_asset(source, target, item["sha256"])
        return item["destination"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, name in enumerate(pool.map(restore, items), 1):
            print(f"ASSET {i}/{len(items)} {name}", flush=True)
    snapshot_download(
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        revision="114c1379950215c8b35dfcd4e90a5c251dde0d32",
        allow_patterns=["config.json", "model.safetensors"],
        cache_dir=cache,
    )


def checkpoint_payload(checkpoint, config_path, recipe_path, *, allow_live=False):
    checkpoint = Path(checkpoint).resolve()
    config = load_config(config_path)
    receipt = json.loads((checkpoint / "complete.json").read_text())
    binding = receipt["binding"]
    if (
        config["schema_version"] != 3
        or binding.get("schema_version") != 3
        or binding.get("config_hash") != identity_hash(config)
        or binding.get("experiment_id") != config["experiment_id"]
        or binding.get("initialization", {}).get("kind") != "scratch"
        or receipt["files"] != {"state.pt": sha256(checkpoint / "state.pt")}
    ):
        raise ValueError("checkpoint identity or weights differ")
    journal_path = next(
        (p / "campaign-state.json" for p in checkpoint.parents
         if (p / "campaign-state.json").is_file()), None
    )
    if journal_path is None:
        raise ValueError("checkpoint has no owning campaign recipe")
    journal = json.loads(journal_path.read_text())
    if (
        journal["binding"]["recipe"] != json.loads(Path(recipe_path).read_text())
        or journal["binding"]["config_sha256"] != sha256(config_path)
    ):
        raise ValueError("publication recipe or config differs from owning campaign")
    required = {
        str(checkpoint / "state.pt"): receipt["files"]["state.pt"],
        str(checkpoint / "complete.json"): sha256(checkpoint / "complete.json"),
    }
    journaled = any(
        all(phase.get("result", {}).get("artifacts", {}).get(k) == v for k, v in required.items())
        for phase in journal["phases"].values()
    )
    relative = checkpoint.relative_to(journal_path.parent).as_posix()
    live_owned = allow_live and (
        (re.fullmatch(r"(?:startup|cycle-[0-9]+/fit)/iteration-[0-9]+", relative)
         and binding.get("trainer_code_hash") == journal["binding"].get("code_hash")
         and binding.get("trainer_code_hash") is not None)
        or (relative == "initial/iteration-0000" and receipt["iteration"] == 0
            and binding.get("seed") == journal["binding"]["recipe"].get("seed"))
    )
    if not journaled and not live_owned:
        raise ValueError("checkpoint is not a completed artifact of its owning campaign")
    return {
        "state.pt": checkpoint / "state.pt",
        "complete.json": checkpoint / "complete.json",
        "config.yaml": Path(config_path),
        "recipe.json": Path(recipe_path),
    }


def publish_checkpoint(checkpoint, config_path, recipe_path, *, repo_id, label):
    from huggingface_hub import HfApi

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("use a unique plain checkpoint label")
    files = checkpoint_payload(checkpoint, config_path, recipe_path)
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model")
    if not info.private and info.gated != "manual":
        raise ValueError("public checkpoint destination requires manual gating")
    prefix = f"checkpoints/{label}"
    existing = api.list_repo_files(repo_id, repo_type="model")
    if any(p.startswith(prefix + "/") for p in existing):
        raise ValueError("checkpoint label already published; use a new immutable label")
    with tempfile.TemporaryDirectory(prefix="meta-checkpoint-") as folder:
        folder = Path(folder)
        for name, source in files.items():
            shutil.copy2(source, folder / name)
        manifest = {name: sha256(folder / name) for name in files}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            path_in_repo=prefix,
            commit_message=f"Publish {label}",
            parent_commit=info.sha,
        )
    return dict(
        repo_id=repo_id,
        revision=commit.oid,
        path=prefix,
        files=manifest,
        scope="Selected Meta state; full resume also needs its journal and retained replay",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recipe = commands.add_parser("recipe")
    recipe.add_argument("--config", required=True, type=Path)
    recipe.add_argument("--starter", required=True, type=Path)
    recipe.add_argument("--root", type=Path, default=Path.cwd())
    recipe.add_argument("--output", required=True, type=Path)
    recipe.add_argument("--acceptance", action="store_true")
    restore = commands.add_parser("restore-assets")
    restore.add_argument("--manifest", required=True, type=Path)
    restore.add_argument("--root", type=Path, default=Path.cwd())
    pack = commands.add_parser("pack-data")
    pack.add_argument("--source", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    data = commands.add_parser("restore-data")
    data.add_argument("--bundle", required=True, type=Path)
    data.add_argument("--output", required=True, type=Path)
    sync = commands.add_parser("sync-checkpoints")
    sync.add_argument("--root", type=Path, required=True)
    sync.add_argument("--config", type=Path, required=True)
    sync.add_argument("--recipe", type=Path, required=True)
    sync.add_argument("--repo", required=True)
    sync.add_argument("--run-id", required=True)
    sync.add_argument("--wandb-url")
    publish = commands.add_parser("publish-checkpoint")
    for name in ("checkpoint", "config", "recipe"):
        publish.add_argument(f"--{name}", required=True, type=Path)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--label", required=True)
    publish.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "recipe":
        result = build_recipe(
            load_config(args.config), args.starter, root=args.root, acceptance=args.acceptance
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            if json.loads(args.output.read_text()) != result:
                raise ValueError("refusing to replace a different recipe")
        else:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)
    elif args.command == "sync-checkpoints":
        from latency_meta_mdp.io.fqi_publication import sync_checkpoints
        status = sync_checkpoints(args.root, args.config, args.recipe, repo_id=args.repo,
                                  run_id=args.run_id, wandb_url=args.wandb_url)
        print(json.dumps({k: status[k] for k in ("uploaded", "pending", "index_pending", "error")}))
        if status["pending"] or status["index_pending"] or status["error"]:
            raise SystemExit(75)
    elif args.command == "restore-assets":
        restore_assets(args.manifest, args.root)
    elif args.command == "pack-data":
        from latency_meta_mdp.io.fqi_dataset import pack_dataset

        print(json.dumps(pack_dataset(args.source, args.output)))
    elif args.command == "restore-data":
        from latency_meta_mdp.io.fqi_dataset import restore_dataset

        restore_dataset(args.bundle, args.output)
    else:
        result = publish_checkpoint(
            args.checkpoint, args.config, args.recipe, repo_id=args.repo, label=args.label
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
