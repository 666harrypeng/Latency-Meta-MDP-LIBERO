"""Train a structured clean policy from a public, versioned job configuration."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
    p.add_argument("--wandb-disabled", action="store_true")
    a = p.parse_args()
    from huggingface_hub import snapshot_download

    from latency_meta_mdp.sft_delivery import (
        load_training_job,
        publish_checkpoints,
        verify_bundle_files,
    )

    job = load_training_job(a.config)
    root = Path(__file__).resolve().parents[3]
    a.work_dir = a.work_dir.resolve()
    bundle = a.work_dir / "bundles" / job["dataset_revision"]
    print("Preparing public training bundle", flush=True)
    snapshot_download(
        repo_id=job["dataset_repo"],
        repo_type="dataset",
        revision=job["dataset_revision"],
        local_dir=bundle,
        token=False,
    )
    bundle_info = verify_bundle_files(bundle)
    if bundle_info["level"] != job["level"]:
        raise ValueError("Training bundle level mismatch")
    a.dataset_root = bundle / bundle_info["dataset_root"]
    a.preparation_root = bundle / bundle_info["preparation_root"]
    a.checkpoint_root = a.work_dir / "checkpoints"
    a.level, a.run_name = job["level"], job["run_name"]
    a.device_count, a.batch_size = job["device_count"], job["batch_size"]
    a.mode = "formal"
    os.environ["HF_LEROBOT_HOME"] = str(a.dataset_root.resolve())
    from latency_meta_mdp.artifacts import sha256_file
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy
    from latency_meta_mdp.sft_launch import SFTLaunchRequest, resolve_sft_schedule
    from latency_meta_mdp.sft_profile import load_sft_profile

    profile_path = job["profile"]
    profile = load_sft_profile(profile_path)
    prep = json.loads((a.preparation_root / "preparation.json").read_text())
    if prep["level"] != a.level or prep["profile_sha256"] != sha256_file(profile_path):
        raise ValueError("level preparation/profile identity mismatch")
    manifest_path = a.dataset_root / "manifest.json"
    if sha256_file(manifest_path) != prep["dataset_manifest_sha256"]:
        raise ValueError("prepared export manifest changed")
    manifest = json.loads(manifest_path.read_text())
    level = next(x for x in manifest["datasets"] if x["level"] == a.level)
    if (
        level["repo_id"] != profile.levels[a.level].repo_id
        or level["frame_count"] != prep["source_count"]
    ):
        raise ValueError("level dataset identity/source count mismatch")
    nested_path = a.dataset_root / level["dataset_manifest"]
    if sha256_file(nested_path) != level["dataset_manifest_sha256"]:
        raise ValueError("level dataset manifest hash mismatch")
    norm = a.preparation_root / prep["norm_stats"]
    if sha256_file(norm) != prep["norm_stats_sha256"]:
        raise ValueError("level norm statistics changed")
    patches = tuple(
        root / "patches/openpi" / name
        for name in (
            "0001-filter-incomplete-action-chunks.patch",
            "0002-save-completed-step-checkpoints.patch",
            "0003-mask-action-tails.patch",
        )
    )
    if any(sha256_file(path) != prep["patches"][path.name] for path in patches):
        raise ValueError("OpenPI patches differ from preparation")
    request = SFTLaunchRequest(
        level=a.level,
        experiment_name=a.run_name,
        mode=a.mode,
        resume=a.resume,
        device_count=a.device_count,
        batch_size_override=a.batch_size,
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        from latency_meta_mdp.openpi_sft import build_level_train_config, run_openpi_training

        config = build_level_train_config(
            profile=profile,
            request=request,
            assets_root=a.preparation_root / "assets",
            checkpoint_root=a.checkpoint_root,
            wandb_enabled=not a.wandb_disabled,
        )
        data = config.data.create(config.assets_dirs, config.model)
        if data.norm_stats is None or data.norm_stats["state"].mean.shape != (16,):
            raise ValueError("model did not load the prepared current16 norm statistics")
        report = {
            "implementation_revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "implementation_tracked_dirty": subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD"]
            ).returncode
            != 0,
            "launcher_sha256": sha256_file(Path(__file__)),
            "dataset_revision": job["dataset_revision"],
            "mode": a.mode,
            "run_name": a.run_name,
            "level": a.level,
            "repo_id": data.repo_id,
            "profile_sha256": sha256_file(profile_path),
            "preparation_sha256": sha256_file(a.preparation_root / "preparation.json"),
            "source_count": prep["source_count"],
            "batch_size": config.batch_size,
            "num_steps": config.num_train_steps,
            "training_examples": config.batch_size * config.num_train_steps,
            "equivalent_source_epochs": config.batch_size
            * config.num_train_steps
            / prep["source_count"],
            "state_tokens_enabled": config.model.discrete_state_input,
            "active_action_dim": config.model.active_action_dim,
            "checkpoint_dir": str(config.checkpoint_dir),
            "norm_stats_sha256": sha256_file(norm),
            "training_parallelism": "replicated_data_parallel",
            "fsdp_devices": config.fsdp_devices,
            "device_count": a.device_count,
            "per_device_batch_size": config.batch_size // a.device_count,
            "warmup_steps": config.lr_schedule.warmup_steps,
            "decay_steps": config.lr_schedule.decay_steps,
            "checkpoint_steps": list(schedule.expected_checkpoint_steps),
            "reference_training_examples": profile.batch_size * profile.num_train_steps,
            "training_started": False,
        }
        print(json.dumps(report, indent=2), flush=True)
        destination = Path(config.checkpoint_dir)
        if a.publish_only:
            revision = publish_checkpoints(
                root=destination,
                steps=schedule.expected_checkpoint_steps,
                repo_id=job["publish_repo"],
                notices=root / "licenses/pi05",
            )
            print(json.dumps({"published_revision": revision}), flush=True)
            return
        import jax

        devices = jax.devices()
        if (
            len(devices) != a.device_count
            or jax.process_count() != 1
            or not all(device.platform == "gpu" for device in devices)
        ):
            raise RuntimeError(
                "Visible GPU devices differ from the replicated data-parallel request"
            )
        import numpy as np
        from openpi.training.data_loader import create_torch_data_loader

        loader = create_torch_data_loader(
            data,
            config.model,
            config.model.action_horizon,
            batch_size=a.device_count,
            num_batches=1,
            num_workers=0,
            shuffle=False,
        )
        observation, actions = next(iter(loader))
        mask = np.asarray(observation.action_loss_mask)
        if actions.shape != (a.device_count, 50, 32) or mask.shape != actions.shape:
            raise ValueError("Incorrect action or loss-mask shape")
        if not mask[..., :7].any() or mask[..., 7:].any():
            raise ValueError("Invalid active action dimensions")
        if (
            observation.tokenized_prompt is None
            or not np.asarray(observation.tokenized_prompt_mask).any()
        ):
            raise ValueError("Prompt/state tokens are missing")
        print("Actual data-loader batch and GPU topology checked", flush=True)
        if a.check_only:
            return
        del loader, observation, actions, mask
        gc.collect()
        if destination.exists() and not a.resume:
            raise FileExistsError(destination)
        contract_path = destination.parent / (destination.name + ".launch-contract.json")
        contract = {
            key: report[key]
            for key in (
                "mode",
                "run_name",
                "level",
                "repo_id",
                "dataset_revision",
                "profile_sha256",
                "preparation_sha256",
                "norm_stats_sha256",
                "batch_size",
                "device_count",
                "training_parallelism",
                "warmup_steps",
                "decay_steps",
            )
        }
        if a.resume:
            if not destination.exists() or not contract_path.is_file():
                raise ValueError("Resume requires a checkpoint directory and its launch contract")
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError(
                    "Resume would change the data, batch, topology or learning-rate contract"
                )
        else:
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            with contract_path.open("x") as out:
                json.dump(contract, out, indent=2)
        started = time.monotonic()
        run_openpi_training(config=config, openpi_root=copied)
        expected = list(schedule.expected_checkpoint_steps)
        found = sorted(
            int(path.name)
            for path in destination.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
        if not set(expected).issubset(found):
            raise ValueError(f"Missing milestones: {expected}; found {found}")
        for step in expected:
            if (
                not (destination / str(step) / "params").exists()
                or not (destination / str(step) / "train_state").exists()
            ):
                raise ValueError("checkpoint bundle is incomplete")
        report.update(
            training_started=True,
            training_completed=True,
            elapsed_seconds=time.monotonic() - started,
            devices=[d.device_kind for d in devices],
            memory_stats=[d.memory_stats() for d in devices],
            checkpoint_steps=found,
        )
        with (destination / f"verified_run-step{found[-1]}.json").open("x") as out:
            json.dump(report, out, indent=2, default=int)
            out.write("\n")
        revision = publish_checkpoints(
            root=destination,
            steps=schedule.expected_checkpoint_steps,
            repo_id=job["publish_repo"],
            notices=root / "licenses/pi05",
        )
        print(json.dumps({"published_revision": revision}), flush=True)
        sys.path.remove(str(copied / "src"))


if __name__ == "__main__":
    main()
