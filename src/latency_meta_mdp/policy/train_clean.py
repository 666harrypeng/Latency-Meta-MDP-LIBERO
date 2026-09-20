"""Train a clean policy from a pinned local or public training bundle."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from latency_meta_mdp.io.paths import repository_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", "--work-dir", dest="work_dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument(
        "--check-data-only",
        action="store_true",
        help="Validate data/config on available devices without a training-topology claim",
    )
    mode.add_argument("--publish-only", action="store_true")
    p.add_argument("--wandb-disabled", action="store_true")
    a = p.parse_args()
    return run(a)


def run(a):
    a.mode = getattr(a, "mode", "formal")
    if a.mode == "smoke" and a.publish_only:
        raise ValueError("smoke checkpoints cannot be published as formal milestones")
    from latency_meta_mdp.io.artifacts import sha256_file
    from latency_meta_mdp.io.policy_publish import publish_checkpoints
    from latency_meta_mdp.policy.clean_data import require_training_source, resolve_clean_inputs
    from latency_meta_mdp.policy.config import load_training_job
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
    from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule

    job = load_training_job(a.config)
    root = repository_root()
    a.work_dir = a.work_dir.resolve()
    inputs = resolve_clean_inputs(job, a.work_dir)
    data_only = getattr(a, "check_data_only", False)
    if not a.publish_only:
        require_training_source(inputs, checking=a.check_only or data_only)
    if a.publish_only and not job.get("publish_repo"):
        raise ValueError("publish-only requires an explicit publish_repo")
    a.dataset_root, a.preparation_root = inputs.dataset_root, inputs.preparation_root
    a.checkpoint_root = a.work_dir / "checkpoints"
    a.level, a.run_name = job.get("level"), job["run_name"]
    a.device_count, a.batch_size = job["device_count"], job["batch_size"]
    os.environ["HF_LEROBOT_HOME"] = str(a.dataset_root.resolve())
    profile_path, profile, prep = job["profile"], inputs.profile, inputs.preparation
    norm = a.preparation_root / prep["norm_stats"]
    patches = tuple(
        root / "patches/openpi" / name
        for name in (
            "0001-filter-incomplete-action-chunks.patch",
            "0002-save-completed-step-checkpoints.patch",
            "0003-mask-action-tails.patch",
        )
    )
    request = SFTLaunchRequest(
        level=a.level,
        experiment_name=a.run_name,
        mode=a.mode,
        resume=a.resume,
        device_count=a.device_count,
        batch_size_override=a.batch_size,
        task_id=job.get("task_id"),
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        from latency_meta_mdp.policy.openpi.training import (
            build_launch_train_config,
            run_openpi_training,
        )

        config = build_launch_train_config(
            profile=profile,
            request=request,
            assets_root=a.preparation_root / "assets",
            checkpoint_root=a.checkpoint_root,
            wandb_enabled=not a.wandb_disabled,
            task_parameters=inputs.task_parameters,
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
            "dataset_revision": job.get("dataset_revision", job.get("dataset", {}).get("revision")),
            "data_identity": inputs.bundle_identity,
            "data_purpose": inputs.purpose,
            "task_id": config.policy_metadata["task_id"],
            "action_contract": config.policy_metadata["action_contract_id"],
            "mode": a.mode,
            "run_name": a.run_name,
            "level": a.level,
            "repo_id": data.repo_id,
            "profile_sha256": sha256_file(profile_path),
            "training_recipe_sha256": sha256_file(job["training"]) if job.get("training") else None,
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
            "finetune_mode": "full",
            "paligemma_variant": config.model.paligemma_variant,
            "action_expert_variant": config.model.action_expert_variant,
            "ema_decay": config.ema_decay,
            "checkpoint_dir": str(config.checkpoint_dir),
            "norm_stats_sha256": sha256_file(norm),
            "training_parallelism": config.policy_metadata["training_parallelism"],
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
        if not data_only and (
            len(devices) != a.device_count
            or jax.process_count() != 1
            or not all(device.platform == "gpu" for device in devices)
        ):
            raise RuntimeError("Visible GPU devices differ from the single-host training request")
        import numpy as np
        from openpi.training.data_loader import create_torch_data_loader

        check_batch_size = min(2, prep["source_count"]) if data_only else a.device_count
        loader = create_torch_data_loader(
            data,
            config.model,
            config.model.action_horizon,
            batch_size=check_batch_size,
            num_batches=1,
            num_workers=0,
            shuffle=False,
            sharding=jax.sharding.SingleDeviceSharding(devices[0]) if data_only else None,
        )
        observation, actions = next(iter(loader))
        mask = np.asarray(observation.action_loss_mask)
        if actions.shape != (check_batch_size, 50, 32) or mask.shape != actions.shape:
            raise ValueError("Incorrect action or loss-mask shape")
        if not mask[..., :7].any() or mask[..., 7:].any():
            raise ValueError("Invalid active action dimensions")
        if (
            observation.tokenized_prompt is None
            or not np.asarray(observation.tokenized_prompt_mask).any()
        ):
            raise ValueError("Prompt/state tokens are missing")
        report.update(
            data_checked=True,
            hardware_checked=not data_only,
            action_batch_shape=list(actions.shape),
            action_mask_shape=list(mask.shape),
        )
        if a.check_only or data_only:
            a.work_dir.mkdir(parents=True, exist_ok=True)
            (a.work_dir / "preflight.json").write_text(json.dumps(report, indent=2))
            print(
                "Actual data batch checked"
                + (
                    "; GPU topology checked" if not data_only else "; training topology not checked"
                ),
                flush=True,
            )
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
                "training_recipe_sha256",
                "preparation_sha256",
                "norm_stats_sha256",
                "batch_size",
                "device_count",
                "training_parallelism",
                "warmup_steps",
                "decay_steps",
            )
        }
        if job["schema_version"] == 2:
            contract.update(
                task_id=report["task_id"],
                action_contract=report["action_contract"],
                data_identity=report["data_identity"],
            )
            contract.pop("level", None)
        if config.fsdp_devices > 1:
            contract["fsdp_devices"] = config.fsdp_devices
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
        if job.get("publish_repo") and a.mode == "formal":
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
