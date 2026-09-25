"""Train the selected forecast input mode from a matched clean milestone."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", "--work-dir", dest="work_dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    p.add_argument("--clean-work-dir", type=Path)
    p.add_argument("--forecast-dir", type=Path)
    p.add_argument("--wandb-disabled", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--check-data-only", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
    a = p.parse_args()
    return run(a)


def run(a):
    a.mode = getattr(a, "mode", "formal")
    data_only = getattr(a, "check_data_only", False)
    if a.mode == "smoke" and a.publish_only:
        raise ValueError("smoke checkpoints cannot be published as formal milestones")
    root = Path.cwd()
    work = a.work_dir.resolve()
    from latency_meta_mdp.data.forecast.assets import load_forecast_job, verify_forecast_models
    from latency_meta_mdp.io.artifacts import sha256_file
    from latency_meta_mdp.io.policy_publish import publish_checkpoints
    from latency_meta_mdp.policy.conditioning import conditioning_contract, conditioning_patches
    from latency_meta_mdp.policy.config import resolve_policy_profile
    from latency_meta_mdp.policy.initialization import conditioned_initialization
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
    from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule

    job = load_forecast_job(a.config, project_root=root)
    clean_job = job["clean"]
    clean_work = getattr(a, "clean_work_dir", None) or work / "clean"
    task_inputs = None
    if clean_job["schema_version"] == 2:
        from latency_meta_mdp.policy.clean_data import require_training_source, resolve_clean_inputs

        task_inputs = resolve_clean_inputs(clean_job, clean_work.resolve())
        require_training_source(task_inputs, checking=a.check_only or a.publish_only or data_only)
    input_mode = job.get("input_mode", "current_and_forecast")
    mode_contract = conditioning_contract(input_mode)
    profile = task_inputs.profile if task_inputs else resolve_policy_profile(clean_job)
    level = clean_job.get("level")
    request = SFTLaunchRequest(
        level,
        clean_job["run_name"],
        "formal",
        False,
        clean_job["device_count"],
        clean_job["batch_size"],
        task_id=clean_job.get("task_id"),
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    bundle, clean_dir, initialization = conditioned_initialization(
        job, profile, schedule, work=work, clean_work=clean_work.resolve(), clean_inputs=task_inputs
    )
    training_profile = dataclasses.replace(
        profile, fsdp_devices=job.get("fsdp_devices", profile.fsdp_devices)
    )
    training_request = dataclasses.replace(
        request,
        device_count=job.get("device_count", request.device_count),
        batch_size_override=job.get("batch_size", request.batch_size_override),
    )
    os.environ["HF_LEROBOT_HOME"] = str(
        task_inputs.dataset_root if task_inputs else bundle / "dataset"
    )
    cache = getattr(a, "forecast_dir", None) or work / "preparation/forecasts"
    cache = cache.resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    bindings = manifest["bindings"]
    verify_forecast_models(job, bindings, work / "preparation")
    identity = {
        k: bindings[k]
        for k in (
            "predictor_architecture",
            "predictor_sha256",
            "decoder_sha256",
            "jepa_normalization_sha256",
        )
    }
    view = {
        "cache_root": str(cache),
        "policy_export_manifest": str(
            task_inputs.policy_export_manifest if task_inputs else bundle / "dataset/manifest.json"
        ),
        "bindings": bindings,
    }
    if input_mode == "forecast_only":
        view["input_mode"] = input_mode
    if a.mode == "smoke":
        view["smoke_subset"] = True
    patches = conditioning_patches(root, input_mode)
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as upstream:
        sys.path.insert(0, str(upstream / "src"))
        from latency_meta_mdp.policy.openpi.training import (
            build_forecast_policy_train_config,
            build_launch_train_config,
            run_openpi_training,
        )

        clean = build_launch_train_config(
            profile=training_profile,
            request=training_request,
            assets_root=(task_inputs.preparation_root if task_inputs else bundle / "preparation")
            / "assets",
            checkpoint_root=work / "conditioned/checkpoints",
            wandb_enabled=not a.wandb_disabled,
            task_parameters=task_inputs.task_parameters if task_inputs else None,
        )
        builder = build_forecast_policy_train_config
        if input_mode == "forecast_only":
            from latency_meta_mdp.policy.openpi.forecast_only import (
                build_forecast_only_train_config,
            )

            builder = build_forecast_only_train_config
        config = builder(
            clean_config=clean,
            clean_checkpoint=clean_dir / "params",
            forecast_identity=identity,
            forecast_view=view,
            experiment_name=job["conditioned_run_name"],
        )
        if "conditioned_source_epochs" in job:
            from latency_meta_mdp.policy.schedule import apply_conditioned_source_budget

            config = apply_conditioned_source_budget(
                config,
                source_count=(
                    task_inputs.preparation["source_count"]
                    if task_inputs
                    else sum(row["frame_count"] for row in manifest["episodes"])
                ),
                source_epochs=job["conditioned_source_epochs"],
            )
        config = dataclasses.replace(config, resume=a.resume)
        steps = tuple(
            config.policy_metadata.get(
                "checkpoint_steps", (config.keep_period, config.num_train_steps)
            )
        )
        if a.mode == "smoke":
            pilot = resolve_sft_schedule(
                profile=training_profile,
                request=dataclasses.replace(training_request, mode="smoke", resume=a.resume),
            )
            config = dataclasses.replace(
                config,
                num_train_steps=pilot.num_train_steps,
                save_interval=pilot.rolling_save_interval,
                keep_period=pilot.milestone_interval,
            )
            steps = pilot.expected_checkpoint_steps
        destination = Path(config.checkpoint_dir)
        report = {
            "level": level,
            "batch_size": config.batch_size,
            "device_count": training_request.device_count,
            "num_train_steps": config.num_train_steps,
            "milestones": steps,
            "clean_checkpoint_step": initialization["step"],
            **(
                {"clean_report_sha256": initialization["clean_report_sha256"]}
                if "clean_report_sha256" in initialization
                else {"initialization": initialization}
            ),
            "forecast_manifest_sha256": sha256_file(cache / "manifest.json"),
            "job_sha256": sha256_file(a.config),
            "training_recipe_sha256": sha256_file(clean_job["training"])
            if clean_job.get("training")
            else None,
            "forecast_identity": identity,
            "openpi_patches": {p.name: sha256_file(p) for p in patches},
            "trainable_scope": config.policy_metadata["trainable_scope"],
        }
        if "conditioned_source_epochs" in job:
            report.update(
                training_budget_unit="source_epochs",
                source_epochs_requested=job["conditioned_source_epochs"],
                training_examples=config.batch_size * config.num_train_steps,
            )
        if a.mode == "smoke":
            report["mode"] = "smoke"
            report["smoke_episode_ids"] = [row["episode_id"] for row in manifest["episodes"]]
        if task_inputs:
            report.update(
                task_id=clean_job["task_id"],
                action_contract=clean_job["action_contract"],
                data_identity=task_inputs.bundle_identity,
            )
        if input_mode == "forecast_only":
            report.update(input_mode=input_mode, **mode_contract)
        if config.fsdp_devices > 1:
            report.update(training_parallelism="fsdp", fsdp_devices=config.fsdp_devices)

        def prepare_publication():
            if input_mode == "forecast_only":
                for step in steps:
                    path = destination / str(step) / "assets/policy_contract.json"
                    path.write_text(
                        json.dumps(
                            {
                                "input_mode": input_mode,
                                **mode_contract,
                                "level": level,
                                "forecast_identity": identity,
                                "action_horizon": 50,
                                "state_dim": 16,
                            },
                            indent=2,
                        )
                    )

        if a.publish_only:
            prepare_publication()
            revision = publish_checkpoints(
                root=destination,
                steps=steps,
                repo_id=job["conditioned_publish_repo"],
                notices=root / "licenses/pi05",
            )
            print(json.dumps({"published_revision": revision}), flush=True)
            return
        import jax
        import numpy as np
        from openpi.training import data_loader

        if not data_only and (
            jax.device_count() != training_request.device_count or jax.process_count() != 1
        ):
            raise ValueError("Unexpected single-host training topology")
        if not data_only and any(d.platform != "gpu" for d in jax.devices()):
            raise ValueError("Training requires CUDA devices")
        data = config.data.create(config.assets_dirs, config.model)
        provider = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
        original = data_loader.create_torch_dataset
        data_loader.create_torch_dataset = lambda *args, **kwargs: provider
        try:
            # Check native shapes and masks before allocating VLA weights.
            check_batch_size = min(2, len(provider)) if data_only else training_request.device_count
            check_config = dataclasses.replace(config, batch_size=check_batch_size, num_workers=0)
            loader = data_loader.create_data_loader(
                check_config,
                shuffle=True,
                num_batches=1,
                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]) if data_only else None,
            )
            obs, actions = next(iter(loader))
            mask = np.asarray(obs.action_loss_mask)
            if (
                set(obs.images) != set(config.model.image_keys)
                or actions.shape != (check_batch_size, 50, 32)
                or mask[..., 7:].any()
                or not mask[:, 0, :7].all()
            ):
                raise ValueError("Forecast image/action-mask contract failed")
            report["coverage"] = provider.coverage()
            print(json.dumps(report), flush=True)
            if a.check_only or data_only:
                report.update(
                    data_checked=True, hardware_checked=not data_only, training_started=False
                )
                target = work / "conditioned/preflight.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2) + "\n")
                return
            contract = destination.parent / (destination.name + ".launch-contract.json")
            # Tuples become lists in persisted JSON.
            comparable = json.loads(json.dumps(report))
            if a.mode == "smoke":
                for key in ("num_train_steps", "milestones", "training_examples"):
                    comparable.pop(key, None)
            if a.resume:
                if json.loads(contract.read_text()) != comparable:
                    raise ValueError("Conditioned resume identity changed")
            else:
                if destination.exists():
                    raise FileExistsError(destination)
                contract.parent.mkdir(parents=True, exist_ok=True)
                with contract.open("x") as f:
                    json.dump(comparable, f, indent=2)
            del loader, obs, actions, mask
            started = time.monotonic()
            run_openpi_training(config=config, openpi_root=upstream)
            report.update(status="completed", seconds=time.monotonic() - started)
            target = work / "conditioned/completion.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2))
        finally:
            data_loader.create_torch_dataset = original
        if a.mode == "smoke":
            return
        prepare_publication()
        revision = publish_checkpoints(
            root=destination,
            steps=steps,
            repo_id=job["conditioned_publish_repo"],
            notices=root / "licenses/pi05",
        )
        print(json.dumps({"published_revision": revision}), flush=True)


if __name__ == "__main__":
    main()
