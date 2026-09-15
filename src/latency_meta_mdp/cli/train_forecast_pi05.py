"""Train the native four-image forecast policy from a matched local clean milestone."""

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
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb-disabled", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
    a = p.parse_args()
    root = Path.cwd()
    work = a.work_dir.resolve()
    from latency_meta_mdp.artifacts import sha256_file
    from latency_meta_mdp.forecast_delivery import load_forecast_job
    from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_copy
    from latency_meta_mdp.sft_delivery import publish_checkpoints, verify_bundle_files
    from latency_meta_mdp.sft_launch import SFTLaunchRequest, resolve_sft_schedule
    from latency_meta_mdp.sft_profile import load_sft_profile

    job = load_forecast_job(a.config, project_root=root)
    clean_job = job["clean"]
    profile = load_sft_profile(clean_job["profile"])
    level = clean_job["level"]
    bundle = work / "clean/bundles" / clean_job["dataset_revision"]
    verify_bundle_files(bundle)
    os.environ["HF_LEROBOT_HOME"] = str(bundle / "dataset")
    request = SFTLaunchRequest(
        level,
        clean_job["run_name"],
        "formal",
        False,
        clean_job["device_count"],
        clean_job["batch_size"],
    )
    schedule = resolve_sft_schedule(profile=profile, request=request)
    clean_dir = (
        work
        / "clean/checkpoints"
        / profile.levels[level].config_name
        / clean_job["run_name"]
        / str(schedule.num_train_steps)
    )
    if not (clean_dir / "params").is_dir():
        raise ValueError("Matched final clean checkpoint is missing")
    cache = work / "preparation/forecasts"
    manifest = json.loads((cache / "manifest.json").read_text())
    bindings = manifest["bindings"]
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
        "policy_export_manifest": str(bundle / "dataset/manifest.json"),
        "bindings": bindings,
    }
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=tuple(sorted((root / "patches/openpi").glob("000[1-9]-*.patch"))),
        expected_revision=profile.openpi_revision,
    ) as upstream:
        sys.path.insert(0, str(upstream / "src"))
        from latency_meta_mdp.openpi_sft import (
            build_forecast_policy_train_config,
            build_level_train_config,
            run_openpi_training,
        )

        clean = build_level_train_config(
            profile=profile,
            request=request,
            assets_root=bundle / "preparation/assets",
            checkpoint_root=work / "conditioned/checkpoints",
            wandb_enabled=not a.wandb_disabled,
        )
        config = build_forecast_policy_train_config(
            clean_config=clean,
            clean_checkpoint=clean_dir / "params",
            forecast_identity=identity,
            forecast_view=view,
            experiment_name=job["conditioned_run_name"],
        )
        config = dataclasses.replace(config, resume=a.resume)
        steps = (config.keep_period, config.num_train_steps)
        destination = Path(config.checkpoint_dir)
        report = {
            "level": level,
            "batch_size": config.batch_size,
            "device_count": clean_job["device_count"],
            "num_train_steps": config.num_train_steps,
            "milestones": steps,
            "clean_checkpoint_step": schedule.num_train_steps,
            "clean_report_sha256": sha256_file(
                clean_dir.parent / f"verified_run-step{schedule.num_train_steps}.json"
            ),
            "forecast_manifest_sha256": sha256_file(cache / "manifest.json"),
            "job_sha256": sha256_file(a.config),
            "forecast_identity": identity,
            "openpi_patches": {
                p.name: sha256_file(p)
                for p in sorted((root / "patches/openpi").glob("000[1-9]-*.patch"))
            },
            "trainable_scope": config.policy_metadata["trainable_scope"],
        }
        if a.publish_only:
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

        if jax.device_count() != clean_job["device_count"] or jax.process_count() != 1:
            raise ValueError("Unexpected replicated-device topology")
        if any(d.platform != "gpu" for d in jax.devices()):
            raise ValueError("Training requires CUDA devices")
        data = config.data.create(config.assets_dirs, config.model)
        provider = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
        original = data_loader.create_torch_dataset
        data_loader.create_torch_dataset = lambda *args, **kwargs: provider
        try:
            # Check the actual normalized four-image batch before allocating VLA weights.
            check_config = dataclasses.replace(
                config, batch_size=clean_job["device_count"], num_workers=0
            )
            loader = data_loader.create_data_loader(check_config, shuffle=True, num_batches=1)
            obs, actions = next(iter(loader))
            mask = np.asarray(obs.action_loss_mask)
            if (
                set(obs.images) != set(config.model.image_keys)
                or actions.shape != (clean_job["device_count"], 50, 32)
                or mask[..., 7:].any()
                or not mask[:, 0, :7].all()
            ):
                raise ValueError("Forecast image/action-mask contract failed")
            report["coverage"] = provider.coverage()
            print(json.dumps(report), flush=True)
            if a.check_only:
                return
            contract = destination.parent / (destination.name + ".launch-contract.json")
            # Tuples become lists in persisted JSON.
            comparable = json.loads(json.dumps(report))
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
        revision = publish_checkpoints(
            root=destination,
            steps=steps,
            repo_id=job["conditioned_publish_repo"],
            notices=root / "licenses/pi05",
        )
        print(json.dumps({"published_revision": revision}), flush=True)


if __name__ == "__main__":
    main()
