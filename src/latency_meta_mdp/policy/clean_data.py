"""Resolve clean-policy data independently of the shared training lifecycle."""

import json
from dataclasses import dataclass
from pathlib import Path

from latency_meta_mdp.data.bundles import verify_bundle_files
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.policy.config import resolve_policy_profile


@dataclass(frozen=True)
class CleanTrainingInputs:
    dataset_root: Path
    preparation_root: Path
    preparation: dict
    profile: object
    bundle_identity: str
    purpose: str
    task_parameters: dict | None


def resolve_clean_inputs(job, work_dir):
    root = repository_root()
    if job["schema_version"] == 1:
        from huggingface_hub import snapshot_download

        bundle = work_dir / "bundles" / job["dataset_revision"]
        print("Preparing public training bundle", flush=True)
        snapshot_download(
            repo_id=job["dataset_repo"],
            repo_type="dataset",
            revision=job["dataset_revision"],
            local_dir=bundle,
            token=False,
        )
        identity = job["dataset_revision"]
    else:
        source = job["dataset"]
        if source["kind"] == "local":
            bundle = source["path"]
        else:
            from huggingface_hub import snapshot_download

            bundle = (
                work_dir / "bundles" / source["repo_id"].replace("/", "--") / source["revision"]
            )
            snapshot_download(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                local_dir=bundle,
                token=False,
            )
        identity = sha256_file(bundle / "bundle.json")
        if identity != source["manifest_sha256"]:
            raise ValueError("training bundle manifest differs from job")
    info = verify_bundle_files(bundle)
    dataset_root = bundle / info["dataset_root"]
    preparation_root = bundle / info["preparation_root"]
    prep = json.loads((preparation_root / "preparation.json").read_text())
    task_parameters = None
    if job["schema_version"] == 1:
        if info["schema_version"] != 1 or info["level"] != job["level"]:
            raise ValueError("Training bundle level mismatch")
        profile = resolve_policy_profile(job)
        if prep["level"] != job["level"]:
            raise ValueError("level preparation identity mismatch")
        manifest_path = dataset_root / "manifest.json"
        if sha256_file(manifest_path) != prep["dataset_manifest_sha256"]:
            raise ValueError("prepared export manifest changed")
        manifest = json.loads(manifest_path.read_text())
        level = next(x for x in manifest["datasets"] if x["level"] == job["level"])
        if (
            level["repo_id"] != profile.levels[job["level"]].repo_id
            or level["frame_count"] != prep["source_count"]
        ):
            raise ValueError("level dataset identity/source count mismatch")
        nested = dataset_root / level["dataset_manifest"]
        if sha256_file(nested) != level["dataset_manifest_sha256"]:
            raise ValueError("level dataset manifest hash mismatch")
    else:
        if info["schema_version"] != 2 or any(
            info[k] != job[k] for k in ("task_id", "variant", "action_contract")
        ):
            raise ValueError("training bundle task/controller mismatch")
        nested = (bundle / info["dataset_manifest"]).resolve()
        if (
            not nested.is_relative_to(dataset_root.resolve())
            or sha256_file(nested) != info["dataset_manifest_sha256"]
        ):
            raise ValueError("task dataset manifest mismatch")
        if sha256_file(preparation_root / "preparation.json") != info["preparation_sha256"]:
            raise ValueError("task preparation manifest mismatch")
        metadata = json.loads(nested.read_text())
        if (
            metadata["split"] != "train"
            or metadata["task_id"] != job["task_id"]
            or metadata["variant"] != job["variant"]
            or metadata["frame_count"] != prep["source_count"]
            or metadata["frame_count"] != info["source_count"]
            or metadata["state_dim"] != 16
            or metadata["action_dim"] != 7
            or metadata["action_contract"] != job["action_contract"]
            or metadata["repo_id"] != info["repo_id"]
            or prep["task_id"] != job["task_id"]
            or prep["action_contract"] != job["action_contract"]
            or prep["purpose"] != metadata["purpose"]
            or prep["purpose"] != info["purpose"]
            or prep["dataset_manifest_sha256"] != info["dataset_manifest_sha256"]
            or metadata["action_target_contract"] != "masked_h50_real_actions_v1"
        ):
            raise ValueError("task data/preparation contracts disagree")
        profile = resolve_policy_profile(job, source_count=prep["source_count"])
        if prep["openpi_revision"] != profile.openpi_revision:
            raise ValueError("preparation OpenPI revision differs from the model profile")
        task_parameters = dict(
            config_name=f"pi05_{job['task_id']}_{job['variant']}_state16_h50",
            repo_id=info["repo_id"],
            task_id=job["task_id"],
            action_contract_id=job["action_contract"],
            extra_metadata={"variant": job["variant"]},
        )
    if prep["profile_sha256"] != sha256_file(job["profile"]):
        raise ValueError("preparation/model profile identity mismatch")
    norm = (preparation_root / prep["norm_stats"]).resolve()
    if (
        not norm.is_relative_to(preparation_root.resolve())
        or sha256_file(norm) != prep["norm_stats_sha256"]
    ):
        raise ValueError("normalization identity mismatch")
    expected_patches = {
        "0001-filter-incomplete-action-chunks.patch",
        "0002-save-completed-step-checkpoints.patch",
        "0003-mask-action-tails.patch",
    }
    if set(prep["patches"]) != expected_patches:
        raise ValueError("prepared OpenPI patch set differs from the clean training contract")
    for name, digest in prep["patches"].items():
        if sha256_file(root / "patches/openpi" / name) != digest:
            raise ValueError("OpenPI patches differ from preparation")
    return CleanTrainingInputs(
        dataset_root,
        preparation_root,
        prep,
        profile,
        identity,
        "training_source" if job["schema_version"] == 1 else info["purpose"],
        task_parameters,
    )


def require_training_source(inputs, *, checking):
    if not checking and inputs.purpose != "training_source":
        raise ValueError(
            "development_smoke bundles are check-only; prepare a training_source corpus for SFT"
        )
