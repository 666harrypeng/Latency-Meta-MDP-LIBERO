"""Validate a local structured policy export and prepare exact train-only norm assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
from latency_meta_mdp.policy.profile import load_sft_profile


def prepare(
    *, dataset_manifest: Path, profile_path: Path, level: int, output_dir: Path, openpi_root: Path
) -> Path:
    profile = load_sft_profile(profile_path)
    if not profile.masked_action_tails or level not in (1, 2, 3):
        raise ValueError("preparation requires a structured state-aware level")
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"preparation output already exists: {output}")
    manifest_path = dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format_id") != "metamdp_lerobot_structured_train_v1"
        or manifest.get("split") != "train"
        or manifest.get("sft_profile_sha256") != sha256_file(profile_path)
    ):
        raise ValueError("dataset must be the profile-bound structured train export")
    matches = [row for row in manifest["datasets"] if row["level"] == level]
    if len(matches) != 1 or matches[0]["repo_id"] != profile.levels[level].repo_id:
        raise ValueError("dataset level/repo identity mismatch")
    row = matches[0]
    nested_path = (manifest_path.parent / row["dataset_manifest"]).resolve()
    if not nested_path.is_relative_to(manifest_path.parent):
        raise ValueError("dataset manifest escapes the export")
    if sha256_file(nested_path) != row["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    nested = json.loads(nested_path.read_text())
    if (
        nested["action_target_contract"] != "masked_h50_real_actions_v1"
        or nested["state_dim"] != 16
        or nested["drop_n_last_frames"] != 0
        or nested["frame_count"] != nested["valid_action_chunk_source_count"]
        or nested["frame_count"] != row["frame_count"]
    ):
        raise ValueError("dataset does not retain every real current16 source")
    if any(
        e["logical_master_task_index"] not in manifest["train_master_task_indices"]
        for e in nested["episodes"]
    ):
        raise ValueError("dataset contains a non-train master")
    for name, digest in nested["artifacts"].items():
        path = (nested_path.parent / name).resolve()
        if not path.is_relative_to(nested_path.parent) or sha256_file(path) != digest:
            raise ValueError("dataset artifact hash mismatch")

    project_root = repository_root()
    patches = tuple(
        project_root / "patches/openpi" / name
        for name in (
            "0001-filter-incomplete-action-chunks.patch",
            "0002-save-completed-step-checkpoints.patch",
            "0003-mask-action-tails.patch",
        )
    )
    if sha256_file(patches[0]) != profile.openpi_patch_sha256:
        raise ValueError("OpenPI data patch does not match profile")
    # Set before importing LeRobot: the pinned loader captures this environment
    # variable as a module constant.
    previous_home = os.environ.get("HF_LEROBOT_HOME")
    os.environ["HF_LEROBOT_HOME"] = str(manifest_path.parent)
    output.mkdir(parents=True)
    try:
        with temporary_patched_openpi_copy(
            openpi_root=openpi_root, patch_paths=patches, expected_revision=profile.openpi_revision
        ) as copied:
            sys.path.insert(0, str(copied / "src"))
            try:
                import jax
                from openpi.shared import normalize
                from openpi.training.data_loader import (
                    create_torch_data_loader,
                    create_torch_dataset,
                )

                from latency_meta_mdp.policy.openpi.training import (
                    _build_config,
                    compute_openpi_norm_stats,
                )

                repo_id = row["repo_id"]
                norm_path = (
                    output
                    / "assets"
                    / profile.levels[level].config_name
                    / repo_id
                    / "norm_stats.json"
                )
                computation = compute_openpi_norm_stats(
                    level=level,
                    repo_id=repo_id,
                    dataset_root=manifest_path.parent,
                    expected_source_count=row["frame_count"],
                    output_path=norm_path,
                    profile=profile,
                    openpi_root=copied,
                )
                config = _build_config(profile, level)
                data = config.data.create(output / "assets" / config.name, config.model)
                raw = create_torch_dataset(data, 50, config.model)
                if len(raw) != row["frame_count"]:
                    raise ValueError("OpenPI loader source count mismatch")
                cursor = 0
                for episode in nested["episodes"]:
                    length = episode["frame_count"]
                    for offset in {0, max(0, length - 50), length - 1}:
                        pad = np.asarray(raw[cursor + offset]["actions_is_pad"])
                        if not np.array_equal(~pad, np.arange(50) < min(50, length - offset)):
                            raise ValueError("OpenPI episode boundary mask mismatch")
                    cursor += length
                loader = create_torch_data_loader(
                    data,
                    config.model,
                    50,
                    min(2, len(raw)),
                    num_batches=1,
                    num_workers=0,
                    # This two-sample interface check is independent of training topology.
                    sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
                )
                observation, actions = next(iter(loader))
                mask = np.asarray(observation.action_loss_mask)
                if mask.shape != actions.shape or not mask[:, 0, :7].all() or mask[..., 7:].any():
                    raise ValueError("training batch lost the real-action mask")
                stats = normalize.load(norm_path.parent)
                report = {
                    "format_id": "structured_pi05_preparation_v1",
                    "level": level,
                    "dataset_manifest_sha256": sha256_file(manifest_path),
                    "profile_sha256": sha256_file(profile_path),
                    "source_manifest_sha256": manifest["source_manifest_sha256"],
                    "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "openpi_revision": profile.openpi_revision,
                    "patches": {p.name: sha256_file(p) for p in patches},
                    "norm_stats": norm_path.relative_to(output).as_posix(),
                    "norm_stats_sha256": sha256_file(norm_path),
                    "source_count": computation.source_count,
                    "episode_boundary_masks_checked": len(nested["episodes"]),
                    "normalization_counts_each_real_frame_once": True,
                    "state_stats_dim": len(stats["state"].mean),
                    "action_stats_dim": len(stats["actions"].mean),
                    "batch_action_shape": list(actions.shape),
                    "batch_mask_shape": list(mask.shape),
                    "state_tokens_enabled": config.model.discrete_state_input,
                    "token_count_range": [
                        int(np.asarray(observation.tokenized_prompt_mask).sum(-1).min()),
                        int(np.asarray(observation.tokenized_prompt_mask).sum(-1).max()),
                    ],
                    "training_started": False,
                    "closed_loop_proven": False,
                }
                path = output / "preparation.json"
                with path.open("x") as handle:
                    json.dump(report, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                return path
            finally:
                sys.path.remove(str(copied / "src"))
    finally:
        if previous_home is None:
            os.environ.pop("HF_LEROBOT_HOME", None)
        else:
            os.environ["HF_LEROBOT_HOME"] = previous_home


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, default=Path("configs/contracts/policy/pi05_state16_h50.yaml")
    )
    parser.add_argument("--openpi-root", type=Path, default=Path("third_party/openpi"))
    args = parser.parse_args()
    print(
        prepare(
            dataset_manifest=args.dataset_manifest,
            profile_path=args.profile,
            level=args.level,
            output_dir=args.output_dir,
            openpi_root=args.openpi_root,
        )
    )


if __name__ == "__main__":
    main()
