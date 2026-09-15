"""Historical 8D SFT entry; use train_structured_pi05 for current state16 data.

The historical L2 dataset/model repositories have been retired.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree
from latency_meta_mdp.sft_asset_lock import load_sft_asset_lock
from latency_meta_mdp.sft_launch import (
    SFTLaunchRequest,
    resolve_sft_schedule,
    stage_level_dataset,
    stage_level_norm_stats,
)
from latency_meta_mdp.sft_profile import load_sft_profile

AssetDownloader = Callable[..., Path]
DatasetDownloader = Callable[..., Path]
TrainBackend = Callable[..., None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"),
    )
    parser.add_argument(
        "--asset-lock",
        type=Path,
        default=Path("configs/data/franka_moving_ball_sft_assets_v1.yaml"),
    )
    parser.add_argument(
        "--data-patch",
        type=Path,
        default=Path("patches/openpi/0001-filter-incomplete-action-chunks.patch"),
    )
    parser.add_argument(
        "--training-patch",
        type=Path,
        default=Path("patches/openpi/0002-save-completed-step-checkpoints.patch"),
    )
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--batch-size", type=int, help="Global batch; formal sample budget is preserved"
    )
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wandb-disabled", action="store_true")
    return parser


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _default_download_asset(*, repo_id: str, revision: str, path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            revision=revision,
        )
    )


def _default_download_dataset(*, repo_id: str, revision: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
    )


def _publish_run_manifest(
    *,
    args: argparse.Namespace,
    request: SFTLaunchRequest,
    config: Any,
    repo_id: str,
    data_revision: str,
    asset_revision: str,
    staged_norm_stats: Path,
    staged_dataset: Path,
    checkpoint_steps: tuple[int, ...],
) -> Path:
    target = args.run_dir.resolve()
    if target.exists():
        raise FileExistsError(f"SFT run artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        provenance = collect_implementation_provenance(args.project_root)
        manifest = {
            "schema_version": 1,
            "format_id": "metamdp_pi05_sft_run_v1",
            "eligible": not provenance.dirty,
            "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
            "implementation_revision": provenance.revision,
            "implementation_source_sha256": provenance.source_sha256,
            "implementation_dirty": provenance.dirty,
            "level": request.level,
            "mode": request.mode,
            "experiment_name": request.experiment_name,
            "resume": request.resume,
            "device_count": request.device_count,
            "config_name": config.name,
            "repo_id": repo_id,
            "data_revision": data_revision,
            "asset_revision": asset_revision,
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "per_device_batch_size": config.batch_size // request.device_count,
            "training_parallelism": "replicated_data_parallel",
            "fsdp_devices": config.fsdp_devices,
            "checkpoint_steps": list(checkpoint_steps),
            "checkpoint_dir": str(Path(config.checkpoint_dir)),
            "dataset_dir": str(staged_dataset),
            "norm_stats_sha256": sha256_file(staged_norm_stats),
            "input_sha256": {
                "profile": sha256_file(args.profile),
                "asset_lock": sha256_file(args.asset_lock),
                "data_patch": sha256_file(args.data_patch),
                "training_patch": sha256_file(args.training_patch),
            },
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def main(
    argv: list[str] | None = None,
    *,
    asset_downloader: AssetDownloader | None = None,
    dataset_downloader: DatasetDownloader | None = None,
    train_backend: TrainBackend | None = None,
    device_count: int | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.run_dir.resolve().exists():
        raise FileExistsError(f"SFT run artifact already exists: {args.run_dir.resolve()}")
    profile = load_sft_profile(args.profile)
    asset_lock = load_sft_asset_lock(args.asset_lock, profile=profile)
    level_asset = asset_lock.level(args.level)
    print(
        f"[sft][L{args.level}] start mode={args.mode}",
        file=sys.stderr,
        flush=True,
    )
    with temporary_patched_openpi_worktree(
        openpi_root=args.openpi_root,
        patch_path=args.data_patch,
        expected_revision=profile.openpi_revision,
        additional_patch_paths=(args.training_patch,),
    ) as patched_root:
        sys.path.insert(0, str(patched_root / "src"))
        try:
            if device_count is None:
                import jax

                observed_devices = jax.device_count()
            else:
                observed_devices = device_count
            request = SFTLaunchRequest(
                level=args.level,
                experiment_name=args.experiment_name,
                mode=args.mode,
                resume=args.resume,
                device_count=observed_devices,
                batch_size_override=args.batch_size,
            )
            staged_dataset = stage_level_dataset(
                profile=profile,
                asset_lock=asset_lock,
                level=args.level,
                dataset_root=args.dataset_root,
                download_dataset=dataset_downloader or _default_download_dataset,
            )
            staged = stage_level_norm_stats(
                profile=profile,
                asset_lock=asset_lock,
                level=args.level,
                assets_root=args.assets_root,
                download_asset=asset_downloader or _default_download_asset,
            )
            from latency_meta_mdp.openpi_sft import (
                build_level_train_config,
                run_openpi_training,
            )

            config = build_level_train_config(
                profile=profile,
                request=request,
                assets_root=args.assets_root,
                checkpoint_root=args.checkpoint_root,
                wandb_enabled=not args.wandb_disabled,
            )
            backend = train_backend or run_openpi_training
            previous_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
            os.environ["HF_LEROBOT_HOME"] = str(args.dataset_root.resolve())
            try:
                backend(config=config, openpi_root=patched_root)
            finally:
                if previous_lerobot_home is None:
                    os.environ.pop("HF_LEROBOT_HOME", None)
                else:
                    os.environ["HF_LEROBOT_HOME"] = previous_lerobot_home
            checkpoint_root = Path(config.checkpoint_dir)
            checkpoint_steps = tuple(
                sorted(
                    int(path.name)
                    for path in checkpoint_root.iterdir()
                    if path.is_dir() and path.name.isdigit()
                )
            )
            schedule = resolve_sft_schedule(profile=profile, request=request)
            if checkpoint_steps != schedule.expected_checkpoint_steps:
                raise ValueError("SFT checkpoint inventory does not match the launch contract")
        finally:
            sys.path.remove(str(patched_root / "src"))
    manifest = _publish_run_manifest(
        args=args,
        request=request,
        config=config,
        repo_id=level_asset.repo_id,
        data_revision=level_asset.data_revision,
        asset_revision=level_asset.asset_revision,
        staged_norm_stats=staged,
        staged_dataset=staged_dataset,
        checkpoint_steps=checkpoint_steps,
    )
    print(
        f"[sft][L{args.level}] done checkpoints="
        + ",".join(str(step) for step in checkpoint_steps),
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
