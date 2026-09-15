"""Registration of level-specific Meta-MDP configs into pinned OpenPI."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.policy.norm_stats import NormStatsComputation
from latency_meta_mdp.policy.profile import SFTProfile
from latency_meta_mdp.policy.schedule import SFTLaunchRequest, resolve_sft_schedule


def _build_config(profile: SFTProfile, level: int) -> Any:
    if profile.fsdp_devices != 1:
        raise ValueError("SFT currently requires replicated data parallelism (fsdp_devices=1)")
    import openpi.models.pi0_config as pi0_config
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import DataConfig, LeRobotLiberoDataConfig, TrainConfig

    level_profile = profile.levels[level]
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=profile.action_horizon,
        discrete_state_input=profile.discrete_state_input,
        **({"active_action_dim": profile.source_action_dim} if profile.masked_action_tails else {}),
    )
    data_factory = LeRobotLiberoDataConfig
    if profile.masked_action_tails:
        from latency_meta_mdp.policy.openpi.data import StructuredPolicyDataConfig

        data_factory = StructuredPolicyDataConfig
    return TrainConfig(
        name=level_profile.config_name,
        project_name="latency-meta-mdp-robosuite",
        model=model,
        data=data_factory(
            repo_id=level_profile.repo_id,
            base_config=DataConfig(
                prompt_from_task=True,
                drop_n_last_frames=profile.drop_n_last_frames,
            ),
            extra_delta_transform=profile.extra_delta_transform,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(profile.base_checkpoint),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=profile.warmup_steps,
            peak_lr=profile.peak_learning_rate,
            decay_steps=profile.num_train_steps,
            decay_lr=profile.decay_learning_rate,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=profile.ema_decay,
        batch_size=profile.batch_size,
        num_workers=profile.num_workers,
        num_train_steps=profile.num_train_steps,
        log_interval=profile.log_interval,
        save_interval=profile.save_interval,
        keep_period=profile.keep_period,
        fsdp_devices=profile.fsdp_devices,
        checkpoint_base_dir="outputs/sft/checkpoints",
        assets_base_dir="outputs/sft/assets",
        wandb_enabled=True,
        policy_metadata={
            "profile_id": profile.profile_id,
            "task_id": "dynamic_grasp_lift",
            "level": level,
            "fps": profile.fps,
            "state_dim": profile.state_dim,
            "discrete_state_input": profile.discrete_state_input,
            "masked_action_tails": profile.masked_action_tails,
            "source_action_dim": profile.source_action_dim,
            "temporal_contract_id": profile.temporal_contract.contract_id,
            "prediction_horizon": profile.action_horizon,
            "launch_trigger_horizon": profile.launch_trigger_horizon,
            "action_contract_id": "panda_osc_pose_delta_v1",
        },
    )


def register_sft_configs(profile: SFTProfile) -> tuple[str, ...]:
    """Register the validated L1-L3 configs into the active OpenPI process."""

    from openpi.training.config import _CONFIGS_DICT

    names: list[str] = []
    for level in (1, 2, 3):
        config = _build_config(profile, level)
        _CONFIGS_DICT[config.name] = config
        names.append(config.name)
    return tuple(names)


def build_forecast_policy_train_config(
    *,
    clean_config: Any,
    clean_checkpoint: Path,
    forecast_identity: dict,
    experiment_name: str,
    forecast_view: dict | None = None,
) -> Any:
    """Native decoded-future post-training; source provider attaches explicit forecasts."""
    import re

    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from latency_meta_mdp.policy.openpi.data import StructuredPolicyDataConfig
    from latency_meta_mdp.policy.openpi.forecast import ForecastPolicyDataConfig

    model = clean_config.model
    if (
        not isinstance(clean_config.data, StructuredPolicyDataConfig)
        or clean_config.fsdp_devices != 1
        or not model.pi05
        or not model.discrete_state_input
        or model.active_action_dim != 7
        or model.action_horizon != 50
        or clean_config.policy_metadata.get("state_dim") != 16
        or any(
            getattr(model, key, False)
            for key in ("use_return_belief", "use_return_belief_prefix", "use_rtc_forecast")
        )
    ):
        raise ValueError("forecast SFT requires the matched clean state-aware checkpoint config")
    expected = {
        "predictor_architecture",
        "predictor_sha256",
        "decoder_sha256",
        "jepa_normalization_sha256",
    }
    if set(forecast_identity) != expected or not isinstance(
        forecast_identity["predictor_architecture"], str
    ):
        raise ValueError("forecast checkpoint identity fields are incomplete")
    if any(
        not isinstance(forecast_identity[k], str)
        or re.fullmatch(r"[0-9a-f]{64}", forecast_identity[k]) is None
        for k in expected - {"predictor_architecture"}
    ):
        raise ValueError("forecast identity requires SHA256 digests")
    if forecast_view is not None and any(
        forecast_view.get("bindings", {}).get(k) != v for k, v in forecast_identity.items()
    ):
        raise ValueError("forecast dataset and policy checkpoint bindings disagree")
    schedule_kwargs = {}
    schedule_metadata = {"forecast_training_budget_resolved": False}
    if forecast_view is not None:
        import json
        from math import gcd

        manifest = json.loads((Path(forecast_view["cache_root"]) / "manifest.json").read_text())
        if (
            manifest.get("complete") is not True
            or manifest.get("bindings") != forecast_view["bindings"]
        ):
            raise ValueError("forecast training budget requires the matching complete cache")
        source_count = sum(e["frame_count"] for e in manifest["episodes"])
        batch = clean_config.batch_size
        multiple = batch // gcd(batch, 20)
        per_query = ((source_count + multiple - 1) // multiple) * multiple
        steps_per_epoch = 20 * per_query // batch
        steps = 2 * steps_per_epoch
        if steps_per_epoch < 1:
            raise ValueError("forecast source inventory cannot form a training epoch")
        schedule_kwargs = {
            "num_train_steps": steps,
            "lr_schedule": dataclasses.replace(
                clean_config.lr_schedule, warmup_steps=max(1, steps // 20), decay_steps=steps
            ),
            "keep_period": steps_per_epoch,
        }
        schedule_metadata = {
            "forecast_training_budget_resolved": True,
            "balanced_pair_epochs": 2,
            "real_action_sources": source_count,
            "balanced_examples_per_epoch": 20 * per_query,
            "training_examples": steps * batch,
        }
    return dataclasses.replace(
        clean_config,
        **schedule_kwargs,
        name=f"{clean_config.name}_rtc_forecast",
        exp_name=experiment_name,
        model=dataclasses.replace(model, use_rtc_forecast=True, max_token_len=256),
        data=ForecastPolicyDataConfig(
            forecast_policy_view=forecast_view,
            **{
                f.name: getattr(clean_config.data, f.name)
                for f in dataclasses.fields(StructuredPolicyDataConfig)
                if f.name != "assets"
            },
            assets=AssetsConfig(
                assets_dir=clean_config.data.assets.assets_dir or str(clean_config.assets_dirs),
                asset_id=clean_config.data.assets.asset_id or clean_config.data.repo_id,
            ),
        ),
        weight_loader=CheckpointWeightLoader(str(clean_checkpoint)),
        freeze_filter=nnx.Any(
            PathRegex("PaliGemma/img/.*"), PathRegex("PaliGemma/llm/embedder/.*")
        ),
        policy_metadata={
            **clean_config.policy_metadata,
            **schedule_metadata,
            "conditioning": "native_rtc_forecast_rgb_v1",
            "policy_alignment": "observation_time",
            "protocol_id": "rtc_observation_time_h50_v1",
            "forecast_identity": dict(forecast_identity),
            "source_tail_policy": "missing_forecast_preserve_action_supervision_v1",
            "trainable_scope": "native_vlm_action_expert",
        },
    )


def build_level_train_config(
    *,
    profile: SFTProfile,
    request: SFTLaunchRequest,
    assets_root: Path,
    checkpoint_root: Path,
    wandb_enabled: bool,
) -> Any:
    """Build one explicit smoke or formal TrainConfig from the canonical profile."""

    base = _build_config(profile, request.level)
    schedule = resolve_sft_schedule(profile=profile, request=request)
    return dataclasses.replace(
        base,
        exp_name=request.experiment_name,
        assets_base_dir=str(assets_root.resolve()),
        checkpoint_base_dir=str(checkpoint_root.resolve()),
        batch_size=schedule.batch_size,
        num_train_steps=schedule.num_train_steps,
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            warmup_steps=schedule.warmup_steps,
            decay_steps=schedule.decay_steps,
        ),
        save_interval=schedule.rolling_save_interval,
        keep_period=schedule.milestone_interval,
        overwrite=False,
        resume=request.resume,
        wandb_enabled=wandb_enabled,
        policy_metadata={
            **base.policy_metadata,
            "training_parallelism": "replicated_data_parallel",
            "training_device_count": request.device_count,
            "per_device_batch_size": schedule.batch_size // request.device_count,
            "reference_training_examples": profile.batch_size * profile.num_train_steps,
            "training_examples": schedule.batch_size * schedule.num_train_steps,
        },
    )


def _load_norm_stats_script(openpi_root: Path) -> Any:
    script_path = openpi_root / "scripts/compute_norm_stats.py"
    spec = importlib.util.spec_from_file_location(
        "_metamdp_openpi_compute_norm_stats",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenPI norm-stat script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_script(openpi_root: Path) -> Any:
    script_path = openpi_root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location(
        "_metamdp_openpi_train",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenPI train script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_openpi_training(*, config: Any, openpi_root: Path) -> None:
    """Run pinned OpenPI training and return only after async checkpoints flush."""

    import jax

    if config.fsdp_devices != 1:
        raise ValueError("SFT currently requires replicated data parallelism (fsdp_devices=1)")
    if jax.process_count() != 1:
        raise ValueError("the SFT launcher supports one process on one multi-GPU host")
    expected = config.policy_metadata.get("training_device_count", jax.device_count())
    if jax.device_count() != expected or config.batch_size % expected:
        raise ValueError("visible devices/global batch differ from the SFT launch contract")
    _load_train_script(openpi_root.resolve()).main(config)


def compute_openpi_norm_stats(
    *,
    level: int,
    repo_id: str,
    dataset_root: Path,
    expected_source_count: int,
    output_path: Path,
    profile: SFTProfile,
    openpi_root: Path,
) -> NormStatsComputation:
    """Compute exact state/action stats through the patched OpenPI data path."""

    if level not in (1, 2, 3) or profile.levels[level].repo_id != repo_id:
        raise ValueError("norm-stat level and repo do not match the SFT profile")
    if output_path.name != "norm_stats.json" or output_path.exists():
        raise ValueError("norm-stat output path must be a new norm_stats.json")
    if expected_source_count <= 0:
        raise ValueError("expected_source_count must be positive")

    previous_home = os.environ.get("HF_LEROBOT_HOME")
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root.resolve())
    try:
        import openpi.shared.normalize as normalize
        from openpi.training.config import _CONFIGS_DICT

        register_sft_configs(profile)
        config = _CONFIGS_DICT[profile.levels[level].config_name]
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_script = _load_norm_stats_script(openpi_root.resolve())
        loader, _ = norm_script.create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            min(config.batch_size, expected_source_count),
            config.model,
            0,
        )
        stats = {
            "state": normalize.RunningStats(),
            "actions": normalize.RunningStats(),
        }
        source_count = 0
        for batch in loader:
            state = np.asarray(batch["state"])
            actions = np.asarray(batch["actions"])
            if state.ndim != 2 or actions.ndim != 3 or state.shape[0] != actions.shape[0]:
                raise ValueError("OpenPI norm-stat batch has invalid state/action shapes")
            source_count += state.shape[0]
            stats["state"].update(state)
            if profile.masked_action_tails:
                mask = np.asarray(batch["action_loss_mask"])
                if mask.shape != actions.shape or not mask[:, 0].all():
                    raise ValueError(
                        "structured norm stats require a real first action at every source"
                    )
                # Every recorded frame is a source: count each real action once,
                # independent of H50 overlap or the amount of terminal padding.
                stats["actions"].update(actions[:, 0])
            else:
                stats["actions"].update(actions)
        if source_count != expected_source_count:
            raise ValueError("OpenPI norm-stat loader did not cover every certified source")
        normalize.save(
            output_path.parent,
            {name: running.get_statistics() for name, running in stats.items()},
        )
    finally:
        if previous_home is None:
            os.environ.pop("HF_LEROBOT_HOME", None)
        else:
            os.environ["HF_LEROBOT_HOME"] = previous_home
    return NormStatsComputation(source_count=source_count)


def probe_sft_pilot_level(
    *,
    level: int,
    repo_id: str,
    lerobot_home: Path,
    profile: SFTProfile,
    openpi_root: Path,
    expected_episode_count: int,
    expected_frame_count: int,
    expected_source_count: int,
) -> dict[str, Any]:
    """Load one pilot dataset through metadata, norm-stat, and SFT paths."""

    os.environ["HF_LEROBOT_HOME"] = str(lerobot_home.resolve())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi.training.config import _CONFIGS_DICT
    from openpi.training.data_loader import create_torch_data_loader, create_torch_dataset

    register_sft_configs(profile)
    config = _CONFIGS_DICT[profile.levels[level].config_name]
    if config.data.repo_id != repo_id:
        raise ValueError("OpenPI config and certification repo id disagree")
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    no_action_padding = all(
        not bool(np.asarray(raw_dataset[index]["actions_is_pad"]).any())
        for index in range(len(raw_dataset))
    )

    norm_script = _load_norm_stats_script(openpi_root.resolve())
    norm_loader, _ = norm_script.create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        0,
    )
    norm_batch_sizes = [int(np.asarray(batch["state"]).shape[0]) for batch in norm_loader]

    train_loader = create_torch_data_loader(
        data_config,
        config.model,
        config.model.action_horizon,
        config.batch_size,
        skip_norm_stats=True,
        num_batches=1,
        num_workers=0,
    )
    observation, actions = next(iter(train_loader))
    metadata = LeRobotDatasetMetadata(repo_id)
    result = {
        "metadata_version": str(metadata._version),
        "metadata_fps": metadata.fps,
        "episode_count": metadata.total_episodes,
        "frame_count": metadata.total_frames,
        "source_count": len(raw_dataset),
        "norm_source_count": sum(norm_batch_sizes),
        "norm_batch_sizes": norm_batch_sizes,
        "no_action_padding": no_action_padding,
        "train_state_shape": list(observation.state.shape),
        "train_action_shape": list(actions.shape),
    }
    expected = (
        expected_episode_count,
        expected_frame_count,
        expected_source_count,
    )
    observed = (
        result["episode_count"],
        result["frame_count"],
        result["source_count"],
    )
    if observed != expected:
        raise ValueError("OpenPI pilot metadata does not match the derived manifest")
    return result
