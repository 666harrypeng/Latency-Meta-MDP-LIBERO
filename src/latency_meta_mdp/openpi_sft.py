"""Registration of level-specific Meta-MDP configs into pinned OpenPI."""

from __future__ import annotations

from typing import Any

from latency_meta_mdp.sft_profile import SFTProfile


def _build_config(profile: SFTProfile, level: int) -> Any:
    import openpi.models.pi0_config as pi0_config
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import DataConfig, LeRobotLiberoDataConfig, TrainConfig

    level_profile = profile.levels[level]
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=profile.action_horizon,
        discrete_state_input=False,
    )
    return TrainConfig(
        name=level_profile.config_name,
        project_name="latency-meta-mdp-robosuite",
        model=model,
        data=LeRobotLiberoDataConfig(
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
            "source_action_dim": profile.source_action_dim,
            "prediction_horizon": profile.action_horizon,
            "execution_horizon": profile.execution_horizon,
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
