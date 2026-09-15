"""Historical return-mixture adapter and prefix policy configurations."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


def build_return_policy_train_config(
    *,
    clean_config: Any,
    clean_checkpoint: Path,
    view_spec: dict,
    experiment_name: str,
    adapter_only: bool = True,
) -> Any:
    """Prepare matched return-mixture controls after a clean checkpoint is available.

    The default trains only the policy-owned adapter, retaining the native model
    for the first decision-usefulness comparison. This constructs configuration;
    it neither downloads weights nor starts SFT. clean_checkpoint names params/.
    """
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from latency_meta_mdp.legacy.policy.openpi_belief_adapter import (
        NativePolicyWithReturnBeliefLoader,
    )
    from latency_meta_mdp.legacy.policy.openpi_belief_data import (
        KnownDelayOracleDataConfig,
        NoFutureControlDataConfig,
        ReturnBeliefDataConfig,
    )

    if clean_config.fsdp_devices != 1:
        raise ValueError("return-policy SFT requires replicated data parallelism")
    model = clean_config.model
    if (
        not model.pi05
        or not model.discrete_state_input
        or model.active_action_dim != 7
        or model.action_horizon != 50
        or clean_config.policy_metadata.get("state_dim") != 16
        or not clean_config.policy_metadata.get("masked_action_tails")
    ):
        raise ValueError(
            "return policy initialization requires the matched clean state-aware policy"
        )
    mode = view_spec.get("mode")
    if mode not in {"predicted_mixture", "gt_mixture", "known_delay_oracle", "no_future_control"}:
        raise ValueError("return-policy control mode is invalid")
    factory = {
        "known_delay_oracle": KnownDelayOracleDataConfig,
        "no_future_control": NoFutureControlDataConfig,
    }.get(mode, ReturnBeliefDataConfig)
    return dataclasses.replace(
        clean_config,
        name=f"{clean_config.name}_{mode}",
        exp_name=experiment_name,
        model=dataclasses.replace(model, use_return_belief=True),
        data=factory(
            repo_id=clean_config.data.repo_id,
            base_config=clean_config.data.base_config,
            assets=AssetsConfig(
                assets_dir=str(clean_config.assets_dirs), asset_id=clean_config.data.repo_id
            ),
            return_policy_view=dict(view_spec),
        ),
        weight_loader=NativePolicyWithReturnBeliefLoader(
            CheckpointWeightLoader(str(clean_checkpoint))
        ),
        freeze_filter=nnx.Not(PathRegex("return_belief_adapter/.*"))
        if adapter_only
        else clean_config.freeze_filter,
        policy_metadata={
            **clean_config.policy_metadata,
            "return_policy_mode": mode,
            "adapter_only": adapter_only,
            "privileged_oracle": mode == "known_delay_oracle",
            "initialization": "matched_clean_policy",
        },
        overwrite=False,
        resume=False,
    )


def build_prefix_return_policy_train_config(
    *,
    clean_config: Any,
    clean_checkpoint: Path,
    view_spec: dict,
    experiment_name: str,
    queries_per_view: int = 4,
) -> Any:
    """Post-train the native VLM/action expert with balanced D20 forecast-prefix data."""
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader

    from latency_meta_mdp.legacy.policy.openpi_belief_adapter import (
        NativePolicyWithReturnBeliefLoader,
    )
    from latency_meta_mdp.legacy.policy.openpi_belief_data import PrefixReturnBeliefDataConfig

    if not hasattr(clean_config.model, "use_return_belief_prefix"):
        raise ValueError("forecast-prefix training requires OpenPI patch0006")
    view = {**view_spec, "conditioning": "prefix"}
    base = build_return_policy_train_config(
        clean_config=clean_config,
        clean_checkpoint=clean_checkpoint,
        view_spec=view,
        experiment_name=experiment_name,
        adapter_only=False,
    )
    return dataclasses.replace(
        base,
        name=f"{base.name}_prefix",
        model=dataclasses.replace(
            base.model,
            use_return_belief=False,
            use_return_belief_prefix=True,
            belief_prefix_queries_per_view=queries_per_view,
        ),
        data=PrefixReturnBeliefDataConfig(
            repo_id=clean_config.data.repo_id,
            base_config=clean_config.data.base_config,
            assets=AssetsConfig(
                assets_dir=str(clean_config.assets_dirs), asset_id=clean_config.data.repo_id
            ),
            return_policy_view=view,
        ),
        weight_loader=NativePolicyWithReturnBeliefLoader(
            CheckpointWeightLoader(str(clean_checkpoint)), parameter_key="return_belief_prefix"
        ),
        freeze_filter=nnx.Any(
            PathRegex("PaliGemma/img/.*"), PathRegex("PaliGemma/llm/embedder/.*")
        ),
        policy_metadata={
            **base.policy_metadata,
            "conditioning": "prefix",
            "prefix_tokens": 1 + 5 * (1 + 2 * queries_per_view),
            "training_delay_objective": "uniform_d20_nonempty_pairs",
            "trainable_scope": "prefix_vlm_action_expert",
        },
    )
