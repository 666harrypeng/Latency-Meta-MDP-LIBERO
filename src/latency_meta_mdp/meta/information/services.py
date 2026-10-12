"""Lifetime-scoped frozen services; clean bootstrap and conditioned VLA load separately."""

import dataclasses
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def frozen_services(root, config, *, bootstrap_only=False):
    assets = config["assets"]
    paths, identity = assets["paths"], assets["identity"]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR", str(root / "outputs/cache/jax-conveyor-conditioned")
    )

    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
    from latency_meta_mdp.policy.profile import load_sft_profile

    profile = load_sft_profile(root / "configs/contracts/policy/pi05_state16_h50.yaml")
    patches = tuple(sorted((root / "patches/openpi").glob("000[1-9]-*.patch")))
    checkpoint, clean = root / paths["conditioned"], root / paths["clean"]
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        from openpi.policies.policy_config import create_trained_policy
        from openpi.training.config import AssetsConfig

        from latency_meta_mdp.policy.openpi.training import (
            build_forecast_policy_train_config,
            build_task_train_config,
        )

        clean_config = build_task_train_config(
            profile,
            config_name="conveyor_information",
            repo_id="metamdp/conveyor_sort_surface",
            task_id="conveyor_sort",
            action_contract_id="panda_osc_pose_delta_conveyor_v2",
        )
        if bootstrap_only:
            yield SimpleNamespace(policy=create_trained_policy(clean_config, clean))
            return
        policy_config = build_forecast_policy_train_config(
            clean_config=clean_config,
            clean_checkpoint=clean / "params",
            forecast_identity=identity["forecast_identity"],
            experiment_name="conveyor-information",
        )
        policy_config = dataclasses.replace(
            policy_config,
            data=dataclasses.replace(
                policy_config.data,
                assets=AssetsConfig(
                    assets_dir=str(checkpoint / "assets"), asset_id="metamdp/conveyor_sort_surface"
                ),
            ),
        )
        print("Restoring frozen conditioned policy", flush=True)
        policy = create_trained_policy(
            policy_config, checkpoint, sample_kwargs={"rtc_max_guidance_weight": 5.0}
        )
        from latency_meta_mdp.belief.decoder.evaluation import load_visual_decoder
        from latency_meta_mdp.belief.jepa.config import load_action_conditioned_jepa_config
        from latency_meta_mdp.belief.jepa.corpus import load_jepa_proprio_normalization
        from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
        from latency_meta_mdp.belief.jepa.model import (
            DirectJepaPredictor,
            load_direct_prediction_weights,
        )
        from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder

        cfg = load_action_conditioned_jepa_config(
            model_path=root / "configs/models/jepa/model.yaml",
            temporal_sampling_path=root / "configs/models/jepa/stride4_80ms_history_160ms.yaml",
            task_id="conveyor_sort",
            control_path=root / "configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml",
        )
        norm = load_jepa_proprio_normalization(root / paths["normalization"])
        predictor = DirectJepaPredictor(
            backbone_config=cfg, proprio_normalization=norm, project_root=root
        )
        load_direct_prediction_weights(predictor, root / paths["predictor_weights"])
        engine = FrozenForecastEngine(
            predictor,
            load_visual_decoder(root / paths["decoder_dir"], device="cuda:0"),
            device="cuda:0",
        )
        encoder = HfDinoPatchEncoder.from_pretrained(
            spec=cfg.vision_encoder, device="cuda:0", local_files_only=True
        )
        yield SimpleNamespace(policy=policy, engine=engine, encoder=encoder, normalization=norm)
